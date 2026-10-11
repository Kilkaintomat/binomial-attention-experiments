"""Resumable fixed-update training; test evaluation is a separate phase."""
import copy
from dataclasses import replace
import json
import os
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .config import Config,allocation,read,root,source_hash,write_json,file_hash
from .data import Dataset
from .model import Encoder
from .metrics import binary_metrics,slices


def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()


def setup(c):
    allocation();torch.set_num_threads(c.threads);random.seed(c.seed);np.random.seed(c.seed);torch.manual_seed(c.seed)
    if c.device=='cuda':
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one allocated GPU required')
        if not os.environ.get('SLURM_JOB_GPUS'):raise RuntimeError('Missing scheduler GPU allocation')
        torch.cuda.manual_seed_all(c.seed);torch.backends.cuda.matmul.allow_tf32=c.tf32;torch.backends.cudnn.allow_tf32=c.tf32
    return torch.device(c.device)


def optimizer_for(model,c):
    optimizer=torch.optim.AdamW(model.parameters(),lr=c.lr,weight_decay=c.weight_decay)
    warmup=round(c.planned_steps*c.warmup_fraction)
    def schedule(step):
        return (step+1)/warmup if step<warmup else .5*(1+np.cos(np.pi*min(1,(step-warmup)/(c.planned_steps-warmup))))
    return optimizer,torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)


def checkpoint(path,state):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    temp.replace(path)


@torch.no_grad()
def predict(model,data,device,batch_size=64,save_path=None,indices=None):
    model.eval();all_y=[];all_z=[];meta=[];max_norm=0.;started=time.perf_counter();forward_seconds=0.
    ids=np.arange(len(data)) if indices is None else np.asarray(indices)
    for offset in range(0,len(ids),batch_size):
        inputs,y,metadata=data.batch(ids[offset:offset+batch_size],device);sync(device);tick=time.perf_counter()
        logits=model(inputs);sync(device);forward_seconds+=time.perf_counter()-tick
        if not torch.isfinite(logits).all():raise FloatingPointError('Non-finite logits')
        if model.state_norms:
            norms=np.array(model.state_norms)
            if not np.isfinite(norms).all():raise FloatingPointError('Non-finite intermediate states')
            max_norm=max(max_norm,float(norms.max()))
        all_y.append(y.cpu().numpy());all_z.append(logits.cpu().numpy());meta.extend(metadata)
    labels=np.concatenate(all_y);logits=np.concatenate(all_z);metrics=binary_metrics(labels,logits)
    metrics.update(elapsed_seconds=time.perf_counter()-started,forward_seconds=forward_seconds,batch_size=batch_size,
                   max_observed_state_norm=max_norm if model.observe_states else None)
    if save_path:
        save_path.parent.mkdir(parents=True,exist_ok=True)
        with save_path.with_suffix('.npz.tmp').open('wb') as f:
            np.savez_compressed(f,labels=labels,logits=logits,graph_ids=np.array([r['graph_id'] for r in meta]),
                                query_ids=np.array([r['query_id'] for r in meta]),n=np.array([r['n'] for r in meta]),
                                distance=np.array([r['distance'] for r in meta]),weak_same=np.array([r['weak_same'] for r in meta]),
                                reverse=np.array([r['reverse'] for r in meta]),ell=np.array([r.get('ell') or -1 for r in meta]),
                                variant=np.array([r.get('variant') or '' for r in meta]))
        save_path.with_suffix('.npz.tmp').replace(save_path)
    return metrics,labels,logits,meta


def train(c,deadline=None,stop_after=None,train_data=None,validation=None):
    c.validate();device=setup(c);directory=c.directory();directory.mkdir(parents=True,exist_ok=True)
    train_data=train_data or Dataset(c.family,'train');validation=validation or Dataset(c.family,'validation')
    small_validation=validation if c.stage in ['overfit','smoke'] else Dataset(c.family,'validation',limit_graphs=128)
    src=source_hash();data_hash=train_data.manifest['data_hash'];result_path=directory/'result.json'
    if result_path.exists():
        old=read(result_path)
        if old['source_hash']!=src or old['config']!=c.dict() or old['data_hash']!=data_hash:raise RuntimeError('Existing result incompatible')
        if old['status']=='completed':return old
    model=Encoder(c).to(device);optimizer,scheduler=optimizer_for(model,c);rng=np.random.default_rng(c.seed+987123)
    step=0;best=(-1.,-float('inf'));best_step=None;last_validation=None;train_peak_bytes=0
    times={'training_seconds':0.,'validation_seconds':0.,'checkpoint_seconds':0.,'elapsed_seconds':0.}
    thresholds={str(t):{'status':'not_reached'} for t in [.8,.9,.95]}
    last=directory/'checkpoint_last.pt'
    if last.exists():
        state=torch.load(last,map_location='cpu',weights_only=False)
        if state['config']!=c.dict() or state['source_hash']!=src or state['data_hash']!=data_hash:raise RuntimeError('Checkpoint config/source/data mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        for values in optimizer.state.values():
            for key,value in values.items():
                if torch.is_tensor(value):values[key]=value.to(device)
        rng.bit_generator.state=state['batch_rng'];random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
        step=state['step'];times=state['times'];best=tuple(state['best']);best_step=state['best_step'];last_validation=state['last_validation'];thresholds=state['thresholds'];train_peak_bytes=state['train_peak_bytes']
        for filename in ['train.jsonl','validation.jsonl']:
            p=directory/filename
            if p.exists():p.write_text(''.join(line+'\n' for line in p.read_text().splitlines() if json.loads(line)['step']<=step))
    write_json(directory/'config.json',c.dict())
    started=time.perf_counter();previous=times['elapsed_seconds'];status='running'
    if c.device=='cuda':torch.cuda.reset_peak_memory_stats()

    def save(is_best=False):
        tick=time.perf_counter();times['elapsed_seconds']=previous+tick-started
        state={'config':c.dict(),'source_hash':src,'data_hash':data_hash,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
               'scheduler':scheduler.state_dict(),'batch_rng':rng.bit_generator.state,'python_rng':random.getstate(),
               'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if c.device=='cuda' else None,
               'step':step,'times':copy.deepcopy(times),'best':best,'best_step':best_step,'last_validation':last_validation,'thresholds':thresholds,'train_peak_bytes':train_peak_bytes}
        checkpoint(last,state)
        if is_best:checkpoint(directory/'checkpoint_best.pt',state)
        times['checkpoint_seconds']+=time.perf_counter()-tick

    target=min(c.completion_steps,stop_after) if stop_after else c.completion_steps
    every=1000 if c.stage=='final' else 500 if c.stage in ['pilot','confirmation'] else 100 if c.stage=='overfit' else 20
    while step<target:
        if deadline and time.perf_counter()>deadline-90:status='paused';save();break
        tick=time.perf_counter()
        if c.device=='cuda':torch.cuda.reset_peak_memory_stats()
        ids=rng.integers(len(train_data),size=c.batch_size);inputs,y,_=train_data.batch(ids,device)
        model.train();optimizer.zero_grad(set_to_none=True);logits=model(inputs);loss=F.binary_cross_entropy_with_logits(logits,y)
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c.clip,error_if_nonfinite=True)
        optimizer.step();scheduler.step();step+=1;sync(device);times['training_seconds']+=time.perf_counter()-tick
        if c.device=='cuda':train_peak_bytes=max(train_peak_bytes,torch.cuda.max_memory_allocated())
        if step==1 or step%250==0 or step==target:
            row={'step':step,'loss':float(loss),'gradient_norm':float(norm),'lr':optimizer.param_groups[0]['lr'],
                 'training_seconds':times['training_seconds'],'elapsed_seconds':previous+time.perf_counter()-started,
                 'train_accuracy':float(((logits>=0)==(y>=.5)).float().mean()),
                 'peak_memory_bytes':torch.cuda.max_memory_allocated() if c.device=='cuda' else None}
            with (directory/'train.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        validation_due=step%every==0 or step==c.completion_steps
        if validation_due:
            tick=time.perf_counter();full=c.stage not in ['pilot','confirmation'] or step==c.completion_steps
            metric,_,_,_=predict(model,validation if full else small_validation,device,c.batch_size)
            times['validation_seconds']+=time.perf_counter()-tick
            metric.update(step=step,full_validation=full,training_seconds=times['training_seconds'],elapsed_seconds=previous+time.perf_counter()-started)
            rank=(metric['balanced_accuracy'] if metric['balanced_accuracy'] is not None else -1.,-metric['bce'])
            improved=rank>best
            if improved:best=rank;best_step=step
            for threshold,entry in thresholds.items():
                if full and metric['balanced_accuracy'] is not None and metric['balanced_accuracy']>=float(threshold) and entry['status']=='not_reached':entry.update(status='reached',step=step,training_seconds=times['training_seconds'])
            last_validation=metric
            with (directory/'validation.jsonl').open('a') as f:f.write(json.dumps(metric,allow_nan=False)+'\n')
            save(improved)
        elif step%250==0:save()
    if step==c.completion_steps:status='completed'
    elif status=='running':status='paused'
    save();times['elapsed_seconds']=previous+time.perf_counter()-started
    result={'status':status,'config':c.dict(),'source_hash':src,'data_hash':data_hash,'step':step,'best_step':best_step,
            'best_validation_ba':best[0],'final_validation':last_validation,'times':times,'thresholds':thresholds,
            'operations':model.operations(),'examples_seen':step*c.batch_size,'batch_rng':rng.bit_generator.state,'train_peak_memory_bytes':train_peak_bytes or None,
            'device_name':torch.cuda.get_device_name() if c.device=='cuda' else 'cpu'}
    write_json(result_path,result);return result


def load_model(c):
    device=setup(c);result=read(c.directory()/'result.json')
    if result['status']!='completed' or result['step']!=c.completion_steps:raise RuntimeError('Unfinished training')
    state=torch.load(c.directory()/'checkpoint_best.pt',map_location='cpu',weights_only=False)
    if state['config']!=c.dict() or state['source_hash']!=source_hash():raise RuntimeError('Checkpoint mismatch')
    if state['data_hash']!=Dataset(c.family,'validation').manifest['data_hash']:raise RuntimeError('Data changed')
    model=Encoder(c).to(device);model.load_state_dict(state['model']);model.eval();return model,device,state


def evaluate_config(c,deadline=None):
    if c.stage!='final':raise ValueError('Test is for finals only')
    if not (root()/'final_selection.json').exists() or not (root()/'frozen_checkpoints.json').exists():raise RuntimeError('Final configs/checkpoints must be frozen before test')
    assert read(root()/'frozen_checkpoints.json')[c.hash()]['checkpoint_sha256']==file_hash(c.directory()/'checkpoint_best.pt')
    model,device,state=load_model(c);directory=c.directory()/'evaluation';directory.mkdir(exist_ok=True)
    workloads=[('fixed','test_ID',None,None,None),('fixed','test_near',None,None,None)]
    workloads += [('fixed','test_far',n,None,None) for n in [32,64,128]]+[('fixed','stress',None,None,None)]
    settings=([('hops',r) for r in [4,8,16,32]] if c.filter in ['binomial','exponential','geometric'] else [('outer',t) for t in [32,64,128]] if c.model=='R32-one' else [])
    for key,value in settings:
        workloads += [('extra','test_far',n,key,value) for n in [32,64,128]]+[('extra','stress',None,key,value)]
    for regime,split,n,key,value in workloads:
        name='_'.join(str(v) for v in [regime,split,n,key,value] if v is not None);path=directory/(name+'.json')
        if path.exists():continue
        if deadline and time.perf_counter()>deadline-180:return False
        limit=(50 if split=='stress' else 500) if regime=='extra' else None
        data=Dataset('stress' if split=='stress' else c.family,split,n=n,limit_graphs=limit)
        model=Encoder(c).to(device);model.load_state_dict(state['model']);model.eval()
        if key=='hops':model.extra_compute(hops=value)
        elif key=='outer':model.extra_compute(outer_steps=value)
        metric,y,z,metadata=predict(model,data,device,c.batch_size,directory/(name+'.npz'))
        groups={}
        for i,row in enumerate(metadata):
            group=(row['n'],row.get('ell'),row.get('variant'));groups.setdefault(group,[]).append(i)
        by_group=[]
        for (size,ell,variant),ids in groups.items():by_group.append({'n':size,'ell':ell,'variant':variant,**binary_metrics(y[ids],z[ids])})
        from .checks import timing
        timing_rows=[timing(model,size,batch,device) for size in sorted({r['n'] for r in metadata}) for batch in [1,64]]
        metric.update(regime=regime,split=split,n=n,setting=key,value=value,subset='fixed first templates/graphs' if limit else 'full',
                      checkpoint_step=state['step'],source_hash=source_hash(),data_hash=state['data_hash'],operations=model.operations(),
                      by_size_or_stress_cell=by_group,path_length_slices=slices(y,z,metadata),timing=timing_rows,
                      device_name=torch.cuda.get_device_name() if c.device=='cuda' else 'cpu')
        write_json(path,metric)
    return True
