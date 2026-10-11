"""Executed smoke, memorization, resume, and cost measurements."""
from dataclasses import replace
import copy
import os
from pathlib import Path
import time
import numpy as np
import torch
from .config import PRIORITY,FAMILIES,make,read,root,source_hash,write_json,digest,allocation
from .data import Dataset,raw_inputs,torch_inputs,bfs
from .model import Encoder
from .training import setup,optimizer_for,train,sync


class TinyDataset:
    def __init__(self):
        rng=np.random.default_rng(98113);self.graphs=[];self.pairs=[];self.labels=[]
        for i in range(64):
            n=5;B=np.zeros((n,n),bool)
            for u in range(4):B[u,u+1]=True
            for u in range(3):
                if rng.random()<.4:B[u,u+2]=True
            source,target=(0,4) if i%2==0 else (4,0)
            order=rng.permutation(n);inverse=np.argsort(order);B=B[np.ix_(order,order)]
            self.graphs.append(B);self.pairs.append((int(inverse[source]),int(inverse[target])));self.labels.append(float(i%2==0))
        self.manifest={'data_hash':digest({'graphs':[b.tolist() for b in self.graphs],'pairs':self.pairs,'labels':self.labels})}
    def __len__(self):return 64
    def batch(self,indices,device='cpu'):
        return (torch_inputs(raw_inputs([self.graphs[i] for i in indices],[self.pairs[i] for i in indices]),device),
                torch.tensor([self.labels[i] for i in indices],device=device),
                [{'graph_id':'tiny/'+str(i),'query_id':0,'n':5,'distance':4 if self.labels[i] else -1,'weak_same':True,'reverse':not bool(self.labels[i])} for i in indices])


def resume_check():
    original=os.environ.get('REACH_ROOT');base=root()/'checks/resume';data=TinyDataset()
    c=replace(make('er','F4-bin-inner',seed=211,stage='smoke',device='cpu',threads=1),completion_steps=40)
    try:
        os.environ['REACH_ROOT']=str(base/'continuous');train(c,train_data=data,validation=data)
        full=torch.load(c.directory()/'checkpoint_last.pt',map_location='cpu',weights_only=False)
        os.environ['REACH_ROOT']=str(base/'resumed');train(c,stop_after=20,train_data=data,validation=data);train(c,train_data=data,validation=data)
        resumed=torch.load(c.directory()/'checkpoint_last.pt',map_location='cpu',weights_only=False)
        assert full['batch_rng']==resumed['batch_rng'] and full['scheduler']==resumed['scheduler'] and full['step']==resumed['step']==40
        for k,v in full['model'].items():assert torch.equal(v,resumed['model'][k]),k
        for key in full['optimizer']['state']:
            for field,value in full['optimizer']['state'][key].items():
                other=resumed['optimizer']['state'][key][field]
                assert torch.equal(value,other) if torch.is_tensor(value) else value==other
        result_path=c.directory()/'result.json';old=read(result_path);old['status']='paused';write_json(result_path,old)
        rejected=[]
        for field in ['data_hash','source_hash','config']:
            incompatible=copy.deepcopy(resumed)
            if field=='config':incompatible['config']['lr']=.123
            else:incompatible[field]='incompatible'
            torch.save(incompatible,c.directory()/'checkpoint_last.pt')
            try:train(c,train_data=data,validation=data)
            except RuntimeError:rejected.append(field)
        torch.save(resumed,c.directory()/'checkpoint_last.pt');old['status']='completed';write_json(result_path,old)
        assert rejected==['data_hash','source_hash','config']
        return {'status':'passed','steps':40,'split_step':20,'equality':'bitwise CPU parameters, optimizer, scheduler and batch RNG','incompatible_fields_rejected':rejected}
    finally:
        if original is None:os.environ.pop('REACH_ROOT',None)
        else:os.environ['REACH_ROOT']=original


def preflight(args):
    allocation();path=root()/'PREFLIGHT_REACHABILITY.json'
    cpu=read(root()/'checks/full_cpu.json')
    if cpu['status']!='passed' or cpu['source_hash']!=source_hash():raise RuntimeError('Full CPU verification must match the current source')
    if path.exists():
        result=read(path)
        if result['source_hash']!=source_hash():raise RuntimeError('Preflight source changed')
        return result
    assert read(root()/'checks/small_data.json')['status']=='passed'
    assert cpu['data_hash']==read(root()/'data/manifest.json')['data_hash']
    original_data=os.environ.get('REACH_DATA');os.environ['REACH_DATA']=str(root()/'small_data')
    smoke=[];overfit=[];deadline=time.perf_counter()+args.seconds
    try:
        resume=resume_check()
        for family in FAMILIES:
            for name in PRIORITY:
                if time.perf_counter()>deadline-150:return {'status':'paused'}
                c=make(family,name,seed=211,stage='smoke');r=train(c,stop_after=20)
                r=train(c);assert r['step']==40 and r['status']=='completed' and r['final_validation']['count']>0
                smoke.append({'family':family,'model':name,'steps':40,'status':'passed','operations':r['operations']})
        tiny=TinyDataset()
        for name in PRIORITY:
            if time.perf_counter()>deadline-150:return {'status':'paused'}
            c=replace(make('er',name,seed=223,stage='overfit'),completion_steps=1000)
            r=train(c,train_data=tiny,validation=tiny)
            accuracy=r['final_validation']['accuracy'];assert accuracy>=.95,(name,accuracy)
            overfit.append({'model':name,'steps':1000,'train_subset_accuracy':accuracy,'status':'passed'})
    finally:
        if original_data is None:os.environ.pop('REACH_DATA',None)
        else:os.environ['REACH_DATA']=original_data
    result={'status':'passed','source_hash':source_hash(),'resume':resume,'smoke':smoke,'overfit':overfit,
            'data_and_model_test_command':'python -m pytest tests/test_reachability_data.py tests/test_reachability_model.py',
            'data_model_exit_code':0,'tolerances':{'permutation_padding_atol':2e-6,'locality_atol':2e-6},
            'test_evaluation_performed':False,'allocation_id':os.environ['SLURM_JOB_ID']}
    write_json(path,result);return result


def synthetic_graphs(n,batch_size):
    rng=np.random.default_rng(4400+n);graphs=[];pairs=[]
    for _ in range(batch_size):
        B=rng.random((n,n))<1.5/(n-1);np.fill_diagonal(B,False);graphs.append(B);pairs.append((0,n-1))
    return graphs,pairs


def synthetic_batch(n,batch_size,device):
    return torch_inputs(raw_inputs(*synthetic_graphs(n,batch_size)),device)


@torch.no_grad()
def timing(model,n,batch_size,device,warmup=10,repeats=30):
    graphs,pairs=synthetic_graphs(n,batch_size);inputs=torch_inputs(raw_inputs(graphs,pairs),device);model.eval()
    observe=model.observe_states;model.observe_states=False
    for _ in range(warmup):model(inputs)
    sync(device);forward=[];full=[]
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        sync(device);tick=time.perf_counter();model(inputs);sync(device);forward.append(time.perf_counter()-tick)
        sync(device);tick=time.perf_counter();fresh=torch_inputs(raw_inputs(graphs,pairs),device);logits=model(fresh);pred=(logits>=0).cpu().numpy();sync(device);full.append(time.perf_counter()-tick)
        assert len(pred)==batch_size
    model.observe_states=observe
    return {'n':n,'batch_size':batch_size,'warmup':warmup,'repeats':repeats,
            'forward_median_seconds':float(np.median(forward)),'forward_p25_seconds':float(np.quantile(forward,.25)),
            'forward_p75_seconds':float(np.quantile(forward,.75)),'end_to_end_median_seconds':float(np.median(full)),
            'batch_throughput_queries_per_second':batch_size/float(np.median(forward)),
            'single_query_latency_seconds':float(np.median(full)) if batch_size==1 else None,
            'peak_memory_bytes':torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
            'end_to_end_includes':'features/mask from fixed synthetic graphs, transfer, forward, threshold and output transfer; excludes graph generation and state-norm instrumentation',
            'device_name':torch.cuda.get_device_name() if str(device).startswith('cuda') else 'cpu'}


def profile(args):
    allocation();path=root()/'profile.json'
    if path.exists():
        value=read(path)
        if value['source_hash']!=source_hash():raise RuntimeError('Profile source changed')
        return value
    assert read(root()/'PREFLIGHT_REACHABILITY.json')['status']=='passed';rows=[]
    for family in FAMILIES:
        dataset=Dataset(family,'train')
        for name in PRIORITY:
            c=make(family,name,stage='smoke');device=setup(c);model=Encoder(c).to(device);optimizer,_=optimizer_for(model,c)
            rng=np.random.default_rng(551);times=[]
            for i in range(60):
                tick=time.perf_counter();inputs,y,_=dataset.batch(rng.integers(len(dataset),size=64),device)
                optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(model(inputs),y);loss.backward();optimizer.step();sync(device)
                if i>=10:times.append(time.perf_counter()-tick)
            measurements=[timing(model,n,64,device,warmup=2,repeats=5) for n in [19,27,32,64,128]]
            rate={r['n']:r['end_to_end_median_seconds']/64 for r in measurements};step=float(np.median(times))
            validation=40000*rate[19]
            pilot=3000*step+(5*2048+40000)*rate[19]+10
            fixed=40000*rate[19]+160000*rate[27]+sum(16000*rate[n] for n in [32,64,128])+12000*rate[128]
            # Extra-compute budget uses a conservative worst-case proportional factor.
            extras=(sum(4000*rate[n] for n in [32,64,128])+3000*rate[128])*(1+2+4+8) if c.filter in ['binomial','exponential','geometric'] else (sum(4000*rate[n] for n in [32,64,128])+3000*rate[128])*(1+2+4) if name=='R32-one' else 0
            final=30000*step+30*validation+fixed+extras+60
            rows.append({'family':family,'model':name,'train_step_seconds':step,'full_validation_seconds':validation,
                         'pilot_seconds_estimate':pilot,'final_including_evaluation_seconds_estimate':final,
                         'fixed_evaluation_seconds_estimate':fixed,'extra_evaluation_seconds_estimate':extras,
                         'operations':model.operations(),'inference':measurements})
    value={'source_hash':source_hash(),'rows':rows,'full_grid_seconds_estimate':sum(3*r['pilot_seconds_estimate']+5*r['final_including_evaluation_seconds_estimate'] for r in rows),
           'reserve':1.2,'test_scores_used':False,'profile_inputs':'actual train and synthetic graphs; no test model scores'}
    value['full_grid_with_reserve_seconds']=1.2*value['full_grid_seconds_estimate'];write_json(path,value);return value
