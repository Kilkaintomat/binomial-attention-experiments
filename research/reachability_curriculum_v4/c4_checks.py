"""Allocation-only data/sampler tests, exact training resume, long resource profile."""
import argparse
import time
from c4_common import *
from c4_data import generate,augmented_graph
from transport_reasoning.reachability.config import allocation
from transport_reasoning.reachability.checks import timing,TinyDataset


def cpu():
    allocation();torch.set_num_threads(1);freeze();install();manifest=generate();rows=[]
    for family in FAMILIES:
        baseline=next(c for c in controls() if c.family==family and c.seed==11)
        original=Encoder(baseline)
        for regime in REGIMES:
            c=make(family,regime,device='cpu');model=Encoder(c)
            for (name,p),(other,q) in zip(model.named_parameters(),original.named_parameters()):assert name==other and torch.equal(p,q)
            data=SampledDataset(family,regime);rng=np.random.default_rng(5271);ids=rng.integers(len(data),size=200000)
            a,index,label,ell=data.mapping(ids)
            assert abs(float(label.mean())-.5)<.005
            assert abs(float(a.mean())-(0 if regime=='class_balanced' else .5))<.005
            assert all(np.array_equal(x,y) for x,y in zip(data.mapping(ids[:1000]),data.mapping(ids[:1000])))
            hist=np.bincount(ell[a],minlength=17)[1:].astype(float)
            if a.any():
                expected=data.short_prob if regime=='short_aug' else np.ones(16)/16
                assert np.max(np.abs(hist/hist.sum()-expected))<.008
            inputs,y,meta=data.batch(ids[:64],'cpu');z=model(inputs)
            assert torch.isfinite(z).all();torch.nn.functional.binary_cross_entropy_with_logits(z,y).backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            for row,target in zip(meta,y.tolist()):assert (row['distance']>=0)==bool(target)
            rows.append({'family':family,'regime':regime,'positive_fraction':float(label.mean()),'augmented_fraction':float(a.mean()),
                'augmented_length_histogram':hist.tolist(),'sampled_examples':200000,'status':'passed'})
        short=SampledDataset(family,'short_aug');long=SampledDataset(family,'length_balanced')
        sa,si,sy,se=short.mapping(ids);la,li,ly,le=long.mapping(ids)
        assert np.array_equal(sa,la) and np.array_equal(sy,ly) and np.array_equal(si[~sa],li[~la])
        assert np.array_equal((si[sa]%2048)//2,(li[la]%2048)//2)
        assert short.pool.manifest['data_hash']==long.pool.manifest['data_hash']
    value={'status':'passed','source_hash':source_hash(),'pool_data_hash':manifest['data_hash'],'sampling':rows,
        'identical_initial_parameters':True,'CD_identical_pool_size_mix_original_examples_and_graph_index':True,
        'finite_forward_backward':True,'deterministic_stateless_sampler':True,'scheduler_job':os.environ['SLURM_JOB_ID']}
    write_json(ROOT/'cpu_preflight.json',value);print(json.dumps(value),flush=True)


def resume():
    saved=os.environ['REACH_ROOT'];c=make('er','length_balanced',seed=211,stage='smoke')
    data=SampledDataset('er','length_balanced');validation=TinyDataset()
    try:
        os.environ['REACH_ROOT']=str(ROOT/'checks/continuous');trainer.train(c,train_data=data,validation=validation)
        full=torch.load(c.directory()/'checkpoint_last.pt',map_location='cpu',weights_only=False)
        os.environ['REACH_ROOT']=str(ROOT/'checks/resumed');trainer.train(c,stop_after=20,train_data=data,validation=validation)
        trainer.train(c,train_data=data,validation=validation)
        second=torch.load(c.directory()/'checkpoint_last.pt',map_location='cpu',weights_only=False)
        assert full['step']==second['step']==40 and full['batch_rng']==second['batch_rng'] and full['scheduler']==second['scheduler']
        for k,v in full['model'].items():assert torch.equal(v,second['model'][k]),k
        for k,v in full['optimizer']['state'].items():
            for field,t in v.items():
                u=second['optimizer']['state'][k][field];assert torch.equal(t,u) if torch.is_tensor(t) else t==u
    finally:os.environ['REACH_ROOT']=saved
    return {'status':'passed','steps':40,'resume_at':20,'comparison':'bitwise model/optimizer/scheduler/batch RNG with actual mixed-size sampler'}


def gpu():
    allocation();freeze();install();assert read(ROOT/'cpu_preflight.json')['source_hash']==source_hash()
    restored=resume();rows=[]
    for family in FAMILIES:
        for regime in REGIMES:
            c=make(family,regime,seed=211,stage='smoke');data=SampledDataset(family,regime)
            smoke=trainer.train(c,train_data=data,validation=Dataset(family,'validation',limit_graphs=32))
            assert smoke['status']=='completed' and smoke['step']==40
            device=trainer.setup(c);model=Encoder(c).to(device);optimizer,scheduler=trainer.optimizer_for(model,c)
            rng=np.random.default_rng(551);times=[]
            for i in range(1000):
                trainer.sync(device);tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                inputs,y,_=data.batch(rng.integers(len(data),size=64),device)
                model.train();optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(model(inputs),y)
                assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();scheduler.step();trainer.sync(device)
                if i>=100:times.append(time.perf_counter()-tick)
            measurements=[timing(model,n,64,device,warmup=10,repeats=30) for n in [19,48,64,128,256]]
            step=float(np.mean(times));val=40000*measurements[0]['end_to_end_median_seconds']/64
            seconds=30000*step+30*val+60
            row={'family':family,'regime':regime,'step_seconds':step,'estimated_final_seconds':seconds,
                'warmup_steps':100,'measured_steps':900,'timing_samples_seconds':times,
                'inference_profile':measurements,'smoke_status':'passed','smoke_steps':40}
            rows.append(row);print(json.dumps({k:row[k] for k in ['family','regime','step_seconds','estimated_final_seconds']}),flush=True)
    # Reserve one full GPU-hour for40-checkpoint inference, larger test graphs and coefficient settings.
    estimate=5*sum(r['estimated_final_seconds'] for r in rows)+3600
    out={'status':'passed','source_hash':source_hash(),'resume':restored,'profile':rows,
        'estimated_full_seconds':estimate,'estimated_full_seconds_with_reserve':1.2*estimate,
        'inference_reserve_seconds':3600,'test_scores_used':False,'scheduler_job':os.environ['SLURM_JOB_ID']}
    write_json(ROOT/'gpu_preflight.json',out)
    print(json.dumps({k:v for k,v in out.items() if k not in ['profile']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','gpu']);a=p.parse_args();cpu() if a.mode=='cpu' else gpu()
