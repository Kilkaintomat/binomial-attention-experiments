"""Fixed pilots/finals, frozen validation calibration, then one untouched test grid."""
import argparse
import time
from fair_common import *
from transport_reasoning.reachability.config import allocation
from transport_reasoning.reachability.metrics import binary_metrics, slices
from transport_reasoning.reachability.checks import timing


def calibrated_threshold(labels,logits):
    y=np.asarray(labels,dtype=np.int64);z=np.asarray(logits,dtype=np.float64)
    assert len(y)>0 and np.isfinite(z).all() and set(np.unique(y))=={0,1}
    order=np.argsort(z,kind='stable');zs=z[order];ys=y[order]
    prefix=np.r_[0,np.cumsum(ys)];p=int(y.sum());n=len(y)-p
    candidates=np.unique(np.r_[0.,zs,np.nextafter(zs[-1],np.inf)])
    cut=np.searchsorted(zs,candidates,side='left');tp=p-prefix[cut];tn=cut-prefix[cut]
    score=tp*n+tn*p
    best=np.flatnonzero(score==score.max())
    i=min(best,key=lambda i:(abs(candidates[i]),candidates[i]))
    return float(candidates[i]),{'validation_BA':float(score[i]/(2*p*n)),'positive_count':p,'negative_count':n,
        'candidate_count':len(candidates),'selection':'maximum BA, closest to0, lower threshold'}


def threshold_metrics(y,z,threshold):
    m=binary_metrics(y,np.asarray(z,dtype=np.float64)-threshold)
    return {k:m[k] for k in ['count','positive_count','negative_count','balanced_accuracy','accuracy','precision','recall','specificity','f1','tp','tn','fp','fn']}


def freeze_checkpoints():
    out={}
    for c,base in all_final_configs():
        directory=base_directory(c) if base else c.directory();result=read(directory/'result.json')
        assert result['status']=='completed' and result['step']==30000
        sha=file_hash(directory/'checkpoint_best.pt')
        if base:assert sha==read(BASE_ROOT/'frozen_checkpoints.json')[c.hash()]['checkpoint_sha256']
        out[c.hash()]={'config':c.dict(),'baseline':base,'checkpoint_sha256':sha,'step':result['best_step']}
    assert len(out)==80
    p=OUT/'frozen_checkpoints.json'
    if p.exists():assert read(p)==out
    else:write_json(p,out)
    return out


def evaluate(deadline):
    frozen=freeze_checkpoints();manifest=read(OUT/'data/manifest.json');cache={}
    configs=sorted(all_final_configs(),key=lambda row:(row[0].seed,row[0].family,row[0].model))
    done=0
    for c,baseline in configs:
        paths=[eval_path(c,split,n) for split,n in workloads()]
        if all(p.exists() for p in paths):
            for p in paths:
                row=read(p)
                assert row['source_hash']==extension_hash() and row['data_hash']==manifest['data_hash']
                assert row['checkpoint_sha256']==frozen[c.hash()]['checkpoint_sha256']
                assert file_hash(p.with_suffix('.npz'))==row['predictions_sha256']
            done+=4;continue
        if time.perf_counter()>deadline-180:return 'paused'
        device=trainer.setup(c);model,record=load_checkpoint(c,baseline,device)
        before=learned_hash(model);model.observe_states=True
        calpath=OUT/'calibration'/c.family/c.model/('seed_'+str(c.seed)+'.json')
        if calpath.exists():
            calibration=read(calpath);assert calibration['checkpoint_sha256']==record['checkpoint_sha256']
        else:
            val=Dataset(c.family,'validation')
            with torch.inference_mode():metric,y,z,_=trainer.predict(model,val,device,64,calpath.with_suffix('.npz'))
            threshold,info=calibrated_threshold(y,z)
            calibration={'threshold':threshold,**info,'source_split':'original validation18/19 only',
                'checkpoint_sha256':record['checkpoint_sha256'],'source_hash':extension_hash(),
                'predictions_sha256':file_hash(calpath.with_suffix('.npz')),'default_validation':metric}
            write_json(calpath,calibration)
        timpath=OUT/'timing'/c.family/c.model/('seed_'+str(c.seed)+'.json')
        if timpath.exists():
            timerows=read(timpath)['rows']
        else:
            timerows=[timing(model,n,batch,device,warmup=10,repeats=30) for n in [32,64,128] for batch in [1,64]]
            write_json(timpath,{'rows':timerows,'checkpoint_sha256':record['checkpoint_sha256'],'source_hash':extension_hash()})
        for split,n in workloads():
            path=eval_path(c,split,n)
            if path.exists():
                row=read(path);assert row['predictions_sha256']==file_hash(path.with_suffix('.npz'));done+=1;continue
            if time.perf_counter()>deadline-150:return 'paused'
            key=('stress' if split=='stress' else c.family,split,n)
            if key not in cache:cache[key]=TestDataset(*key)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():metric,y,z,meta=trainer.predict(model,cache[key],device,64,path.with_suffix('.npz'))
            assert learned_hash(model)==before
            groups={};by_group=[]
            for i,m in enumerate(meta):groups.setdefault((m['n'],m['ell'],m['variant']),[]).append(i)
            threshold=calibration['threshold']
            for (size,ell,variant),ids in groups.items():
                ids=np.array(ids);cell={'n':size,'ell':ell,'variant':variant,**binary_metrics(y[ids],z[ids]),
                    'validation_threshold_metrics':threshold_metrics(y[ids],z[ids],threshold)}
                if split=='stress':
                    assert np.array_equal(y[ids].reshape(-1,2),np.tile([1.,0.],(len(ids)//2,1)))
                    assert all(meta[int(a)]['graph_id']==meta[int(b)]['graph_id'] for a,b in ids.reshape(-1,2))
                    zz=z[ids].astype(np.float64);margin=zz.reshape(-1,2)[:,0]-zz.reshape(-1,2)[:,1]
                    cell.update(both_queries_correct=float(((zz>=0)==(y[ids]==1)).reshape(-1,2).all(1).mean()),
                        both_queries_correct_calibrated=float(((zz>=threshold)==(y[ids]==1)).reshape(-1,2).all(1).mean()),
                        logit_margin={'median':float(np.median(margin)),'p10':float(np.quantile(margin,.1)),
                            'p90':float(np.quantile(margin,.9)),'minimum':float(margin.min()),'maximum':float(margin.max()),
                            'fraction_positive':float((margin>0).mean()),'fraction_gt_1e-6':float((margin>1e-6).mean())})
                by_group.append(cell)
            metric.update(family=c.family,model=c.model,seed=c.seed,baseline_reused=baseline,config=c.dict(),split=split,n=n,
                source_hash=extension_hash(),training_source_hash=record['training_source_hash'],data_hash=manifest['data_hash'],
                checkpoint_sha256=record['checkpoint_sha256'],checkpoint_step=record['step'],learned_parameter_sha256=before,
                learned_parameters_unchanged=True,operations=model.operations(),radius=c.outer_steps*c.hops,
                coefficients=[b.fixed_coeff.detach().cpu().tolist() for b in model.blocks],
                by_group=by_group,path_length_slices=slices(y,z,meta),validation_threshold=threshold,
                validation_threshold_metrics=threshold_metrics(y,z,threshold),timing=timerows,
                peak_memory_bytes=torch.cuda.max_memory_allocated(),device_name=torch.cuda.get_device_name(),
                predictions_sha256=file_hash(path.with_suffix('.npz')),scheduler_job=os.environ['SLURM_JOB_ID'])
            write_json(path,metric);done+=1
            write_json(OUT/'progress.json',{'stage':'evaluation','completed':done,'expected':320,'family':c.family,'model':c.model,'seed':c.seed})
        assert file_hash((base_directory(c) if baseline else c.directory())/'checkpoint_best.pt')==record['checkpoint_sha256']
        del model;torch.cuda.empty_cache()
    assert done==320
    write_json(OUT/'completion.json',{'status':'completed','new_finals':30,'reused_finals':50,'pilots':18,'evaluations':320,
        'source_hash':extension_hash(),'data_hash':manifest['data_hash'],'scheduler_job':os.environ['SLURM_JOB_ID']})
    return 'completed'


def run(seconds,remaining):
    allocation();freeze_protocol();install_training_adapter();deadline=time.perf_counter()+seconds
    check=read(OUT/'gpu_preflight.json');assert check['status']=='passed' and check['source_hash']==extension_hash()
    if not (OUT/'admission.json').exists():
        allowed=check['estimated_full_seconds_with_reserve']<=remaining
        write_json(OUT/'admission.json',{'status':'admitted' if allowed else 'budget_stopped','estimated_seconds_with_reserve':check['estimated_full_seconds_with_reserve'],
            'remaining_seconds':remaining,'models':NEW_MODELS,'families':FAMILIES,'seeds':SEEDS,'test_scores_used':False})
    if read(OUT/'admission.json')['status']!='admitted':return 'budget_stopped'
    selected={};pilot_rows=[]
    for family in FAMILIES:
        for name in NEW_MODELS:
            candidates=[]
            for lr in [.0003,.001]:
                if time.perf_counter()>deadline-120:return 'paused'
                c=make(family,name,101,'pilot',lr)
                write_json(OUT/'progress.json',{'stage':'pilot','family':family,'model':name,'lr':lr,'config_hash':c.hash()})
                result=trainer.train(c,deadline=deadline)
                if result['status']!='completed':return 'paused'
                m=result['final_validation'];assert m['full_validation']
                candidates.append((m['balanced_accuracy'],-m['bce'],-lr,lr));pilot_rows.append({'config':c.dict(),'validation':m})
            selected[family+'/'+name]=max(candidates)[-1]
    write_json(OUT/'pilot_selection.json',{'LRs':selected,'pilots':pilot_rows,'test_scores_used':False,'source_hash':extension_hash()})
    for family in FAMILIES:
        for name in NEW_MODELS:
            if time.perf_counter()>deadline-120:return 'paused'
            c=make(family,name,103,'confirmation',selected[family+'/'+name])
            write_json(OUT/'progress.json',{'stage':'confirmation','family':family,'model':name,'config_hash':c.hash()})
            result=trainer.train(c,deadline=deadline)
            if result['status']!='completed':return 'paused'
    finals=[make(family,name,seed,'final',selected[family+'/'+name]) for seed in SEEDS for name in NEW_MODELS for family in FAMILIES]
    selection={'configs':[c.dict() for c in finals],'source_hash':extension_hash(),'before_new_test_predictions':True}
    p=OUT/'final_selection.json'
    if p.exists():assert read(p)==selection
    else:write_json(p,selection)
    for c in finals:
        if time.perf_counter()>deadline-120:return 'paused'
        write_json(OUT/'progress.json',{'stage':'final','family':c.family,'model':c.model,'seed':c.seed,'config_hash':c.hash()})
        result=trainer.train(c,deadline=deadline)
        if result['status']!='completed':return 'paused'
    return evaluate(deadline)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,required=True);p.add_argument('--remaining',type=int,required=True);a=p.parse_args()
    result=run(a.seconds,a.remaining)
    write_json(OUT/('slot_'+os.environ['SLURM_JOB_ID']+'.json'),{'status':result})
    print(json.dumps({'status':result}),flush=True)
