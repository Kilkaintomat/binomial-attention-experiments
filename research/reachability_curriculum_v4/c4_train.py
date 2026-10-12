"""Fixed-step data ablation; never modify or inspect the current fair-v3 tests."""
import argparse
import time
from c4_common import *
from transport_reasoning.reachability.config import allocation


def run(seconds,remaining):
    allocation();freeze();install();deadline=time.perf_counter()+seconds
    check=read(ROOT/'gpu_preflight.json');assert check['status']=='passed' and check['source_hash']==source_hash()
    if not (ROOT/'admission.json').exists():
        admitted=check['estimated_full_seconds_with_reserve']<=remaining
        write_json(ROOT/'admission.json',{'status':'admitted' if admitted else 'budget_stopped',
            'estimated_seconds_with_reserve':check['estimated_full_seconds_with_reserve'],'remaining_seconds':remaining,
            'finals':30,'controls':10,'test_scores_used':False})
    if read(ROOT/'admission.json')['status']!='admitted':return 'budget_stopped'
    configs=[make(f,r,s) for s in SEEDS for r in REGIMES for f in FAMILIES]
    selection={'configs':[c.dict() for c in configs],'source_hash':source_hash(),'checkpoint':'last30000','test_scores_used':False}
    if (ROOT/'final_selection.json').exists():assert read(ROOT/'final_selection.json')==selection
    else:write_json(ROOT/'final_selection.json',selection)
    cache={}
    for c in configs:
        if time.perf_counter()>deadline-120:return 'paused'
        existing=c.directory()/'result.json'
        if existing.exists():
            old=read(existing)
            assert old['source_hash']==source_hash() and old['config']==c.dict()
            if old['status']=='completed':continue
        key=(c.family,c.regime)
        if key not in cache:cache[key]=SampledDataset(*key)
        write_json(ROOT/'progress.json',{'stage':'final','family':c.family,'regime':c.regime,'seed':c.seed,'config_hash':c.hash()})
        result=trainer.train(c,deadline=deadline,train_data=cache[key],validation=Dataset(c.family,'validation'))
        if result['status']!='completed':return 'paused'
    frozen={}
    for c in configs:
        state=torch.load(c.directory()/'checkpoint_last.pt',map_location='cpu',weights_only=False)
        assert state['step']==30000 and state['source_hash']==source_hash() and state['config']==c.dict()
        frozen[c.hash()]={'config':c.dict(),'regime':c.regime,'step':30000,'checkpoint_sha256':file_hash(c.directory()/'checkpoint_last.pt')}
    for c in controls():
        p=control_directory(c)/'checkpoint_last.pt';state=torch.load(p,map_location='cpu',weights_only=False)
        assert state['step']==30000 and state['source_hash']==BASE_SOURCE and state['data_hash']==BASE_DATA and state['config']==c.dict()
        frozen[c.hash()]={'config':c.dict(),'regime':'original','step':30000,'checkpoint_sha256':file_hash(p)}
    assert len(frozen)==40
    if (ROOT/'frozen_checkpoints.json').exists():assert read(ROOT/'frozen_checkpoints.json')==frozen
    else:write_json(ROOT/'frozen_checkpoints.json',frozen)
    write_json(ROOT/'training_completed.json',{'status':'completed','new_finals':30,'controls':10,'source_hash':source_hash()})
    return 'training_completed'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,required=True);p.add_argument('--remaining',type=int,required=True);a=p.parse_args()
    status=run(a.seconds,a.remaining)
    write_json(ROOT/('slot_'+os.environ['SLURM_JOB_ID']+'.json'),{'status':status});print(json.dumps({'status':status}),flush=True)
