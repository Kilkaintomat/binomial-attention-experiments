"""Pre-test resource estimate refinement; preserve original preflight and all scientific code.

The initial 60-step timing gave inconsistent costs between equal-shaped ER/DAG
workloads (terminal filter15.4 vs24.8ms). Measure900 consecutive updates after100
warmup, using the same device/dtype/threads/optimizer/batch as the frozen trainer.
This diagnostic does not select hyperparameters or evaluate new test outcomes.
"""
import time
from fair_common import *
from transport_reasoning.reachability.config import allocation
from transport_reasoning.reachability.checks import timing


def refine():
    allocation();freeze_protocol();install_training_adapter()
    old=read(OUT/'gpu_preflight.json');assert old['status']=='passed' and 'refinement' not in old
    records=[]
    for family in FAMILIES:
        dataset=Dataset(family,'train')
        for name in NEW_MODELS:
            c=make(family,name,seed=211,stage='smoke');device=trainer.setup(c);model=Encoder(c).to(device)
            optimizer,scheduler=trainer.optimizer_for(model,c);rng=np.random.default_rng(551)
            samples=[]
            for i in range(1000):
                trainer.sync(device);tick=time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                inputs,y,_=dataset.batch(rng.integers(len(dataset),size=64),device)
                model.train();optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(model(inputs),y)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();scheduler.step();trainer.sync(device)
                if i>=100:samples.append(time.perf_counter()-tick)
            measurements=[timing(model,n,64,device,warmup=10,repeats=30) for n in [19,32,64,128]]
            step=float(np.mean(samples));rate={r['n']:r['end_to_end_median_seconds']/64 for r in measurements}
            val=40000*rate[19];pilot=3000*step+(5*2048+40000)*rate[19]+10
            final=30000*step+30*val+60
            evaluation=sum(4000*rate[n] for n in [32,64,128])+9600*rate[128]+val+8
            row={'family':family,'model':name,'step_seconds':step,'pilot_seconds':pilot,'final_seconds':final,
                'evaluation_seconds':evaluation,'operations':model.operations(),'profile_inference':measurements,
                'timing_samples_seconds':samples,'warmup_steps':100,'measured_steps':900,
                'timing_estimator':'arithmetic mean of900 consecutive actual optimizer/scheduler updates',
                'block_means_seconds':[float(np.mean(samples[j:j+100])) for j in range(0,900,100)]}
            records.append(row)
            print(json.dumps({k:row[k] for k in ['family','model','step_seconds','block_means_seconds']}),flush=True)
    estimate=sum(3*r['pilot_seconds']+5*(r['final_seconds']+r['evaluation_seconds']) for r in records)+1800
    refinement={'status':'passed','profile':records,'estimated_full_seconds':estimate,
        'estimated_full_seconds_with_reserve':1.2*estimate,'test_scores_used':False,
        'scientific_source_hash':extension_hash(),'diagnostic_source_hash':file_hash(__file__),
        'scheduler_job':os.environ['SLURM_JOB_ID'],'original_preflight_job':old['scheduler_job'],
        'reason':'First short profile costs varied markedly for equal-shaped workloads; longer mean estimator before admission.'}
    write_json(OUT/'profile_refinement.json',refinement)
    # The scientific protocol, grid, admission reserve and initial evidence remain immutable.
    # The admission reader uses the current estimate with an explicit diagnostic provenance.
    original_path=OUT/'gpu_preflight_initial.json';assert not original_path.exists()
    write_json(original_path,old)
    revised={**old,'profile':records,'estimated_full_seconds':estimate,'estimated_full_seconds_with_reserve':1.2*estimate,
        'refinement':{k:v for k,v in refinement.items() if k!='profile'},
        'initial_preflight_record_sha256':file_hash(original_path)}
    write_json(OUT/'gpu_preflight.json',revised)
    print(json.dumps({k:v for k,v in refinement.items() if k!='profile'}),flush=True)


if __name__=='__main__':refine()
