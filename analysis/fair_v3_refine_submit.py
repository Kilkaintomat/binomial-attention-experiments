"""One bounded timing refinement allocation, recorded in the original10h ledger."""
import fcntl
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(os.environ['FAIR_PROJECT'])/'research/reachability_fair_v3'))
from fair_queue import OUT, PROJECT, state, save, read, command, submit_job

with (OUT/'queue/lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    value=state();assert value['status']=='budget_stopped'
    assert all(r['accounted'] for r in value['gpu_jobs'])
    assert not any(r['mode']=='profile_refinement' for r in value['gpu_jobs'])
    remaining=read(OUT/'queue/budget_grant.json')['gpu_seconds_limit']-value['gpu_seconds_used']
    assert remaining>=600
    value['previous_admission_stop_reason']=value.pop('reason',None)
    job=command(['sbatch','--parsable','--job-name=fair3_long_profile','--account='+os.environ['RUN_ACCOUNT'],
        '--partition='+os.environ['RUN_GPU_PARTITION'],'--gres='+os.environ['RUN_GPU_GRES'],
        '--cpus-per-task=4','--mem=8G','--time=10','--chdir='+str(PROJECT),
        '--output='+str(PROJECT/'.cache/long_profile_%j.out'),str(PROJECT/'analysis/fair_v3_refine_host.sh')]).split(';')[0]
    value['gpu_jobs'].append({'job_id':job,'mode':'profile_refinement','reserved_seconds':600,'accounted':False})
    value.update(status='queued',active_gpu_job=job);save(value)
    value['accounting_job']=submit_job('account',seconds=180,dependency=job,argument=job);save(value)
    print(json.dumps(value),flush=True)
