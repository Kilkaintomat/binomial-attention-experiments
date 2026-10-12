"""Serial short Slurm allocations with an explicit 10 GPU-hour ceiling."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

PROJECT=Path(os.environ['FAIR_PROJECT'])
OUT=PROJECT/'experiments/fair_v3'


def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def command(args):return subprocess.check_output(args,text=True).strip()
def state():return read(OUT/'queue/state.json') if (OUT/'queue/state.json').exists() else {'status':'prepared','gpu_jobs':[],'gpu_seconds_used':0}
def save(value):write(OUT/'queue/state.json',value)


def submit_job(mode,seconds=1800,gpu=False,dependency=None,argument=None):
    args=['sbatch','--parsable','--job-name=fair3_'+mode,'--account='+os.environ['RUN_ACCOUNT'],
          '--partition='+os.environ['RUN_GPU_PARTITION' if gpu else 'RUN_CPU_PARTITION'],
          '--cpus-per-task='+('4' if gpu else '1'),'--mem='+('8G' if gpu else '4G'),
          '--time='+str(math.ceil(seconds/60)),'--chdir='+str(PROJECT),
          '--output='+str(PROJECT/'.cache'/(mode+'_%j.out'))]
    if gpu:args.append('--gres='+os.environ['RUN_GPU_GRES'])
    if dependency:args.append('--dependency=afterany:'+str(dependency))
    args += [str(PROJECT/'research/reachability_fair_v3/host.sh'),mode]
    if argument is not None:args.append(str(argument))
    return command(args).split(';')[0]


def next_gpu(value,mode='slot'):
    grant=read(OUT/'queue/budget_grant.json');remaining=grant['gpu_seconds_limit']-value['gpu_seconds_used']
    seconds=math.floor(min(600 if mode=='preflight' else 1800,remaining)/60)*60
    if seconds<300:value['status']='budget_stopped';save(value);return value
    job=submit_job(mode,seconds,gpu=True)
    value['gpu_jobs'].append({'job_id':job,'mode':mode,'reserved_seconds':seconds,'accounted':False})
    value.update(status='queued',active_gpu_job=job);save(value)
    value['accounting_job']=submit_job('account',seconds=180,dependency=job,argument=job);save(value)
    return value


def start():
    value=state()
    if value['gpu_jobs']:return value
    check=read(OUT/'cpu_preflight.json');assert check['status']=='passed'
    write(OUT/'queue/budget_grant.json',{'gpu_seconds_limit':36000,
        'authorization':'User explicitly approved up to10 GPU-hours for this separate fair-comparison series.',
        'recorded_unix':time.time(),'includes':'all GPU allocations, smoke, profiling, pilots, finals, inference, failures and overhead'})
    return next_gpu(value,'preflight')


def gpu(mode):
    value=state();job=os.environ['SLURM_JOB_ID'];row=next(r for r in value['gpu_jobs'] if r['job_id']==job)
    remaining=read(OUT/'queue/budget_grant.json')['gpu_seconds_limit']-value['gpu_seconds_used']
    assert row['reserved_seconds']<=remaining and row['mode']==mode
    live=set(command(['squeue','-h','-o','%i']).splitlines())
    assert not any(r['job_id']!=job and r['job_id'] in live for r in value['gpu_jobs'])
    value.update(status='running',current_job=job);save(value)
    args=['bash',str(PROJECT/'research/reachability_fair_v3/container.sh'),mode]
    if mode=='slot':args += ['--seconds',str(row['reserved_seconds']-60),'--remaining',str(remaining)]
    subprocess.run(args,check=True)
    result=read(OUT/'gpu_preflight.json') if mode=='preflight' else read(OUT/('slot_'+job+'.json'))
    value=state();value['status']='preflight_passed' if mode=='preflight' and result['status']=='passed' else result['status']
    save(value);return value


def account(job):
    value=state();row=next(r for r in value['gpu_jobs'] if r['job_id']==str(job))
    if row['accounted']:return value
    found=[]
    for attempt in range(5):
        raw=command(['sacct','-X','-j',str(job),'--noheader','--parsable2','--format=JobIDRaw,State,ElapsedRaw,ExitCode,AllocTRES'])
        found=[line.split('|') for line in raw.splitlines() if line.split('|')[0]==str(job)]
        if len(found)==1 and found[0][1] not in ['RUNNING','PENDING','COMPLETING']:break
        time.sleep(2)
    if len(found)!=1 or found[0][1] in ['RUNNING','PENDING','COMPLETING']:
        value['status']='accounting_unavailable';save(value);return value
    record=found[0];elapsed=int(record[2]);tres=dict(part.split('=',1) for part in record[4].split(',') if '=' in part)
    gpu_count=int(tres.get('gres/gpu',0))
    if elapsed and gpu_count!=1:
        value['status']='accounting_unavailable';value['accounting_error']='Cannot verify one allocated GPU';save(value);return value
    row.update(accounted=True,allocated_seconds=elapsed*gpu_count,scheduler_state=record[1],exit_code=record[3],gpu_count=gpu_count)
    value['gpu_seconds_used']+=elapsed*gpu_count
    if record[1]!='COMPLETED' or record[3]!='0:0':value['status']='failed'
    save(value)
    if value['status']=='completed':
        if not value.get('postprocess_job'):value['postprocess_job']=submit_job('postprocess',seconds=1800);save(value)
        return value
    if value['status']=='preflight_passed':
        check=read(OUT/'gpu_preflight.json');remaining=36000-value['gpu_seconds_used']
        if check['estimated_full_seconds_with_reserve']>remaining:
            value.update(status='budget_stopped',reason='Full prespecified grid estimate exceeds remaining10 GPU-hours');save(value);return value
    if value['status'] not in ['paused','preflight_passed']:return value
    return next_gpu(value)


def postprocess():
    assert state()['status']=='completed'
    subprocess.run(['bash',str(PROJECT/'research/reachability_fair_v3/container.sh'),'postprocess'],check=True)
    write(OUT/'postprocess_completed.json',{'status':'completed','audit':True,'report':True})
    return {'status':'completed'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','start','preflight','slot','account','status','postprocess'])
    parser.add_argument('argument',nargs='?');args=parser.parse_args()
    (OUT/'queue').mkdir(parents=True,exist_ok=True)
    with (OUT/'queue/lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.mode=='prepare':result={'cpu_prepare_job':submit_job('cpu',seconds=1800)};write(OUT/'queue/prepare.json',result)
        elif args.mode=='start':result=start()
        elif args.mode in ['preflight','slot']:result=gpu(args.mode)
        elif args.mode=='account':result=account(args.argument)
        elif args.mode=='postprocess':result=postprocess()
        else:result=state()
        print(json.dumps(result),flush=True)
