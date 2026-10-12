"""Single-GPU Slurm continuation, separate explicit grant, then closed evaluation."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

PROJECT=Path(os.environ['CURRICULUM_PROJECT']);ROOT=PROJECT/'experiments/curriculum_v4'
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def command(args):return subprocess.check_output(args,text=True).strip()
def state():return read(ROOT/'queue/state.json') if (ROOT/'queue/state.json').exists() else {'status':'prepared','gpu_jobs':[],'gpu_seconds_used':0}
def save(s):write(ROOT/'queue/state.json',s)


def submit_job(mode,seconds=1800,gpu=False,dependency=None,argument=None):
    args=['sbatch','--parsable','--job-name=curr4_'+mode,'--account='+os.environ['RUN_ACCOUNT'],
        '--partition='+os.environ['RUN_GPU_PARTITION' if gpu else 'RUN_CPU_PARTITION'],
        '--cpus-per-task='+('4' if gpu else '1'),'--mem='+('8G' if gpu else '4G'),
        '--time='+str(math.ceil(seconds/60)),'--chdir='+str(PROJECT),
        '--output='+str(PROJECT/'.cache'/(mode+'_%j.out'))]
    if gpu:args.append('--gres='+os.environ['RUN_GPU_GRES'])
    if dependency:args.append('--dependency=afterany:'+str(dependency))
    args += [str(PROJECT/'research/reachability_curriculum_v4/host.sh'),mode]
    if argument is not None:args.append(str(argument))
    return command(args).split(';')[0]


def next_gpu(s,mode='slot'):
    grant=read(ROOT/'queue/budget_grant.json');remaining=grant['gpu_seconds_limit']-s['gpu_seconds_used']
    seconds=math.floor(min(600 if mode=='preflight' else 1800,remaining)/60)*60
    if seconds<300:s['status']='budget_stopped';save(s);return s
    job=submit_job(mode,seconds,gpu=True)
    s['gpu_jobs'].append({'job_id':job,'mode':mode,'reserved_seconds':seconds,'accounted':False})
    s.update(status='queued',active_gpu_job=job);save(s)
    s['accounting_job']=submit_job('account',seconds=180,dependency=job,argument=job);save(s);return s


def start():
    s=state()
    if s['gpu_jobs']:return s
    grant=read(ROOT/'queue/budget_grant.json')
    assert grant['explicit_user_approval'] and grant['gpu_seconds_limit']>0 and grant['series']=='reachability_curriculum_v4'
    assert read(ROOT/'cpu_preflight.json')['status']=='passed'
    return next_gpu(s,'preflight')


def gpu(mode):
    s=state();job=os.environ['SLURM_JOB_ID'];row=next(r for r in s['gpu_jobs'] if r['job_id']==job)
    remaining=read(ROOT/'queue/budget_grant.json')['gpu_seconds_limit']-s['gpu_seconds_used']
    assert row['mode']==mode and row['reserved_seconds']<=remaining
    live=set(command(['squeue','-h','-o','%i']).splitlines())
    assert not any(r['job_id']!=job and r['job_id'] in live for r in s['gpu_jobs'])
    s.update(status='running',current_job=job);save(s)
    args=['bash',str(PROJECT/'research/reachability_curriculum_v4/container.sh'),mode]
    if mode in ['slot','eval_slot']:args += ['--seconds',str(row['reserved_seconds']-60),'--remaining',str(remaining)]
    subprocess.run(args,check=True)
    result=read(ROOT/'gpu_preflight.json') if mode=='preflight' else read(ROOT/('slot_'+job+'.json'))
    s=state();s['status']='preflight_passed' if mode=='preflight' and result['status']=='passed' else result['status'];save(s);return s


def account(job):
    s=state();row=next(r for r in s['gpu_jobs'] if r['job_id']==str(job))
    if row['accounted']:return s
    found=[]
    for attempt in range(5):
        lines=command(['sacct','-X','-j',str(job),'--noheader','--parsable2','--format=JobIDRaw,State,ElapsedRaw,ExitCode,AllocTRES']).splitlines()
        found=[line.split('|') for line in lines if line.split('|')[0]==str(job)]
        if len(found)==1 and found[0][1] not in ['RUNNING','PENDING','COMPLETING']:break
        time.sleep(2)
    if len(found)!=1 or found[0][1] in ['RUNNING','PENDING','COMPLETING']:
        s['status']='accounting_unavailable';save(s);return s
    r=found[0];elapsed=int(r[2]);tres=dict(part.split('=',1) for part in r[4].split(',') if '=' in part);gpus=int(tres.get('gres/gpu',0))
    if elapsed and gpus!=1:s['status']='accounting_unavailable';save(s);return s
    row.update(accounted=True,allocated_seconds=elapsed*gpus,gpu_count=gpus,scheduler_state=r[1],exit_code=r[3]);s['gpu_seconds_used']+=elapsed*gpus
    if r[1]!='COMPLETED' or r[3]!='0:0':s['status']='failed'
    save(s)
    if s['status']=='completed':
        if not s.get('postprocess_job'):s['postprocess_job']=submit_job('postprocess',seconds=1800);save(s)
        return s
    if s['status']=='training_completed':
        if (ROOT/'test_ready.json').exists():return next_gpu(s,'eval_slot')
        s['test_prepare_job']=submit_job('test_prepare',seconds=1800)
        s['evaluation_start_job']=submit_job('evaluation_start',seconds=180,dependency=s['test_prepare_job'],argument=s['test_prepare_job'])
        s['status']='preparing_tests';save(s);return s
    if s['status']=='preflight_passed':
        remaining=read(ROOT/'queue/budget_grant.json')['gpu_seconds_limit']-s['gpu_seconds_used']
        if read(ROOT/'gpu_preflight.json')['estimated_full_seconds_with_reserve']>remaining:
            s.update(status='budget_stopped',reason='Full grid estimate with20percent reserve exceeds remaining approved cap');save(s);return s
    if s['status'] not in ['paused','preflight_passed']:return s
    return next_gpu(s,'eval_slot' if row['mode']=='eval_slot' else 'slot')


def evaluation_start(job):
    rows=command(['sacct','-X','-j',str(job),'--noheader','--parsable2','--format=JobIDRaw,State,ExitCode']).splitlines()
    assert any(row.split('|')[:3]==[str(job),'COMPLETED','0:0'] for row in rows)
    assert read(ROOT/'test_ready.json')['status']=='passed' and read(ROOT/'training_completed.json')['status']=='completed'
    return next_gpu(state(),'eval_slot')


def postprocess():
    assert state()['status']=='completed'
    subprocess.run(['bash',str(PROJECT/'research/reachability_curriculum_v4/container.sh'),'postprocess'],check=True)
    write(ROOT/'postprocess_completed.json',{'status':'completed','audit':True,'report':True});return {'status':'completed'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','start','preflight','slot','eval_slot','account','evaluation_start','postprocess','status'])
    p.add_argument('argument',nargs='?');a=p.parse_args();(ROOT/'queue').mkdir(parents=True,exist_ok=True)
    with (ROOT/'queue/lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if a.mode=='prepare':result={'cpu_prepare_job':submit_job('cpu',seconds=1800)};write(ROOT/'queue/prepare.json',result)
        elif a.mode=='start':result=start()
        elif a.mode in ['preflight','slot','eval_slot']:result=gpu(a.mode)
        elif a.mode=='account':result=account(a.argument)
        elif a.mode=='evaluation_start':result=evaluation_start(a.argument)
        elif a.mode=='postprocess':result=postprocess()
        else:result=state()
        print(json.dumps(result),flush=True)
