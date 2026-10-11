"""One GPU per campaign, short allocations, exact scheduler accounting."""
import argparse
import json
import math
import os
import subprocess
from pathlib import Path
import time
from .config import read,root,write_json


def command(args):return subprocess.check_output(args,text=True).strip()
def state_path():return root()/'queue/state.json'
def state():return read(state_path()) if state_path().exists() else {'status':'prepared','gpu_jobs':[],'gpu_seconds_used':0}
def save(value):write_json(state_path(),value)
def active(job):
    if not job:return False
    ids=set(command(['squeue','-h','-o','%i']).splitlines())
    return str(job) in ids


def submit_job(mode,seconds=1800,gpu=False,dependency=None,job=None):
    project=Path(os.environ['RUN_PROJECT'])
    args=['sbatch','--parsable','--job-name=reach_'+mode,'--account='+os.environ['RUN_ACCOUNT'],
          '--partition='+os.environ['RUN_GPU_PARTITION' if gpu else 'RUN_CPU_PARTITION'],
          '--cpus-per-task='+('4' if gpu else '1'),'--mem='+('8G' if gpu else '4G' if mode=='postprocess' else '1G'),
          '--time='+str(math.ceil(seconds/60)),'--chdir='+str(project),
          '--output='+str(project/'.cache'/(mode+'_%j.out'))]
    if gpu:args.append('--gres='+os.environ['RUN_GPU_GRES'])
    if dependency:args.append('--dependency=afterok:'+str(dependency) if mode=='slot' else '--dependency=afterany:'+str(dependency))
    args += [str(project/'scripts/reachability/host.sh'),mode]
    if job:args.append(str(job))
    return command(args).split(';')[0]


def next_slot(value,dependency=None):
    grant=read(root()/'queue/budget_grant.json');remaining=grant['gpu_seconds_limit']-value['gpu_seconds_used']
    seconds=math.floor(min(1800,remaining)/60)*60
    if seconds<300:value['status']='budget_stopped';save(value);return value
    job=submit_job('slot',seconds,gpu=True,dependency=dependency)
    value['gpu_jobs'].append({'job_id':job,'reserved_seconds':seconds,'accounted':False})
    value.update(status='queued',active_gpu_job=job);save(value)
    value['accounting_job']=submit_job('account',seconds=180,dependency=job,job=job);save(value)
    return value


def submit(dependency):
    value=state()
    if value['gpu_jobs']:return value
    grant={'gpu_seconds_limit':86400,'authorization':'user requested this protocol; section 2 technical ceiling 24 GPU-hours',
           'recorded_unix':time.time(),'includes':'all GPU allocations: preflight, profile, pilots, finals, evaluation, failures'}
    write_json(root()/'queue/budget_grant.json',grant)
    return next_slot(value,dependency)


def slot():
    value=state();job=os.environ['SLURM_JOB_ID'];row=next(r for r in value['gpu_jobs'] if r['job_id']==job)
    grant=read(root()/'queue/budget_grant.json');remaining=grant['gpu_seconds_limit']-value['gpu_seconds_used']
    if row['reserved_seconds']>remaining:raise RuntimeError('Allocation exceeds remaining ceiling')
    for other in value['gpu_jobs']:
        if other['job_id']!=job and active(other['job_id']):raise RuntimeError('Overlapping allocations in this campaign')
    value.update(status='running',current_job=job);save(value)
    env=','.join(['PYTHONPATH=/workspace/src:/workspace','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',
                  'SLURM_JOB_ID='+job,'SLURM_JOB_GPUS='+os.environ['SLURM_JOB_GPUS'],'MPLCONFIGDIR=/workspace/.cache/matplotlib'])
    cmd=['apptainer','exec','--nv','--cleanenv','--no-home','--bind',os.environ['RUN_LEGACY_PROJECT']+':/legacy:ro',
         '--bind',os.environ['RUN_PROJECT']+':/workspace','--pwd','/workspace','--env',env,os.environ['RUN_IMAGE'],
         os.environ['RUN_PYTHON_IN_CONTAINER'],'/workspace/scripts/reachability/gpu_entry.py',
         '--seconds',str(row['reserved_seconds']-45),'--remaining-seconds',str(remaining)]
    try:subprocess.run(cmd,check=True)
    except Exception:
        value=state();value['status']='failed';save(value);raise
    result=read(root()/('queue/slot_result_'+job+'.json'));value=state();value['status']=result['status'];save(value);return value


def account(job):
    value=state();row=next(r for r in value['gpu_jobs'] if r['job_id']==str(job))
    if row['accounted']:return value
    rows=command(['sacct','-X','-j',str(job),'--noheader','--parsable2','--format=JobIDRaw,State,ElapsedRaw,ExitCode']).splitlines()
    found=[line.split('|') for line in rows if line.split('|')[0]==str(job)]
    if len(found)!=1 or found[0][1] in ['RUNNING','PENDING','COMPLETING']:
        value['status']='accounting_unavailable';save(value);return value
    record=found[0];row.update(accounted=True,allocated_seconds=int(record[2]),scheduler_state=record[1],exit_code=record[3]);value['gpu_seconds_used']+=int(record[2])
    if record[1]!='COMPLETED' or record[3]!='0:0':value['status']='failed'
    save(value)
    if value['status']=='completed':
        if not value.get('postprocess_job'):value['postprocess_job']=submit_job('postprocess',seconds=1200);save(value)
        return value
    if value['status'] in ['failed','budget_stopped','accounting_unavailable']:return value
    return next_slot(value)


def postprocess():
    if state()['status']!='completed':raise RuntimeError('Neural test phase must finish before postprocessing')
    project=Path(os.environ['RUN_PROJECT'])
    for action in ['baselines','report']:
        subprocess.run([str(project/'scripts/reachability/cpu.sh'),'-m','transport_reasoning.reachability',action],check=True)
    write_json(root()/'postprocess_completed.json',{'status':'completed','baselines_and_report':True})
    return {'status':'completed'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['submit','slot','account','status','postprocess']);parser.add_argument('argument',nargs='?');args=parser.parse_args()
    result=submit(args.argument) if args.mode=='submit' else slot() if args.mode=='slot' else account(args.argument) if args.mode=='account' else postprocess() if args.mode=='postprocess' else state()
    print(json.dumps(result),flush=True)
