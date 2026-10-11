"""Own-job scheduler: predecessors are read-only, with independent absolute limits."""
import json
import math
import os
import subprocess
import time
from pathlib import Path
from transport_reasoning.config import write_json
from .config import root
from .campaign import read,preflight,admission


def command(args): return subprocess.check_output(args,text=True).strip()
def path(): return root()/'queue/state.json'
def state(): return read(path()) if path().exists() else {'status':'prepared','gpu_jobs':[],'gpu_seconds_used':0}
def save(value): write_json(path(),value)
def active(job): return bool(command(['squeue','-h','-j',str(job),'-o','%i'])) if job else False


def batch(mode,dependency=None,hold=False,seconds=7200,gpu=False,job=None):
    project=Path(os.environ['RUN_PROJECT'])
    args=['sbatch','--parsable','--job-name=local_mask_'+mode,'--account='+os.environ['RUN_ACCOUNT'],
          '--partition='+os.environ['RUN_GPU_PARTITION' if gpu else 'RUN_CPU_PARTITION'],
          '--cpus-per-task='+('4' if gpu else '1'),'--mem='+('24G' if gpu else '1G'),
          '--time='+str(max(1,math.ceil(seconds/60))),'--chdir='+str(project),
          '--output='+str(project/'.cache'/(mode+'_%j.out'))]
    if gpu: args.append('--gres='+os.environ['RUN_GPU_GRES'])
    if dependency: args.append('--dependency=afterany:'+str(dependency))
    if hold: args.append('--hold')
    args += [str(project/'research/sudoku_local_mask_v1/job.sh'),mode]
    if job: args.append(str(job))
    return command(args).split(';')[0]


def predecessor():
    old=read(Path(os.environ['RUN_LEGACY_PROJECT'])/'experiments/transport_reasoning/mlp_final_submission.json')
    if active(old.get('gpu_job')) or active(old.get('post_job_accounting_job')):
        return {'ready':False,'dependency':old.get('post_job_accounting_job') or old['gpu_job'],'reason':'existing_mlp_campaign'}
    prior_root=Path(os.environ['RUN_PRIOR_SERIES_PROJECT'])/'experiments/recurrent_filter_sudoku_v1'
    prior=read(prior_root/'queue/state.json')
    if active(prior.get('accounting_job')):
        return {'ready':False,'dependency':prior['accounting_job'],'reason':'existing_recurrent_series'}
    for record in prior['gpu_jobs']:
        if active(record['job_id']): return {'ready':False,'dependency':record['job_id'],'reason':'existing_recurrent_series'}
    if prior.get('status')!='completed' or not (prior_root/'completion.json').exists():
        return {'ready':False,'dependency':None,'reason':'existing_series_incomplete'}
    return {'ready':True,'dependency':None,'reason':None}


def schedule_gate(value):
    if active(value.get('gate_job')): return
    previous=predecessor()
    value['gate_job']=batch('gate',dependency=previous['dependency'],seconds=120);save(value)


def submit(args):
    value=state()
    if value['gpu_jobs']: return value
    job=batch('slot',hold=True,seconds=1200,gpu=True)
    value.update(status='held_pending_checks_predecessors_and_budget',held_job=job)
    value['gpu_jobs'].append({'job_id':job,'phase':'profile','reserved_seconds':1200,'accounted':False})
    save(value);value['accounting_job']=batch('account',dependency=job,seconds=180,job=job);save(value)
    schedule_gate(value);return value


def grant_path(): return root()/'queue/budget_grant.json'


def allow(args):
    if not args.gpu_hours or not args.wall_hours or min(args.gpu_hours,args.wall_hours)<=0: raise ValueError('Explicit positive limits required')
    grant={'gpu_seconds_limit':round(args.gpu_hours*3600),'wall_seconds_limit':round(args.wall_hours*3600),
           'recorded_unix':time.time(),'authorization':'requires explicit user instruction for this grant'}
    if grant_path().exists(): grant['previous_limit_seconds']=read(grant_path())['gpu_seconds_limit']
    write_json(grant_path(),grant);value=state()
    if any(active(r['job_id']) for r in value['gpu_jobs']):
        if active(value.get('held_job')): schedule_gate(value)
        return grant
    if value.get('first_allocation_unix') and value['status']=='budget_stopped':
        preflight()
        remaining=grant['gpu_seconds_limit']-value['gpu_seconds_used']
        if admission(remaining) is None: return {'status':'budget_stopped','reason':'approved_limit_still_insufficient'}
        return next_slot(value,grant)
    schedule_gate(value);return grant


def gate(args):
    value=state()
    if value.get('status')=='released': return value
    previous=predecessor()
    if not previous['ready']:
        value.update(status='held_waiting_for_'+previous['reason'])
        if previous['dependency']: value['gate_job']=batch('gate',dependency=previous['dependency'],seconds=120)
        save(value);return value
    try: preflight()
    except (RuntimeError,FileNotFoundError) as error:
        value.update(status='held_preflight_incomplete',reason=type(error).__name__);save(value);return value
    if not grant_path().exists():
        value['status']='held_awaiting_confirmed_additional_budget';save(value);return value
    grant=read(grant_path())
    if min(grant['gpu_seconds_limit']-value['gpu_seconds_used'],grant['wall_seconds_limit'])<1200:
        value['status']='held_insufficient_profile_reserve';save(value);return value
    command(['scontrol','release',str(value['held_job'])])
    value.update(status='released',release_unix=time.time());save(value);return value


def slot(args):
    value=state();grant=read(grant_path());job=os.environ['SLURM_JOB_ID']
    if not predecessor()['ready']: raise RuntimeError('Predecessor work is not complete; no overlap allowed')
    record=next(r for r in value['gpu_jobs'] if str(r['job_id'])==job)
    if 'first_allocation_unix' not in value: value['first_allocation_unix']=time.time()
    remaining=grant['gpu_seconds_limit']-value['gpu_seconds_used']
    wall=value['first_allocation_unix']+grant['wall_seconds_limit']-time.time()
    if record['reserved_seconds']>remaining or wall<180:
        value['status']='budget_stopped';save(value);return value
    save(value)
    env=','.join(['PYTHONPATH=/workspace/src:/workspace','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',
                  'SLURM_JOB_ID='+job,'SLURM_JOB_GPUS='+os.environ['SLURM_JOB_GPUS'],
                  'LOCAL_MASK_DATA_DIR='+os.environ['RUN_DATA_IN_CONTAINER'],'LOCAL_MASK_LEGACY_ROOT=/legacy/experiments/transport_reasoning',
                  'LOCAL_MASK_CODE_COMMIT='+os.environ.get('LOCAL_MASK_CODE_COMMIT',''),'TRANSPORT_COMMIT='+os.environ.get('LOCAL_MASK_CODE_COMMIT','')])
    try:
        subprocess.run(['apptainer','exec','--nv','--cleanenv','--no-home',
                        '--bind',os.environ['RUN_LEGACY_PROJECT']+':/legacy:ro','--bind',os.environ['RUN_PROJECT']+':/workspace',
                        '--pwd','/workspace','--env',env,os.environ['RUN_IMAGE'],os.environ['RUN_PYTHON_IN_CONTAINER'],
                        '-m','research.sudoku_local_mask_v1','run','--seconds',str(int(min(record['reserved_seconds']-90,wall-60))),
                        '--remaining-seconds',str(remaining)],check=True)
        status=read(root()/('queue/slot_result_'+job+'.json'))['status']
    except Exception:
        value=state();value['status']='failed';save(value);raise
    value=state();value['status']=status;save(value);return {'status':status}


def next_slot(value,grant):
    remaining=grant['gpu_seconds_limit']-value['gpu_seconds_used']
    wall=value['first_allocation_unix']+grant['wall_seconds_limit']-time.time()
    seconds=math.floor(min(7200,remaining,wall)/60)*60
    if seconds<600:
        value['status']='budget_stopped';save(value);return value
    job=batch('slot',seconds=seconds,gpu=True)
    value['gpu_jobs'].append({'job_id':job,'phase':'campaign','reserved_seconds':seconds,'accounted':False})
    value['status']='queued';save(value)
    value['accounting_job']=batch('account',dependency=job,seconds=180,job=job);save(value);return value


def account(args):
    value=state();job=str(args.job);record=next(r for r in value['gpu_jobs'] if str(r['job_id'])==job)
    if record['accounted']: return value
    lines=command(['sacct','-X','-j',job,'--noheader','--parsable2','--format=JobIDRaw,State,ElapsedRaw,AllocTRES,ExitCode']).splitlines()
    rows=[line.split('|') for line in lines if line.split('|')[0]==job]
    if len(rows)!=1 or rows[0][1] in ['PENDING','RUNNING','COMPLETING']:
        value['status']='accounting_unavailable_no_automatic_restart';save(value);return value
    row=rows[0];elapsed=int(row[2]);record.update(accounted=True,allocated_seconds=elapsed,scheduler_state=row[1],exit_code=row[4])
    value['gpu_seconds_used']+=elapsed;grant=read(grant_path())
    if value['status'] in ['completed','failed','budget_stopped'] or row[1]!='COMPLETED' or row[4]!='0:0':
        if row[1]!='COMPLETED' or row[4]!='0:0': value['status']='failed'
        save(value);return value
    remaining=grant['gpu_seconds_limit']-value['gpu_seconds_used']
    if admission(remaining) is None:
        value['status']='budget_stopped';save(value);return value
    return next_slot(value,grant)
