"""CPU-only Slurm dispatcher, compatible with the cluster's Python 3.6.

Old jobs/ledgers are read-only. No new GPU job is submitted until all 36 MLP
runs complete and no older GPU allocation remains in the user's queue.
"""
import json
import math
import os
import subprocess
import time
import traceback

from common import ART,SIDE,ROOT,CODE,atomic,read,validate
from plan import next_stage,complete,completed_training

TERMINAL={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def capture(args):return subprocess.check_output(args,universal_newlines=True,stderr=subprocess.STDOUT).strip()


def queue():
    raw=capture(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%T|%b'])
    return {p[0]:dict(zip(['id','name','state','gres'],p)) for p in (s.split('|') for s in raw.splitlines()) if len(p)==4}


def watch(dependency=None,delay=False):
    args=['sbatch','--parsable','--account='+os.environ['QBM_ACCOUNT'],
          '--partition='+os.environ['QBM_CPU_PARTITION'],'--cpus-per-task=1','--mem=512M','--time=5',
          '--job-name=lt-language-watch','--output='+str(SIDE/'logs/watch-%j.log')]
    if dependency:args+=['--dependency=afterany:'+dependency]
    if delay:args+=['--begin=now+30minutes']
    job=capture(args+[str(CODE/'watch.sbatch')]).split(';')[0]
    atomic(ART/'watcher.json',dict(job_id=job,epoch=time.time(),dependency=dependency,delayed=delay))
    return job


def prior_complete():
    old=ROOT/'research/qbm_mlp_20261102'
    manifest=read(old/'artifacts/manifest.json')
    completed=0
    for item in manifest['plan']:
        run=old/'runs'/item['key'];a=run/'training.json';b=run/'result.json'
        if a.exists() and b.exists() and read(a).get('status')=='completed' and read(b).get('status')=='completed':completed+=1
    return completed==36,completed


def account(ledger):
    ids=[x['job_id'] for x in ledger['allocations']]
    if ids:
        raw=capture(['sacct','-nP','-X','-j',','.join(ids),'--format=JobID,State,ElapsedRaw'])
        records={}
        for line in raw.splitlines():
            p=line.split('|')
            if len(p)>=3 and p[0] in ids:records[p[0]]=(p[1].split()[0].rstrip('+'),int(p[2]))
        for job in ledger['allocations']:
            if job['job_id'] in records:
                state,seconds=records[job['job_id']];job['scheduler_state']=state
                if state in TERMINAL:job['actual_seconds']=seconds
    return ledger


def charged(ledger):
    return sum(j.get('actual_seconds',j['reserved_seconds']) for j in ledger['allocations'])


def recover_claims(ledger,visible):
    ended={j['job_id'] for j in ledger['allocations'] if j.get('scheduler_state') in TERMINAL}
    for claim in (ART/'claims').iterdir():
        owner=claim/'owner.json'
        if not owner.exists():raise RuntimeError('Incomplete claim requires inspection: '+claim.name)
        row=read(owner)
        if row['job_id'] in ended and row['job_id'] not in visible:
            atomic(ART/'recovered_claims'/(claim.name+'-'+row['job_id']+'.json'),row)
            owner.unlink();claim.rmdir()


def cost_estimate():
    p=ART/'benchmarks.json'
    if not p.exists():return None
    b=read(p);selected=ART/'selected_protocol.json'
    steps=read(selected)['steps'] if selected.exists() else 16000
    train=sum(x['seconds_per_microbatch']*4*steps*12 for x in b if x['model']=='scratch')
    train+=sum(x['seconds_per_microbatch']*8*2000*6 for x in b if x['model']=='qwen')
    estimate=dict(training_gpu_hours=train/3600,planned_scratch_steps=steps,
        reservation_with_evaluation_and_margin_gpu_hours=math.ceil((train/3600*1.6+24)/10)*10,
        basis='Actual single-GPU update benchmarks; long-context diagnostics and evaluation use 60% margin plus 24 hours; not a measured total')
    atomic(ART/'cost_estimate.json',estimate)
    return estimate


def tick():
    plan=validate();check=read(ART/'cpu_preflight.json')
    if check['status']!='passed' or check['source_sha256']!=plan['source_sha256']:raise RuntimeError('CPU verification does not match frozen source')
    ledger=account(read(ART/'budget.json'));atomic(ART/'budget.json',ledger)
    visible=queue();recover_claims(ledger,visible)
    active=[j for j in ledger['allocations'] if 'actual_seconds' not in j]
    workers={p.stem:read(p) for p in (ART/'workers').glob('*.json')}
    def publish(status,**kw):
        atomic(ART/'status.json',dict(status=status,epoch=time.time(),charged_gpu_hours=charged(ledger)/3600,
            actual_gpu_hours=sum(j.get('actual_seconds',0) for j in ledger['allocations'])/3600,
            active_jobs=[j['job_id'] for j in active],**kw))
    failed=[j['job_id'] for j in ledger['allocations'] if workers.get(j['job_id'],{}).get('status')=='failed'
            or j.get('scheduler_state') in ['FAILED','CANCELLED','OUT_OF_MEMORY','BOOT_FAIL','DEADLINE','REVOKED']]
    if failed:publish('failed',failed_jobs=failed);return
    interruptions=[j for j in ledger['allocations'] if j.get('scheduler_state') in ['NODE_FAIL','PREEMPTED','TIMEOUT']]
    if len(interruptions)>3:publish('repeated_scheduler_failures');return
    if any(j['job_id'] not in visible for j in active):watch(delay=True);publish('waiting_accounting');return
    done,n=prior_complete()
    if not done:
        watch(delay=True);publish('waiting_existing_mlp',predecessor_completed=n,predecessor_total=36);return
    foreign=[j['id'] for j in visible.values() if 'gpu' in j['gres'].lower() and j['id'] not in {x['job_id'] for x in active}]
    if foreign:watch(dependency=foreign[0]);publish('waiting_existing_gpu',jobs=foreign);return
    stage,items=next_stage()
    if stage.startswith('blocked'):
        publish(stage);return
    if stage=='completed':
        if active:publish('draining');return
        publish('completed',completed=68,total=68)
        if (ART/'report_job.json').exists():return
        # A small CPU allocation creates final tables/figures only from saved results.
        args=['sbatch','--parsable','--account='+os.environ['QBM_ACCOUNT'],
            '--partition='+os.environ['QBM_CPU_PARTITION'],'--cpus-per-task=2','--mem=4G','--time=20',
            '--job-name=lt-language-report','--output='+str(SIDE/'logs/report-%j.log'),str(CODE/'runtime.sh'),
            'research/language_transport_20261105/code/report.py']
        atomic(ART/'report_job.json',dict(job_id=capture(args).split(';')[0]));return
    auth=read(ART/'budget_authorization.json');estimate=cost_estimate()
    if estimate and estimate['reservation_with_evaluation_and_margin_gpu_hours']>auth['campaign_gpu_hours']:
        # User explicitly authorizes selecting/increasing the necessary budget.
        auth['campaign_gpu_hours']=estimate['reservation_with_evaluation_and_margin_gpu_hours']
        auth['global_gpu_hours']=max(auth['global_gpu_hours'],200+auth['campaign_gpu_hours']+20)
        auth['increase_basis']=estimate;atomic(ART/'budget_authorization.json',auth)
    seconds=3600 if stage=='preflight' else 7200
    slots=1 if stage=='preflight' else 2
    submitted=[]
    for _ in range(max(0,slots-len(active))):
        ceiling=min(auth['campaign_gpu_hours'],auth['global_gpu_hours']-auth['prior_reserved_gpu_hours'])
        if charged(ledger)+seconds>ceiling*3600:break
        command=['sbatch','--hold','--parsable','--account='+os.environ['QBM_ACCOUNT'],
            '--partition='+os.environ['QBM_GPU_PARTITION'],'--gres=gpu:a100:1','--cpus-per-task=4',
            '--mem=32G','--time='+str(seconds//60),'--signal=B:USR1@180',
            '--job-name=lt-language-'+stage,'--output='+str(SIDE/'logs/gpu-%j.log'),str(CODE/'runtime.sh'),
            'research/language_transport_20261105/code/worker.py','--seconds',str(seconds),'--role',stage]
        job=capture(command).split(';')[0]
        entry=dict(job_id=job,gpu_count=1,reserved_seconds=seconds,role=stage,submitted_epoch=time.time())
        ledger['allocations'].append(entry);atomic(ART/'budget.json',ledger)
        watcher=watch(dependency=job);entry['watcher_job_id']=watcher;atomic(ART/'budget.json',ledger)
        capture(['scontrol','release',job]);submitted.append(job)
    completed=sum(completed_training(s) if stage=='pilot' else complete(s) for s in items)
    publish('queued_'+stage if active or submitted else 'budget_paused',submitted=submitted,
        stage_completed=completed,stage_total=len(items),campaign_limit_gpu_hours=auth['campaign_gpu_hours'])


def main():
    assert os.environ.get('SLURM_JOB_ID')
    claim=ART/'controller_claim'
    try:claim.mkdir()
    except FileExistsError:
        # Never steal a live dispatcher lock. A terminal predecessor can be reclaimed.
        owner=claim/'owner.json'
        if owner.exists() and read(owner)['job_id'] not in queue():
            atomic(ART/'controller_lock_recovery.json',read(owner));owner.unlink();claim.rmdir();claim.mkdir()
        else:watch(delay=True);return
    atomic(claim/'owner.json',dict(job_id=os.environ['SLURM_JOB_ID'],epoch=time.time()))
    try:tick()
    except BaseException:
        atomic(ART/'controller_error.json',dict(traceback=traceback.format_exc(),epoch=time.time()));raise
    finally:(claim/'owner.json').unlink();claim.rmdir()


if __name__=='__main__':main()
