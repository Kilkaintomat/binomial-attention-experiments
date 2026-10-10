"""Own-job scheduler only. Site values are supplied by an ignored runtime file."""
import fcntl
import json
import math
import os
import subprocess
import time
from pathlib import Path
from transport_reasoning.config import write_json
from .config import root
from .campaign import read


def command(args): return subprocess.check_output(args, text=True).strip()


def queue_path(): return root() / 'queue/state.json'


def state(): return read(queue_path()) if queue_path().exists() else {'gpu_jobs': [], 'gpu_seconds_used': 0, 'status': 'prepared'}


def active(job): return bool(command(['squeue','-h','-j',str(job),'-o','%i'])) if job else False


def batch(mode, dependency=None, hold=False, seconds=7200, gpu=False, job=None):
    project = Path(os.environ['RUN_PROJECT'])
    args = ['sbatch','--parsable','--job-name=rec_sudoku_' + mode,
            '--account=' + os.environ['RUN_ACCOUNT'], '--partition=' + os.environ['RUN_GPU_PARTITION' if gpu else 'RUN_CPU_PARTITION'],
            '--cpus-per-task=' + ('4' if gpu else '1'), '--mem=' + ('24G' if gpu else '1G'),
            '--time=' + str(max(1, math.ceil(seconds / 60))), '--chdir=' + str(project),
            '--output=' + str(project / '.cache' / (mode + '_%j.out'))]
    if gpu: args += ['--gres=' + os.environ['RUN_GPU_GRES']]
    if dependency: args += ['--dependency=afterany:' + str(dependency)]
    if hold: args += ['--hold']
    args += [str(project / 'research/recurrent_filter_sudoku_v1/job.sh'), mode]
    if job: args.append(str(job))
    return command(args).split(';')[0]


def save(value): write_json(queue_path(), value)


def ensure_gate(value):
    if active(value.get('gate_job')): return
    old = read(Path(os.environ['RUN_LEGACY_PROJECT']) / 'experiments/transport_reasoning/mlp_final_submission.json')
    dependency = old.get('post_job_accounting_job')
    value['gate_job'] = batch('gate', dependency=dependency if active(dependency) else None, seconds=120)
    save(value)


def submit(args):
    value = state()
    if value.get('gpu_jobs'): return value
    # Submission occupies no GPU while held. No modification of any prior job.
    job = batch('slot', hold=True, seconds=1200, gpu=True)
    value.update(status='held_pending_old_campaign_checks_and_budget', held_job=job)
    value['gpu_jobs'].append({'job_id': job, 'reserved_seconds': 1200, 'phase': 'profile', 'accounted': False})
    save(value)
    value['accounting_job'] = batch('account', dependency=job, seconds=180, job=job)
    save(value); ensure_gate(value); return value


def allow(args):
    if not args.gpu_hours or not args.wall_hours or args.gpu_hours <= 0 or args.wall_hours <= 0:
        raise ValueError('Explicit positive GPU-hour and wall-hour limits required')
    path = root() / 'queue/budget_grant.json'
    grant = {'gpu_seconds_limit': round(args.gpu_hours*3600), 'wall_seconds_limit': round(args.wall_hours*3600),
             'authorization': 'explicit_user_instruction_required_for_this_command', 'recorded_unix': time.time()}
    if path.exists():
        old = read(path)
        grant['previous_limit_seconds'] = old['gpu_seconds_limit']
    write_json(path, grant)
    value = state()
    if any(active(r['job_id']) for r in value['gpu_jobs']):
        if value.get('held_job') and active(value['held_job']): ensure_gate(value)
        return grant
    if value.get('first_allocation_unix') and value['status'] == 'budget_stopped':
        from .campaign import preflight
        preflight()
        remaining = grant['gpu_seconds_limit'] - value['gpu_seconds_used']
        wall = value['first_allocation_unix'] + grant['wall_seconds_limit'] - time.time()
        seconds = math.floor(min(7200, remaining, wall)/60)*60
        if seconds < 600: raise ValueError('Approved remaining budget is less than a minimum slot')
        job = batch('slot', seconds=seconds, gpu=True)
        value['gpu_jobs'].append({'job_id':job,'reserved_seconds':seconds,'phase':'campaign','accounted':False})
        value['status']='queued'; save(value)
        value['accounting_job']=batch('account',dependency=job,seconds=180,job=job); save(value)
    else: ensure_gate(value)
    return grant


def gate(args):
    value = state()
    if value.get('status') == 'released': return value
    project = Path(os.environ['RUN_LEGACY_PROJECT'])
    old = read(project / 'experiments/transport_reasoning/mlp_final_submission.json')
    dependency = old.get('post_job_accounting_job')
    if active(old.get('gpu_job')) or active(dependency):
        value['status'] = 'held_waiting_for_existing_campaign'
        value['gate_job'] = batch('gate', dependency=dependency, seconds=120)
        save(value); return value
    observed = json.loads(subprocess.check_output(['/usr/bin/python3', str(project / 'scripts/monitor_mlp_final.py')], cwd=project, text=True))
    if observed['completed_runs'] != observed['expected_runs']:
        value['status'] = 'held_existing_campaign_incomplete'; save(value); return value
    try:
        from .campaign import preflight
        preflight()
    except Exception as error:
        value.update(status='held_preflight_incomplete', reason=type(error).__name__); save(value); return value
    if not (root() / 'queue/budget_grant.json').exists():
        value['status'] = 'held_awaiting_confirmed_additional_budget'; save(value); return value
    grant = read(root() / 'queue/budget_grant.json')
    if min(grant['gpu_seconds_limit'] - value['gpu_seconds_used'], grant['wall_seconds_limit']) < 1200:
        value['status'] = 'held_insufficient_profile_reserve'; save(value); return value
    command(['scontrol','release',str(value['held_job'])])
    value.update(status='released', release_unix=time.time()); save(value); return value


def slot(args):
    value = state(); grant = read(root() / 'queue/budget_grant.json')
    # Recheck previous work even after release; never overlap GPU campaigns.
    old = read(Path(os.environ['RUN_LEGACY_PROJECT']) / 'experiments/transport_reasoning/mlp_final_submission.json')
    if active(old.get('gpu_job')): raise RuntimeError('Prior campaign still has an allocated or pending GPU job')
    job = os.environ['SLURM_JOB_ID']
    record = next(r for r in value['gpu_jobs'] if str(r['job_id']) == job)
    if 'first_allocation_unix' not in value: value['first_allocation_unix'] = time.time()
    remaining = grant['gpu_seconds_limit'] - value['gpu_seconds_used']
    wall_remaining = value['first_allocation_unix'] + grant['wall_seconds_limit'] - time.time()
    if record['reserved_seconds'] > remaining or wall_remaining < 180:
        value['status'] = 'budget_stopped'; save(value); return value
    save(value)
    try:
        env = ','.join(['PYTHONPATH=/workspace/src:/workspace', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
                        'SLURM_JOB_ID=' + job, 'SLURM_JOB_GPUS=' + os.environ['SLURM_JOB_GPUS'],
                        'RECURRENT_CODE_COMMIT=' + os.environ.get('RECURRENT_CODE_COMMIT',''),
                        'TRANSPORT_COMMIT=' + os.environ.get('RECURRENT_CODE_COMMIT',''),
                        'RECURRENT_DATA_DIR=' + os.environ['RUN_DATA_IN_CONTAINER'],
                        'RECURRENT_LEGACY_ROOT=/legacy/experiments/transport_reasoning'])
        subprocess.run(['apptainer','exec','--nv','--cleanenv','--no-home',
                        '--bind',os.environ['RUN_LEGACY_PROJECT'] + ':/legacy:ro',
                        '--bind',os.environ['RUN_PROJECT'] + ':/workspace','--pwd','/workspace','--env',env,
                        os.environ['RUN_IMAGE'], os.environ['RUN_PYTHON_IN_CONTAINER'],
                        '-m','research.recurrent_filter_sudoku_v1','run',
                        '--seconds',str(int(min(record['reserved_seconds'] - 60, wall_remaining))),
                        '--remaining-seconds',str(remaining)], check=True)
        status = read(root() / ('queue/slot_result_' + job + '.json'))['status']
    except Exception:
        value = state(); value['status'] = 'failed'; save(value); raise
    value = state(); value['status'] = status; save(value); return {'status': status}


def account(args):
    value = state(); job = str(args.job)
    record = next(r for r in value['gpu_jobs'] if str(r['job_id']) == job)
    if record['accounted']: return value
    rows = command(['sacct','-X','-j',job,'--noheader','--parsable2','--format=JobIDRaw,State,ElapsedRaw,AllocTRES,ExitCode']).splitlines()
    matching = [r.split('|') for r in rows if r.split('|')[0] == job]
    if len(matching) != 1 or matching[0][1] in ['RUNNING','PENDING','COMPLETING']:
        value['status'] = 'accounting_unavailable_no_automatic_restart'; save(value); return value
    fields = matching[0]; elapsed = int(fields[2]); scheduler_state = fields[1]
    record.update(accounted=True, scheduler_state=scheduler_state, allocated_seconds=elapsed, exit_code=fields[4])
    value['gpu_seconds_used'] += elapsed
    grant = read(root() / 'queue/budget_grant.json')
    remaining = grant['gpu_seconds_limit'] - value['gpu_seconds_used']
    if value['status'] in ['completed','failed','budget_stopped'] or scheduler_state != 'COMPLETED' or fields[4] != '0:0':
        if scheduler_state != 'COMPLETED' or fields[4] != '0:0': value['status'] = 'failed'
        save(value); return value
    wall_remaining = value['first_allocation_unix'] + grant['wall_seconds_limit'] - time.time()
    seconds = math.floor(min(7200, remaining, wall_remaining) / 60) * 60
    if seconds < 600:
        value['status'] = 'budget_stopped'; save(value); return value
    next_job = batch('slot', seconds=seconds, gpu=True, dependency=job)
    value['gpu_jobs'].append({'job_id': next_job, 'reserved_seconds': seconds, 'phase': 'campaign', 'accounted': False})
    value['status'] = 'queued'; save(value)
    value['accounting_job'] = batch('account', dependency=next_job, seconds=180, job=next_job)
    save(value); return value
