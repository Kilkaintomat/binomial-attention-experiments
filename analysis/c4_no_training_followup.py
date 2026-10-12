"""Bounded CPU follow-up while the existing fair-v3 queue produces calibration."""
import json
import os
from pathlib import Path
import subprocess
import sys

project=Path(os.environ['CURRICULUM_PROJECT']);attempt=int(sys.argv[1]);assert 0<=attempt<=20
root=project/'experiments/curriculum_v4'
subprocess.run(['bash',str(project/'research/reachability_curriculum_v4/container.sh'),'no_training'],check=True)
status=json.loads((root/'no_training/status.json').read_text())
result={'attempt':attempt,'status':status['status'],'completed_checkpoints':status['completed_checkpoints']}
if status['status']!='completed' and attempt<20:
    job=subprocess.check_output(['sbatch','--parsable','--job-name=curr4_threshold','--account='+os.environ['RUN_ACCOUNT'],
        '--partition='+os.environ['RUN_CPU_PARTITION'],'--cpus-per-task=1','--mem=4G','--time=10','--begin=now+30minutes',
        '--chdir='+str(project),'--output='+str(project/'.cache/threshold_%j.out'),
        str(project/'analysis/c4_no_training_followup.sh'),str(attempt+1)],text=True).strip().split(';')[0]
    result['next_cpu_job']=job
elif status['status']!='completed':result['status']='waiting_limit_reached_no_more_automatic_jobs'
p=root/'no_training/followup.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(p)
print(json.dumps(result),flush=True)
