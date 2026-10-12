#!/bin/bash
set -euo pipefail
source "$HOME/directed-reachability-v1/.cache/site.sh"
export FAIR_PROJECT="$HOME/reachability-fair-v3"
apptainer exec --nv --cleanenv --no-home \
  --bind "$RUN_LEGACY_PROJECT:/legacy:ro" --bind "$RUN_PROJECT:/original:ro" \
  --bind "$HOME/reachability-reweight-v2:/reweight_previous:ro" --bind "$FAIR_PROJECT:/workspace" \
  --pwd /workspace \
  --env "PYTHONPATH=/original/src:/workspace/research/reachability_fair_v3,REACH_ROOT=/workspace/experiments/fair_v3,FAIR_ROOT=/workspace/experiments/fair_v3,REACH_DATA=/original/experiments/reachability/data,BASE_REACH_ROOT=/original/experiments/reachability,PREVIOUS_REWEIGHT_DATA=/reweight_previous/experiments/reweight_v2/data,OMP_NUM_THREADS=4,OPENBLAS_NUM_THREADS=4,SLURM_JOB_ID=$SLURM_JOB_ID,SLURM_JOB_GPUS=$SLURM_JOB_GPUS" \
  "$RUN_IMAGE" "$RUN_PYTHON_IN_CONTAINER" /workspace/analysis/fair_v3_long_profile.py
"$RUN_HOST_PYTHON" - <<'PY'
import fcntl,json,os
from pathlib import Path
root=Path(os.environ['FAIR_PROJECT'])/'experiments/fair_v3'
with (root/'queue/lock').open('a') as f:
    fcntl.flock(f,fcntl.LOCK_EX)
    p=root/'queue/state.json';value=json.loads(p.read_text())
    assert value['active_gpu_job']==os.environ['SLURM_JOB_ID']
    value.update(status='preflight_passed',current_job=os.environ['SLURM_JOB_ID'])
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(p)
PY
