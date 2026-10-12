#!/bin/bash
set -euo pipefail
source "$HOME/directed-reachability-v1/.cache/site.sh"
REWEIGHT_PROJECT="$HOME/reachability-reweight-v2"
REWEIGHT_ACTION="${1:?action required}"
REWEIGHT_NV=()
REWEIGHT_THREADS=1
REWEIGHT_GPU_ENV=""
if [[ "$REWEIGHT_ACTION" == evaluate ]]; then
  if [[ -f "$REWEIGHT_PROJECT/experiments/reweight_v2/completion.json" ]]; then
    echo 'All evaluations already completed.'
    exit 0
  fi
  REWEIGHT_NV=(--nv)
  REWEIGHT_THREADS=4
  REWEIGHT_GPU_ENV=",SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
fi
exec apptainer exec "${REWEIGHT_NV[@]}" --cleanenv --no-home \
  --bind "$RUN_LEGACY_PROJECT:/legacy:ro" \
  --bind "$RUN_PROJECT:/original:ro" \
  --bind "$REWEIGHT_PROJECT:/workspace" --pwd /workspace \
  --env "PYTHONPATH=/original/src:/workspace/research/reachability_reweight_v2,REACH_ROOT=/original/experiments/reachability,REACH_DATA=/original/experiments/reachability/data,REWEIGHT_OUT=/workspace/experiments/reweight_v2,OMP_NUM_THREADS=$REWEIGHT_THREADS,OPENBLAS_NUM_THREADS=$REWEIGHT_THREADS,SLURM_JOB_ID=$SLURM_JOB_ID$REWEIGHT_GPU_ENV" \
  "$RUN_IMAGE" "$RUN_PYTHON_IN_CONTAINER" "/workspace/research/reachability_reweight_v2/$REWEIGHT_ACTION.py"
