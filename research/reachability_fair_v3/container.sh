#!/bin/bash
set -euo pipefail
source "$HOME/directed-reachability-v1/.cache/site.sh"
FAIR_PROJECT="$HOME/reachability-fair-v3"
FAIR_ACTION="${1:?action required}"
shift
FAIR_EXEC=(apptainer exec)
FAIR_THREADS=1
FAIR_GPU_ENV=""
FAIR_ARGS=()
if [[ "$FAIR_ACTION" == preflight || "$FAIR_ACTION" == slot ]]; then
  FAIR_EXEC+=(--nv)
  FAIR_THREADS=4
  FAIR_GPU_ENV=",SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
fi
case "$FAIR_ACTION" in
  cpu) FAIR_ARGS=(fair_checks.py cpu);;
  preflight) FAIR_ARGS=(fair_checks.py gpu);;
  slot) FAIR_ARGS=(fair_runner.py "$@");;
  postprocess) FAIR_ARGS=(fair_report.py);;
  *) echo "Unknown action: $FAIR_ACTION" >&2; exit 2;;
esac
exec "${FAIR_EXEC[@]}" --cleanenv --no-home \
  --bind "$RUN_LEGACY_PROJECT:/legacy:ro" \
  --bind "$RUN_PROJECT:/original:ro" \
  --bind "$HOME/reachability-reweight-v2:/reweight_previous:ro" \
  --bind "$FAIR_PROJECT:/workspace" --pwd /workspace/research/reachability_fair_v3 \
  --env "PYTHONPATH=/original/src:/workspace/research/reachability_fair_v3,REACH_ROOT=/workspace/experiments/fair_v3,FAIR_ROOT=/workspace/experiments/fair_v3,REACH_DATA=/original/experiments/reachability/data,BASE_REACH_ROOT=/original/experiments/reachability,PREVIOUS_REWEIGHT_DATA=/reweight_previous/experiments/reweight_v2/data,OMP_NUM_THREADS=$FAIR_THREADS,OPENBLAS_NUM_THREADS=$FAIR_THREADS,SLURM_JOB_ID=$SLURM_JOB_ID$FAIR_GPU_ENV" \
  "$RUN_IMAGE" "$RUN_PYTHON_IN_CONTAINER" "${FAIR_ARGS[@]}"
