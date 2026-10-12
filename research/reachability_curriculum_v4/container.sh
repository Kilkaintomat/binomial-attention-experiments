#!/bin/bash
set -euo pipefail
source "$HOME/directed-reachability-v1/.cache/site.sh"
CURRICULUM_PROJECT="$HOME/reachability-curriculum-v4"
CURRICULUM_ACTION="${1:?action required}"
shift
CURRICULUM_EXEC=(apptainer exec)
CURRICULUM_THREADS=1
CURRICULUM_GPU_ENV=""
if [[ "$CURRICULUM_ACTION" == preflight || "$CURRICULUM_ACTION" == slot || "$CURRICULUM_ACTION" == eval_slot ]]; then
  CURRICULUM_EXEC+=(--nv)
  CURRICULUM_THREADS=4
  CURRICULUM_GPU_ENV=",SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
fi
case "$CURRICULUM_ACTION" in
  cpu) CURRICULUM_ARGS=(c4_checks.py cpu);;
  preflight) CURRICULUM_ARGS=(c4_checks.py gpu);;
  slot) CURRICULUM_ARGS=(c4_train.py "$@");;
  test_prepare) CURRICULUM_ARGS=(c4_evaluate.py prepare);;
  eval_slot) CURRICULUM_ARGS=(c4_evaluate.py evaluate "$@");;
  postprocess) CURRICULUM_ARGS=(c4_evaluate.py report);;
  no_training) CURRICULUM_ARGS=(c4_no_training.py "$@");;
  *) echo "Unknown action: $CURRICULUM_ACTION" >&2; exit 2;;
esac
exec "${CURRICULUM_EXEC[@]}" --cleanenv --no-home \
  --bind "$RUN_LEGACY_PROJECT:/legacy:ro" --bind "$RUN_PROJECT:/original:ro" \
  --bind "$HOME/reachability-fair-v3:/fair_previous:ro" \
  --bind "$HOME/reachability-reweight-v2:/reweight_previous:ro" \
  --bind "$CURRICULUM_PROJECT:/workspace" --pwd /workspace/research/reachability_curriculum_v4 \
  --env "PYTHONPATH=/original/src:/workspace/research/reachability_curriculum_v4,REACH_ROOT=/workspace/experiments/curriculum_v4,CURRICULUM_ROOT=/workspace/experiments/curriculum_v4,REACH_DATA=/original/experiments/reachability/data,OMP_NUM_THREADS=$CURRICULUM_THREADS,OPENBLAS_NUM_THREADS=$CURRICULUM_THREADS,SLURM_JOB_ID=$SLURM_JOB_ID$CURRICULUM_GPU_ENV" \
  "$RUN_IMAGE" "$RUN_PYTHON_IN_CONTAINER" "${CURRICULUM_ARGS[@]}"
