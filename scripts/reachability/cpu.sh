#!/bin/bash
set -euo pipefail
source .cache/site.sh
exec apptainer exec --cleanenv --no-home \
  --bind "$RUN_LEGACY_PROJECT:/legacy:ro" --bind "$RUN_PROJECT:/workspace" --pwd /workspace \
  --env "PYTHONPATH=/workspace/src:/workspace,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,SLURM_JOB_ID=$SLURM_JOB_ID,SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-},MPLCONFIGDIR=/workspace/.cache/matplotlib" \
  "$RUN_IMAGE" "$RUN_PYTHON_IN_CONTAINER" "$@"
