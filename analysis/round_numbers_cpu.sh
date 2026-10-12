#!/bin/bash
set -euo pipefail
source "$HOME/directed-reachability-v1/.cache/site.sh"
ROUND_DIAG="$HOME/reachability-round-diagnosis"
exec apptainer exec --cleanenv --no-home \
  --bind "$RUN_LEGACY_PROJECT:/legacy:ro" --bind "$RUN_PROJECT:/original:ro" \
  --bind "$ROUND_DIAG:/diagnosis" --pwd /diagnosis \
  --env "PYTHONPATH=/original/src,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,SLURM_JOB_ID=$SLURM_JOB_ID" \
  "$RUN_IMAGE" "$RUN_PYTHON_IN_CONTAINER" /diagnosis/round_numbers_diagnosis.py
