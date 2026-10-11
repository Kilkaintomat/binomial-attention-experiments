#!/bin/bash
set -euo pipefail
source .cache/site.sh
export PYTHONPATH="$RUN_PROJECT/src:$RUN_PROJECT"
if [[ $# -gt 1 ]]; then
  exec "$RUN_HOST_PYTHON" -m research.sudoku_local_mask_v1 "$1" --job "$2"
fi
exec "$RUN_HOST_PYTHON" -m research.sudoku_local_mask_v1 "$1"
