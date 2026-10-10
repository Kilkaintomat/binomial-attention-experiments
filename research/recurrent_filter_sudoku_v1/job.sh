#!/bin/bash
set -euo pipefail
source .cache/site.sh
export PYTHONPATH="$RUN_PROJECT/src:$RUN_PROJECT"
mode="$1"
args=()
if [[ $# -gt 1 ]]; then args+=(--job "$2"); fi
exec /usr/bin/python3 -m research.recurrent_filter_sudoku_v1 "$mode" "${args[@]}"
