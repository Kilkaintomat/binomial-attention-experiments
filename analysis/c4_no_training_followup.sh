#!/bin/bash
set -euo pipefail
source "$HOME/directed-reachability-v1/.cache/site.sh"
export CURRICULUM_PROJECT="$HOME/reachability-curriculum-v4"
exec "$RUN_HOST_PYTHON" "$CURRICULUM_PROJECT/analysis/c4_no_training_followup.py" "${1:-0}"
