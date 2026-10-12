#!/bin/bash
set -euo pipefail
source "$HOME/directed-reachability-v1/.cache/site.sh"
export CURRICULUM_PROJECT="$HOME/reachability-curriculum-v4"
CURRICULUM_ACTION="${1:?action required}"
shift
if [[ "$CURRICULUM_ACTION" == cpu || "$CURRICULUM_ACTION" == test_prepare || "$CURRICULUM_ACTION" == no_training ]]; then
  exec bash "$CURRICULUM_PROJECT/research/reachability_curriculum_v4/container.sh" "$CURRICULUM_ACTION" "$@"
fi
exec "$RUN_HOST_PYTHON" "$CURRICULUM_PROJECT/research/reachability_curriculum_v4/c4_queue.py" "$CURRICULUM_ACTION" "$@"
