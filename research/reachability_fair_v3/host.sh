#!/bin/bash
set -euo pipefail
source "$HOME/directed-reachability-v1/.cache/site.sh"
export FAIR_PROJECT="$HOME/reachability-fair-v3"
FAIR_ACTION="${1:?action required}"
shift
if [[ "$FAIR_ACTION" == cpu ]]; then
  exec bash "$FAIR_PROJECT/research/reachability_fair_v3/container.sh" cpu
fi
exec "$RUN_HOST_PYTHON" "$FAIR_PROJECT/research/reachability_fair_v3/fair_queue.py" "$FAIR_ACTION" "$@"
