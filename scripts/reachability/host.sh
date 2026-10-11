#!/bin/bash
set -euo pipefail
source .cache/site.sh
export PYTHONPATH="$RUN_PROJECT/src:$RUN_PROJECT"
exec "$RUN_HOST_PYTHON" -m transport_reasoning.reachability.scheduler "$@"
