#!/usr/bin/env bash
set -euo pipefail
cd "${QBM_ROOT:?}"
source ./slurm.env
export PYTHONDONTWRITEBYTECODE=1
exec python3 -B research/language_transport_20261105/code/controller.py
