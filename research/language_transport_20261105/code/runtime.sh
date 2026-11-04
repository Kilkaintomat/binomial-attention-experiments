#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C LANG=C
lt_root="${QBM_ROOT:?}"
lt_side="$lt_root/research/language_transport_20261105"
lt_args=(--no-home)
if [ -n "${SLURM_JOB_GPUS:-}" ]; then lt_args+=(--nv); fi
exec apptainer exec --cleanenv "${lt_args[@]}" \
  --bind "$lt_root:/workspace:ro" \
  --bind "$lt_side:/workspace/research/language_transport_20261105:rw" --pwd /workspace \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID:-},CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}},LT_PROJECT_ROOT=/workspace,OMP_NUM_THREADS=4,OPENBLAS_NUM_THREADS=4,PYTHONPATH=/workspace/research/language_transport_20261105/code:/workspace/research/language_transport_20261105/.deps,PYTHONDONTWRITEBYTECODE=1,PYTHONUNBUFFERED=1,HF_HOME=/workspace/research/language_transport_20261105/.cache/huggingface,XDG_CACHE_HOME=/workspace/research/language_transport_20261105/.cache,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,TOKENIZERS_PARALLELISM=false" \
  "${QBM_IMAGE:?}" .venv/bin/python "$@"
