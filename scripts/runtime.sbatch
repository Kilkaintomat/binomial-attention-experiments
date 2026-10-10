#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C LANG=C
transport_root="$(cd "${TRANSPORT_ROOT:?}" && pwd -P)"
transport_inputs="$(cd "${TRANSPORT_INPUTS:?}" && pwd -P)"
if [ ! -f "$transport_root/experiments/transport_reasoning/container_identity.json" ]; then
    transport_image_sha="$(sha256sum "${TRANSPORT_IMAGE:?}" | cut -d ' ' -f 1)"
    printf '{"sha256":"%s","description":"existing compatible compute container"}\n' "$transport_image_sha" > "$transport_root/experiments/transport_reasoning/container_identity.json"
fi
transport_gpu_args=(--no-home)
if [ -n "${SLURM_JOB_GPUS:-}" ] || [ -n "${SLURM_STEP_GPUS:-}" ]; then
    transport_gpu_args+=(--nv)
fi
exec apptainer exec --cleanenv "${transport_gpu_args[@]}" \
    --bind "$transport_root:/workspace" --bind "$transport_inputs:/inputs" \
    --pwd /workspace --env "SLURM_JOB_ID=${SLURM_JOB_ID:-}" \
    --env "TRANSPORT_COMMIT=${TRANSPORT_COMMIT:-}" \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}" \
    --env "OMP_NUM_THREADS=4,OPENBLAS_NUM_THREADS=4,MPLCONFIGDIR=/workspace/.cache/matplotlib,XDG_CACHE_HOME=/workspace/.cache" \
    "${TRANSPORT_IMAGE:?}" /workspace/.venv/bin/python "$@"
