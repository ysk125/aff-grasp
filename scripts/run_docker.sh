#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${AFF_GRASP_IMAGE:-$(id -un)-aff-grasp:cu121}"
RUNTIME_TMP="/dev/shm/affgrasp-tmp"

cd "${ROOT_DIR}"
docker run --gpus "device=${GPU_ID}" --shm-size=8g -it --rm \
  --mount "type=bind,source=${ROOT_DIR},target=${ROOT_DIR}" \
  --workdir "${ROOT_DIR}" \
  --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH:-}" \
  --env TMPDIR="${RUNTIME_TMP}" \
  --env TMP="${RUNTIME_TMP}" \
  --env TEMP="${RUNTIME_TMP}" \
  --env TORCHINDUCTOR_CACHE_DIR="${RUNTIME_TMP}/torchinductor" \
  --env HF_HOME="${RUNTIME_TMP}/huggingface" \
  --env HF_HUB_CACHE="${RUNTIME_TMP}/huggingface/hub" \
  --env OMP_NUM_THREADS=4 \
  --env MKL_NUM_THREADS=4 \
  --env OPENBLAS_NUM_THREADS=4 \
  --env NUMEXPR_NUM_THREADS=4 \
  "${IMAGE_NAME}" \
  bash -lc 'mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${HF_HUB_CACHE}"; source scripts/affgrasp_env.sh; exec bash'
