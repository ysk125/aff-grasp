#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
CONFIG_PATH="${2:-experiments/affgrasp_mmseg/configs/segformer_affgrasp/segformer_a.py}"
CONTAINER_NAME="${3:-affgrasp-mmseg-$(basename "${CONFIG_PATH}" .py)}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${AFF_GRASP_IMAGE:-$(id -un)-aff-grasp:cu121}"

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container already exists: ${CONTAINER_NAME}" >&2
  echo "Inspect it with: docker ps -a --filter name=${CONTAINER_NAME}" >&2
  echo "Remove it after checking logs with: docker rm ${CONTAINER_NAME}" >&2
  exit 1
fi

cd "${ROOT_DIR}"
docker run --gpus "device=${GPU_ID}" --shm-size=8g -d \
  --name "${CONTAINER_NAME}" \
  --mount "type=bind,source=${ROOT_DIR},target=${ROOT_DIR}" \
  --workdir "${ROOT_DIR}" \
  --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env OMP_NUM_THREADS=4 \
  --env MKL_NUM_THREADS=4 \
  --env OPENBLAS_NUM_THREADS=4 \
  --env NUMEXPR_NUM_THREADS=4 \
  "${IMAGE_NAME}" \
  bash -lc "
    set -euo pipefail
    python experiments/affgrasp_mmseg/preflight.py --check-timm-models
    python experiments/affgrasp_mmseg/train_affgrasp_mmseg.py \
      --config '${CONFIG_PATH}' \
      --gpu 0 \
      --test-after
  "

echo "Started ${CONTAINER_NAME} on GPU ${GPU_ID} with ${CONFIG_PATH}."
echo "Follow logs: docker logs -f ${CONTAINER_NAME}"

