#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
OUTPUT_ROOT="${AFFGRASP_OUTPUT_ROOT:-outputs}"
CONTAINER_NAME="${AFFGRASP_DIAGNOSTICS_CONTAINER:-affgrasp-mmseg-diagnostics}"
IMAGE="${AFFGRASP_DOCKER_IMAGE:-saka-aff-grasp:cu121}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run --gpus "device=${GPU_ID}" --shm-size=8g -d \
  --name "${CONTAINER_NAME}" \
  --mount "type=bind,source=${WORKSPACE},target=${WORKSPACE}" \
  --workdir "${WORKSPACE}" \
  --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env PYTHONPATH="${WORKSPACE}" \
  "${IMAGE}" \
  bash -lc "python tools/run_affgrasp_diagnostics.py --output-root '${OUTPUT_ROOT}' --gpu 0"

echo "Started ${CONTAINER_NAME} on physical GPU ${GPU_ID}."
echo "Follow logs: docker logs -f ${CONTAINER_NAME}"
