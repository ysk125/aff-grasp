#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
CONTAINER_NAME="${2:-affgrasp-mmseg-all}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${AFF_GRASP_IMAGE:-$(id -un)-aff-grasp:cu121}"
INCLUDE_INTERNIMAGE="${AFFGRASP_INCLUDE_EXPERIMENTAL_INTERNIMAGE:-0}"

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
  --env PYTHONPATH="${ROOT_DIR}" \
  --env OMP_NUM_THREADS=4 \
  --env MKL_NUM_THREADS=4 \
  --env OPENBLAS_NUM_THREADS=4 \
  --env NUMEXPR_NUM_THREADS=4 \
  --env AFFGRASP_INCLUDE_EXPERIMENTAL_INTERNIMAGE="${INCLUDE_INTERNIMAGE}" \
  "${IMAGE_NAME}" \
  bash -lc "
    set -euo pipefail
    bash experiments/affgrasp_mmseg/run_all_experiments.sh 0
  "

echo "Started ${CONTAINER_NAME} on GPU ${GPU_ID}."
if [ "${INCLUDE_INTERNIMAGE}" = "1" ]; then
  echo "This runs all seven SegFormer/InternImage experiments sequentially."
else
  echo "This runs the four SegFormer experiments sequentially."
fi
echo "Follow logs: docker logs -f ${CONTAINER_NAME}"
echo "Check status: docker ps -a --filter name=${CONTAINER_NAME}"
