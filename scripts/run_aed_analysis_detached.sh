#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
CONTAINER_NAME="${2:-affgrasp-aed-analysis}"
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
  bash -lc '
    set -euo pipefail
    source scripts/affgrasp_env.sh
    python -c "import torch; print(\"CUDA available:\", torch.cuda.is_available())"
    python -c "import MultiScaleDeformableAttention; print(\"MSDA ok\")"
    python tools/run_aff_grasp_eval_maps.py --artifact-profile analysis --threshold 0.8
    ANALYSIS_ROOT="$(find analysis -mindepth 1 -maxdepth 1 -type d -printf "%T@ %p\n" | sort -n | tail -1 | cut -d" " -f2-)"
    test -n "${ANALYSIS_ROOT}"
    python tools/analyze_aed_metrics.py --analysis-root "${ANALYSIS_ROOT}"
    python tools/build_aed_review_bundle.py --analysis-root "${ANALYSIS_ROOT}"
  '

echo "Started ${CONTAINER_NAME} on GPU ${GPU_ID}."
echo "Follow logs: docker logs -f ${CONTAINER_NAME}"
echo "Check status: docker ps -a --filter name=${CONTAINER_NAME}"
