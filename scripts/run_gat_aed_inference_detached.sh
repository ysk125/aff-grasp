#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
CHECKPOINT="${2:-pretrained_aff_grasp.pth}"
CONTAINER_NAME="${3:-affgrasp-gat-aed}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${AFF_GRASP_IMAGE:-$(id -un)-aff-grasp:cu121}"
OUTPUT_DIR="${GAT_AED_OUTPUT_DIR:-analysis/gat_aed_$(date +%Y%m%d_%H%M%S)}"
THRESHOLD="${GAT_AED_THRESHOLD:-0.8}"
RUNTIME_TMP="${GAT_RUNTIME_TMP:-${ROOT_DIR}/.tmp/gat-aed-${CONTAINER_NAME}}"
if [[ "${CHECKPOINT}" == /* || "${CHECKPOINT}" == "pretrained_aff_grasp.pth" ]]; then
  CHECKPOINT_ARG="${CHECKPOINT}"
else
  CHECKPOINT_ARG="${ROOT_DIR}/${CHECKPOINT}"
fi

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container already exists: ${CONTAINER_NAME}" >&2
  echo "Inspect it with: docker ps -a --filter name=${CONTAINER_NAME}" >&2
  echo "Follow logs with: docker logs -f ${CONTAINER_NAME}" >&2
  exit 1
fi

cd "${ROOT_DIR}"
mkdir -p "${RUNTIME_TMP}"
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
  --env TMPDIR="${RUNTIME_TMP}" \
  --env TMP="${RUNTIME_TMP}" \
  --env TEMP="${RUNTIME_TMP}" \
  --env TORCHINDUCTOR_CACHE_DIR="${RUNTIME_TMP}/torchinductor" \
  "${IMAGE_NAME}" \
  bash -lc "
    set -euo pipefail
    mkdir -p '${RUNTIME_TMP}' '${RUNTIME_TMP}/torchinductor'
    bash scripts/setup_gat_runtime.sh
    bash scripts/build_gat_ops.sh
    source scripts/affgrasp_env.sh
    python tools/run_aff_grasp_eval_maps.py \
      --aff-root affordance-learning \
      --model-file ${CHECKPOINT_ARG} \
      --output-dir ${OUTPUT_DIR} \
      --artifact-profile analysis \
      --threshold ${THRESHOLD}
    python tools/analyze_aed_metrics.py --analysis-root ${OUTPUT_DIR}
    python tools/build_aed_review_bundle.py --analysis-root ${OUTPUT_DIR}
  "

echo "Started ${CONTAINER_NAME} on physical GPU ${GPU_ID}."
echo "Checkpoint: ${CHECKPOINT_ARG}"
echo "Follow logs: docker logs -f ${CONTAINER_NAME}"
echo "Outputs: ${OUTPUT_DIR}"
