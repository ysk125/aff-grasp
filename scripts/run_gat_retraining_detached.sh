#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
CONTAINER_NAME="${2:-affgrasp-gat-train}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${AFF_GRASP_IMAGE:-$(id -un)-aff-grasp:cu121}"
OUTPUT_DIR="${GAT_OUTPUT_DIR:-outputs/gat_retraining_cosine}"
NUM_WORKERS="${GAT_NUM_WORKERS:-4}"
BATCH_SIZE="${GAT_BATCH_SIZE:-8}"
EPOCHS="${GAT_EPOCHS:-15}"
SCHEDULER="${GAT_SCHEDULER:-cosine}"

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container already exists: ${CONTAINER_NAME}" >&2
  echo "Inspect it with: docker ps -a --filter name=${CONTAINER_NAME}" >&2
  echo "Follow logs with: docker logs -f ${CONTAINER_NAME}" >&2
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
    bash scripts/setup_gat_runtime.sh
    bash scripts/build_gat_ops.sh
    source scripts/affgrasp_env.sh
    python -m experiments.affgrasp_gat.validate_data \
      --data-root affordance-learning/ag_dataset \
      --output-dir ${OUTPUT_DIR}/data_validation \
      --visualization-count 50
    python -m experiments.affgrasp_gat.train_gat \
      --source-root upstream-aff-grasp/affordance-learning \
      --runtime-root affordance-learning \
      --data-root affordance-learning/ag_dataset \
      --output-dir ${OUTPUT_DIR} \
      --scheduler ${SCHEDULER} \
      --epochs ${EPOCHS} \
      --batch-size ${BATCH_SIZE} \
      --num-workers ${NUM_WORKERS} \
      --gpu 0
  "

echo "Started ${CONTAINER_NAME} on physical GPU ${GPU_ID}."
echo "Follow logs: docker logs -f ${CONTAINER_NAME}"
echo "Outputs: ${OUTPUT_DIR}"

