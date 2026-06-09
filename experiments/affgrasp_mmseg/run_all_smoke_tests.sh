#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
EPOCHS="${AFFGRASP_SMOKE_EPOCHS:-1}"
MAX_TRAIN="${AFFGRASP_SMOKE_MAX_TRAIN:-8}"
MAX_VAL="${AFFGRASP_SMOKE_MAX_VAL:-4}"
OUTPUT_ROOT="${AFFGRASP_SMOKE_OUTPUT_ROOT:-outputs_smoke}"

CONFIGS=(
  "experiments/affgrasp_mmseg/configs/segformer_affgrasp/segformer_a.py"
  "experiments/affgrasp_mmseg/configs/segformer_affgrasp/segformer_d.py"
  "experiments/affgrasp_mmseg/configs/segformer_affgrasp/segformer_b.py"
  "experiments/affgrasp_mmseg/configs/segformer_affgrasp/segformer_c.py"
  "experiments/affgrasp_mmseg/configs/internimage_affgrasp/internimage_a.py"
  "experiments/affgrasp_mmseg/configs/internimage_affgrasp/internimage_d.py"
  "experiments/affgrasp_mmseg/configs/internimage_affgrasp/internimage_c.py"
)

echo "== preflight =="
python experiments/affgrasp_mmseg/preflight.py --check-timm-models

for config in "${CONFIGS[@]}"; do
  name="$(basename "${config}" .py)"
  echo
  echo "== smoke test: ${name} =="
  python experiments/affgrasp_mmseg/train_affgrasp_mmseg.py \
    --config "${config}" \
    --output-root "${OUTPUT_ROOT}" \
    --epochs "${EPOCHS}" \
    --max-train-samples "${MAX_TRAIN}" \
    --max-val-samples "${MAX_VAL}" \
    --gpu "${GPU_ID}"
done

echo
echo "All smoke tests finished. Outputs: ${OUTPUT_ROOT}/"

