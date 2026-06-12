#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
OUTPUT_ROOT="${AFFGRASP_OUTPUT_ROOT:-outputs}"
EXPERIMENT_FAMILY="${AFFGRASP_EXPERIMENT_FAMILY:-all}"

SEGFORMER_CONFIGS=(
  "experiments/affgrasp_mmseg/configs/segformer_affgrasp/segformer_a.py"
  "experiments/affgrasp_mmseg/configs/segformer_affgrasp/segformer_d.py"
  "experiments/affgrasp_mmseg/configs/segformer_affgrasp/segformer_b.py"
  "experiments/affgrasp_mmseg/configs/segformer_affgrasp/segformer_c.py"
)

INTERNIMAGE_CONFIGS=(
  "experiments/affgrasp_mmseg/configs/internimage_affgrasp/internimage_a.py"
  "experiments/affgrasp_mmseg/configs/internimage_affgrasp/internimage_d.py"
  "experiments/affgrasp_mmseg/configs/internimage_affgrasp/internimage_c.py"
)

case "${EXPERIMENT_FAMILY}" in
  segformer)
    CONFIGS=("${SEGFORMER_CONFIGS[@]}")
    ;;
  internimage)
    CONFIGS=("${INTERNIMAGE_CONFIGS[@]}")
    ;;
  all)
    CONFIGS=("${SEGFORMER_CONFIGS[@]}")
    if [ "${AFFGRASP_INCLUDE_EXPERIMENTAL_INTERNIMAGE:-0}" = "1" ]; then
      CONFIGS+=("${INTERNIMAGE_CONFIGS[@]}")
    fi
    ;;
  *)
    echo "Unknown AFFGRASP_EXPERIMENT_FAMILY: ${EXPERIMENT_FAMILY}" >&2
    exit 2
    ;;
esac

echo "== preflight =="
python experiments/affgrasp_mmseg/preflight.py --check-timm-models

mkdir -p "${OUTPUT_ROOT}/_logs"
summary="${OUTPUT_ROOT}/${EXPERIMENT_FAMILY}_experiments_status.tsv"
printf "experiment\tconfig\tstatus\tstarted_at\tfinished_at\n" > "${summary}"

for config in "${CONFIGS[@]}"; do
  name="$(basename "${config}" .py)"
  started_at="$(date --iso-8601=seconds)"
  echo
  echo "== full experiment: ${name} =="
  printf "%s\t%s\trunning\t%s\t\n" "${name}" "${config}" "${started_at}" >> "${summary}"
  log_path="${OUTPUT_ROOT}/_logs/${name}.log"
  if python experiments/affgrasp_mmseg/train_affgrasp_mmseg.py \
      --config "${config}" \
      --output-root "${OUTPUT_ROOT}" \
      --gpu "${GPU_ID}" \
      --test-after 2>&1 | tee "${log_path}"; then
    finished_at="$(date --iso-8601=seconds)"
    printf "%s\t%s\tsucceeded\t%s\t%s\n" "${name}" "${config}" "${started_at}" "${finished_at}" >> "${summary}"
  else
    finished_at="$(date --iso-8601=seconds)"
    printf "%s\t%s\tfailed\t%s\t%s\n" "${name}" "${config}" "${started_at}" "${finished_at}" >> "${summary}"
    echo "Experiment failed: ${name}. See ${log_path}" >&2
    exit 1
  fi
done

DIAGNOSTIC_EXPERIMENTS=()
for config in "${CONFIGS[@]}"; do
  DIAGNOSTIC_EXPERIMENTS+=("$(basename "${config}" .py)")
done
python tools/run_affgrasp_diagnostics.py \
  --output-root "${OUTPUT_ROOT}" \
  --experiments "${DIAGNOSTIC_EXPERIMENTS[@]}" \
  --summarize-only

echo
echo "All experiments finished. Summary: ${summary}"
