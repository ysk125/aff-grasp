#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${ROOT_DIR}/upstream-aff-grasp"
UPSTREAM_AFF_DIR="${UPSTREAM_DIR}/affordance-learning"
AFF_ROOT="${AFF_ROOT:-${ROOT_DIR}/affordance-learning}"
AFFGRASP_OFFLINE="${AFFGRASP_OFFLINE:-0}"

clone_or_update_upstream() {
  if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
    git clone https://github.com/Reagan1311/Aff-Grasp.git "${UPSTREAM_DIR}"
  elif [[ "${AFFGRASP_OFFLINE}" != "1" ]]; then
    git -C "${UPSTREAM_DIR}" pull --ff-only
  else
    echo "Using existing upstream source without git pull: ${UPSTREAM_DIR}"
  fi
}

link_if_missing() {
  local source_path="$1"
  local target_path="$2"
  if [[ -e "${target_path}" || -L "${target_path}" ]]; then
    return
  fi
  ln -s "${source_path}" "${target_path}"
}

clone_or_update_upstream

if [[ ! -d "${UPSTREAM_AFF_DIR}" ]]; then
  echo "Missing upstream affordance-learning directory: ${UPSTREAM_AFF_DIR}" >&2
  exit 1
fi

mkdir -p "${AFF_ROOT}"

for name in data models utils train.py test.py; do
  link_if_missing "${UPSTREAM_AFF_DIR}/${name}" "${AFF_ROOT}/${name}"
done

echo "GAT runtime ready:"
echo "  official source: ${UPSTREAM_AFF_DIR}"
echo "  runtime/assets : ${AFF_ROOT}"
echo
echo "Expected assets:"
echo "  ${AFF_ROOT}/dinov2_vitb14_pretrain.pth"
echo "  ${AFF_ROOT}/pretrained_aff_grasp.pth"
echo "  ${AFF_ROOT}/ag_dataset/ego_train"
echo "  ${AFF_ROOT}/ag_dataset/Affordance_Evaluation_Dataset"
echo "  ${AFF_ROOT}/depth or ${AFF_ROOT}/ag_dataset/depth"
