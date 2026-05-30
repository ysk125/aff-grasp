#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"
bash tools/setup_upstream_aff_grasp.sh

echo "== build Aff-Grasp CUDA extension =="
cd "${ROOT_DIR}/affordance-learning/models/dino/ops"
rm -rf build
python setup.py build

source "${ROOT_DIR}/scripts/affgrasp_env.sh"
python -c "import MultiScaleDeformableAttention; print('MSDA ok')"

echo "== download AED/model/DINOv2 assets =="
cd "${ROOT_DIR}"
python tools/download_aff_grasp_assets.py --eval-only

echo "Bootstrap complete."
