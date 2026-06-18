#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AFF_ROOT="${AFF_ROOT:-${ROOT_DIR}/affordance-learning}"
OPS_DIR="${AFF_ROOT}/models/dino/ops"

if [[ ! -d "${OPS_DIR}" ]]; then
  echo "Missing GAT CUDA ops source: ${OPS_DIR}" >&2
  echo "Run: bash scripts/setup_gat_runtime.sh" >&2
  exit 1
fi

cd "${OPS_DIR}"
python -c "import torch; print('torch:', torch.__version__, 'torch cuda:', torch.version.cuda); assert torch.version.cuda and torch.version.cuda.startswith('12.1'), torch.version.cuda"
python setup.py build

cd "${ROOT_DIR}"
source scripts/affgrasp_env.sh
python -c "import MultiScaleDeformableAttention; print('MSDA ok')"
