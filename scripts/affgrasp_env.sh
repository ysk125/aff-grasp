#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPS_DIR="${ROOT_DIR}/affordance-learning/models/dino/ops"
MSDA_SO="$(find "${OPS_DIR}/build" -type f -name 'MultiScaleDeformableAttention*.so' -print -quit 2>/dev/null || true)"

export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH:-}"
if [[ -n "${MSDA_SO}" ]]; then
  export PYTHONPATH="$(dirname "${MSDA_SO}"):${PYTHONPATH:-}"
fi

