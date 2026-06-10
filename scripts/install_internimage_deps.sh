#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERNIMAGE_DIR="${ROOT_DIR}/third_party/InternImage"

if [ ! -d "${INTERNIMAGE_DIR}/.git" ]; then
  git clone --depth 1 https://github.com/OpenGVLab/InternImage.git "${INTERNIMAGE_DIR}"
fi

cd "${INTERNIMAGE_DIR}/classification/ops_dcnv3"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5}" MAX_JOBS="${MAX_JOBS:-4}" sh ./make.sh
python test.py

echo "Set before running experiments:"
echo "export INTERNIMAGE_ROOT=${INTERNIMAGE_DIR}/classification"
