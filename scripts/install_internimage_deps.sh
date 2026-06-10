#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERNIMAGE_DIR="${ROOT_DIR}/third_party/InternImage"

if [ ! -d "${INTERNIMAGE_DIR}/.git" ]; then
  git clone --depth 1 https://github.com/OpenGVLab/InternImage.git "${INTERNIMAGE_DIR}"
fi

cd "${INTERNIMAGE_DIR}/classification/ops_dcnv3"
sed -i 's/if torch.cuda.is_available() and CUDA_HOME is not None:/if (torch.cuda.is_available() or os.getenv("FORCE_CUDA") == "1") and CUDA_HOME is not None:/' setup.py
grep -q 'FORCE_CUDA' setup.py
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
FORCE_CUDA=1 \
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5}" \
MAX_JOBS="${MAX_JOBS:-4}" \
sh ./make.sh

if python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  python test.py
else
  echo "DCNv3 was compiled without a visible runtime GPU; run test.py inside a GPU-enabled container."
fi

echo "Set before running experiments:"
echo "export INTERNIMAGE_ROOT=${INTERNIMAGE_DIR}/classification"
