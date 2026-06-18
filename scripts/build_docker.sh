#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${AFF_GRASP_IMAGE:-$(id -un)-aff-grasp:cu121}"

cd "${ROOT_DIR}"
docker build \
  --build-arg USER_ID="$(id -u)" \
  --build-arg USER_NAME="$(id -un)" \
  --build-arg GROUP_ID="$(id -g)" \
  --build-arg GROUP_NAME="$(id -gn)" \
  -t "${IMAGE_NAME}" \
  .

echo "Built ${IMAGE_NAME}"
echo "Purpose: GAT retraining, AED inference, and metric/history artifacts only"
