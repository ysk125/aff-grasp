#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${AFF_GRASP_IMAGE:-$(id -un)-aff-grasp:cu121}"
INSTALL_INTERNIMAGE="${AFFGRASP_WITH_INTERNIMAGE:-0}"

cd "${ROOT_DIR}"
docker build \
  --build-arg USER_ID="$(id -u)" \
  --build-arg USER_NAME="$(id -un)" \
  --build-arg GROUP_ID="$(id -g)" \
  --build-arg GROUP_NAME="$(id -gn)" \
  --build-arg INSTALL_INTERNIMAGE="${INSTALL_INTERNIMAGE}" \
  -t "${IMAGE_NAME}" \
  .

echo "Built ${IMAGE_NAME}"
echo "InternImage dependencies: ${INSTALL_INTERNIMAGE}"
