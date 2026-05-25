#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${ROOT_DIR}/upstream-aff-grasp"
AFF_LINK="${ROOT_DIR}/affordance-learning"

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  git clone https://github.com/Reagan1311/Aff-Grasp.git "${UPSTREAM_DIR}"
else
  git -C "${UPSTREAM_DIR}" pull --ff-only
fi

if [[ -e "${AFF_LINK}" && ! -L "${AFF_LINK}" ]]; then
  echo "affordance-learning already exists and is not a symlink: ${AFF_LINK}" >&2
  echo "Move it aside or keep using the existing directory." >&2
  exit 1
fi

if [[ ! -e "${AFF_LINK}" ]]; then
  ln -s upstream-aff-grasp/affordance-learning "${AFF_LINK}"
fi

echo "Ready: ${AFF_LINK}"
