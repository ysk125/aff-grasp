#!/usr/bin/env bash
set -euo pipefail

echo "== identity =="
hostname
whoami
pwd

echo "== gpu =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi

echo "== memory =="
free -h || true

echo "== disk =="
df -h .
df -h "$HOME" || true

echo "== docker =="
docker --version
docker info --format 'Docker root: {{.DockerRootDir}}' || true
