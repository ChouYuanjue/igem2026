#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/runnel/miniconda3/envs/enzymecage/bin/python}"

echo "[info] This script prepares only the lightweight exploration package."
echo "[info] EnzymeCAGE dependencies should be installed according to upstream instructions."
echo "[info] No files in external_repos/ will be modified."

if [[ ! -d "${ROOT_DIR}/.venv" ]]; then
  echo "[create] ${ROOT_DIR}/.venv"
  "${PYTHON}" -m venv "${ROOT_DIR}/.venv"
fi

echo "[install] exploration package dependencies"
"${ROOT_DIR}/.venv/bin/python" -m pip install --upgrade pip
"${ROOT_DIR}/.venv/bin/python" -m pip install -e "${ROOT_DIR}"

echo "[done] Activate with: source .venv/bin/activate"
