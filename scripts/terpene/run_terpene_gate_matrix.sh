#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -n "${PYTHON:-}" ]]; then
  :
elif [[ -x "/home/zhangruncheng/miniconda3/envs/chem/bin/python" ]]; then
  PYTHON="/home/zhangruncheng/miniconda3/envs/chem/bin/python"
elif [[ -x "/home/zhangruncheng/miniconda3/envs/sci-mol/bin/python" ]]; then
  PYTHON="/home/zhangruncheng/miniconda3/envs/sci-mol/bin/python"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON="python3"
fi

cd "${PROJECT_ROOT}"
"${PYTHON}" projects/active/terpene_screening/gate_matrix.py "$@"
