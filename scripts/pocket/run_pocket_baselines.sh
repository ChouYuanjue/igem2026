#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

PYTHON="${PYTHON:-/home/runnel/miniconda3/envs/enzymecage/bin/python}"

"${PYTHON}" projects/active/pocket_robustness/runners/run_compare_baselines.py \
  --experiment_config projects/active/pocket_robustness/configs/demo_p2rank_top1.yaml

"${PYTHON}" projects/active/pocket_robustness/runners/run_compare_baselines.py \
  --experiment_config projects/active/pocket_robustness/configs/demo_p2rank_topk.yaml

echo "[done] Results are under results/pocket/demo_p2rank_top1 and results/pocket/demo_p2rank_topk"
