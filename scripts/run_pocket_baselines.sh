#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-/home/runnel/miniconda3/envs/enzymecage/bin/python}"

"${PYTHON}" explorations/pocket/runners/run_compare_baselines.py \
  --experiment_config explorations/pocket/configs/demo_p2rank_top1.yaml

"${PYTHON}" explorations/pocket/runners/run_compare_baselines.py \
  --experiment_config explorations/pocket/configs/demo_p2rank_topk.yaml

echo "[done] Results are under results/pocket/demo_p2rank_top1 and results/pocket/demo_p2rank_topk"
