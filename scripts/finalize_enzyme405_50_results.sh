#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}" || exit 1

export PYTHON="/home/runnel/miniconda3/envs/enzymecage/bin/python"

mkdir -p results/pocket

"${PYTHON}" explorations/pocket/analysis/build_best_available_result_matrix.py \
  --results_root results/pocket \
  --dataset_scales enzyme405_50 \
  --output_csv results/pocket/enzyme405_50_result_matrix.csv \
  --output_md results/pocket/enzyme405_50_result_matrix.md \
  --conclusion_md results/pocket/enzyme405_50_conclusion.md

"${PYTHON}" explorations/pocket/analysis/compare_all_baselines.py \
  --results_root results/pocket \
  --baseline_matrix explorations/pocket/configs/baseline_matrix.yaml \
  --dataset_scale enzyme405_50 \
  --output_dir results/pocket/comparison

"${PYTHON}" explorations/pocket/analysis/summarize_experiment_status.py \
  --results_root results/pocket \
  --matrix_csv results/pocket/enzyme405_50_result_matrix.csv \
  --output results/pocket/experiment_status.md

"${PYTHON}" explorations/pocket/analysis/render_enzyme405_50_chinese_report.py \
  --results_root results/pocket \
  --matrix_csv results/pocket/enzyme405_50_result_matrix.csv \
  --matrix_md results/pocket/enzyme405_50_result_matrix.md \
  --output results/pocket/enzyme405_50_中文实验报告.md

echo "[done] Finalized enzyme405_50 results."
echo "[done] Matrix CSV: results/pocket/enzyme405_50_result_matrix.csv"
echo "[done] Matrix MD: results/pocket/enzyme405_50_result_matrix.md"
echo "[done] Conclusion MD: results/pocket/enzyme405_50_conclusion.md"
echo "[done] Experiment status: results/pocket/experiment_status.md"
echo "[done] Chinese report: results/pocket/enzyme405_50_中文实验报告.md"
