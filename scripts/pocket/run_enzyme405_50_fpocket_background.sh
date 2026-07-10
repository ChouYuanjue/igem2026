#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

source /home/runnel/miniconda3/etc/profile.d/conda.sh
conda activate enzymecage

export PYTHON="/home/runnel/miniconda3/envs/enzymecage/bin/python"
export FPOCKET_BIN="${FPOCKET_BIN:-/home/runnel/miniconda3/envs/enzymecage/bin/fpocket}"

LOG_DIR="results/pocket/background_logs"
mkdir -p "${LOG_DIR}" "results/pocket/patches" "data/assets/fpocket"

printf '%s\n' "${FPOCKET_BIN}" > data/assets/fpocket/fpocket_path.txt

PIDS_FILE="${LOG_DIR}/fpocket_pids.txt"
touch "${PIDS_FILE}"

smoke_test_env() {
  "${PYTHON}" - <<'PY'
import drfp
import esm
import rdkit
import torch
import torch_geometric
print("env ok")
PY
}

result_complete() {
  local run_id="$1"
  "${PYTHON}" - "${ROOT_DIR}" "${run_id}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_id = sys.argv[2]
run_dir = root / "results/pocket" / run_id
summary_path = run_dir / "run_summary.json"
predictions = run_dir / "predictions/pocket_level_predictions.csv"
aggregation_files = list((run_dir / "aggregation").glob("enzyme_level_*.csv"))
metrics = run_dir / "metrics/metrics_top5_top10.json"

if not summary_path.exists() or not predictions.exists() or not aggregation_files or not metrics.exists():
    print("0")
    raise SystemExit(0)

try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
except Exception:
    print("0")
    raise SystemExit(0)

print("1" if summary.get("status") == "completed" else "0")
PY
}

append_pid() {
  local run_id="$1"
  local pid="$2"
  local log_path="$3"
  echo "${run_id} ${pid} ${log_path}" >> "${PIDS_FILE}"
}

launch_compare_baseline() {
  local run_id="$1"
  local config_path="$2"
  local log_path="${LOG_DIR}/${run_id}.nohup.log"
  echo "[launch] ${run_id} -> ${log_path}"
  nohup setsid "${PYTHON}" projects/active/pocket_robustness/runners/run_compare_baselines.py \
    --experiment_config "${config_path}" \
    --resume < /dev/null > "${log_path}" 2>&1 &
  append_pid "${run_id}" "$!" "${log_path}"
}

launch_compare_baseline_when_source_ready() {
  local run_id="$1"
  local config_path="$2"
  local source_run_id="$3"
  local log_path="${LOG_DIR}/${run_id}.nohup.log"
  echo "[launch-wait] ${run_id} waits for ${source_run_id} -> ${log_path}"
  nohup setsid env \
    ROOT_DIR="${ROOT_DIR}" \
    PYTHON="${PYTHON}" \
    CONFIG_PATH="${config_path}" \
    SOURCE_RUN_ID="${source_run_id}" \
    bash -c '
      set -euo pipefail
      while true; do
        if [[ -f "${ROOT_DIR}/results/pocket/${SOURCE_RUN_ID}/run_summary.json" ]] \
          && [[ -f "${ROOT_DIR}/results/pocket/${SOURCE_RUN_ID}/predictions/pocket_level_predictions.csv" ]] \
          && compgen -G "${ROOT_DIR}/results/pocket/${SOURCE_RUN_ID}/aggregation/enzyme_level_*.csv" > /dev/null \
          && [[ -f "${ROOT_DIR}/results/pocket/${SOURCE_RUN_ID}/metrics/metrics_top5_top10.json" ]] \
          && grep -q "\"status\": \"completed\"" "${ROOT_DIR}/results/pocket/${SOURCE_RUN_ID}/run_summary.json"; then
          break
        fi
        sleep 60
      done
      exec "${PYTHON}" projects/active/pocket_robustness/runners/run_compare_baselines.py --experiment_config "${CONFIG_PATH}" --resume
    ' < /dev/null > "${log_path}" 2>&1 &
  append_pid "${run_id}" "$!" "${log_path}"
}

smoke_test_env

if result_complete "enzyme405_50_fpocket_topk_rank_weighted" | grep -q '^1$'; then
  echo "[skip] enzyme405_50_fpocket_topk_rank_weighted already completed."
else
  launch_compare_baseline \
    "enzyme405_50_fpocket_topk_rank_weighted" \
    "projects/active/pocket_robustness/configs/generated_best_available/enzyme405_50_fpocket_topk_rank_weighted.yaml"
fi

if result_complete "enzyme405_50_fpocket_top1" | grep -q '^1$'; then
  echo "[skip] enzyme405_50_fpocket_top1 already completed."
else
  launch_compare_baseline_when_source_ready \
    "enzyme405_50_fpocket_top1" \
    "projects/active/pocket_robustness/configs/generated_best_available/enzyme405_50_fpocket_top1.yaml" \
    "enzyme405_50_fpocket_topk_rank_weighted"
fi

if result_complete "enzyme405_50_p2rank_fpocket_union_max" | grep -q '^1$'; then
  echo "[skip] enzyme405_50_p2rank_fpocket_union_max already completed."
else
  launch_compare_baseline \
    "enzyme405_50_p2rank_fpocket_union_max" \
    "projects/active/pocket_robustness/configs/generated_best_available/enzyme405_50_p2rank_fpocket_union_max.yaml"
fi

if result_complete "enzyme405_50_p2rank_fpocket_union_source_weighted" | grep -q '^1$'; then
  echo "[skip] enzyme405_50_p2rank_fpocket_union_source_weighted already completed."
else
  launch_compare_baseline \
    "enzyme405_50_p2rank_fpocket_union_source_weighted" \
    "projects/active/pocket_robustness/configs/generated_best_available/enzyme405_50_p2rank_fpocket_union_source_weighted.yaml"
fi

echo "[done] fpocket background launch complete."
echo "[done] PID log: ${PIDS_FILE}"
