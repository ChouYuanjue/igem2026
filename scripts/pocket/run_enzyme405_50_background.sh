#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

export PYTHON="/home/runnel/miniconda3/envs/enzymecage/bin/python"
LOG_DIR="results/pocket/background_logs"
PATCH_DIR="results/pocket/patches"
mkdir -p "${LOG_DIR}" "${PATCH_DIR}"

PIDS_FILE="${LOG_DIR}/pids.txt"
touch "${PIDS_FILE}"

kill_stale_runs() {
  while IFS= read -r pid cmd; do
    pid="${pid//[[:space:]]/}"
    if [[ "${cmd}" == *"enzyme405_100"* || "${cmd}" == *"all_feasible"* ]]; then
      if [[ "${pid}" != "$$" ]]; then
        kill "${pid}" 2>/dev/null || true
      fi
    fi
  done < <(ps -eo pid=,cmd=)
}

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
  nohup "${PYTHON}" projects/active/pocket_robustness/runners/run_compare_baselines.py \
    --experiment_config "${config_path}" \
    --resume > "${log_path}" 2>&1 &
  append_pid "${run_id}" "$!" "${log_path}"
}

launch_derived_baseline() {
  local run_id="$1"
  local aggregation_method="$2"
  local top1_only="$3"
  local log_path="${LOG_DIR}/${run_id}.derive.nohup.log"
  local source_run_dir="results/pocket/enzyme405_50_p2rank_topk_softmax_pool"
  local label_csv="data/pocket_runs/enzyme405_50_p2rank_topk_softmax_pool/smallset_pairs.csv"

  if [[ ! -f "${source_run_dir}/predictions/pocket_level_predictions.csv" ]]; then
    echo "[warning] Source predictions missing for ${run_id}; skipping derivation."
    return 0
  fi

  echo "[derive] ${run_id} <- ${source_run_dir} (${aggregation_method})"
  local cmd=(
    "${PYTHON}" projects/active/pocket_robustness/analysis/derive_baseline_from_topk.py
    --source_run_dir "${source_run_dir}"
    --target_run_dir "results/pocket/${run_id}"
    --baseline_name "${run_id}"
    --aggregation_method "${aggregation_method}"
    --label_csv "${label_csv}"
  )
  if [[ "${top1_only}" == "true" ]]; then
    cmd+=(--top1_only)
  fi

  nohup "${cmd[@]}" > "${log_path}" 2>&1 &
  append_pid "${run_id}" "$!" "${log_path}"
}

write_blocked_fpocket_summary() {
  local run_id="$1"
  local run_dir="results/pocket/${run_id}"
  local config_path="projects/active/pocket_robustness/configs/generated_best_available/${run_id}.yaml"
  mkdir -p "${run_dir}"
  "${PYTHON}" - "${run_dir}" "${run_id}" "${config_path}" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(sys.argv[1])
run_id = sys.argv[2]
config_path = sys.argv[3]

run_dir.mkdir(parents=True, exist_ok=True)
payload = {
    "run_id": run_id,
    "baseline_name": run_id,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "config_path": str(Path(config_path).resolve()),
    "output_dir": str(run_dir.resolve()),
    "status": "blocked_fpocket_missing",
    "failed_step": "pocket_extraction",
    "error": "fpocket executable is not available",
    "warnings": ["fpocket missing; skipped without installation."],
    "generated_files": [str((run_dir / "run_summary.json").resolve())],
    "commands": [],
}
(run_dir / "run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
  echo "[blocked] ${run_id}: fpocket executable is not available"
}

kill_stale_runs
smoke_test_env

if result_complete "enzyme405_50_official_precomputed_pocket" | grep -q '^1$'; then
  echo "[skip] enzyme405_50_official_precomputed_pocket already completed."
else
  launch_compare_baseline \
    "enzyme405_50_official_precomputed_pocket" \
    "projects/active/pocket_robustness/configs/generated_best_available/enzyme405_50_official_precomputed_pocket.yaml"
fi

if result_complete "enzyme405_50_p2rank_top1" | grep -q '^1$'; then
  echo "[skip] enzyme405_50_p2rank_top1 already completed."
else
  launch_derived_baseline "enzyme405_50_p2rank_top1" "max" "true"
fi

if result_complete "enzyme405_50_p2rank_topk_max" | grep -q '^1$'; then
  echo "[skip] enzyme405_50_p2rank_topk_max already completed."
else
  launch_derived_baseline "enzyme405_50_p2rank_topk_max" "max" "false"
fi

if result_complete "enzyme405_50_p2rank_topk_mean" | grep -q '^1$'; then
  echo "[skip] enzyme405_50_p2rank_topk_mean already completed."
else
  launch_derived_baseline "enzyme405_50_p2rank_topk_mean" "mean" "false"
fi

if result_complete "enzyme405_50_p2rank_topk_rank_weighted" | grep -q '^1$'; then
  echo "[skip] enzyme405_50_p2rank_topk_rank_weighted already completed."
else
  launch_derived_baseline "enzyme405_50_p2rank_topk_rank_weighted" "rank_weighted" "false"
fi

if result_complete "enzyme405_50_p2rank_topk_softmax_pool" | grep -q '^1$'; then
  echo "[skip] enzyme405_50_p2rank_topk_softmax_pool already completed."
else
  launch_compare_baseline \
    "enzyme405_50_p2rank_topk_softmax_pool" \
    "projects/active/pocket_robustness/configs/generated_best_available/enzyme405_50_p2rank_topk_softmax_pool.yaml"
fi

if command -v fpocket >/dev/null 2>&1; then
  for run_id in \
    enzyme405_50_fpocket_top1 \
    enzyme405_50_fpocket_topk_rank_weighted \
    enzyme405_50_p2rank_fpocket_union_max \
    enzyme405_50_p2rank_fpocket_union_source_weighted
  do
    if result_complete "${run_id}" | grep -q '^1$'; then
      echo "[skip] ${run_id} already completed."
      continue
    fi
    launch_compare_baseline \
      "${run_id}" \
      "projects/active/pocket_robustness/configs/generated_best_available/${run_id}.yaml"
  done
else
  for run_id in \
    enzyme405_50_fpocket_top1 \
    enzyme405_50_fpocket_topk_rank_weighted \
    enzyme405_50_p2rank_fpocket_union_max \
    enzyme405_50_p2rank_fpocket_union_source_weighted
  do
    if [[ -f "results/pocket/${run_id}/run_summary.json" ]]; then
      echo "[skip] ${run_id} already has a run_summary.json."
      continue
    fi
    write_blocked_fpocket_summary "${run_id}"
  done
fi

echo "[done] Background launch complete."
echo "[done] PID log: ${PIDS_FILE}"
