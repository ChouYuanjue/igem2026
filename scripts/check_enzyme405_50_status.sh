#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}" || exit 1

export PYTHON="/home/runnel/miniconda3/envs/enzymecage/bin/python"

LOG_DIR="results/pocket/background_logs"
PIDS_FILES=(
  "${LOG_DIR}/pids.txt"
  "${LOG_DIR}/fpocket_pids.txt"
)
RUN_IDS=(
  enzyme405_50_official_precomputed_pocket
  enzyme405_50_p2rank_top1
  enzyme405_50_p2rank_topk_max
  enzyme405_50_p2rank_topk_mean
  enzyme405_50_p2rank_topk_rank_weighted
  enzyme405_50_p2rank_topk_softmax_pool
  enzyme405_50_fpocket_top1
  enzyme405_50_fpocket_topk_rank_weighted
  enzyme405_50_p2rank_fpocket_union_max
  enzyme405_50_p2rank_fpocket_union_source_weighted
  enzyme405_50_p2rank_fpocket_union_source_balanced_mean
  enzyme405_50_p2rank_fpocket_union_source_balanced_rank_weighted
  enzyme405_50_p2rank_fpocket_union_source_balanced_softmax_pool
)

echo "## Background PIDs"
pid_found=0
for pid_file in "${PIDS_FILES[@]}"; do
  if [[ ! -f "${pid_file}" ]]; then
    continue
  fi
  while read -r run_id pid log_path; do
    [[ -z "${run_id:-}" ]] && continue
    if kill -0 "${pid}" 2>/dev/null; then
      alive="alive"
    else
      alive="stopped"
    fi
    echo "- ${run_id}: pid=${pid}, ${alive}, log=${log_path}"
    pid_found=1
  done < "${pid_file}"
done
if [[ "${pid_found}" -eq 0 ]]; then
  echo "- no pid file entries found"
fi

echo
echo "## Baseline Status"
"${PYTHON}" - "${ROOT_DIR}" "${RUN_IDS[@]}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_ids = sys.argv[2:]

for run_id in run_ids:
    run_dir = root / "results/pocket" / run_id
    summary_path = run_dir / "run_summary.json"
    manifest_path = run_dir / "manifests/pocket_manifest.csv"
    predictions = run_dir / "predictions/pocket_level_predictions.csv"
    aggregation_files = sorted((run_dir / "aggregation").glob("enzyme_level_*.csv"))
    metrics_path = run_dir / "metrics/metrics_top5_top10.json"

    summary_status = "missing"
    failed_step = "NA"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_status = str(summary.get("status", "unknown"))
            failed_step = str(summary.get("failed_step") or "NA")
        except Exception as exc:  # noqa: BLE001
            summary_status = f"unreadable({exc})"

    print(
        f"- {run_id}: summary={summary_status}, run_summary={summary_path.exists()}, "
        f"pocket_manifest={manifest_path.exists()}, predictions={predictions.exists()}, "
        f"aggregation={bool(aggregation_files)}, metrics={metrics_path.exists()}, failed_step={failed_step}"
    )
PY

echo
echo "## Recent Logs"
shopt -s nullglob
logs=("${LOG_DIR}"/*.nohup.log "${LOG_DIR}"/*.derive.nohup.log)
if (( ${#logs[@]} == 0 )); then
  echo "- no nohup logs yet"
else
  for log in "${logs[@]}"; do
    echo "### $(basename "${log}")"
    tail -n 30 "${log}"
    echo
  done
fi
