#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results/terpene_cage_screen"
DATA_DIR="${PROJECT_ROOT}/data/terpene_cage_screen"
LOG_DIR="${RESULTS_DIR}/background_logs"
LOG_FILE="${LOG_DIR}/terpene_screen.nohup.log"
PID_FILE="${LOG_DIR}/pids.txt"
PAIRS_CSV="${DATA_DIR}/terpene_candidate_pairs.csv"
STRUCTURE_DIR="${DATA_DIR}/structures"
POCKET_DIR="${DATA_DIR}/pockets"
PREDICTIONS_CSV="${RESULTS_DIR}/predictions/all_pair_scores.csv"
METRICS_JSON="${RESULTS_DIR}/metrics/topk_metrics.json"
REPORT_MD="${RESULTS_DIR}/terpene_screen_report.md"

echo "== Pipeline =="
if [[ -f "${PID_FILE}" ]]; then
  pid="$(head -n 1 "${PID_FILE}" || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "PID: ${pid} (running)"
  else
    echo "PID: ${pid:-NA} (not running)"
  fi
else
  echo "PID: NA"
fi

echo
echo "== Files =="
if [[ -f "${PAIRS_CSV}" ]]; then
  echo "candidate_pairs: present"
  echo "candidate_pairs_rows: $(( $(wc -l < "${PAIRS_CSV}") - 1 ))"
else
  echo "candidate_pairs: missing"
fi

if [[ -d "${STRUCTURE_DIR}" ]]; then
  echo "structures: present"
  echo "structures_count: $(find "${STRUCTURE_DIR}" -maxdepth 1 -type f \( -name '*.cif' -o -name '*.pdb' \) | wc -l)"
else
  echo "structures: missing"
fi

if [[ -d "${POCKET_DIR}" ]]; then
  echo "pockets: present"
  echo "pockets_count: $(find "${POCKET_DIR}" -maxdepth 1 -type f -name '*.pdb' | wc -l)"
else
  echo "pockets: missing"
fi

if [[ -f "${PREDICTIONS_CSV}" ]]; then
  echo "predictions: present"
  echo "predictions_rows: $(( $(wc -l < "${PREDICTIONS_CSV}") - 1 ))"
else
  echo "predictions: missing"
fi

if [[ -f "${METRICS_JSON}" ]]; then
  echo "metrics: present"
else
  echo "metrics: missing"
fi

if [[ -f "${REPORT_MD}" ]]; then
  echo "report: present"
else
  echo "report: missing"
fi

echo
echo "== Log Tail =="
if [[ -f "${LOG_FILE}" ]]; then
  tail -n 40 "${LOG_FILE}"
else
  echo "(no nohup log yet)"
fi

