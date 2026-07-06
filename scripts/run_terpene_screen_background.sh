#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/runnel/miniconda3/envs/enzymecage/bin/python"
RESULTS_DIR="${PROJECT_ROOT}/results/terpene_cage_screen"
LOG_DIR="${RESULTS_DIR}/background_logs"
LOG_FILE="${LOG_DIR}/terpene_screen.nohup.log"
PID_FILE="${LOG_DIR}/pids.txt"

mkdir -p "${LOG_DIR}"

if [[ "${1:-}" == "--run-pipeline" ]]; then
  cd "${PROJECT_ROOT}"
  export PYTHON

  "${PYTHON}" explorations/terpene_screen/inspect_terpene_data.py
  "${PYTHON}" explorations/terpene_screen/build_terpene_pairs.py
  "${PYTHON}" explorations/terpene_screen/download_structures.py
  "${PYTHON}" explorations/terpene_screen/run_p2rank_top1.py --threads 8
  "${PYTHON}" explorations/terpene_screen/run_cage_inference.py
  "${PYTHON}" explorations/terpene_screen/evaluate_terpene_screen.py
  "${PYTHON}" explorations/terpene_screen/write_report.py
  exit 0
fi

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(head -n 1 "${PID_FILE}" || true)"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "${existing_pid}"
    exit 0
  fi
fi

PIPELINE_CMD=$(cat <<EOF
set -euo pipefail
cd "${PROJECT_ROOT}"
export PYTHON="${PYTHON}"
"${PYTHON}" explorations/terpene_screen/inspect_terpene_data.py
"${PYTHON}" explorations/terpene_screen/build_terpene_pairs.py
"${PYTHON}" explorations/terpene_screen/download_structures.py
"${PYTHON}" explorations/terpene_screen/run_p2rank_top1.py --threads 8
"${PYTHON}" explorations/terpene_screen/run_cage_inference.py
"${PYTHON}" explorations/terpene_screen/evaluate_terpene_screen.py
"${PYTHON}" explorations/terpene_screen/write_report.py
EOF
)

nohup setsid bash -lc "${PIPELINE_CMD}" </dev/null >"${LOG_FILE}" 2>&1 &
PIPELINE_PID=$!
printf '%s\n' "${PIPELINE_PID}" >"${PID_FILE}"
echo "${PIPELINE_PID}"
