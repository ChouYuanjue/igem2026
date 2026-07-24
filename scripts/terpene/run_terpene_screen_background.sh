#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -n "${ENZYMECAGE_PYTHON:-}" ]]; then
  PYTHON="${ENZYMECAGE_PYTHON}"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON="python3"
fi

JAVA_HOME_FILE="${PROJECT_ROOT}/data/assets/java17/JAVA_HOME"
if [[ -f "${JAVA_HOME_FILE}" ]]; then
  JAVA_HOME="$(cat "${JAVA_HOME_FILE}")"
  export JAVA_HOME
  export PATH="${JAVA_HOME}/bin:${PATH}"
fi

export PYTHON
export ENZYMECAGE_PYTHON="${PYTHON}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

RESULTS_DIR="${PROJECT_ROOT}/results/terpene_cage_screen"
LOG_DIR="${RESULTS_DIR}/background_logs"
LOG_FILE="${LOG_DIR}/terpene_screen.nohup.log"
PID_FILE="${LOG_DIR}/pids.txt"

mkdir -p "${LOG_DIR}"

if [[ "${1:-}" == "--run-pipeline" ]]; then
  cd "${PROJECT_ROOT}"
  export PYTHON

  "${PYTHON}" projects/active/terpene_screening/inspect_terpene_data.py
  "${PYTHON}" projects/active/terpene_screening/build_terpene_pairs.py
  "${PYTHON}" projects/active/terpene_screening/download_structures.py
  "${PYTHON}" projects/active/terpene_screening/run_p2rank_top1.py --threads 8
  "${PYTHON}" projects/active/terpene_screening/run_cage_inference.py
  "${PYTHON}" projects/active/terpene_screening/evaluate_terpene_screen.py
  "${PYTHON}" projects/active/terpene_screening/write_report.py
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
"${PYTHON}" projects/active/terpene_screening/inspect_terpene_data.py
"${PYTHON}" projects/active/terpene_screening/build_terpene_pairs.py
"${PYTHON}" projects/active/terpene_screening/download_structures.py
"${PYTHON}" projects/active/terpene_screening/run_p2rank_top1.py --threads 8
"${PYTHON}" projects/active/terpene_screening/run_cage_inference.py
"${PYTHON}" projects/active/terpene_screening/evaluate_terpene_screen.py
"${PYTHON}" projects/active/terpene_screening/write_report.py
EOF
)

nohup setsid bash -lc "${PIPELINE_CMD}" </dev/null >"${LOG_FILE}" 2>&1 &
PIPELINE_PID=$!
printf '%s\n' "${PIPELINE_PID}" >"${PID_FILE}"
echo "${PIPELINE_PID}"
