#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="${CATALYST_FINDER_RUNTIME_DIR:-${ROOT_DIR}/results/catalyst_finder_runtime}"
PID_FILE="${RUNTIME_DIR}/server.pid"
LOG_FILE="${RUNTIME_DIR}/server.log"
ENV_FILE="${RUNTIME_DIR}/deepseek.env"
HOST="${CATALYST_FINDER_HOST:-0.0.0.0}"
PORT="${CATALYST_FINDER_PORT:-8791}"
PYTHON="${CATALYST_FINDER_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
SERVER="${ROOT_DIR}/scripts/catalyst_finder/serve.py"

usage() { echo "Usage: $0 {start|stop|restart|status|logs|configure-key|feedback-summary|feedback-tail|feedback-json}"; }
pid_value() { [[ -f "${PID_FILE}" ]] && cat "${PID_FILE}" || true; }
is_running() { local pid; pid="$(pid_value)"; [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; }

load_env() {
  if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
  fi
}

start() {
  mkdir -p "${RUNTIME_DIR}"
  chmod 700 "${RUNTIME_DIR}" 2>/dev/null || true
  [[ -f "${RUNTIME_DIR}/feedback.jsonl" ]] && chmod 600 "${RUNTIME_DIR}/feedback.jsonl" 2>/dev/null || true
  load_env
  if is_running; then
    echo "[ready] catalyst finder already running pid=$(pid_value)"
    return
  fi
  if [[ ! -x "${PYTHON}" ]]; then
    echo "[error] Python environment not found: ${PYTHON}" >&2
    exit 1
  fi
  if ! "${PYTHON}" -c 'import langgraph' >/dev/null 2>&1; then
    echo "[error] LangGraph is missing. Run: ${PYTHON} -m pip install -r ${ROOT_DIR}/scripts/catalyst_finder/requirements.txt" >&2
    exit 1
  fi
  if ss -ltn "( sport = :${PORT} )" | grep -q ":${PORT}"; then
    echo "[error] port ${PORT} is already in use" >&2
    ss -ltnp "( sport = :${PORT} )" >&2 || true
    exit 1
  fi
  cd "${ROOT_DIR}"
  nohup "${PYTHON}" "${SERVER}" --host "${HOST}" --port "${PORT}" >>"${LOG_FILE}" 2>&1 </dev/null &
  echo $! > "${PID_FILE}"
  for _ in $(seq 1 40); do
    if curl -fsS "http://127.0.0.1:${PORT}/api/status" >/dev/null 2>&1; then
      echo "[started] pid=$(pid_value) url=http://${HOST}:${PORT}/"
      curl -fsS "http://127.0.0.1:${PORT}/api/status"
      echo
      return
    fi
    sleep 0.25
  done
  echo "[error] catalyst finder failed to become ready" >&2
  tail -80 "${LOG_FILE}" >&2 || true
  exit 1
}

stop() {
  local pid; pid="$(pid_value)"
  if [[ -z "${pid}" ]]; then
    echo "[stopped] no pid file"
    return
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}"
    for _ in $(seq 1 20); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 0.25
    done
    kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
  echo "[stopped] pid=${pid}"
}

status() {
  if is_running; then
    echo "[running] pid=$(pid_value)"
    curl -fsS "http://127.0.0.1:${PORT}/api/status" | "${PYTHON}" -m json.tool
  else
    echo "[not running]"
    exit 1
  fi
}

configure_key() {
  mkdir -p "${RUNTIME_DIR}"
  chmod 700 "${RUNTIME_DIR}" 2>/dev/null || true
  local key
  read -r -s -p "DeepSeek API key: " key
  echo
  if [[ -z "${key}" ]]; then
    echo "[error] empty key; nothing changed" >&2
    exit 1
  fi
  {
    printf 'DEEPSEEK_API_KEY=%q\n' "${key}"
    printf 'DEEPSEEK_MODEL=%q\n' "deepseek-v4-flash"
  } > "${ENV_FILE}"
  chmod 600 "${ENV_FILE}"
  unset key
  echo "[saved] ${ENV_FILE} (mode 600)"
  if is_running; then
    stop
  fi
  start
}

feedback_summary() {
  "${PYTHON}" "${ROOT_DIR}/scripts/catalyst_finder/feedback_report.py" --limit "${1:-10}"
}

feedback_tail() {
  "${PYTHON}" "${ROOT_DIR}/scripts/catalyst_finder/feedback_report.py" --raw --limit "${1:-20}"
}

feedback_json() {
  "${PYTHON}" "${ROOT_DIR}/scripts/catalyst_finder/feedback_report.py" --json --limit "${1:-20}"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  logs) mkdir -p "${RUNTIME_DIR}"; touch "${LOG_FILE}"; tail -n 120 "${LOG_FILE}" ;;
  configure-key) configure_key ;;
  feedback-summary) feedback_summary "${2:-10}" ;;
  feedback-tail) feedback_tail "${2:-20}" ;;
  feedback-json) feedback_json "${2:-20}" ;;
  *) usage; exit 2 ;;
esac
