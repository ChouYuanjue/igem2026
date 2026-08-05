#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="${TERPENE_PORTAL_RUNTIME_DIR:-${ROOT_DIR}/results/terpene_portal_runtime}"
PID_FILE="${RUNTIME_DIR}/portal.pid"
LOG_FILE="${RUNTIME_DIR}/portal.log"
HOST="${TERPENE_PORTAL_HOST:-0.0.0.0}"
PORT="${TERPENE_PORTAL_PORT:-8787}"
PYTHON="${TERPENE_PORTAL_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
SERVER="${ROOT_DIR}/scripts/terpene_portal/serve.py"

usage() {
  echo "Usage: $0 {start|stop|restart|status|logs}"
}

pid_value() {
  [[ -f "${PID_FILE}" ]] && cat "${PID_FILE}" || true
}

is_running() {
  local pid
  pid="$(pid_value)"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

start() {
  mkdir -p "${RUNTIME_DIR}"
  if is_running; then
    echo "[ready] portal already running pid=$(pid_value)"
    return
  fi
  if [[ ! -x "${PYTHON}" ]]; then
    echo "[error] Python environment not found: ${PYTHON}" >&2
    exit 1
  fi
  if [[ ! -f "${ROOT_DIR}/frontend/terpene_portal/dist/index.html" ]]; then
    echo "[error] portal build missing; run scripts/terpene_portal/build.sh" >&2
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
  for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${PORT}/api/portal/status" >/dev/null 2>&1; then
      echo "[started] pid=$(pid_value) url=http://${HOST}:${PORT}/portal/"
      return
    fi
    sleep 0.25
  done
  echo "[error] portal failed to become ready" >&2
  tail -80 "${LOG_FILE}" >&2 || true
  exit 1
}

stop() {
  local pid
  pid="$(pid_value)"
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
    curl -fsS "http://127.0.0.1:${PORT}/api/portal/status" | "${PYTHON}" -m json.tool
  else
    echo "[not running]"
    exit 1
  fi
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  logs) mkdir -p "${RUNTIME_DIR}"; touch "${LOG_FILE}"; tail -n 120 "${LOG_FILE}" ;;
  *) usage; exit 2 ;;
esac
