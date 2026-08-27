#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="${CATALYST_FINDER_RUNTIME_DIR:-${ROOT_DIR}/results/catalyst_finder_runtime}"
PID_FILE="${RUNTIME_DIR}/server.pid"
LOG_FILE="${RUNTIME_DIR}/server.log"
ENV_FILE="${RUNTIME_DIR}/deepseek.env"
HOST="${CATALYST_FINDER_HOST:-127.0.0.1}"
PORT="${CATALYST_FINDER_PORT:-8791}"
PYTHON="${CATALYST_FINDER_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
SERVER="${ROOT_DIR}/scripts/catalyst_finder/serve.py"
SYSTEMD_UNIT_NAME="catalyst-finder.service"
SYSTEMD_UNIT_SOURCE="${ROOT_DIR}/scripts/catalyst_finder/catalyst-finder.service"
SYSTEMD_UNIT_PATH="${HOME}/.config/systemd/user/${SYSTEMD_UNIT_NAME}"

usage() { echo "Usage: $0 {install-service|start|stop|restart|status|logs|configure-key|feedback-summary|feedback-tail|feedback-json}"; }
pid_value() { [[ -f "${PID_FILE}" ]] && cat "${PID_FILE}" || true; }
is_running() {
  local pid cmdline
  pid="$(pid_value)"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
  [[ "${cmdline}" == *"scripts/catalyst_finder/serve.py"* ]]
}
systemd_installed() { systemctl --user cat "${SYSTEMD_UNIT_NAME}" >/dev/null 2>&1; }
service_running() {
  if systemd_installed && systemctl --user is-active --quiet "${SYSTEMD_UNIT_NAME}"; then
    return 0
  fi
  is_running
}

load_env() {
  if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
  fi
}

wait_ready() {
  for _ in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:${PORT}/api/status" >/dev/null 2>&1; then
      curl -fsS "http://127.0.0.1:${PORT}/api/status"
      echo
      return 0
    fi
    sleep 0.25
  done
  return 1
}

install_service() {
  if [[ ! -f "${SYSTEMD_UNIT_SOURCE}" ]]; then
    echo "[error] systemd unit template not found: ${SYSTEMD_UNIT_SOURCE}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${SYSTEMD_UNIT_PATH}")" "${RUNTIME_DIR}"
  chmod 700 "${RUNTIME_DIR}" 2>/dev/null || true
  install -m 0644 "${SYSTEMD_UNIT_SOURCE}" "${SYSTEMD_UNIT_PATH}"
  systemctl --user daemon-reload
  systemctl --user enable "${SYSTEMD_UNIT_NAME}" >/dev/null
  echo "[installed] ${SYSTEMD_UNIT_PATH}"
}

start() {
  mkdir -p "${RUNTIME_DIR}"
  chmod 700 "${RUNTIME_DIR}" 2>/dev/null || true
  [[ -f "${RUNTIME_DIR}/feedback.jsonl" ]] && chmod 600 "${RUNTIME_DIR}/feedback.jsonl" 2>/dev/null || true
  if systemd_installed; then
    if systemctl --user is-active --quiet "${SYSTEMD_UNIT_NAME}"; then
      echo "[ready] ${SYSTEMD_UNIT_NAME} already running"
    else
      systemctl --user start "${SYSTEMD_UNIT_NAME}"
    fi
    if wait_ready; then
      echo "[started] supervised by ${SYSTEMD_UNIT_NAME}"
      return
    fi
    echo "[error] catalyst finder systemd service failed to become ready" >&2
    systemctl --user status "${SYSTEMD_UNIT_NAME}" --no-pager -l >&2 || true
    journalctl --user -u "${SYSTEMD_UNIT_NAME}" -n 120 --no-pager >&2 || true
    exit 1
  fi
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
  if wait_ready; then
    echo "[started] pid=$(pid_value) url=http://${HOST}:${PORT}/"
    return
  fi
  echo "[error] catalyst finder failed to become ready" >&2
  tail -80 "${LOG_FILE}" >&2 || true
  exit 1
}

stop_legacy() {
  local pid; pid="$(pid_value)"
  if [[ -z "${pid}" ]]; then
    return
  fi
  if ! is_running; then
    rm -f "${PID_FILE}"
    return
  fi
  kill "${pid}"
  for _ in $(seq 1 80); do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 0.25
  done
  kill -9 "${pid}" 2>/dev/null || true
  rm -f "${PID_FILE}"
  echo "[stopped] legacy pid=${pid}"
}

stop() {
  if systemd_installed && systemctl --user is-active --quiet "${SYSTEMD_UNIT_NAME}"; then
    systemctl --user stop "${SYSTEMD_UNIT_NAME}"
    echo "[stopped] ${SYSTEMD_UNIT_NAME}"
  fi
  stop_legacy
}

status() {
  if service_running; then
    if systemd_installed && systemctl --user is-active --quiet "${SYSTEMD_UNIT_NAME}"; then
      echo "[running] ${SYSTEMD_UNIT_NAME} pid=$(systemctl --user show -p MainPID --value "${SYSTEMD_UNIT_NAME}")"
    else
      echo "[running] pid=$(pid_value)"
    fi
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
  if service_running; then
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
  install-service) install_service ;;
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  logs)
    if systemd_installed; then
      journalctl --user -u "${SYSTEMD_UNIT_NAME}" -n 120 --no-pager
    else
      mkdir -p "${RUNTIME_DIR}"; touch "${LOG_FILE}"; tail -n 120 "${LOG_FILE}"
    fi
    ;;
  configure-key) configure_key ;;
  feedback-summary) feedback_summary "${2:-10}" ;;
  feedback-tail) feedback_tail "${2:-20}" ;;
  feedback-json) feedback_json "${2:-20}" ;;
  *) usage; exit 2 ;;
esac
