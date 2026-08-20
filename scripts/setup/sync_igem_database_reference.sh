#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK_FILE="${ROOT_DIR}/reproducibility/external_repos/igem_database.lock.json"
TARGET="${ROOT_DIR}/external_repos/igem_database"
VERIFY_ONLY=0

usage() {
  cat <<'USAGE'
Usage: scripts/setup/sync_igem_database_reference.sh [--verify-only]

Clone or synchronize the read-only igem_database reference to the exact commit
recorded in reproducibility/external_repos/igem_database.lock.json.

Environment overrides:
  IGEM_DATABASE_REPO_URL  Alternate Git URL. Defaults to the lock's SSH URL.
  GIT_SSH_COMMAND         Optional SSH command/proxy settings for Git.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verify-only) VERIFY_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] unsupported argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ ! -f "${LOCK_FILE}" ]]; then
  echo "[error] lock file not found: ${LOCK_FILE}" >&2
  exit 1
fi

readarray -t LOCK_VALUES < <(
  python3 - "${LOCK_FILE}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["repository"]["ssh_url"])
print(payload["pin"]["commit"])
print(payload["checkout"]["partial_clone_filter"])
for pattern in payload["checkout"]["sparse_patterns"]:
    print(pattern)
PY
)

LOCK_URL="${LOCK_VALUES[0]}"
LOCK_COMMIT="${LOCK_VALUES[1]}"
PARTIAL_FILTER="${LOCK_VALUES[2]}"
SPARSE_PATTERNS=("${LOCK_VALUES[@]:3}")
REPO_URL="${IGEM_DATABASE_REPO_URL:-${LOCK_URL}}"

mkdir -p "$(dirname "${TARGET}")"

if [[ ${VERIFY_ONLY} -eq 0 ]]; then
  if [[ -e "${TARGET}" && ! -d "${TARGET}/.git" ]]; then
    echo "[error] target exists but is not a Git worktree: ${TARGET}" >&2
    exit 1
  fi

  if [[ ! -d "${TARGET}/.git" ]]; then
    echo "[clone] ${REPO_URL} -> ${TARGET}"
    git clone --depth 1 --no-checkout --filter="${PARTIAL_FILTER}" "${REPO_URL}" "${TARGET}"
  fi

  git -C "${TARGET}" remote set-url origin "${REPO_URL}"
  git -C "${TARGET}" sparse-checkout init --no-cone
  printf '%s\n' "${SPARSE_PATTERNS[@]}" > "${TARGET}/.git/info/sparse-checkout"

  if ! git -C "${TARGET}" cat-file -e "${LOCK_COMMIT}^{commit}" 2>/dev/null; then
    echo "[fetch] pinned commit ${LOCK_COMMIT}"
    git -C "${TARGET}" fetch --depth 1 origin "${LOCK_COMMIT}"
  fi
  git -C "${TARGET}" checkout --detach "${LOCK_COMMIT}"
fi

if [[ ! -d "${TARGET}/.git" ]]; then
  echo "[error] nested repository missing: ${TARGET}" >&2
  exit 1
fi

ACTUAL_COMMIT="$(git -C "${TARGET}" rev-parse HEAD)"
ACTUAL_BRANCH="$(git -C "${TARGET}" branch --show-current)"
ACTUAL_ORIGIN="$(git -C "${TARGET}" remote get-url origin)"
ACTUAL_FILTER="$(git -C "${TARGET}" config --get remote.origin.partialclonefilter || true)"
ACTUAL_SPARSE="$(git -C "${TARGET}" config --bool --get core.sparseCheckout || true)"
ACTUAL_SHALLOW="$(git -C "${TARGET}" rev-parse --is-shallow-repository)"
ACTUAL_SPARSE_PATTERNS="$(cat "${TARGET}/.git/info/sparse-checkout" 2>/dev/null || true)"
EXPECTED_SPARSE_PATTERNS="$(printf '%s\n' "${SPARSE_PATTERNS[@]}")"
NODE_MODULES_CHECKED_OUT=0
if [[ -e "${TARGET}/frontend/node_modules" || -e "${TARGET}/node_modules" ]]; then
  NODE_MODULES_CHECKED_OUT=1
fi

fail=0
if [[ "${ACTUAL_ORIGIN}" != "${REPO_URL}" ]]; then
  echo "[error] origin mismatch: expected ${REPO_URL}, got ${ACTUAL_ORIGIN}" >&2
  fail=1
fi
if [[ "${ACTUAL_COMMIT}" != "${LOCK_COMMIT}" ]]; then
  echo "[error] HEAD mismatch: expected ${LOCK_COMMIT}, got ${ACTUAL_COMMIT}" >&2
  fail=1
fi
if [[ -n "${ACTUAL_BRANCH}" ]]; then
  echo "[error] nested repository must use detached HEAD, found branch ${ACTUAL_BRANCH}" >&2
  fail=1
fi
if [[ "${ACTUAL_SPARSE}" != "true" ]]; then
  echo "[error] sparse checkout is not enabled" >&2
  fail=1
fi
if [[ "${ACTUAL_FILTER}" != "${PARTIAL_FILTER}" ]]; then
  echo "[error] partial clone filter mismatch: expected ${PARTIAL_FILTER}, got ${ACTUAL_FILTER:-none}" >&2
  fail=1
fi
if [[ "${ACTUAL_SHALLOW}" != "true" ]]; then
  echo "[error] nested repository is expected to remain shallow" >&2
  fail=1
fi
if [[ "${ACTUAL_SPARSE_PATTERNS}" != "${EXPECTED_SPARSE_PATTERNS}" ]]; then
  echo "[error] sparse checkout patterns differ from the lock" >&2
  fail=1
fi
if [[ ${NODE_MODULES_CHECKED_OUT} -eq 1 ]]; then
  echo "[error] node_modules must remain excluded from the sparse worktree" >&2
  fail=1
fi
if ! git -C "${TARGET}" diff --quiet || ! git -C "${TARGET}" diff --cached --quiet; then
  echo "[error] nested reference has local modifications" >&2
  fail=1
fi

python3 - <<PY
import json
print(json.dumps({
  "status": "valid" if ${fail} == 0 else "invalid",
  "path": "external_repos/igem_database",
  "head": "${ACTUAL_COMMIT}",
  "detached": ${ACTUAL_BRANCH@Q} == '',
  "origin": "${ACTUAL_ORIGIN}",
  "partial_clone_filter": "${ACTUAL_FILTER}",
  "sparse_checkout": "${ACTUAL_SPARSE}" == "true",
  "shallow": "${ACTUAL_SHALLOW}" == "true",
  "node_modules_checked_out": bool(${NODE_MODULES_CHECKED_OUT}),
}, indent=2))
PY

exit ${fail}
