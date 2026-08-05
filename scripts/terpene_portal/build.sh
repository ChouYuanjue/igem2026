#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_DIR="${ROOT_DIR}/frontend/terpene_portal"
NODE_VERSION="${TERPENE_PORTAL_NODE_VERSION:-24.18.0}"
NODE_CACHE="${TERPENE_PORTAL_NODE_ROOT:-${HOME}/.cache/terpene-portal-node-v${NODE_VERSION}}"

ensure_node() {
  if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    return
  fi
  if [[ -x "${NODE_CACHE}/bin/node" && -x "${NODE_CACHE}/bin/npm" ]]; then
    export PATH="${NODE_CACHE}/bin:${PATH}"
    return
  fi
  local archive="/tmp/node-v${NODE_VERSION}-linux-x64.tar.xz"
  local url="https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz"
  echo "[download] portable Node ${NODE_VERSION} -> ${NODE_CACHE}"
  rm -rf "${NODE_CACHE}" "${archive}"
  curl -fL --retry 3 "${url}" -o "${archive}"
  mkdir -p "${NODE_CACHE}"
  tar -xJf "${archive}" --strip-components=1 -C "${NODE_CACHE}"
  rm -f "${archive}"
  export PATH="${NODE_CACHE}/bin:${PATH}"
}

ensure_node
printf '[toolchain] node=%s npm=%s\n' "$(node --version)" "$(npm --version)"
cd "${APP_DIR}"
npm ci --no-audit --no-fund
npm run build
printf '[built] %s\n' "${APP_DIR}/dist"
