#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" != "--yes" ]]; then
  echo "[warning] This removes generated files under data/pocket_runs/ and results/pocket/."
  echo "[warning] It will not touch external_repos/."
  echo "Run again with: bash scripts/clean_generated.sh --yes"
  exit 1
fi

echo "[clean] data/pocket_runs generated files"
find "${ROOT_DIR}/data/pocket_runs" -mindepth 1 ! -name ".gitkeep" -exec rm -rf {} +

echo "[clean] results/pocket generated files"
find "${ROOT_DIR}/results/pocket" -mindepth 1 ! -name ".gitkeep" -exec rm -rf {} +

mkdir -p \
  "${ROOT_DIR}/data/pocket_runs" \
  "${ROOT_DIR}/results/pocket/predictions" \
  "${ROOT_DIR}/results/pocket/metrics" \
  "${ROOT_DIR}/results/pocket/rank_shift" \
  "${ROOT_DIR}/results/pocket/failure_cases"

touch \
  "${ROOT_DIR}/data/pocket_runs/.gitkeep" \
  "${ROOT_DIR}/results/pocket/predictions/.gitkeep" \
  "${ROOT_DIR}/results/pocket/metrics/.gitkeep" \
  "${ROOT_DIR}/results/pocket/rank_shift/.gitkeep" \
  "${ROOT_DIR}/results/pocket/failure_cases/.gitkeep"

echo "[done] Generated pocket data cleaned."
