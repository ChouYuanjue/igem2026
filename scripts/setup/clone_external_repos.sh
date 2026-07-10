#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external_repos"
SUMMARY_JSON="${EXTERNAL_DIR}/clone_summary.json"
SUMMARY_TSV="${EXTERNAL_DIR}/.clone_summary.tsv"
PYTHON="${PYTHON:-/home/runnel/miniconda3/envs/enzymecage/bin/python}"

mkdir -p "${EXTERNAL_DIR}"

echo "[info] external_repos/ is read-only dependency space."
echo "[info] Do not edit cloned repositories directly."
echo -e "repo_name\turl\tlocal_path\tstatus\tcurrent_commit\tcurrent_branch\twarning" > "${SUMMARY_TSV}"

git_commit() {
  local target="$1"
  if [[ -d "${target}/.git" ]]; then
    git -C "${target}" rev-parse HEAD 2>/dev/null || true
  fi
}

git_branch() {
  local target="$1"
  if [[ -d "${target}/.git" ]]; then
    git -C "${target}" branch --show-current 2>/dev/null || true
  fi
}

record_repo() {
  local name="$1"
  local url="$2"
  local target="$3"
  local status="$4"
  local warning="${5:-}"
  local commit
  local branch

  commit="$(git_commit "${target}")"
  branch="$(git_branch "${target}")"
  echo -e "${name}\t${url}\t${target}\t${status}\t${commit}\t${branch}\t${warning}" >> "${SUMMARY_TSV}"
}

clone_if_missing() {
  local name="$1"
  local url="$2"
  local required="$3"
  local target="${EXTERNAL_DIR}/${name}"

  if [[ -d "${target}/.git" || -e "${target}" ]]; then
    echo "[skip] ${name} already exists at ${target}"
    record_repo "${name}" "${url}" "${target}" "already_exists"
    return 0
  fi

  echo "[clone] ${url} -> ${target}"
  if git clone "${url}" "${target}"; then
    record_repo "${name}" "${url}" "${target}" "cloned"
    return 0
  fi

  local warning="clone failed"
  if [[ "${required}" == "required" ]]; then
    warning="required repository clone failed"
  fi
  echo "[warning] ${name}: ${warning}"
  record_repo "${name}" "${url}" "${target}" "failed" "${warning}"
  return 0
}

clone_if_missing "EnzymeCAGE" "https://github.com/GENTEL-lab/EnzymeCAGE.git" "required"
clone_if_missing "p2rank" "https://github.com/rdk/p2rank.git" "optional"
clone_if_missing "alphafill" "https://github.com/PDB-REDO/alphafill.git" "optional"
clone_if_missing "fpocket" "https://github.com/Discngine/fpocket.git" "optional"
clone_if_missing "DeepSurf" "https://github.com/stemylonas/DeepSurf.git" "optional"
clone_if_missing "masif" "https://github.com/LPDI-EPFL/masif.git" "optional"
clone_if_missing "ScanNet" "https://github.com/jertubiana/ScanNet.git" "optional"
clone_if_missing "ReactZyme" "https://github.com/WillHua127/ReactZyme.git" "optional"
clone_if_missing "GENzyme" "https://github.com/WillHua127/GENzyme.git" "optional"

"${PYTHON}" - "${SUMMARY_TSV}" "${SUMMARY_JSON}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

tsv_path = Path(sys.argv[1])
json_path = Path(sys.argv[2])

with tsv_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

for row in rows:
    for key, value in list(row.items()):
        if value == "":
            row[key] = None

json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
PY

rm -f "${SUMMARY_TSV}"

echo "[done] External repositories collected."
echo "[done] Clone summary written to ${SUMMARY_JSON}"
echo "[reminder] Keep external_repos/ read-only. Put adapters in projects/active/<project>/ or shared scripts/."
