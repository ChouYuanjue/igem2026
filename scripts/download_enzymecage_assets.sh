#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="${ROOT_DIR}/data/assets/enzymecage_official"
ENZYMECAGE_ROOT="${ROOT_DIR}/external_repos/EnzymeCAGE"
STATUS_JSON="${ROOT_DIR}/results/pocket/enzymecage_assets_status.json"
PYTHON="${PYTHON:-/home/runnel/miniconda3/envs/enzymecage/bin/python}"

DATASET_URL="https://drive.google.com/file/d/1IcuoqpEGhKdLAG9zorKEHQ9RDXSgU3_C/view?usp=sharing"
DATASET_ID="1IcuoqpEGhKdLAG9zorKEHQ9RDXSgU3_C"
CHECKPOINTS_URL="https://drive.google.com/file/d/1LLsS_MMKEbFpU2iIOF9ro46cO86S-SCt/view?usp=sharing"
CHECKPOINTS_ID="1LLsS_MMKEbFpU2iIOF9ro46cO86S-SCt"

DATASET_ZIP="${ASSET_DIR}/dataset.zip"
CHECKPOINTS_ZIP="${ASSET_DIR}/checkpoints.zip"

mkdir -p "${ASSET_DIR}" "$(dirname "${STATUS_JSON}")"

write_status() {
  local status="$1"
  local message="$2"
  "${PYTHON}" - "$STATUS_JSON" "$status" "$message" "$ASSET_DIR" "$ENZYMECAGE_ROOT" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

status_path = Path(sys.argv[1])
status = sys.argv[2]
message = sys.argv[3]
asset_dir = Path(sys.argv[4])
enzymecage_root = Path(sys.argv[5])

required_paths = [
    enzymecage_root / "dataset/demo/reaction.csv",
    enzymecage_root / "dataset/demo/structures",
    enzymecage_root / "checkpoints/pretrain/seed_42/best_model.pth",
]
checkpoint_dir = enzymecage_root / "checkpoints/pretrain/seed_42"
dataset_dir = enzymecage_root / "dataset"

payload = {
    "status": status,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "message": message,
    "asset_dir": str(asset_dir),
    "enzymecage_root": str(enzymecage_root),
    "zip_files_present": {
        str(asset_dir / "dataset.zip"): (asset_dir / "dataset.zip").exists(),
        str(asset_dir / "checkpoints.zip"): (asset_dir / "checkpoints.zip").exists(),
    },
    "required_paths": {str(path): path.exists() for path in required_paths},
    "missing_paths": [str(path) for path in required_paths if not path.exists()],
    "dataset_present": dataset_dir.exists(),
    "checkpoint_candidates_seed_42": [
        str(path) for path in sorted(checkpoint_dir.glob("*.pth"))
    ] if checkpoint_dir.exists() else [],
    "manual_download_instructions": {
        "dataset_zip": str(asset_dir / "dataset.zip"),
        "checkpoints_zip": str(asset_dir / "checkpoints.zip"),
    },
}
status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

download_with_gdown() {
  local file_id="$1"
  local output_path="$2"
  if command -v gdown >/dev/null 2>&1; then
    echo "[download:gdown] ${output_path}"
    gdown "https://drive.google.com/uc?id=${file_id}" -O "${output_path}"
    return $?
  fi
  if "${PYTHON}" -c "import gdown" >/dev/null 2>&1; then
    echo "[download:python-gdown] ${output_path}"
    "${PYTHON}" -m gdown "https://drive.google.com/uc?id=${file_id}" -O "${output_path}"
    return $?
  fi
  return 127
}

download_with_wget_or_curl() {
  local url="$1"
  local file_id="$2"
  local output_path="$3"
  if command -v wget >/dev/null 2>&1; then
    echo "[download:wget] ${output_path}"
    wget -O "${output_path}" "https://drive.google.com/uc?export=download&id=${file_id}" ||
      wget -O "${output_path}" "${url}"
    return $?
  fi
  if command -v curl >/dev/null 2>&1; then
    echo "[download:curl] ${output_path}"
    curl -L "https://drive.google.com/uc?export=download&id=${file_id}" -o "${output_path}" ||
      curl -L "${url}" -o "${output_path}"
    return $?
  fi
  return 127
}

is_valid_zip() {
  local path="$1"
  [[ -s "${path}" ]] && unzip -tq "${path}" >/dev/null 2>&1
}

remove_invalid_zip_if_present() {
  local name="$1"
  local path="$2"
  if [[ -e "${path}" ]] && ! is_valid_zip "${path}"; then
    echo "[warning] Existing ${name} is not a valid zip; removing ${path}"
    rm -f "${path}"
  fi
}

ensure_zip() {
  local name="$1"
  local url="$2"
  local file_id="$3"
  local output_path="$4"

  if is_valid_zip "${output_path}"; then
    echo "[skip] ${name} already exists at ${output_path}"
    return 0
  elif [[ -e "${output_path}" ]]; then
    echo "[warning] Existing ${name} is not a valid zip; removing ${output_path}"
    rm -f "${output_path}"
  fi

  rm -f "${output_path}"
  if download_with_gdown "${file_id}" "${output_path}" && is_valid_zip "${output_path}"; then
    return 0
  fi

  echo "[warning] gdown download failed or unavailable for ${name}; trying wget/curl."
  rm -f "${output_path}"
  if download_with_wget_or_curl "${url}" "${file_id}" "${output_path}" && is_valid_zip "${output_path}"; then
    return 0
  fi

  echo "[warning] Downloaded ${name} is missing or not a valid zip."
  rm -f "${output_path}"
  return 1
}

check_required_paths() {
  [[ -f "${ENZYMECAGE_ROOT}/dataset/demo/reaction.csv" ]] &&
  [[ -d "${ENZYMECAGE_ROOT}/dataset/demo/structures" ]] &&
  [[ -f "${ENZYMECAGE_ROOT}/checkpoints/pretrain/seed_42/best_model.pth" ]]
}

check_extracted_assets_present() {
  [[ -d "${ENZYMECAGE_ROOT}/dataset" ]] &&
  find "${ENZYMECAGE_ROOT}/checkpoints/pretrain/seed_42" -maxdepth 1 -type f -name "*.pth" 2>/dev/null | grep -q .
}

if [[ ! -d "${ENZYMECAGE_ROOT}" ]]; then
  write_status "blocked_enzymecage_repo_missing" "EnzymeCAGE repository is missing. Run scripts/clone_external_repos.sh first."
  echo "[blocked] EnzymeCAGE repository is missing: ${ENZYMECAGE_ROOT}"
  exit 0
fi

echo "[info] Download directory: ${ASSET_DIR}"
echo "[info] Extraction target: ${ENZYMECAGE_ROOT}"
echo "[info] This extracts official assets into EnzymeCAGE but does not modify source code."

remove_invalid_zip_if_present "dataset.zip" "${DATASET_ZIP}"
remove_invalid_zip_if_present "checkpoints.zip" "${CHECKPOINTS_ZIP}"

if check_required_paths; then
  rm -f "${DATASET_ZIP}" "${CHECKPOINTS_ZIP}"
  write_status "completed" "Official EnzymeCAGE assets are already present and verified. Zip files removed."
  echo "[done] Assets already verified. Zip files removed."
  exit 0
fi

if check_extracted_assets_present && [[ ! -e "${DATASET_ZIP}" ]] && [[ ! -e "${CHECKPOINTS_ZIP}" ]]; then
  write_status "failed_extracted_assets_missing_required_paths" "Official assets appear to be extracted, but required demo/checkpoint paths are missing. Not re-downloading because zip files were already removed."
  echo "[failed] Assets are extracted, but required demo/checkpoint paths are missing. See ${STATUS_JSON}"
  exit 1
fi

if ! ensure_zip "dataset.zip" "${DATASET_URL}" "${DATASET_ID}" "${DATASET_ZIP}"; then
  write_status "blocked_google_drive_download" "Failed to download dataset.zip. Manually place dataset.zip and checkpoints.zip in data/assets/enzymecage_official/ and rerun."
  echo "[blocked] Google Drive download failed for dataset.zip."
  echo "[manual] Put dataset.zip at ${DATASET_ZIP}"
  echo "[manual] Put checkpoints.zip at ${CHECKPOINTS_ZIP}"
  exit 0
fi

if ! ensure_zip "checkpoints.zip" "${CHECKPOINTS_URL}" "${CHECKPOINTS_ID}" "${CHECKPOINTS_ZIP}"; then
  write_status "blocked_google_drive_download" "Failed to download checkpoints.zip. Manually place dataset.zip and checkpoints.zip in data/assets/enzymecage_official/ and rerun."
  echo "[blocked] Google Drive download failed for checkpoints.zip."
  echo "[manual] Put dataset.zip at ${DATASET_ZIP}"
  echo "[manual] Put checkpoints.zip at ${CHECKPOINTS_ZIP}"
  exit 0
fi

echo "[extract] dataset.zip -> ${ENZYMECAGE_ROOT}"
if ! unzip -oq "${DATASET_ZIP}" -d "${ENZYMECAGE_ROOT}"; then
  write_status "blocked_google_drive_download" "dataset.zip could not be extracted. Manually download valid dataset.zip and checkpoints.zip into data/assets/enzymecage_official/ and rerun."
  echo "[blocked] dataset.zip could not be extracted."
  exit 0
fi

echo "[extract] checkpoints.zip -> ${ENZYMECAGE_ROOT}"
if ! unzip -oq "${CHECKPOINTS_ZIP}" -d "${ENZYMECAGE_ROOT}"; then
  write_status "blocked_google_drive_download" "checkpoints.zip could not be extracted. Manually download valid dataset.zip and checkpoints.zip into data/assets/enzymecage_official/ and rerun."
  echo "[blocked] checkpoints.zip could not be extracted."
  exit 0
fi

if check_required_paths; then
  rm -f "${DATASET_ZIP}" "${CHECKPOINTS_ZIP}"
  write_status "completed" "Official EnzymeCAGE assets downloaded, extracted, verified, and zip files removed."
  echo "[done] Assets verified. Zip files removed."
else
  rm -f "${DATASET_ZIP}" "${CHECKPOINTS_ZIP}"
  write_status "failed_extracted_assets_missing_required_paths" "Assets were extracted, but required demo/checkpoint paths are still missing. Zip files were removed after extraction."
  echo "[failed] Assets extracted, but required paths are missing. See ${STATUS_JSON}"
  exit 1
fi
