#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
HORIZYN_REPO="https://github.com/dayhofflabs/horizyn.git"
HORIZYN_COMMIT="e6655e732f574c8bfa0488b9bc5068b67e382745"
HORIZYN_SHA256="31bb9b6d73241b7807050377799de8b4bfb17f42a6cd652c8b17b65faf754c25"
HORIZYN_TARGET="results/terpene_production_models/marts_adapted_drfp_pu_r2e_exact_residual/horizyn_v1_0_dev.ckpt"

SKIP_INSTALL=0
VERIFY_ONLY=0
FULL_CHECK=0
for arg in "$@"; do
  case "$arg" in
    --skip-install) SKIP_INSTALL=1 ;;
    --verify-only) VERIFY_ONLY=1; SKIP_INSTALL=1 ;;
    --full-check) FULL_CHECK=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$VERIFY_ONLY" -eq 0 && "$SKIP_INSTALL" -eq 0 ]]; then
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
  "$VENV_DIR/bin/python" -m pip install -r requirements-terpene-runtime.txt
  # DRFP 0.3.6 still declares the obsolete package name rdkit-pypi. Python 3.12
  # has no compatible rdkit-pypi distribution, while production is validated with
  # rdkit==2026.3.2. Install DRFP without its stale dependency metadata.
  "$VENV_DIR/bin/python" -m pip install --no-deps drfp==0.3.6
  "$VENV_DIR/bin/python" -m pip install -e .
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Missing $VENV_DIR/bin/python. Run without --skip-install or set VENV_DIR." >&2
  exit 1
fi
PY="$VENV_DIR/bin/python"

if [[ "$VERIFY_ONLY" -eq 0 ]]; then
  if [[ ! -d external/horizyn/.git ]]; then
    mkdir -p external
    git clone --filter=blob:none "$HORIZYN_REPO" external/horizyn
  fi
  git -C external/horizyn fetch --depth 1 origin "$HORIZYN_COMMIT"
  git -C external/horizyn checkout --detach "$HORIZYN_COMMIT"
  "$PY" -m pip install --no-deps -e external/horizyn

  mkdir -p "$(dirname "$HORIZYN_TARGET")"
  if [[ ! -f "$HORIZYN_TARGET" ]] || [[ "$(sha256sum "$HORIZYN_TARGET" | awk '{print $1}')" != "$HORIZYN_SHA256" ]]; then
    tmp_dir="$(mktemp -d)"
    trap 'rm -rf "$tmp_dir"' EXIT
    "$PY" external/horizyn/scripts/download_checkpoint.py --only dev --output-dir "$tmp_dir"
    downloaded="$tmp_dir/horizyn_v1_0_dev.ckpt"
    actual="$(sha256sum "$downloaded" | awk '{print $1}')"
    if [[ "$actual" != "$HORIZYN_SHA256" ]]; then
      echo "Horizyn checkpoint SHA-256 mismatch: $actual" >&2
      exit 1
    fi
    cp "$downloaded" "$HORIZYN_TARGET"
    rm -rf "$tmp_dir"
    trap - EXIT
  fi
fi

"$PY" scripts/verify_terpene_runtime.py

if [[ "$FULL_CHECK" -eq 1 ]]; then
  for deployment in \
    marts_adapted_drfp_pu \
    marts_adapted_drfp_pu_r2e075 \
    marts_adapted_drfp_pu_r2e_exact_residual \
    marts_adapted_drfp_pu_e2r \
    marts_adapted_drfp_pu_e2r_hardneg128; do
    "$PY" projects/active/terpene_screening/validate_open_world_deployment.py \
      --deployment-dir "results/terpene_production_models/$deployment" \
      --output "/tmp/${deployment}_validation.json"
  done
  "$PY" projects/active/terpene_screening/validate_dual_kernel_deployment.py \
    --output /tmp/terpene_dual_kernel_validation.json
  "$PY" -m pytest -q projects/active/terpene_screening/tests
  "$PY" scripts/validate_terpene_system_health.py     --smoke --output /tmp/terpene_system_health_bootstrap.json
  "$PY" scripts/validate_terpene_single_batch_parity.py     --output /tmp/terpene_single_batch_parity_bootstrap.json
fi

echo "Terpene runtime is ready under: $ROOT"
echo "For new enzyme sequences, ESM-C 600M will be downloaded/cached on first use if absent."
