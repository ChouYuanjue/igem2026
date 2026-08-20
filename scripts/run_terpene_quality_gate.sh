#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-$ROOT/.venv/bin/python}"
FULL=0
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ ! -x "$PY" ]]; then
  echo "Missing Python environment: $PY" >&2
  exit 1
fi

# A clone retains the compatibility registry. Promote it to an immutable
# snapshot before validating the production workflow.
if [[ ! -f data/terpene_open_world_registry/CURRENT ]]; then
  "$PY" projects/active/terpene_screening/manage_open_world_registry.py snapshot >/tmp/terpene_registry_migration.json
fi

"$PY" -m compileall -q projects/active/terpene_screening scripts
"$PY" -m pytest -q projects/active/terpene_screening/tests
"$PY" scripts/verify_terpene_runtime.py

for deployment in \
  marts_adapted_drfp_pu \
  marts_adapted_drfp_pu_r2e075 \
  marts_adapted_drfp_pu_r2e_exact_residual \
  marts_adapted_drfp_pu_e2r \
  marts_adapted_drfp_pu_e2r_hardneg128; do
  "$PY" projects/active/terpene_screening/validate_open_world_deployment.py \
    --deployment-dir "results/terpene_production_models/$deployment" \
    --output "/tmp/${deployment}_validation.json" >/dev/null
done
"$PY" projects/active/terpene_screening/validate_dual_kernel_deployment.py \
  --output /tmp/terpene_dual_kernel_validation.json >/dev/null
"$PY" scripts/validate_terpene_system_health.py \
  --output /tmp/terpene_system_health.json
"$PY" scripts/prepare_terpene_conformal_retrieval_sets.py \
  --output-dir /tmp/terpene_conformal_retrieval_gate \
  >/tmp/terpene_conformal_retrieval_gate.json

# Generated research-readiness workflows must execute even when the temporal
# data gate correctly refuses to create an under-covered split.
"$PY" projects/active/terpene_screening/prepare_marts_mechanism_features.py \
  --output-dir /tmp/terpene_mechanism_features_gate >/dev/null
"$PY" projects/active/terpene_screening/prepare_temporal_holdout.py \
  --output-dir /tmp/terpene_temporal_readiness_gate >/dev/null

if command -v git >/dev/null 2>&1; then
  git diff --check
fi

if [[ "$FULL" -eq 1 ]]; then
  "$PY" scripts/validate_terpene_system_health.py \
    --smoke --output /tmp/terpene_system_health_full.json >/dev/null
  "$PY" scripts/validate_terpene_single_batch_parity.py \
    --output /tmp/terpene_single_batch_parity.json
  "$PY" scripts/validate_terpene_golden_routes.py \
    >/tmp/terpene_golden_routes.json
  "$PY" scripts/analyze_terpene_cycle_consistency.py \
    --direction reaction_to_enzyme \
    --reaction-id RHEA:54512 \
    --top-k 2 \
    --cycle-top-n 1 \
    --reverse-top-k 10 \
    --device cpu \
    --output /tmp/terpene_cycle_consistency_gate.csv \
    >/tmp/terpene_cycle_consistency_gate.json
  "$PY" scripts/evaluate_terpene_cycle_rerank_grid.py \
    --max-queries-per-direction 2 \
    --objectives 3 \
    --forward-top-k 5 \
    --cycle-top-n 3 \
    --reverse-top-k 10 \
    --weights 0,0.10 \
    --gates all \
    --device cpu \
    --output-dir /tmp/terpene_cycle_rerank_grid_gate \
    >/tmp/terpene_cycle_rerank_grid_gate.json
fi

echo "Terpene quality gate passed (full=$FULL)."
