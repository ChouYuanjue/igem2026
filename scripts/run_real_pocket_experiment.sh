#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}" || exit 1
PYTHON="${PYTHON:-/home/runnel/miniconda3/envs/enzymecage/bin/python}"

MATRIX="explorations/pocket/configs/baseline_matrix.yaml"
GENERATED_DIR="explorations/pocket/configs/generated"
COMPARISON_DIR="results/pocket/comparison"

run_step() {
  local name="$1"
  shift
  echo "[step] ${name}: $*"
  "$@"
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "[warning] ${name} failed with exit code ${rc}; continuing."
  fi
  return 0
}

run_step "inspect_enzymecage_assets" bash scripts/inspect_enzymecage_assets.sh
run_step "inspect_enzymecage_configs" bash scripts/inspect_enzymecage_configs.sh
run_step "inspect_enzymecage" bash scripts/inspect_enzymecage.sh
run_step "generate_baseline_configs" \
  "${PYTHON}" explorations/pocket/runners/generate_baseline_configs.py \
    --matrix "${MATRIX}" \
    --output_dir "${GENERATED_DIR}"

run_baseline_config() {
  local config="$1"
  if [[ ! -f "${config}" ]]; then
    echo "[warning] Missing generated config: ${config}"
    return 0
  fi
  echo "[baseline:dry_run] ${config}"
  "${PYTHON}" explorations/pocket/runners/run_compare_baselines.py \
    --experiment_config "${config}" \
    --dry_run
  local dry_rc=$?
  if [[ ${dry_rc} -ne 0 ]]; then
    echo "[warning] dry_run failed for ${config}; continuing."
    return 0
  fi

  echo "[baseline:run] ${config}"
  "${PYTHON}" explorations/pocket/runners/run_compare_baselines.py \
    --experiment_config "${config}" \
    --resume
  local run_rc=$?
  if [[ ${run_rc} -ne 0 ]]; then
    echo "[warning] run failed for ${config}; continuing."
  fi
  return 0
}

if [[ -d "${GENERATED_DIR}" ]]; then
  run_baseline_config "${GENERATED_DIR}/demo_official_eval_enzyme405.yaml"
  run_baseline_config "${GENERATED_DIR}/demo_official_eval_orphan335.yaml"
  run_baseline_config "${GENERATED_DIR}/demo_pocket_smallset_from_enzyme405.yaml"
  run_baseline_config "${GENERATED_DIR}/demo_pocket_smallset_from_enzyme405_p2rank_topk_softmax.yaml"
  run_baseline_config "${GENERATED_DIR}/demo_pocket_smallset_from_orphan335_p2rank_top1.yaml"
  run_baseline_config "${GENERATED_DIR}/demo_pocket_smallset_from_orphan335_p2rank_topk_softmax.yaml"
fi

run_step "compare_all_baselines" \
  "${PYTHON}" explorations/pocket/analysis/compare_all_baselines.py \
    --results_root results/pocket \
    --baseline_matrix "${MATRIX}" \
    --output_dir "${COMPARISON_DIR}"

P2RANK_MANIFEST="$(find results/pocket -path '*/manifests/p2rank_pocket_manifest.csv' -print | head -1)"
FPOCKET_MANIFEST="$(find results/pocket -path '*/manifests/fpocket_pocket_manifest.csv' -print | head -1)"
if [[ -n "${P2RANK_MANIFEST}" && -n "${FPOCKET_MANIFEST}" ]]; then
  run_step "compare_pocket_sources" \
    "${PYTHON}" explorations/pocket/analysis/compare_pocket_sources.py \
      --p2rank_manifest "${P2RANK_MANIFEST}" \
      --fpocket_manifest "${FPOCKET_MANIFEST}" \
      --output_dir results/pocket/comparison/pocket_source_overlap
else
  echo "[info] Skipping pocket source overlap; both manifests are not available yet."
fi

run_step "summarize_experiment_status" \
  "${PYTHON}" explorations/pocket/analysis/summarize_experiment_status.py \
    --results_root results/pocket \
    --baseline_matrix "${MATRIX}" \
    --output results/pocket/experiment_status.md

echo "[done] Comparison report: results/pocket/comparison/comparison_report.md"
