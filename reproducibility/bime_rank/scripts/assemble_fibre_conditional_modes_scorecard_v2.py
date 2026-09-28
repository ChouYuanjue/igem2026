from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "reproducibility/bime_rank/records/FIBRE_CONDITIONAL_MODES_SCORECARD_V2.json"

PATHS = {
    "development_baseline": ROOT / "results/fibre_fair_baseline_glue002_dev_v1",
    "development_no_glue": ROOT / "results/fibre_fair_noglue_dev_v1",
    "development_conditional_mixture": ROOT / "results/fibre_fair_conditional_mixture_dev_v1",
    "development_dimension_scaled": ROOT / "results/fibre_fair_dimscaled_dev_v1",
    "frozen_baseline": ROOT / "results/fibre_fair_baseline_glue002_frozen_v1",
    "frozen_no_glue": ROOT / "results/fibre_fair_noglue_frozen_v1",
    "full_v1": ROOT / "results/fibre_interaction_atlas_full_v1",
    "full_v2": ROOT / "results/fibre_conditional_modes_full_v2",
    "broad_v1": ROOT / "results/fibre_interaction_atlas_tps_broad_universe_v1",
    "broad_v2": ROOT / "results/fibre_conditional_modes_tps_broad_v2",
}

METRICS = [
    "mean_reciprocal_rank",
    "median_best_positive_rank",
    "hit_probability_at_3",
    "hit_probability_at_5",
    "hit_probability_at_10",
    "hit_probability_at_20",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metric_rows(path: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(path / "metrics.csv")
    rows: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        keep: dict[str, object] = {}
        for key in ("protocol", "direction", "method", "n_query_cells", "n_unique_queries", "n_unique_reactions"):
            if key in row and not pd.isna(row[key]):
                keep[key] = row[key]
        for key in METRICS:
            if key in row and not pd.isna(row[key]):
                keep[key] = float(row[key])
        rows.append(keep)
    return rows


def row_map(rows: list[dict[str, object]], *, protocol: str | None = None, method: str | None = None) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        if protocol is not None and row.get("protocol") != protocol:
            continue
        if method is not None and row.get("method") != method:
            continue
        direction = str(row.get("direction", method or ""))
        result[direction] = row
    return result


def deltas(base: dict[str, dict[str, object]], candidate: dict[str, dict[str, object]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key in sorted(set(base) & set(candidate)):
        result[key] = {
            metric: float(candidate[key][metric]) - float(base[key][metric])
            for metric in METRICS
            if metric in base[key] and metric in candidate[key]
        }
    return result


def paired_bootstrap(base_path: Path, candidate_path: Path, *, seed: int = 20260928, n_boot: int = 10000) -> dict[str, object]:
    base = pd.read_csv(base_path / "query_metrics.csv")
    cand = pd.read_csv(candidate_path / "query_metrics.csv")
    keys = ["protocol", "direction", "query_id", "protein_fold", "reaction_fold"]
    if base.duplicated(keys).any() or cand.duplicated(keys).any():
        raise ValueError("paired bootstrap keys are not unique")
    merged = base.merge(cand, on=keys, suffixes=("_base", "_candidate"), validate="one_to_one")
    if len(merged) != len(base) or len(merged) != len(cand):
        raise ValueError("baseline/candidate query-cell support differs")
    rng = np.random.default_rng(seed)
    result: dict[str, object] = {}
    column_map = {
        "mrr": "reciprocal_rank",
        "hit3": "hit_at_3",
        "hit10": "hit_at_10",
        "hit20": "hit_at_20",
    }
    for direction, group in merged.groupby("direction", sort=True):
        local: dict[str, object] = {"n_query_cells": int(len(group))}
        n = len(group)
        for label, column in column_map.items():
            diff = (
                pd.to_numeric(group[f"{column}_candidate"], errors="raise").to_numpy(float)
                - pd.to_numeric(group[f"{column}_base"], errors="raise").to_numpy(float)
            )
            draws = rng.integers(0, n, size=(n_boot, n))
            means = diff[draws].mean(axis=1)
            local[label] = {
                "delta": float(diff.mean()),
                "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
                "p_boot_gt_zero": float((means > 0).mean()),
            }
        result[str(direction)] = local
    return result


def main() -> None:
    for label, path in PATHS.items():
        if not (path / "metrics.csv").exists():
            raise FileNotFoundError(f"missing {label}: {path / 'metrics.csv'}")

    dev_base = row_map(metric_rows(PATHS["development_baseline"]))
    dev_no_glue = row_map(metric_rows(PATHS["development_no_glue"]))
    dev_conditional = row_map(metric_rows(PATHS["development_conditional_mixture"]))
    dev_dimscaled = row_map(metric_rows(PATHS["development_dimension_scaled"]))
    frozen_base = row_map(metric_rows(PATHS["frozen_baseline"]))
    frozen_no_glue = row_map(metric_rows(PATHS["frozen_no_glue"]))

    full_v1_rows = metric_rows(PATHS["full_v1"])
    full_v2_rows = metric_rows(PATHS["full_v2"])
    broad_v1_rows = metric_rows(PATHS["broad_v1"])
    broad_v2_rows = metric_rows(PATHS["broad_v2"])

    full_comparison: dict[str, object] = {}
    for protocol in ("legacy_exact", "double_cold_25cell"):
        b = row_map(full_v1_rows, protocol=protocol)
        c = row_map(full_v2_rows, protocol=protocol)
        full_comparison[protocol] = {
            "v1": b,
            "v2": c,
            "delta_v2_minus_v1": deltas(b, c),
        }

    broad_v1_full = row_map(broad_v1_rows, method="full_atlas")
    broad_v2_full = row_map(broad_v2_rows, method="full_atlas")
    broad_v2_universal = row_map(broad_v2_rows, method="universal_chart_only")

    scorecard = {
        "version": "fibre-conditional-catalytic-modes-scorecard-v2",
        "status": "promoted_after_development_selection_and_single_frozen_confirmation",
        "predecessor": "fibre-interaction-atlas-reproduction-v1",
        "selected_change": {
            "cross_expert_consistency_weight": {"v1": 0.02, "v2": 0.0},
            "ranking_formula_changed": False,
            "encoder_architecture_changed": False,
            "input_features_changed": False,
            "random_seed": 20260723,
        },
        "mathematical_interpretation": {
            "interaction_space": "universal 128-d computational baseline plus eight 32-d learned local catalytic-mode blocks",
            "hierarchy": (
                "expert mass selects universal versus specialized branch; "
                "query-side gates distribute mass within the eight local modes"
            ),
            "reaction_to_enzyme": "reaction-conditioned positive block-diagonal readout",
            "enzyme_to_reaction": "protein-conditioned positive block-diagonal readout",
            "uncertainty_decomposition": (
                "law of total variance: within-local-mode variance plus "
                "universal-versus-specialized mean disagreement"
            ),
            "physical_anchor": "activation free-energy barrier and transition-state differential stabilization",
            "score_energy_calibrated": False,
            "training_temperature_is_thermodynamic": False,
            "kinetic_log_mean_exp_status": "single frozen confirmation failed; not promoted",
            "mode_posterior_update": "implemented information-geometric primitive; not frozen inference",
            "ranking_confidence": "implemented perturbation-stability primitive; not activity probability",
            "finite_rank": "computational corollary only, not a biological claim",
        },
        "development_selection": {
            "partition": "protein_fold==4 OR reaction_fold==4",
            "frozen_metrics_not_used_for_selection": True,
            "baseline_glue_0p02": dev_base,
            "candidate_no_glue": dev_no_glue,
            "candidate_no_glue_delta": deltas(dev_base, dev_no_glue),
            "rejected_conditional_probability_mixture": dev_conditional,
            "rejected_dimension_scaled_glue_1_over_64": dev_dimscaled,
            "selection_reason": (
                "no-glue removes an empirical cross-mode agreement assumption; "
                "development R2E MRR/Hit@3/Hit@20 improve and Hit@10 is retained, "
                "while E2R changes are small enough to justify one frozen confirmation"
            ),
        },
        "single_frozen_confirmation": {
            "partition": "protein_fold in 0..3 AND reaction_fold in 0..3",
            "retuning_after_reveal": False,
            "baseline_glue_0p02": frozen_base,
            "candidate_no_glue": frozen_no_glue,
            "delta_candidate_minus_baseline": deltas(frozen_base, frozen_no_glue),
            "paired_query_cell_bootstrap": paired_bootstrap(
                PATHS["frozen_baseline"], PATHS["frozen_no_glue"]
            ),
            "interpretation": (
                "MRR improves in both directions; R2E Hit@5/10/20 improve; "
                "E2R Hit@10 and Hit@20 decrease only slightly while Hit@3/5 improve"
            ),
        },
        "post_confirmation_full_protocols": full_comparison,
        "post_confirmation_broad_185918": {
            "candidate_proteins": 185918,
            "v1_full_model": broad_v1_full,
            "v2_full_modes": broad_v2_full,
            "v2_universal_component_only": broad_v2_universal,
            "delta_v2_minus_v1": deltas(broad_v1_full, broad_v2_full),
        },
        "decision": {
            "promote_v2": True,
            "reason": (
                "the empirical 0.02 agreement penalty is unnecessary for robust retrieval; "
                "removing it simplifies the mathematical object while preserving or improving "
                "the major frozen and broad-universe evidence surfaces"
            ),
            "v1_remains_historical_frozen_record": True,
        },
        "provenance": {
            "config": "reproducibility/bime_rank/configs/fibre_conditional_modes_v2.yaml",
            "canonical_evaluator": "reproducibility/bime_rank/support/evaluate_multi_expert_protocol_comparison.py",
            "development_baseline": "results/fibre_fair_baseline_glue002_dev_v1",
            "development_candidate": "results/fibre_fair_noglue_dev_v1",
            "frozen_baseline": "results/fibre_fair_baseline_glue002_frozen_v1",
            "frozen_candidate": "results/fibre_fair_noglue_frozen_v1",
            "full_v2": "results/fibre_conditional_modes_full_v2",
            "broad_v2": "results/fibre_conditional_modes_tps_broad_v2",
        },
        "sha256": {
            "config": sha256(ROOT / "reproducibility/bime_rank/configs/fibre_conditional_modes_v2.yaml"),
            "canonical_evaluator": sha256(ROOT / "reproducibility/bime_rank/support/evaluate_multi_expert_protocol_comparison.py"),
            "kernel_atlas": sha256(ROOT / "projects/active/fibre/kernel/atlas.py"),
            "full_v2_metrics": sha256(PATHS["full_v2"] / "metrics.csv"),
            "broad_v2_metrics": sha256(PATHS["broad_v2"] / "metrics.csv"),
            "frozen_candidate_query_metrics": sha256(PATHS["frozen_no_glue"] / "query_metrics.csv"),
        },
    }
    OUTPUT.write_text(json.dumps(scorecard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scorecard, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
