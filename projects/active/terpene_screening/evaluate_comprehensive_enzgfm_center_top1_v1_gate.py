from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.terpene_screening.evaluate_comprehensive_center_top2000_v1_gate import (
    _delta,
    _fallback_invariant,
    _fold_stability,
    _material_gain,
    _summary,
)
from projects.active.terpene_screening.run_internal_top2000_pair_reranker_v1 import (
    query_metrics_from_positive_rank_frame,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = ROOT / "projects/active/terpene_screening/CATALYST_COMPREHENSIVE_ENZGFM_CENTER_TOP1_V1.json"
DEFAULT_DEV_ROOT = ROOT / "results/comprehensive_enzgfm_center_top1_v1/dev"
DEFAULT_CONFIRM_ROOT = ROOT / "results/comprehensive_enzgfm_center_top1_v1/confirmation"


def _query_metrics_from_eval(eval_dir: Path) -> pd.DataFrame:
    positive = pd.read_csv(
        eval_dir / "positive_ranks.csv", dtype={"query_id": str, "positive_id": str}
    )
    positive = positive[positive["direction"] == "reaction_to_enzyme"].copy()
    return query_metrics_from_positive_rank_frame(positive)


def _align(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    ids = [set(frame["query_id"].astype(str)) for frame in frames]
    if any(values != ids[0] for values in ids[1:]):
        raise AssertionError("query sets differ across matched baseline/candidate/routed artifacts")
    return tuple(frame.sort_values("query_id").reset_index(drop=True) for frame in frames)


def _rank1_protection_invariant(
    support: pd.DataFrame, *, threshold: float, protected_prefix: int
) -> dict[str, object]:
    support = support.copy()
    support["max_train_drfp_tanimoto"] = pd.to_numeric(
        support["max_train_drfp_tanimoto"], errors="raise"
    )
    selected = support[support["max_train_drfp_tanimoto"] >= threshold].copy()
    if selected.empty:
        raise ValueError("No selected queries at or above the frozen similarity threshold")
    prefix_ok = bool(
        (pd.to_numeric(selected["protected_coarse_prefix"], errors="raise") == protected_prefix).all()
    )
    selected_flags = bool(
        (pd.to_numeric(selected["pair_reranker_selected"], errors="raise") == 1).all()
        and (pd.to_numeric(selected["routed_residual_scale"], errors="raise") > 0).all()
    )
    id_equal = bool(
        (selected["coarse_top1_id"].astype(str) == selected["routed_top1_id"].astype(str)).all()
    )
    flag_equal = bool(
        (pd.to_numeric(selected["coarse_top1_preserved"], errors="raise") == 1).all()
    )
    return {
        "query_count": int(len(selected)),
        "protected_prefix_exact": prefix_ok,
        "selected_queries_use_residual": selected_flags,
        "coarse_and_routed_top1_ids_exact": id_equal,
        "coarse_top1_preserved_flags_exact": flag_equal,
        "pass": bool(prefix_ok and selected_flags and id_equal and flag_equal),
    }


def _load_fold(root: Path, *, fold: int, cell: str, threshold: float, protected_prefix: int) -> dict[str, object]:
    baseline = _query_metrics_from_eval(root / "baseline_eval" / cell)
    candidate_coarse = _query_metrics_from_eval(root / "candidate_coarse_eval" / cell)
    routed_dir = root / "candidate_reranked" / f"fold{fold}"
    routed = pd.read_csv(routed_dir / "query_metrics.csv", dtype={"query_id": str}).fillna(np.nan)
    baseline, candidate_coarse, routed = _align(baseline, candidate_coarse, routed)
    support = pd.read_csv(routed_dir / "support_audit.csv", dtype={"query_id": str}).fillna("")
    support["max_train_drfp_tanimoto"] = pd.to_numeric(
        support["max_train_drfp_tanimoto"], errors="raise"
    )
    slices = pd.read_csv(root / "difficulty" / cell / "reaction_slices.csv", dtype={"reaction_id": str})
    slices = slices.rename(columns={"reaction_id": "query_id"})
    if "reaction_similarity_bucket" not in slices.columns:
        raise ValueError("reaction_slices.csv missing reaction_similarity_bucket")
    low_ids = set(slices.loc[slices["reaction_similarity_bucket"].astype(str) == "lt0p3", "query_id"].astype(str))
    if not low_ids:
        raise ValueError("No lt0p3 reaction-similarity queries in frozen fold")
    low_baseline = baseline[baseline["query_id"].astype(str).isin(low_ids)].copy()
    low_routed = routed[routed["query_id"].astype(str).isin(low_ids)].copy()
    return {
        "baseline_query": baseline,
        "candidate_coarse_query": candidate_coarse,
        "routed_query": routed,
        "low_baseline": low_baseline,
        "low_routed": low_routed,
        "support": support,
        "fallback": _fallback_invariant(candidate_coarse, routed, support, threshold=threshold),
        "rank1": _rank1_protection_invariant(
            support, threshold=threshold, protected_prefix=protected_prefix
        ),
        "baseline_summary": _summary(baseline),
        "candidate_coarse_summary": _summary(candidate_coarse),
        "routed_summary": _summary(routed),
        "low_baseline_summary": _summary(low_baseline),
        "low_routed_summary": _summary(low_routed),
    }


def _low_slice_checks(candidate: dict[str, object], baseline: dict[str, object]) -> dict[str, bool]:
    return {
        "mrr": float(candidate["mrr"]) >= float(baseline["mrr"]) - 1e-12,
        "map": float(candidate["map"]) >= float(baseline["map"]) - 1e-12,
        "hit_at_10_floor_minus_0p005": float(candidate["hit_at_10"])
        >= float(baseline["hit_at_10"]) - 0.005 - 1e-12,
        "median_best_positive_rank": float(candidate["median_best_positive_rank"])
        <= float(baseline["median_best_positive_rank"]) + 1e-12,
    }


def _development_result(protocol: dict[str, object], root: Path) -> dict[str, object]:
    split = dict(protocol["development_split"])
    folds = [int(value) for value in split["development_folds"]]
    threshold = float(protocol["top2000_refinement"]["router_min_reaction_similarity"])
    prefix = int(protocol["top2000_refinement"]["protected_coarse_prefix"])
    template = f"clean2023_internal_double_cold_salted_{split['split_salt']}_fold{{fold}}"
    rows = {
        fold: _load_fold(
            root, fold=fold, cell=template.format(fold=fold), threshold=threshold, protected_prefix=prefix
        )
        for fold in folds
    }
    baseline_all = pd.concat([rows[f]["baseline_query"] for f in folds], ignore_index=True)
    coarse_all = pd.concat([rows[f]["candidate_coarse_query"] for f in folds], ignore_index=True)
    routed_all = pd.concat([rows[f]["routed_query"] for f in folds], ignore_index=True)
    low_baseline = pd.concat([rows[f]["low_baseline"] for f in folds], ignore_index=True)
    low_routed = pd.concat([rows[f]["low_routed"] for f in folds], ignore_index=True)
    pooled = {
        "matched_current_mainline": _summary(baseline_all),
        "candidate_coarse": _summary(coarse_all),
        "candidate_route": _summary(routed_all),
        "lt0p3_matched_current_mainline": _summary(low_baseline),
        "lt0p3_candidate_route": _summary(low_routed),
    }
    delta = _delta(pooled["candidate_route"], pooled["matched_current_mainline"])
    coarse_delta = _delta(pooled["candidate_coarse"], pooled["matched_current_mainline"])
    low_delta = _delta(pooled["lt0p3_candidate_route"], pooled["lt0p3_matched_current_mainline"])
    overall = {
        key: float(pooled["candidate_route"][key]) >= float(pooled["matched_current_mainline"][key]) - 1e-12
        for key in ("mrr", "map", "ndcg_at_10", "hit_at_10", "hit_at_20", "hit_at_50")
    }
    low_checks = _low_slice_checks(
        pooled["lt0p3_candidate_route"], pooled["lt0p3_matched_current_mainline"]
    )
    baseline_fold = {fold: rows[fold]["baseline_summary"] for fold in folds}
    candidate_fold = {fold: rows[fold]["routed_summary"] for fold in folds}
    stability = _fold_stability(candidate_fold, baseline_fold)
    fallback = {str(fold): rows[fold]["fallback"] for fold in folds}
    rank1 = {str(fold): rows[fold]["rank1"] for fold in folds}
    router_pass = all(v["pass"] for v in fallback.values()) and all(v["pass"] for v in rank1.values())
    material = _material_gain(delta)
    passed = bool(
        all(overall.values())
        and all(low_checks.values())
        and stability["pass"]
        and router_pass
        and material
    )
    return {
        "mode": "development",
        "protocol_status": protocol["status"],
        "selection_uses_external_or_revealed_outer": False,
        "folds": folds,
        "pooled_metrics": pooled,
        "candidate_route_minus_current_mainline": delta,
        "candidate_coarse_minus_current_mainline": coarse_delta,
        "lt0p3_candidate_route_minus_current_mainline": low_delta,
        "fold_metrics": {
            "matched_current_mainline": {str(f): baseline_fold[f] for f in folds},
            "candidate_route": {str(f): candidate_fold[f] for f in folds},
        },
        "checks": {
            "overall_vs_current_mainline": overall,
            "lt0p3_vs_current_mainline": low_checks,
            "exact_candidate_coarse_fallback_by_fold": fallback,
            "protected_coarse_rank1_by_fold": rank1,
            "fold_stability": stability,
            "material_gain": material,
        },
        "development_gate_pass": passed,
        "next_action": "run_single_frozen_confirmation" if passed else "reject_family_no_retuning",
    }


def _confirmation_result(protocol: dict[str, object], root: Path) -> dict[str, object]:
    split = dict(protocol["confirmation_split"])
    fold = int(split["dev_fold"])
    threshold = float(protocol["top2000_refinement"]["router_min_reaction_similarity"])
    prefix = int(protocol["top2000_refinement"]["protected_coarse_prefix"])
    cell = f"clean2023_internal_double_cold_salted_{split['split_salt']}_fold{fold}"
    row = _load_fold(root, fold=fold, cell=cell, threshold=threshold, protected_prefix=prefix)
    baseline = row["baseline_summary"]
    routed = row["routed_summary"]
    low_baseline = row["low_baseline_summary"]
    low_routed = row["low_routed_summary"]
    delta = _delta(routed, baseline)
    overall = {
        "mrr_strict": delta["mrr"] > 1e-12,
        "map_strict": delta["map"] > 1e-12,
        "ndcg_at_10_strict": delta["ndcg_at_10"] > 1e-12,
        "hit_at_10_strict": delta["hit_at_10"] > 1e-12,
        "hit_at_20_no_regress": delta["hit_at_20"] >= -1e-12,
        "hit_at_50_no_regress": delta["hit_at_50"] >= -1e-12,
    }
    low_delta = _delta(low_routed, low_baseline)
    low_checks = {
        "mrr": low_delta["mrr"] >= -1e-12,
        "map": low_delta["map"] >= -1e-12,
        "hit_at_10_floor_minus_0p005": low_delta["hit_at_10"] >= -0.005 - 1e-12,
    }
    router_pass = bool(row["fallback"]["pass"] and row["rank1"]["pass"])
    passed = bool(all(overall.values()) and all(low_checks.values()) and router_pass)
    return {
        "mode": "confirmation",
        "protocol_status": protocol["status"],
        "selection_uses_external_or_revealed_outer": False,
        "fold": fold,
        "metrics": {
            "matched_current_mainline": baseline,
            "candidate_coarse": row["candidate_coarse_summary"],
            "candidate_route": routed,
            "lt0p3_matched_current_mainline": low_baseline,
            "lt0p3_candidate_route": low_routed,
        },
        "candidate_route_minus_current_mainline": delta,
        "lt0p3_candidate_route_minus_current_mainline": low_delta,
        "checks": {
            "overall_vs_current_mainline": overall,
            "lt0p3_vs_current_mainline": low_checks,
            "exact_candidate_coarse_fallback": row["fallback"],
            "protected_coarse_rank1": row["rank1"],
        },
        "confirmation_gate_pass": passed,
        "next_action": "promote_protected_enzgfm_center_route" if passed else "reject_no_post_confirmation_adjustment",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply the frozen protected EnzGFM+center R2E development/confirmation gate.")
    ap.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    ap.add_argument("--mode", choices=("development", "confirmation"), default="development")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    protocol = json.loads(args.protocol.resolve().read_text(encoding="utf-8"))
    root = (args.root or (DEFAULT_DEV_ROOT if args.mode == "development" else DEFAULT_CONFIRM_ROOT)).resolve()
    result = _development_result(protocol, root) if args.mode == "development" else _confirmation_result(protocol, root)
    output = args.output or (root / f"{args.mode}_gate.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
