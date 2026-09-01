from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.terpene_screening.broad_rhea_metrics import (
    DEFAULT_BUDGETS,
    DEFAULT_TOP_PERCENTS,
    summarize_query_metrics,
)
from projects.active.terpene_screening.run_internal_top2000_pair_reranker_v1 import (
    query_metrics_from_positive_rank_frame,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = ROOT / "projects/active/terpene_screening/CATALYST_COMPREHENSIVE_CENTER_TOP2000_V1.json"
DEFAULT_DEV_ROOT = ROOT / "results/comprehensive_center_top2000_v1/dev"
DEFAULT_CONFIRM_ROOT = ROOT / "results/comprehensive_center_top2000_v1/confirmation"


def _summary(frame: pd.DataFrame) -> dict[str, float | int]:
    return summarize_query_metrics(frame, budgets=DEFAULT_BUDGETS, top_percents=DEFAULT_TOP_PERCENTS)


def _delta(candidate: dict[str, object], baseline: dict[str, object]) -> dict[str, float]:
    keys = ("mrr", "map", "macro_roc_auc", "ndcg_at_10", "hit_at_10", "hit_at_20", "hit_at_50")
    return {key: float(candidate[key]) - float(baseline[key]) for key in keys}


def _fold_stability(
    candidate: dict[int, dict[str, object]], baseline: dict[int, dict[str, object]]
) -> dict[str, object]:
    deltas: dict[str, dict[str, float]] = {}
    improvement_count = {"mrr": 0, "map": 0}
    no_bad_regression = True
    for fold in sorted(candidate):
        row = {
            key: float(candidate[fold][key]) - float(baseline[fold][key])
            for key in ("mrr", "map")
        }
        deltas[str(fold)] = row
        for key in improvement_count:
            improvement_count[key] += int(row[key] > 1e-12)
            if row[key] < -0.005 - 1e-12:
                no_bad_regression = False
    enough_improvement = max(improvement_count.values()) >= 2
    return {
        "per_fold_deltas": deltas,
        "no_mrr_or_map_regression_gt_0p005": no_bad_regression,
        "improvement_fold_counts": improvement_count,
        "improves_mrr_or_map_on_at_least_2_of_3_folds": enough_improvement,
        "pass": bool(no_bad_regression and enough_improvement),
    }


def _material_gain(delta: dict[str, float]) -> bool:
    return bool(
        delta["mrr"] >= 0.01 - 1e-12
        or delta["map"] >= 0.01 - 1e-12
        or delta["hit_at_10"] >= 0.02 - 1e-12
    )


def _fallback_invariant(
    coarse: pd.DataFrame,
    routed: pd.DataFrame,
    support: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, object]:
    support = support.copy()
    support["max_train_drfp_tanimoto"] = pd.to_numeric(
        support["max_train_drfp_tanimoto"], errors="raise"
    )
    fallback = support[support["max_train_drfp_tanimoto"] < threshold].copy()
    if fallback.empty:
        raise ValueError("No fallback queries below the frozen similarity threshold")
    if not (pd.to_numeric(fallback["routed_residual_scale"], errors="raise") == 0.0).all():
        raise AssertionError("Below-threshold query received a nonzero residual scale")
    if not (pd.to_numeric(fallback["pair_reranker_selected"], errors="raise") == 0).all():
        raise AssertionError("Below-threshold query was marked as pair-reranker selected")
    exact_rank_flags = pd.to_numeric(fallback["positive_ranks_exactly_preserved"], errors="raise")
    signatures_equal = (
        fallback["coarse_positive_rank_signature"].astype(str)
        == fallback["reranked_positive_rank_signature"].astype(str)
    )
    rank_identity = bool((exact_rank_flags == 1).all() and signatures_equal.all())

    query_ids = fallback["query_id"].astype(str).tolist()
    c = coarse[coarse["query_id"].astype(str).isin(query_ids)].sort_values("query_id").reset_index(drop=True)
    r = routed[routed["query_id"].astype(str).isin(query_ids)].sort_values("query_id").reset_index(drop=True)
    if c["query_id"].astype(str).tolist() != r["query_id"].astype(str).tolist():
        raise AssertionError("Fallback coarse/routed query IDs differ")
    common = [
        col
        for col in c.columns
        if col in r.columns and col not in {"direction", "query_id"}
    ]
    numeric_common = [col for col in common if pd.api.types.is_numeric_dtype(c[col]) and pd.api.types.is_numeric_dtype(r[col])]
    max_abs_diff = 0.0
    for col in numeric_common:
        cv = pd.to_numeric(c[col], errors="raise").to_numpy(dtype=np.float64)
        rv = pd.to_numeric(r[col], errors="raise").to_numpy(dtype=np.float64)
        diff = np.abs(cv - rv)
        if np.isnan(diff).all():
            continue
        max_abs_diff = max(max_abs_diff, float(np.nanmax(diff)))
    metrics_exact = bool(max_abs_diff <= 1e-15)
    return {
        "query_count": int(len(fallback)),
        "positive_rank_signatures_exact": rank_identity,
        "query_metrics_max_abs_diff": max_abs_diff,
        "query_metrics_exact": metrics_exact,
        "pass": bool(rank_identity and metrics_exact),
    }


def _load_fold(root: Path, *, fold: int, cell: str, threshold: float) -> dict[str, object]:
    direct_coarse_dir = root / "coarse_eval" / cell
    folded_coarse_dir = root / "coarse_eval" / f"fold{fold}" / cell
    coarse_dir = direct_coarse_dir if direct_coarse_dir.is_dir() else folded_coarse_dir
    rerank_dir = root / "reranked" / f"fold{fold}"
    coarse_positive = pd.read_csv(coarse_dir / "positive_ranks.csv", dtype={"query_id": str, "positive_id": str})
    coarse_positive = coarse_positive[coarse_positive["direction"] == "reaction_to_enzyme"].copy()
    coarse_query = query_metrics_from_positive_rank_frame(coarse_positive)
    routed_query = pd.read_csv(rerank_dir / "query_metrics.csv", dtype={"query_id": str}).fillna(np.nan)
    support = pd.read_csv(rerank_dir / "support_audit.csv", dtype={"query_id": str}).fillna("")
    if coarse_query["query_id"].astype(str).tolist() != routed_query["query_id"].astype(str).tolist():
        coarse_ids = set(coarse_query["query_id"].astype(str))
        routed_ids = set(routed_query["query_id"].astype(str))
        if coarse_ids != routed_ids:
            raise AssertionError(f"fold{fold} coarse/routed query sets differ")
        coarse_query = coarse_query.sort_values("query_id").reset_index(drop=True)
        routed_query = routed_query.sort_values("query_id").reset_index(drop=True)
    support["max_train_drfp_tanimoto"] = pd.to_numeric(support["max_train_drfp_tanimoto"], errors="raise")
    high_ids = set(support.loc[support["max_train_drfp_tanimoto"] >= threshold, "query_id"].astype(str))
    high_coarse = coarse_query[coarse_query["query_id"].astype(str).isin(high_ids)].copy()
    high_routed = routed_query[routed_query["query_id"].astype(str).isin(high_ids)].copy()
    return {
        "coarse_query": coarse_query,
        "routed_query": routed_query,
        "high_coarse": high_coarse,
        "high_routed": high_routed,
        "support": support,
        "fallback": _fallback_invariant(coarse_query, routed_query, support, threshold=threshold),
        "coarse_summary": _summary(coarse_query),
        "routed_summary": _summary(routed_query),
        "high_coarse_summary": _summary(high_coarse),
        "high_routed_summary": _summary(high_routed),
    }


def _development_result(protocol: dict[str, object], root: Path) -> dict[str, object]:
    split = dict(protocol["development_split"])
    folds = [int(v) for v in split["development_folds"]]
    threshold = float(protocol["top2000_refinement"]["router_min_reaction_similarity"])
    template = f"clean2023_internal_double_cold_salted_{split['split_salt']}_fold{{fold}}"
    rows = {fold: _load_fold(root, fold=fold, cell=template.format(fold=fold), threshold=threshold) for fold in folds}
    coarse_all = pd.concat([rows[f]["coarse_query"] for f in folds], ignore_index=True)
    routed_all = pd.concat([rows[f]["routed_query"] for f in folds], ignore_index=True)
    high_coarse = pd.concat([rows[f]["high_coarse"] for f in folds], ignore_index=True)
    high_routed = pd.concat([rows[f]["high_routed"] for f in folds], ignore_index=True)
    pooled = {
        "coarse": _summary(coarse_all),
        "routed": _summary(routed_all),
        "high_similarity_ge0p9_coarse": _summary(high_coarse),
        "high_similarity_ge0p9_routed": _summary(high_routed),
    }
    delta = _delta(pooled["routed"], pooled["coarse"])
    high_delta = _delta(pooled["high_similarity_ge0p9_routed"], pooled["high_similarity_ge0p9_coarse"])
    overall_checks = {
        key: float(pooled["routed"][key]) >= float(pooled["coarse"][key]) - 1e-12
        for key in ("mrr", "map", "ndcg_at_10", "hit_at_10", "hit_at_20", "hit_at_50")
    }
    high_checks = {
        "mrr_strict": float(high_delta["mrr"]) > 1e-12,
        "map_strict": float(high_delta["map"]) > 1e-12,
        "ndcg_at_10": float(high_delta["ndcg_at_10"]) >= -1e-12,
        "hit_at_10": float(high_delta["hit_at_10"]) >= -1e-12,
    }
    baseline_fold = {fold: rows[fold]["coarse_summary"] for fold in folds}
    candidate_fold = {fold: rows[fold]["routed_summary"] for fold in folds}
    stability = _fold_stability(candidate_fold, baseline_fold)
    fallback = {str(fold): rows[fold]["fallback"] for fold in folds}
    fallback_pass = all(v["pass"] for v in fallback.values())
    material = _material_gain(delta)
    passed = bool(
        all(overall_checks.values())
        and all(high_checks.values())
        and stability["pass"]
        and fallback_pass
        and material
    )
    return {
        "mode": "development",
        "protocol_status": protocol["status"],
        "selection_uses_external_or_revealed_outer": False,
        "folds": folds,
        "pooled_metrics": pooled,
        "paired_delta_routed_minus_center": delta,
        "high_similarity_ge0p9_delta": high_delta,
        "fold_metrics": {
            "coarse": {str(f): baseline_fold[f] for f in folds},
            "routed": {str(f): candidate_fold[f] for f in folds},
        },
        "checks": {
            "overall": overall_checks,
            "high_similarity_ge0p9": high_checks,
            "fallback_invariant_by_fold": fallback,
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
    cell = f"clean2023_internal_double_cold_salted_{split['split_salt']}_fold{fold}"
    row = _load_fold(root, fold=fold, cell=cell, threshold=threshold)
    coarse, routed = row["coarse_summary"], row["routed_summary"]
    high_coarse, high_routed = row["high_coarse_summary"], row["high_routed_summary"]
    delta = _delta(routed, coarse)
    high_delta = _delta(high_routed, high_coarse)
    overall_checks = {
        "mrr_strict": delta["mrr"] > 1e-12,
        "map_strict": delta["map"] > 1e-12,
        "ndcg_at_10_strict": delta["ndcg_at_10"] > 1e-12,
        "hit_at_10_strict": delta["hit_at_10"] > 1e-12,
        "hit_at_20_no_regress": delta["hit_at_20"] >= -1e-12,
        "hit_at_50_no_regress": delta["hit_at_50"] >= -1e-12,
    }
    high_check = bool(high_delta["mrr"] >= -1e-12 and high_delta["map"] >= -1e-12 and (high_delta["mrr"] > 1e-12 or high_delta["map"] > 1e-12))
    passed = bool(all(overall_checks.values()) and high_check and row["fallback"]["pass"])
    return {
        "mode": "confirmation",
        "protocol_status": protocol["status"],
        "selection_uses_external_or_revealed_outer": False,
        "fold": fold,
        "metrics": {"coarse": coarse, "routed": routed, "high_similarity_ge0p9_coarse": high_coarse, "high_similarity_ge0p9_routed": high_routed},
        "paired_delta_routed_minus_center": delta,
        "high_similarity_ge0p9_delta": high_delta,
        "checks": {"overall": overall_checks, "high_similarity_ge0p9": high_check, "fallback_invariant": row["fallback"]},
        "confirmation_gate_pass": passed,
        "next_action": "promote_center_top2000_production_route" if passed else "reject_no_post_confirmation_adjustment",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply the frozen center+Top-2000 comprehensive development or confirmation gate.")
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
