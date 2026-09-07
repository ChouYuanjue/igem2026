from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_enzymecage_official_aligned import evaluate_scores  # noqa: E402
from projects.active.terpene_screening.fair_benchmark import sha256_file  # noqa: E402


def _stable_order(scores: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return np.lexsort((ids.astype(str), -scores.astype(float)))


def _rank_percentile(scores: np.ndarray, ids: np.ndarray) -> np.ndarray:
    order = _stable_order(scores, ids)
    out = np.empty(len(order), dtype=np.float64)
    if len(order) == 1:
        out[order[0]] = 1.0
        return out
    out[order] = 1.0 - np.arange(len(order), dtype=np.float64) / (len(order) - 1)
    return out


def _topk_set(scores: np.ndarray, ids: np.ndarray, k: int) -> set[str]:
    order = _stable_order(scores, ids)[: min(k, len(scores))]
    return set(ids[order].astype(str))


def _pairwise_jaccard(sets: list[set[str]]) -> float:
    if len(sets) < 2:
        return 1.0
    values: list[float] = []
    for left, right in combinations(sets, 2):
        union = left | right
        values.append(1.0 if not union else len(left & right) / len(union))
    return float(np.mean(values))


def _gap_confidence(scores: np.ndarray) -> float:
    values = np.asarray(scores, dtype=float)
    if len(values) < 2:
        return 1.0
    ordered = np.sort(values)[::-1]
    scale = float(np.std(values))
    if scale <= 1e-12:
        return 0.0
    z = max(0.0, float((ordered[0] - ordered[1]) / scale))
    return float(1.0 - np.exp(-z))


def cage_query_confidence(scores: np.ndarray) -> dict[str, float]:
    values = np.asarray(scores, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {
            "confidence": 0.0,
            "unique_score_fraction": 0.0,
            "gap_confidence": 0.0,
            "top_tie_fraction": 1.0,
            "spread": 0.0,
            "relative_spread": 0.0,
            "dynamic_confidence": 0.0,
        }
    unique = len(np.unique(finite))
    normalizer = np.log1p(max(2, min(len(finite), 32)))
    unique_conf = float(np.log1p(unique) / normalizer) if normalizer > 0 else 0.0
    unique_conf = float(np.clip(unique_conf, 0.0, 1.0))
    top = float(np.max(finite))
    top_tie_fraction = float(np.mean(finite == top))
    gap_conf = _gap_confidence(finite)
    spread = float(np.max(finite) - np.min(finite))
    relative_spread = spread / max(abs(top), 1e-12)
    dynamic_conf = float(np.clip(relative_spread / 0.1, 0.0, 1.0))
    # EnzymeCAGE sigmoid scores often saturate. Unique-score structure and an
    # actual top gap are not enough when all values differ only at numerical
    # near-zero scale. Require meaningful dynamic range as a multiplicative
    # sanity check; this is label-free and catches the legacy saturation mode.
    base = 0.55 * unique_conf + 0.45 * gap_conf
    confidence = float(np.clip(base * (0.25 + 0.75 * dynamic_conf) * (1.0 - 0.5 * top_tie_fraction), 0.0, 1.0))
    return {
        "confidence": confidence,
        "unique_score_fraction": unique_conf,
        "gap_confidence": gap_conf,
        "top_tie_fraction": top_tie_fraction,
        "spread": spread,
        "relative_spread": float(relative_spread),
        "dynamic_confidence": dynamic_conf,
    }


def neural_query_confidence(group: pd.DataFrame, candidate_col: str) -> dict[str, float]:
    ids = group[candidate_col].astype(str).to_numpy()
    member_cols = sorted(column for column in group.columns if column.startswith("neural_member_") and column.endswith("_score"))
    aggregate = group["neural_score"].to_numpy(float)
    gap_conf = _gap_confidence(aggregate)
    if not member_cols:
        return {
            "confidence": gap_conf,
            "top1_vote_fraction": 1.0,
            "top10_jaccard": 1.0,
            "gap_confidence": gap_conf,
            "member_count": 1,
        }
    top1s: list[str] = []
    top_sets: list[set[str]] = []
    for column in member_cols:
        values = group[column].to_numpy(float)
        order = _stable_order(values, ids)
        top1s.append(str(ids[order[0]]))
        top_sets.append(_topk_set(values, ids, 10))
    counts = pd.Series(top1s).value_counts()
    vote = float(counts.iloc[0] / len(top1s))
    jaccard = _pairwise_jaccard(top_sets)
    confidence = float(np.clip(0.4 * vote + 0.35 * jaccard + 0.25 * gap_conf, 0.0, 1.0))
    return {
        "confidence": confidence,
        "top1_vote_fraction": vote,
        "top10_jaccard": jaccard,
        "gap_confidence": gap_conf,
        "member_count": len(member_cols),
    }


def _expert_weights(cage_conf: float, neural_conf: float, agreement: float) -> tuple[float, float]:
    """Conservative target-label-free gate.

    Catalyst remains the backbone. CAGE gains weight when Catalyst's ensemble is
    uncertain and CAGE is self-consistent, or when the experts already agree.
    This avoids letting a saturated CAGE score vector dominate merely because it
    contains many numerically distinct tiny probabilities.
    """
    agreement = float(np.clip(agreement, 0.0, 1.0))
    cage_weight = 0.10 + 0.35 * cage_conf * (1.0 - neural_conf) + 0.15 * agreement
    cage_weight = float(np.clip(cage_weight, 0.10, 0.60))
    neural_weight = 1.0 - cage_weight
    return cage_weight, neural_weight


def route_direction(frame: pd.DataFrame, *, query_col: str, candidate_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {query_col, candidate_col, "cage_score", "neural_score", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"pair score table missing columns: {sorted(missing)}")
    routed_parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for query_id, group in frame.groupby(query_col, sort=True):
        group = group.copy().reset_index(drop=True)
        ids = group[candidate_col].astype(str).to_numpy()
        cage = group["cage_score"].to_numpy(float)
        neural = group["neural_score"].to_numpy(float)
        cage_top10 = _topk_set(cage, ids, 10)
        neural_top10 = _topk_set(neural, ids, 10)
        overlap = 1.0 if not (cage_top10 | neural_top10) else len(cage_top10 & neural_top10) / len(cage_top10 | neural_top10)
        cage_top1 = str(ids[_stable_order(cage, ids)[0]])
        neural_top1 = str(ids[_stable_order(neural, ids)[0]])
        agreement = 0.5 * float(overlap) + 0.5 * float(cage_top1 == neural_top1)
        cage_diag = cage_query_confidence(cage)
        neural_diag = neural_query_confidence(group, candidate_col)
        cage_weight, neural_weight = _expert_weights(cage_diag["confidence"], neural_diag["confidence"], agreement)
        cage_pct = _rank_percentile(cage, ids)
        neural_pct = _rank_percentile(neural, ids)
        fused = cage_weight * cage_pct + neural_weight * neural_pct
        route = "fusion"
        if neural_weight >= 0.65:
            route = "neural_dominant"
        elif cage_weight >= 0.65:
            route = "cage_dominant"
        group["cage_rank_percentile"] = cage_pct
        group["neural_rank_percentile"] = neural_pct
        group["routed_score"] = fused
        group["cage_weight"] = cage_weight
        group["neural_weight"] = neural_weight
        group["route"] = route
        routed_parts.append(group)
        diagnostics.append({
            "query_id": str(query_id),
            "candidate_count": int(len(group)),
            "route": route,
            "cage_weight": cage_weight,
            "neural_weight": neural_weight,
            "cage_confidence": cage_diag["confidence"],
            "cage_unique_score_fraction": cage_diag["unique_score_fraction"],
            "cage_gap_confidence": cage_diag["gap_confidence"],
            "cage_top_tie_fraction": cage_diag["top_tie_fraction"],
            "cage_spread": cage_diag["spread"],
            "cage_relative_spread": cage_diag["relative_spread"],
            "cage_dynamic_confidence": cage_diag["dynamic_confidence"],
            "neural_confidence": neural_diag["confidence"],
            "neural_top1_vote_fraction": neural_diag["top1_vote_fraction"],
            "neural_top10_jaccard": neural_diag["top10_jaccard"],
            "neural_gap_confidence": neural_diag["gap_confidence"],
            "neural_member_count": neural_diag["member_count"],
            "expert_top1_agree": cage_top1 == neural_top1,
            "expert_top10_jaccard": float(overlap),
        })
    return pd.concat(routed_parts, ignore_index=True), pd.DataFrame(diagnostics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label-free reliability routing between EnzymeCAGE and Catalyst experts on one official candidate reservoir.")
    parser.add_argument("--pair-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proof-status", choices=["strict", "diagnostic_only"], default="strict")
    args = parser.parse_args()
    source = args.pair_scores.resolve(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(source)
    summaries: dict[str, object] = {}
    all_routed: list[pd.DataFrame] = []
    all_diag: list[pd.DataFrame] = []
    for direction, query_col, candidate_col in [
        ("reaction_to_enzyme", "reaction_id", "protein_id"),
        ("enzyme_to_reaction", "protein_id", "reaction_id"),
    ]:
        routed, diag = route_direction(pairs, query_col=query_col, candidate_col=candidate_col)
        routed["direction"] = direction; diag["direction"] = direction
        all_routed.append(routed); all_diag.append(diag)
        routed_metrics, _ = evaluate_scores(routed, "routed_score")
        cage_metrics, _ = evaluate_scores(routed, "cage_score")
        neural_metrics, _ = evaluate_scores(routed, "neural_score")
        key = "reaction_to_enzyme" if direction == "reaction_to_enzyme" else "enzyme_to_reaction"
        summaries[direction] = {
            "routed": routed_metrics[key],
            "pure_cage": cage_metrics[key],
            "neural": neural_metrics[key],
            "route_counts": diag["route"].value_counts().to_dict(),
            "mean_cage_weight": float(diag["cage_weight"].mean()),
            "mean_neural_weight": float(diag["neural_weight"].mean()),
            "top1_agreement_fraction": float(diag["expert_top1_agree"].mean()),
            "mean_top10_jaccard": float(diag["expert_top10_jaccard"].mean()),
        }
    routed_frame = pd.concat(all_routed, ignore_index=True)
    diagnostics = pd.concat(all_diag, ignore_index=True)
    routed_frame.to_csv(output / "routed_pair_scores.csv", index=False)
    diagnostics.to_csv(output / "query_routing_diagnostics.csv", index=False)
    payload = {
        "method": "label_free_reliability_weighted_expert_routing",
        "pair_scores": str(source),
        "pair_scores_sha256": sha256_file(source),
        "proof_status": args.proof_status,
        "target_labels_used_for_routing": False,
        "experts": ["EnzymeCAGE", "Catalyst neural ensemble"],
        "routing_features": [
            "CAGE unique-score structure/top-gap/tie fraction/spread",
            "neural ensemble top1 vote/top10 Jaccard/top-gap",
            "cross-expert top1/top10 agreement (audit only; weights are expert-confidence driven)",
        ],
        "fusion": "conservative reliability-weighted per-query rank percentile; Catalyst is the backbone and EnzymeCAGE receives 0.10-0.60 weight based only on label-free self-consistency/ensemble uncertainty/agreement diagnostics",
        "summaries": summaries,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
