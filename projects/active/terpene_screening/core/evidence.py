from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd


EVIDENCE_PASSPORT_VERSION = "terpene-candidate-evidence-passport-v1"
APPLICABILITY_MODEL_VERSION = "terpene-open-world-applicability-v1"


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clip01(value: object, default: float = 0.0) -> float:
    return float(np.clip(_finite_float(value, default), 0.0, 1.0))


def _inverse_dispersion(value: object) -> float:
    dispersion = max(_finite_float(value, float("inf")), 0.0)
    if not math.isfinite(dispersion):
        return 0.0
    return 1.0 / (1.0 + dispersion)


def _positive_margin(value: object) -> float:
    margin = max(_finite_float(value, 0.0), 0.0)
    return margin / (1.0 + margin)


def applicability_tier(score: float, *, current_entity: bool = False) -> str:
    if current_entity:
        return "reference_library"
    if score >= 0.80:
        return "in_domain"
    if score >= 0.60:
        return "near_domain"
    if score >= 0.40:
        return "weakly_supported"
    return "far_out_of_domain"


def compute_query_applicability(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Build a transparent, non-probabilistic open-world applicability proxy.

    The score intentionally uses only diagnostics already produced by the
    retrieval route. It is not trained against biochemical outcomes and must
    not be interpreted as a probability of catalytic activity.
    """

    current_entity = bool(row.get("query_is_current_entity", False))
    nearest_similarity = _clip01(row.get("query_nearest_library_similarity"), 0.0)
    top1_vote = _clip01(row.get("ensemble_top1_vote_fraction"), 0.0)
    topk_jaccard = _clip01(row.get("ensemble_topk_jaccard"), 0.0)
    topk_vote = _clip01(row.get("ensemble_topk_vote_mean"), 0.0)
    rank_stability = _inverse_dispersion(row.get("ensemble_top1_rank_std"))
    boundary_separation = _positive_margin(row.get("ensemble_boundary_margin_z"))

    components = {
        "nearest_library_similarity": nearest_similarity,
        "ensemble_top1_consensus": top1_vote,
        "ensemble_topk_set_stability": topk_jaccard,
        "ensemble_topk_membership_support": topk_vote,
        "top1_rank_stability": rank_stability,
        "topk_boundary_separation": boundary_separation,
    }
    weights = {
        "nearest_library_similarity": 0.40,
        "ensemble_top1_consensus": 0.15,
        "ensemble_topk_set_stability": 0.15,
        "ensemble_topk_membership_support": 0.15,
        "top1_rank_stability": 0.10,
        "topk_boundary_separation": 0.05,
    }
    score = sum(components[key] * weights[key] for key in weights)
    if current_entity:
        score = max(score, 0.95)
    score = _clip01(score)
    tier = applicability_tier(score, current_entity=current_entity)
    recommendation = {
        "reference_library": "standard_shortlist",
        "in_domain": "standard_shortlist",
        "near_domain": "shortlist_with_manual_review",
        "weakly_supported": "expand_candidates_or_add_seed",
        "far_out_of_domain": "abstain_or_collect_supporting_evidence",
    }[tier]
    return {
        "version": APPLICABILITY_MODEL_VERSION,
        "score": score,
        "tier": tier,
        "recommendation": recommendation,
        "components": components,
        "interpretation": "diagnostic_applicability_not_activity_probability",
    }


def _candidate_tier(score: float) -> str:
    if score >= 0.78:
        return "priority_candidate"
    if score >= 0.58:
        return "supported_candidate"
    if score >= 0.38:
        return "review_candidate"
    return "exploratory_candidate"


def _candidate_paths(row: pd.Series) -> list[str]:
    paths = ["production_retrieval"]
    source = str(row.get("score_source", ""))
    if "rrf" in source:
        paths.append("rank_fusion")
    if "dual_kernel" in source:
        paths.append("collaborative_dual_kernel")
    if "neighbor" in source or "hybrid" in source:
        paths.append("neighbor_transfer")
    if _finite_float(row.get("ensemble_topk_vote_fraction"), 0.0) >= 2 / 3:
        paths.append("ensemble_consensus")
    if str(row.get("empirical_reliability_status", "")) == "validated_external_double_cold":
        paths.append("validated_reliability_calibration")
    return paths


def _candidate_warnings(row: pd.Series, applicability_tier_value: str) -> list[str]:
    warnings: list[str] = []
    if applicability_tier_value in {"weakly_supported", "far_out_of_domain"}:
        warnings.append("query_outside_strong_applicability_domain")
    if _finite_float(row.get("ensemble_topk_vote_fraction"), 0.0) < 2 / 3:
        warnings.append("limited_ensemble_membership_consensus")
    if str(row.get("empirical_reliability_tier", "")) == "lower_evidence":
        warnings.append("lower_empirical_ranking_reliability")
    if bool(row.get("is_external_candidate", False)):
        warnings.append("external_candidate_requires_identity_and_input_audit")
    return warnings


def apply_evidence_passport(result: pd.DataFrame) -> pd.DataFrame:
    """Annotate a ranking with query applicability and candidate evidence fields.

    The function is deliberately side-effect free with respect to candidate
    score and rank. It only appends columns, preserving the validated production
    ordering and all existing score sources.
    """

    if result.empty:
        return result
    result = result.copy()
    applicability = compute_query_applicability(result.iloc[0])
    result["evidence_passport_version"] = EVIDENCE_PASSPORT_VERSION
    result["applicability_model_version"] = applicability["version"]
    result["query_applicability_score"] = applicability["score"]
    result["query_applicability_tier"] = applicability["tier"]
    result["query_applicability_recommendation"] = applicability["recommendation"]
    result["query_applicability_components"] = json.dumps(
        applicability["components"], sort_keys=True, separators=(",", ":")
    )
    result["query_applicability_interpretation"] = applicability["interpretation"]

    maximum_rank = max(int(_finite_float(result["rank"].max(), len(result))), 1)
    evidence_scores: list[float] = []
    evidence_tiers: list[str] = []
    evidence_paths: list[str] = []
    evidence_warnings: list[str] = []
    evidence_interpretations: list[str] = []
    for _, row in result.iterrows():
        rank = max(_finite_float(row.get("rank"), maximum_rank), 1.0)
        rank_priority = 1.0 if maximum_rank <= 1 else 1.0 - (rank - 1.0) / (maximum_rank - 1.0)
        membership_support = _clip01(row.get("ensemble_topk_vote_fraction"), 0.0)
        rank_stability = _inverse_dispersion(row.get("ensemble_rank_std"))
        reliability = row.get("empirical_reliability_score")
        reliability_support = (
            _clip01(reliability)
            if math.isfinite(_finite_float(reliability, float("nan")))
            else applicability["score"]
        )
        score = _clip01(
            0.35 * rank_priority
            + 0.25 * membership_support
            + 0.20 * rank_stability
            + 0.10 * applicability["score"]
            + 0.10 * reliability_support
        )
        paths = _candidate_paths(row)
        warnings = _candidate_warnings(row, applicability["tier"])
        evidence_scores.append(score)
        evidence_tiers.append(_candidate_tier(score))
        evidence_paths.append(";".join(paths))
        evidence_warnings.append(";".join(warnings))
        evidence_interpretations.append("evidence_strength_not_activity_probability")

    result["candidate_evidence_score"] = evidence_scores
    result["candidate_evidence_tier"] = evidence_tiers
    result["candidate_evidence_paths"] = evidence_paths
    result["candidate_evidence_warnings"] = evidence_warnings
    result["candidate_evidence_interpretation"] = evidence_interpretations
    return result


def cycle_consistency_score(
    forward_rank: int,
    reverse_rank: int | None,
    *,
    reciprocal_rank_constant: float = 10.0,
) -> float:
    """Return a normalized bidirectional reciprocal-rank consistency score."""

    if forward_rank <= 0:
        raise ValueError("forward_rank must be positive")
    if reverse_rank is None:
        return 0.0
    if reverse_rank <= 0:
        raise ValueError("reverse_rank must be positive when supplied")
    if reciprocal_rank_constant < 0:
        raise ValueError("reciprocal_rank_constant must be non-negative")
    raw = 0.5 * (
        1.0 / (reciprocal_rank_constant + forward_rank)
        + 1.0 / (reciprocal_rank_constant + reverse_rank)
    )
    maximum = 1.0 / (reciprocal_rank_constant + 1.0)
    return _clip01(raw / maximum)
