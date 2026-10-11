"""Stable presentation-scale scores for a small number of ranking-score spaces.

Display calibration only. No success probability, score reranking, query-max
normalization, current-top-k statistics, or per-query fitted parameters.
The production BRIDGE final score spaces have different scales in the two
directions, so they need different fixed display temperatures.
"""
from __future__ import annotations

import math
from typing import Any

SCALE_VERSION = "fixed_score_space_display_v1"
# The BRIDGE final R2E score includes query-local broad z and a sizeable
# reciprocal-context term; E2R uses frozen global affine broad calibration.
# These soft display temperatures temper saturation, without claiming
# experimental probability or fitting a scale to the current request.
_BRIDGE_FINAL_TEMPERATURE = {
    "reaction_to_enzyme": 12.0,
    "enzyme_to_reaction": 8.0,
}
_LEGACY_TEMPERATURE = 3.0


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def score_display_fields(
    row: dict[str, Any], *, direction: str, score_source: str,
) -> dict[str, Any]:
    """Convert a raw rank score using an immutable, coarse score-space scale.

    Identical scores from one model/direction always get identical displayed
    values across candidate pools, query IDs, taxonomy scopes and Top-K sizes.
    """
    raw = _finite(row.get("score"))
    support = _finite(row.get("model_support_index"))
    is_final = str(score_source).startswith("broad+gated_experts+reciprocal_context")
    if raw is not None and is_final and direction in _BRIDGE_FINAL_TEMPERATURE:
        temperature = _BRIDGE_FINAL_TEMPERATURE[direction]
        profile = "bridge_final_r2e" if direction == "reaction_to_enzyme" else "bridge_final_e2r"
    elif support is not None and 0.0 <= support <= 100.0:
        # The engine's separately produced evidence-support index has its own
        # unit and should never be mixed row-by-row with raw rank logits.
        return {
            "display_score_percent": round(support, 1),
            "display_score_profile": "engine_evidence_support",
        }
    elif raw is not None:
        temperature = _LEGACY_TEMPERATURE
        profile = "legacy_rank_score"
    else:
        return {
            "display_score_percent": None,
            "display_score_profile": "unavailable",
        }
    # Fixed-scale soft logistic: no query-dependent maximum and no artificial
    # 100.0 for the top item. The raw scorer's original ordering is untouched.
    scaled = max(-50.0, min(50.0, raw / temperature))
    percent = 100.0 / (1.0 + math.exp(-scaled))
    return {
        "display_score_percent": round(percent, 1),
        "display_score_profile": profile,
    }


def display_scale_metadata(*, direction: str, score_source: str) -> dict[str, Any]:
    final = str(score_source).startswith("broad+gated_experts+reciprocal_context")
    if final and direction in _BRIDGE_FINAL_TEMPERATURE:
        profile = "bridge_final_r2e" if direction == "reaction_to_enzyme" else "bridge_final_e2r"
        temperature: float | None = _BRIDGE_FINAL_TEMPERATURE[direction]
    else:
        profile = "engine_evidence_support_or_legacy_raw"
        temperature = None
    return {
        "version": SCALE_VERSION,
        "profile": profile,
        "temperature": temperature,
        "mapping": "fixed_soft_logistic",
        "grouping": "score_source_family_and_direction",
        "query_max_normalization": False,
        "per_query_fitting": False,
        "reranks_candidates": False,
        "interpretation": "relative_model_display_index_not_activity_probability",
        "empirical_probability_calibration": False,
    }
