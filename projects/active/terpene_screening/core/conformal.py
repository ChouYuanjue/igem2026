from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFORMAL_CALIBRATORS = (
    ROOT / "results/terpene_conformal_retrieval_sets/calibrators.json"
)
CONFORMAL_RETRIEVAL_VERSION = "terpene-conformal-retrieval-sets-v1"
CONFORMAL_METHOD = "normalized_best-positive-rank_split-conformal"
SUPPORTED_CONFORMAL_MODES = {"disabled", "annotate", "expand"}


def finite_sample_quantile(values: Iterable[float], alpha: float) -> float:
    """Return the conservative split-conformal order statistic.

    The returned value is the ceil((n + 1) * (1 - alpha))-th smallest score,
    capped at the largest observed score when the nominal order exceeds n.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    scores = np.asarray(list(values), dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if not len(scores):
        raise ValueError("finite_sample_quantile requires at least one finite score")
    scores.sort()
    order = min(len(scores), int(math.ceil((len(scores) + 1) * (1.0 - alpha))))
    return float(scores[order - 1])


def normalized_rank_score(rank: int, candidate_count: int) -> float:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if rank <= 0 or rank > candidate_count:
        raise ValueError("rank must be within the candidate universe")
    if candidate_count == 1:
        return 0.0
    return float((rank - 1) / (candidate_count - 1))


def conformal_set_size(qhat: float, candidate_count: int) -> int:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if not math.isfinite(float(qhat)):
        raise ValueError("qhat must be finite")
    clipped = float(np.clip(qhat, 0.0, 1.0))
    if candidate_count == 1:
        return 1
    return min(candidate_count, max(1, int(math.ceil(clipped * (candidate_count - 1) + 1))))


def applicability_group(tier: object) -> str:
    value = str(tier or "")
    if value in {"reference_library", "in_domain"}:
        return "strong"
    if value == "near_domain":
        return "moderate"
    return "weak"


def _alpha_entry(entries: dict[str, Any], alpha: float) -> tuple[str, dict[str, Any]] | None:
    for key, value in entries.items():
        try:
            numeric = float(key)
        except (TypeError, ValueError):
            continue
        if abs(numeric - alpha) <= 1e-12:
            return key, dict(value)
    return None


@lru_cache(maxsize=8)
def load_conformal_calibrators_cached(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("manifest_version", 0)) != 1:
        raise ValueError(f"Unsupported conformal calibrator manifest: {path}")
    if str(payload.get("conformal_retrieval_version", "")) != CONFORMAL_RETRIEVAL_VERSION:
        raise ValueError(f"Unsupported conformal retrieval version in {path}")
    return payload


def _initialize_columns(result: pd.DataFrame, *, alpha: float, mode: str) -> pd.DataFrame:
    result = result.copy()
    result["conformal_retrieval_version"] = CONFORMAL_RETRIEVAL_VERSION
    result["conformal_method"] = CONFORMAL_METHOD
    result["conformal_mode"] = mode
    result["conformal_alpha"] = float(alpha)
    result["conformal_target_coverage"] = float(1.0 - alpha)
    result["conformal_calibrator"] = ""
    result["conformal_binding_status"] = "not_checked"
    result["conformal_status"] = "unavailable"
    result["conformal_group"] = ""
    result["conformal_group_source"] = ""
    result["conformal_qhat"] = np.nan
    result["conformal_set_size"] = np.nan
    result["conformal_set_fraction"] = np.nan
    result["conformal_set_truncated"] = False
    result["conformal_validation_coverage"] = np.nan
    result["conformal_validation_n"] = np.nan
    result["conformal_set_member"] = False
    result["conformal_guarantee_scope"] = (
        "marginal_at_least_one_known_positive_under_exchangeability_in_locked_double_cold_protocol"
    )
    result["conformal_interpretation"] = (
        "retrieval_set_coverage_not_candidate_activity_probability"
    )
    result["conformal_recommendation"] = "manual_review_required"
    return result


def apply_conformal_retrieval_set(
    result: pd.DataFrame,
    *,
    calibrators_path: Path = DEFAULT_CONFORMAL_CALIBRATORS,
    alpha: float = 0.10,
    mode: str = "annotate",
) -> pd.DataFrame:
    """Append a route-bound conformal retrieval set without changing rank or score.

    The set contains a prefix of the production ranking. Calibrators are built
    from query-disjoint, double-cold best-positive ranks and transported to the
    bound production candidate universe through normalized rank percentiles.
    """

    if result.empty:
        return result
    if mode not in SUPPORTED_CONFORMAL_MODES:
        raise ValueError(f"Unsupported conformal mode: {mode}")
    if not 0.0 < alpha < 1.0:
        raise ValueError("conformal alpha must be strictly between 0 and 1")
    result = _initialize_columns(result, alpha=alpha, mode=mode)
    if mode == "disabled":
        result["conformal_status"] = "disabled"
        result["conformal_binding_status"] = "not_requested"
        result["conformal_recommendation"] = "use_ranked_shortlist"
        return result

    row = result.iloc[0]
    if bool(row.get("query_is_current_entity", False)):
        result["conformal_status"] = "not_applicable_current_entity"
        result["conformal_binding_status"] = "not_applicable"
        result["conformal_recommendation"] = "use_ranked_shortlist"
        return result
    if not calibrators_path.exists():
        result["conformal_status"] = "calibrator_missing"
        result["conformal_binding_status"] = "missing"
        return result

    payload = load_conformal_calibrators_cached(str(calibrators_path.resolve()))
    key = f"{row.get('direction', '')}_{row.get('ranking_objective', '')}"
    result["conformal_calibrator"] = key
    calibrator = payload.get("calibrators", {}).get(key)
    if not calibrator:
        result["conformal_status"] = "calibrator_unavailable"
        result["conformal_binding_status"] = "unavailable"
        return result

    compatibility = dict(calibrator.get("compatibility", {}))
    mismatches: list[str] = []
    for field in ["route_id", "candidate_universe_hash", "model_bundle_version"]:
        expected = str(compatibility.get(field, ""))
        actual = str(row.get(field, ""))
        if expected and expected != actual:
            mismatches.append(f"{field}:{actual}!={expected}")
    if mismatches:
        result["conformal_status"] = "incompatible_calibrator"
        result["conformal_binding_status"] = ";".join(mismatches)
        return result
    result["conformal_binding_status"] = "compatible"

    matched = _alpha_entry(dict(calibrator.get("alphas", {})), alpha)
    if matched is None:
        result["conformal_status"] = "unsupported_alpha"
        return result
    _, alpha_spec = matched
    query_group = applicability_group(row.get("query_applicability_tier", ""))
    selected = dict(alpha_spec.get("global", {}))
    source = "global"
    group_spec = dict(alpha_spec.get("groups", {}).get(query_group, {}))
    if bool(group_spec.get("enabled", False)):
        selected = group_spec
        source = f"mondrian:{query_group}"

    candidate_count = int(row.get("candidate_universe_size", 0) or 0)
    if candidate_count <= 0:
        candidate_count = int(calibrator.get("production_candidate_count", 0) or 0)
    if candidate_count <= 0:
        result["conformal_status"] = "candidate_universe_size_missing"
        return result
    qhat = float(selected["qhat"])
    set_size = conformal_set_size(qhat, candidate_count)
    validation = dict(selected.get("validation", alpha_spec.get("validation", {})))

    result["conformal_status"] = "validated_external_double_cold_transport"
    result["conformal_group"] = query_group
    result["conformal_group_source"] = source
    result["conformal_qhat"] = qhat
    result["conformal_set_size"] = set_size
    result["conformal_set_fraction"] = float(set_size / candidate_count)
    result["conformal_set_truncated"] = bool(len(result) < set_size)
    result["conformal_validation_coverage"] = float(
        validation.get("empirical_coverage", np.nan)
    )
    result["conformal_validation_n"] = float(validation.get("n_test", np.nan))
    result["conformal_set_member"] = result["rank"].astype(int).le(set_size)
    result["conformal_recommendation"] = (
        "expand_output_to_conformal_set" if len(result) < set_size else "review_conformal_set"
    )
    return result
