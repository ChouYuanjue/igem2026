"""Coarse, fixed display scales for independent scientific ranking score spaces."""
from __future__ import annotations

from scripts.starase_navigator.retrieval.score_display import (
    SCALE_VERSION,
    display_scale_metadata,
    score_display_fields,
)

SOURCE = "broad+gated_experts+reciprocal_context"


def score(value: float, direction: str = "reaction_to_enzyme", source: str = SOURCE) -> float:
    result = score_display_fields(
        {"score": value, "score_fraction": 1.0}, direction=direction, score_source=source,
    )
    return result["display_score_percent"]


def test_bridge_final_r2e_high_raw_scores_no_longer_fake_near_certain_percent() -> None:
    assert 70 < score(15.9489) < 85
    assert 65 < score(14.1579) < score(15.9489)
    assert 50 < score(11.3188) < score(14.1579)
    assert score(15.9489) != 100
    # Raw score and rank are untouched by presentation calibration.
    row = {"rank": 1, "score": 15.9489, "score_fraction": 1.0}
    _ = score_display_fields(row, direction="reaction_to_enzyme", score_source=SOURCE)
    assert row == {"rank": 1, "score": 15.9489, "score_fraction": 1.0}


def test_identical_model_score_uses_identical_scale_across_any_queries_and_pools() -> None:
    for raw in (-15.0, -1.0, 0.0, 1.0, 5.0, 10.0, 15.9489):
        a = score_display_fields({
            "score": raw, "score_fraction": 1.0,
            "retrieval_pool": "all", "rank": 1,
        }, direction="reaction_to_enzyme", score_source=SOURCE)
        b = score_display_fields({
            "score": raw, "score_fraction": 0.02,
            "retrieval_pool": "prokaryote", "rank": 10,
        }, direction="reaction_to_enzyme", score_source=SOURCE)
        assert a == b
    values = [score(v) for v in [-8, -3, 0, 5, 10, 15, 20]]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_only_score_space_family_and_direction_split_bridge_final() -> None:
    r2e = score_display_fields(
        {"score": 8.59}, direction="reaction_to_enzyme", score_source=SOURCE,
    )
    e2r = score_display_fields(
        {"score": 8.59}, direction="enzyme_to_reaction",
        score_source=SOURCE + "+episodic_memory",
    )
    assert r2e["display_score_profile"] == "bridge_final_r2e"
    assert e2r["display_score_profile"] == "bridge_final_e2r"
    assert r2e["display_score_percent"] != e2r["display_score_percent"]
    meta = display_scale_metadata(direction="reaction_to_enzyme", score_source=SOURCE)
    assert meta["version"] == SCALE_VERSION
    assert meta["temperature"] == 12.0
    assert meta["query_max_normalization"] is False
    assert meta["per_query_fitting"] is False
    assert meta["empirical_probability_calibration"] is False


def test_external_evidence_support_preserved_as_distinct_score_space() -> None:
    existing = score_display_fields(
        {"score": 0.99, "model_support_index": 66.7},
        direction="enzyme_to_reaction", score_source="hybrid_direct_0.99",
    )
    assert existing == {
        "display_score_percent": 66.7,
        "display_score_profile": "engine_evidence_support",
    }
    assert score_display_fields(
        {"score": 0.99, "model_support_index": 66.7},
        direction="reaction_to_enzyme", score_source=SOURCE,
    )["display_score_profile"] == "bridge_final_r2e"


def test_unavailable_and_legacy_display_are_deterministic() -> None:
    assert score_display_fields(
        {"score": None, "model_support_index": None},
        direction="reaction_to_enzyme", score_source="unknown",
    )["display_score_percent"] is None
    assert score(0, source="legacy_frozen_rank") == 50.0
    assert score(-3, source="legacy_frozen_rank") < 50
    assert score(3, source="legacy_frozen_rank") > 50
    assert score(float("nan"), source="legacy_frozen_rank") is None

def test_r2e_source_is_read_before_later_metadata_materialization() -> None:
    # The production ranker returns the query alongside candidates, but R2E
    # materializes candidate rows *before* its later query metadata assignment.
    # This regression previously broke the entire AI-native candidate tool.
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "retrieval" / "service.py").read_text()
    r2e = source[source.index('        raw_rows = raw_rows[:selected_top_k]'):]
    first_row = r2e.index('                **score_display_fields({')
    later_query = r2e.index('        query = dict(result.get("query", {}))')
    assert first_row < later_query
    assert 'score_source=str((result.get("query") or {}).get("score_source") or ""))' in r2e[:later_query]
