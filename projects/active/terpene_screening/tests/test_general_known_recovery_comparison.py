from __future__ import annotations

import pandas as pd
import pytest

from projects.active.terpene_screening.compare_general_known_recovery import compare_matched_queries


def _rows(hit_candidate: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.DataFrame(
        [
            {
                "direction": "reaction_to_enzyme",
                "stratum": "unseen_to_historical_training",
                "query_id": "R1",
                "n_positives": 1,
                "reciprocal_rank": 0.1,
                "hit_at_1": 0,
                "hit_at_3": 0,
                "hit_at_5": 0,
                "hit_at_10": 0,
                "hit_at_20": 1,
            },
            {
                "direction": "reaction_to_enzyme",
                "stratum": "historical_training_pair",
                "query_id": "R2",
                "n_positives": 1,
                "reciprocal_rank": 1.0,
                "hit_at_1": 1,
                "hit_at_3": 1,
                "hit_at_5": 1,
                "hit_at_10": 1,
                "hit_at_20": 1,
            },
        ]
    )
    candidate = base.copy()
    candidate.loc[candidate.query_id.eq("R1"), ["reciprocal_rank", "hit_at_10"]] = [0.2, hit_candidate]
    return base, candidate


def test_compare_requires_exact_query_population():
    base, candidate = _rows()
    candidate = candidate[candidate.query_id.ne("R1")]
    with pytest.raises(ValueError, match="query populations do not match exactly"):
        compare_matched_queries(base, candidate)


def test_compare_reports_unseen_gain_and_passes_unchanged_history():
    base, candidate = _rows()
    comparison, summary = compare_matched_queries(base, candidate)
    unseen = comparison[comparison.stratum.eq("unseen_to_historical_training")].iloc[0]
    assert unseen["delta_hit_at_10"] == pytest.approx(1.0)
    assert unseen["ratio_reciprocal_rank"] == pytest.approx(2.0)
    assert summary["retention_guard_passed"] is True


def test_compare_fails_guard_on_any_historical_metric_drop():
    base, candidate = _rows()
    candidate.loc[candidate.query_id.eq("R2"), "hit_at_10"] = 0
    _, summary = compare_matched_queries(base, candidate)
    assert summary["retention_guard_passed"] is False
    assert any(item["metric"] == "hit_at_10" for item in summary["retention_violations"])
