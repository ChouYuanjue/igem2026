import pandas as pd
import pytest

from projects.active.terpene_screening.audit_epoch_capability_by_difficulty import (
    _validate_pairing,
    compare_metric_dicts,
)


def test_compare_metric_dicts_respects_rank_direction() -> None:
    frame = compare_metric_dicts(
        {"mrr": 0.2, "median_best_positive_rank": 10.0},
        {"mrr": 0.3, "median_best_positive_rank": 8.0},
        direction="enzyme_to_reaction",
        slice_name="protein_identity_bucket",
        slice_value="ge80",
        n_queries=3,
    )
    assert set(frame["status"]) == {"improved"}
    assert frame.loc[frame.metric.eq("mrr"), "improvement_delta"].iloc[0] == pytest.approx(0.1)
    assert frame.loc[frame.metric.eq("median_best_positive_rank"), "improvement_delta"].iloc[0] == pytest.approx(2.0)


def test_validate_pairing_rejects_changed_difficulty_labels() -> None:
    one = pd.DataFrame(
        [
            {"direction": "enzyme_to_reaction", "query_id": "p1", "protein_identity_bucket": "ge80", "reaction_similarity_bucket": None},
            {"direction": "reaction_to_enzyme", "query_id": "r1", "protein_identity_bucket": None, "reaction_similarity_bucket": "lt0p3"},
        ]
    )
    two = one.copy()
    two.loc[two.direction.eq("enzyme_to_reaction"), "protein_identity_bucket"] = "60_80"
    with pytest.raises(ValueError, match="difficulty labels changed"):
        _validate_pairing(one, two)


def test_validate_pairing_requires_identical_queries() -> None:
    one = pd.DataFrame(
        [
            {"direction": "enzyme_to_reaction", "query_id": "p1", "protein_identity_bucket": "ge80", "reaction_similarity_bucket": None},
            {"direction": "reaction_to_enzyme", "query_id": "r1", "protein_identity_bucket": None, "reaction_similarity_bucket": "ge0p9"},
        ]
    )
    two = one.copy()
    two.loc[two.direction.eq("enzyme_to_reaction"), "query_id"] = "p2"
    with pytest.raises(ValueError, match="query sets differ"):
        _validate_pairing(one, two)
