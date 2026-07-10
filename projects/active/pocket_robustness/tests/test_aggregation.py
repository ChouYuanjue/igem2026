import numpy as np
import pandas as pd

from projects.active.pocket_robustness.analysis.aggregate_pocket_scores import aggregate_scores


def _demo_df():
    return pd.DataFrame(
        {
            "enzyme_id": ["A", "A", "A", "B", "B"],
            "pocket_rank": [1, 2, 3, 1, 2],
            "pocket_source": ["p2rank", "p2rank", "fpocket", "p2rank", "fpocket"],
            "score": [0.1, 0.8, 0.3, 0.2, 0.4],
        }
    )


def _score_for(result, enzyme_id):
    return result.loc[result["enzyme_id"] == enzyme_id, "aggregated_score"].iloc[0]


def test_max_aggregation():
    result = aggregate_scores(_demo_df(), method="max")

    assert np.isclose(_score_for(result, "A"), 0.8)
    assert np.isclose(_score_for(result, "B"), 0.4)
    assert {"aggregated_score", "n_pockets", "best_pocket_rank"}.issubset(result.columns)


def test_mean_aggregation():
    result = aggregate_scores(_demo_df(), method="mean")

    assert np.isclose(_score_for(result, "A"), (0.1 + 0.8 + 0.3) / 3)
    assert np.isclose(_score_for(result, "B"), 0.3)


def test_rank_weighted_aggregation():
    result = aggregate_scores(_demo_df(), method="rank_weighted")

    expected_a = np.average([0.1, 0.8, 0.3], weights=[1.0, 0.5, 1.0 / 3.0])
    expected_b = np.average([0.2, 0.4], weights=[1.0, 0.5])
    assert np.isclose(_score_for(result, "A"), expected_a)
    assert np.isclose(_score_for(result, "B"), expected_b)


def test_softmax_pool_aggregation():
    result = aggregate_scores(_demo_df(), method="softmax_pool", temperature=0.2)

    assert _score_for(result, "A") > 0.6
    assert _score_for(result, "B") > 0.3
    assert set(result["n_pockets"]) == {2, 3}
    assert result.loc[result["enzyme_id"] == "A", "best_pocket_rank"].iloc[0] == 2


def test_source_weighted_aggregation():
    result = aggregate_scores(
        _demo_df(),
        method="source_weighted",
        source_weights={"p2rank": 0.6, "fpocket": 0.4},
    )

    expected_a = np.average([0.1, 0.8, 0.3], weights=[0.6, 0.6, 0.4])
    assert np.isclose(_score_for(result, "A"), expected_a)
    assert result["used_fallback"].eq(False).all()


def test_source_balanced_aggregations():
    mean_result = aggregate_scores(_demo_df(), method="source_balanced_mean")
    rank_result = aggregate_scores(_demo_df(), method="source_balanced_rank_weighted")
    softmax_result = aggregate_scores(_demo_df(), method="source_balanced_softmax_pool", temperature=0.2)

    expected_a_mean = np.mean([np.mean([0.1, 0.8]), np.mean([0.3])])
    expected_b_mean = np.mean([np.mean([0.2]), np.mean([0.4])])
    expected_a_rank = np.mean([
        np.average([0.1, 0.8], weights=[1.0, 0.5]),
        np.average([0.3], weights=[1.0]),
    ])
    expected_b_rank = np.mean([
        np.average([0.2], weights=[1.0]),
        np.average([0.4], weights=[1.0]),
    ])

    assert np.isclose(_score_for(mean_result, "A"), expected_a_mean)
    assert np.isclose(_score_for(mean_result, "B"), expected_b_mean)
    assert np.isclose(_score_for(rank_result, "A"), expected_a_rank)
    assert np.isclose(_score_for(rank_result, "B"), expected_b_rank)
    assert softmax_result["used_fallback"].eq(False).all()
    assert _score_for(softmax_result, "A") > _score_for(mean_result, "A")


def test_prior_scaffold_fallback():
    result = aggregate_scores(_demo_df(), method="residue_prior_weighted")

    assert result["used_fallback"].eq(True).all()
    assert result["fallback_reason"].str.contains("fallback").all()


if __name__ == "__main__":
    pass
