from __future__ import annotations

import pandas as pd

from projects.active.terpene_screening.evaluate_exact_entity_protocols import (
    build_balanced_exact_folds,
)


def test_balanced_exact_folds_are_deterministic_and_complete() -> None:
    pairs = pd.DataFrame(
        {
            "Entry": ["p1", "p1", "p2", "p3", "p3", "p4", "p5", "p6"],
            "rhea_id": ["r1", "r2", "r1", "r2", "r3", "r3", "r4", "r4"],
        }
    )
    first = build_balanced_exact_folds(pairs, "Entry", seed=17, n_folds=3)
    second = build_balanced_exact_folds(pairs, "Entry", seed=17, n_folds=3)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["Entry"]) == set(pairs["Entry"])
    assert not first["Entry"].duplicated().any()
    fold_pair_load = first.groupby("exact_fold")["n_pairs"].sum()
    assert int(fold_pair_load.max() - fold_pair_load.min()) <= 1


def test_exact_holdout_does_not_imply_cluster_holdout() -> None:
    pairs = pd.DataFrame(
        {
            "Entry": ["p_test", "p_homolog", "p_other"],
            "rhea_id": ["r1", "r2", "r3"],
            "protein_cluster": ["cluster_a", "cluster_a", "cluster_b"],
        }
    )
    test_entities = {"p_test"}
    train = pairs[~pairs["Entry"].isin(test_entities)]
    assert "p_test" not in set(train["Entry"])
    assert "p_homolog" in set(train["Entry"])
    assert "cluster_a" in set(train["protein_cluster"])
