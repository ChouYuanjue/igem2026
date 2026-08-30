from __future__ import annotations

import pandas as pd

from projects.active.terpene_screening.prepare_nested_broad_rhea_dev import (
    build_nested_temporal_double_cold,
)


def test_nested_temporal_double_cold_stays_inside_parent_train_and_isolates_entities():
    dated = pd.DataFrame(
        [
            {"protein_id": "P_train", "reaction_id": "R_train", "creation_date": 20160101},
            {"protein_id": "P_dev", "reaction_id": "R_dev", "creation_date": 20180101},
            # Later pair shares a train protein: must not enter double-cold dev.
            {"protein_id": "P_train", "reaction_id": "R_new", "creation_date": 20180102},
            # Later pair is absent from the parent outer-train partition: must be ignored.
            {"protein_id": "P_outside", "reaction_id": "R_outside", "creation_date": 20180103},
            # Both entities are new to inner train but protein was exposed by the base checkpoint.
            {"protein_id": "P_base", "reaction_id": "R_clean", "creation_date": 20180104},
        ]
    )
    parent_train = pd.DataFrame(
        [
            {"protein_id": "P_train", "reaction_id": "R_train"},
            {"protein_id": "P_dev", "reaction_id": "R_dev"},
            {"protein_id": "P_train", "reaction_id": "R_new"},
            {"protein_id": "P_base", "reaction_id": "R_clean"},
        ]
    )
    train, dev = build_nested_temporal_double_cold(
        parent_train,
        dated,
        cutoff=20161231,
        base_proteins={"P_base"},
        base_reactions=set(),
        base_pairs=set(),
    )
    assert set(map(tuple, train[["protein_id", "reaction_id"]].itertuples(index=False, name=None))) == {
        ("P_train", "R_train")
    }
    assert set(map(tuple, dev[["protein_id", "reaction_id"]].itertuples(index=False, name=None))) == {
        ("P_dev", "R_dev")
    }


def test_nested_temporal_double_cold_rejects_parent_pair_without_dated_provenance():
    dated = pd.DataFrame(
        [{"protein_id": "P1", "reaction_id": "R1", "creation_date": 20160101}]
    )
    parent_train = pd.DataFrame(
        [
            {"protein_id": "P1", "reaction_id": "R1"},
            {"protein_id": "P_missing", "reaction_id": "R_missing"},
        ]
    )
    import pytest

    with pytest.raises(ValueError, match="without dated Rhea provenance"):
        build_nested_temporal_double_cold(
            parent_train,
            dated,
            cutoff=20161231,
            base_proteins=set(),
            base_reactions=set(),
            base_pairs=set(),
        )
