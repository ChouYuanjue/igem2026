from __future__ import annotations

import pandas as pd

from projects.active.terpene_screening.prepare_nested_clean_dev import (
    nested_double_cold,
    nested_protein_cold,
)


def _dense_pairs() -> pd.DataFrame:
    rows = []
    for p in range(30):
        for r in range(15):
            if (p * 7 + r * 3) % 5 != 0:
                rows.append({"protein_id": f"P{p}", "reaction_id": f"R{r}"})
    return pd.DataFrame(rows)


def test_nested_double_cold_has_no_entity_or_pair_overlap():
    pairs = _dense_pairs()
    train, dev = nested_double_cold(
        pairs,
        base_proteins={"P0"},
        base_reactions={"R0"},
        base_pairs={("P0", "R0")},
        modulo=3,
        holdout_bucket=0,
    )
    assert len(train) and len(dev)
    assert set(train["protein_id"]).isdisjoint(set(dev["protein_id"]))
    assert set(train["reaction_id"]).isdisjoint(set(dev["reaction_id"]))
    assert set(map(tuple, train.itertuples(index=False, name=None))).isdisjoint(
        set(map(tuple, dev.itertuples(index=False, name=None)))
    )
    assert "P0" not in set(dev["protein_id"])
    assert "R0" not in set(dev["reaction_id"])


def test_nested_protein_cold_keeps_dev_reactions_seen():
    pairs = _dense_pairs()
    train, dev = nested_protein_cold(
        pairs,
        base_proteins={"P0"},
        base_pairs={("P0", "R0")},
        modulo=3,
        holdout_bucket=0,
    )
    assert len(train) and len(dev)
    assert set(train["protein_id"]).isdisjoint(set(dev["protein_id"]))
    assert set(dev["reaction_id"]).issubset(set(train["reaction_id"]))
    assert "P0" not in set(dev["protein_id"])
