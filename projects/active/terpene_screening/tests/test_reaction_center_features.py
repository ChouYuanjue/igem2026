from __future__ import annotations

import numpy as np

from projects.active.terpene_screening.build_reaction_center_augmented_features import reaction_center_features


def test_reaction_center_features_detect_mapped_bond_order_change() -> None:
    # Oxidation-like C-O single -> double change with stable atom maps.
    rxn = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
    feature, audit = reaction_center_features(rxn)
    assert feature.shape == (1280,)
    assert feature.dtype == np.float32
    assert audit["changed_bond_count"] >= 1
    assert audit["changed_map_count"] >= 2
    assert audit["feature_nonzero"] > 0


def test_reaction_center_identity_has_zero_extension() -> None:
    rxn = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]"
    feature, audit = reaction_center_features(rxn)
    assert np.count_nonzero(feature) == 0
    assert audit["changed_bond_count"] == 0
    assert audit["changed_atom_state_count"] == 0


def test_reaction_center_features_are_deterministic() -> None:
    rxn = "[CH3:1][CH2:2][Br:3]>>[CH3:1][CH2:2][OH:3]"
    a, aa = reaction_center_features(rxn)
    b, bb = reaction_center_features(rxn)
    assert np.array_equal(a, b)
    assert aa == bb
