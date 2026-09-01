from projects.active.terpene_screening.audit_clipzyme_reactzyme_direction_support_v1 import (
    canonical_bag,
    sideblind_key_from_directed,
)


def test_sideblind_identity_ignores_side_and_component_order_but_not_chemistry():
    bag = 'CCO.O=C=O.CN'
    directed = 'CN.CCO>>O=C=O'
    assert canonical_bag(bag) == sideblind_key_from_directed(directed)
    assert canonical_bag('CCO.O=C=O.CC') != sideblind_key_from_directed(directed)


def test_sideblind_identity_removes_atom_maps_and_uses_reactzyme_wildcard_semantics():
    bag = 'C*.O'
    mapped_directed = '[CH3:1][CH3:2]>>[OH2:3]'
    # ReactZyme author preprocessing replaces wildcard * with carbon before molecular encoding.
    assert canonical_bag(bag) == sideblind_key_from_directed(mapped_directed)


def test_malformed_directed_reaction_is_not_recovered():
    assert sideblind_key_from_directed('CCO.O') is None
    assert canonical_bag('not-a-smiles') is None
