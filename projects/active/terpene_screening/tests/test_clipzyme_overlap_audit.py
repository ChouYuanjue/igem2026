from __future__ import annotations

from projects.active.terpene_screening.audit_clipzyme_outer_overlap import (
    assign_rule_splits,
    reaction_keys,
    reaction_keys_from_smiles,
)


def test_reaction_key_ignores_atom_map_and_component_order():
    a,_=reaction_keys(["[CH3:1][OH:2]","O"],["[CH2:1]=O","O"])
    b,_=reaction_keys_from_smiles("O.CO>>O.C=O")
    assert a==b


def test_directionless_reaction_key_matches_reverse():
    f,fu=reaction_keys_from_smiles("CCO>>CC=O")
    r,ru=reaction_keys_from_smiles("CC=O>>CCO")
    assert f!=r
    assert fu==ru


def test_rule_split_is_grouped_and_deterministic():
    samples=[]
    for rule,count in [("A",20),("B",15),("C",10),("D",5),("E",4),("F",3),("G",2),("H",1)]:
        samples += [{"rule_id":rule}] * count
    first=assign_rule_splits(samples); second=assign_rule_splits(samples)
    assert first==second
    assert set(first)==set("ABCDEFGH")
    assert set(first.values()) <= {"train","dev","test"}
