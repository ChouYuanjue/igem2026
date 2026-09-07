from __future__ import annotations

import pickle

import pandas as pd

from projects.active.terpene_screening.audit_clipzyme_protein_support_v1 import (
    load_screening_uniprots,
    reactzyme_audit,
)


def test_screening_asset_uses_exact_uniprot_ids(tmp_path):
    path = tmp_path / "screen.p"
    pickle.dump({"uniprots": ["P1", "P2"], "hiddens": [[1.0], [2.0]]}, path.open("wb"))
    ids, meta = load_screening_uniprots(path)
    assert ids == {"P1", "P2"}
    assert meta["uniprot_count"] == 2
    assert meta["hidden_shape"] == [2, 1]


def test_identical_sequence_alias_is_context_not_executable_support(tmp_path):
    test = pd.DataFrame({
        "source_key": ["UNKNOWN", "P1"],
        "reaction_bag": ["CC.O", "CC.O"],
        "sequence": ["AAAA", "AAAA"],
    })
    summary, detail = reactzyme_audit(
        test,
        seq_to_clip_ids={"AAAA": ["P1"]},
        screening_uniprots={"P1"},
        eligible_reaction_path=tmp_path / "missing.csv",
    )
    assert detail["author_exact_sequence_match"].tolist() == [True, True]
    assert detail["executable_precomputed_protein_support"].tolist() == [False, True]
    assert summary["exact_sequence_alias_is_executable_support"] is False
    assert summary["common_executable_positive_rows"] == 1
