from pathlib import Path

import pandas as pd
import pytest

from projects.active.terpene_screening.build_enzgfm_protein_features import requested_proteins


def _seqs() -> pd.DataFrame:
    return pd.DataFrame({"protein_id": ["P3", "P1", "P2"], "sequence": ["CCC", "AAA", "BBB"]})


def test_requested_proteins_from_explicit_scope(tmp_path: Path) -> None:
    scope = tmp_path / "scope.csv"
    pd.DataFrame({"protein_id": ["P3", "P1", "P3"]}).to_csv(scope, index=False)
    out = requested_proteins(_seqs(), None, scope, 0)
    assert out["protein_id"].tolist() == ["P1", "P3"]
    assert out["sequence"].tolist() == ["AAA", "CCC"]
    assert out["row"].tolist() == [0, 1]


def test_requested_proteins_scope_accepts_entry(tmp_path: Path) -> None:
    scope = tmp_path / "scope.csv"
    pd.DataFrame({"Entry": ["P2"]}).to_csv(scope, index=False)
    out = requested_proteins(_seqs(), None, scope, 0)
    assert out["protein_id"].tolist() == ["P2"]


def test_requested_proteins_rejects_mixed_scopes(tmp_path: Path) -> None:
    scope = tmp_path / "scope.csv"
    assoc = tmp_path / "pairs.csv"
    pd.DataFrame({"protein_id": ["P2"]}).to_csv(scope, index=False)
    pd.DataFrame({"protein_id": ["P1"]}).to_csv(assoc, index=False)
    with pytest.raises(ValueError, match="mutually exclusive"):
        requested_proteins(_seqs(), assoc, scope, 0)
