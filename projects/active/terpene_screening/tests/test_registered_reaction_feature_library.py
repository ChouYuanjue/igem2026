import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from projects.active.terpene_screening.rank_open_world import (
    _load_registered_reaction_feature_library_cached,
    load_registered_reaction_feature_library,
)


def _write_library(tmp_path: Path) -> Path:
    root = tmp_path / "reaction_features"
    root.mkdir()
    pd.DataFrame({"row": [0, 1], "reaction_id": ["R1", "R2"]}).to_csv(
        root / "entries.csv", index=False
    )
    np.save(root / "reaction_feature_matrix.npy", np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "contract": {
                    "feature_mode": "test_mode",
                    "drfp_dimension": 2,
                    "reaction_feature_dimension": 3,
                    "protein_feature_dimension": 1152,
                }
            }
        ),
        encoding="utf-8",
    )
    return root


def _schema(**updates: object) -> dict[str, object]:
    schema: dict[str, object] = {
        "feature_mode": "test_mode",
        "drfp_dimension": 2,
        "reaction_feature_dimension": 3,
        "protein_feature_dimension": 3200,
    }
    schema.update(updates)
    return schema


def test_registered_reaction_library_ignores_protein_dimension(tmp_path: Path) -> None:
    root = _write_library(tmp_path)
    _load_registered_reaction_feature_library_cached.cache_clear()
    matrix, ids = load_registered_reaction_feature_library(root, _schema())
    assert matrix.shape == (2, 3)
    assert ids == ["R1", "R2"]


def test_registered_reaction_library_rejects_reaction_contract_mismatch(tmp_path: Path) -> None:
    root = _write_library(tmp_path)
    _load_registered_reaction_feature_library_cached.cache_clear()
    with pytest.raises(ValueError, match="schema mismatch for feature_mode"):
        load_registered_reaction_feature_library(root, _schema(feature_mode="other"))


def test_registered_reaction_library_rejects_reaction_width_mismatch(tmp_path: Path) -> None:
    root = _write_library(tmp_path)
    _load_registered_reaction_feature_library_cached.cache_clear()
    with pytest.raises(ValueError, match="schema mismatch for reaction_feature_dimension"):
        load_registered_reaction_feature_library(root, _schema(reaction_feature_dimension=4))
