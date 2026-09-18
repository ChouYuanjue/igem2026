from __future__ import annotations

import numpy as np
import pandas as pd

from projects.active.terpene_screening.build_terpene_pocket_ot_view import build_protein_ot_similarity


def test_protein_ot_similarity_identity_symmetry_and_missing() -> None:
    proteins = pd.DataFrame({"protein_id": ["a", "b", "missing"]})
    pockets = pd.DataFrame(
        {
            "instance_id": ["a1", "a2", "b1", "b2"],
            "protein_id": ["a", "a", "b", "b"],
            "pocket_rank": [1, 2, 1, 2],
            "p2rank_probability": [0.75, 0.25, 0.5, 0.5],
        }
    )
    pocket_similarity = np.array(
        [
            [1.0, 0.2, 0.8, 0.1],
            [0.2, 1.0, 0.3, 0.7],
            [0.8, 0.3, 1.0, 0.2],
            [0.1, 0.7, 0.2, 1.0],
        ],
        dtype=np.float32,
    )
    similarity, available, mass, stats = build_protein_ot_similarity(proteins, pockets, pocket_similarity)
    assert available.tolist() == [True, True, False]
    assert np.allclose(similarity, similarity.T)
    assert np.allclose(np.diag(similarity)[:2], 1.0)
    assert np.allclose(similarity[2], 0.0)
    assert np.allclose(similarity[:, 2], 0.0)
    assert np.all((similarity >= 0.0) & (similarity <= 1.0))
    assert np.allclose(mass[:2].sum(axis=1), 1.0)
    assert stats["available_count"] == 2


def test_identical_pocket_measures_have_unit_similarity() -> None:
    proteins = pd.DataFrame({"protein_id": ["a", "b"]})
    pockets = pd.DataFrame(
        {
            "instance_id": ["a1", "a2", "b1", "b2"],
            "protein_id": ["a", "a", "b", "b"],
            "pocket_rank": [1, 2, 1, 2],
            "p2rank_probability": [0.6, 0.4, 0.6, 0.4],
        }
    )
    pocket_similarity = np.array(
        [
            [1.0, 0.25, 1.0, 0.25],
            [0.25, 1.0, 0.25, 1.0],
            [1.0, 0.25, 1.0, 0.25],
            [0.25, 1.0, 0.25, 1.0],
        ],
        dtype=np.float32,
    )
    similarity, *_ = build_protein_ot_similarity(proteins, pockets, pocket_similarity)
    assert np.isclose(similarity[0, 1], 1.0, atol=1e-7)


def test_three_pocket_mass_is_normalized() -> None:
    proteins = pd.DataFrame({"protein_id": ["a"]})
    pockets = pd.DataFrame(
        {
            "instance_id": ["a1", "a2", "a3"],
            "protein_id": ["a", "a", "a"],
            "pocket_rank": [1, 2, 3],
            "p2rank_probability": [0.9, 0.6, 0.3],
        }
    )
    pocket_similarity = np.eye(3, dtype=np.float32)
    similarity, available, mass, _ = build_protein_ot_similarity(proteins, pockets, pocket_similarity)
    assert available.tolist() == [True]
    assert np.isclose(similarity[0, 0], 1.0)
    assert np.allclose(mass[0], np.array([0.5, 1 / 3, 1 / 6], dtype=np.float32), atol=1e-7)
