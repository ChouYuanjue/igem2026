from __future__ import annotations

import numpy as np

from projects.active.terpene_screening.bime_context_experts import (
    protein_seed_similarity_expert,
    reaction_seed_similarity_expert,
    wrap_context_scores,
)


def test_protein_seed_expert_matches_legacy_max_cosine_formula() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(12, 5)).astype(np.float32)
    ids = [f"P{i}" for i in range(len(features))]
    seeds = ["P2", "P8"]
    rows = np.asarray([2, 8], dtype=np.int64)
    legacy = (features @ features[rows].T).max(axis=1)
    expert = protein_seed_similarity_expert(features, ids, seeds)
    assert expert.available
    assert expert.name == "seed_sequence_similarity"
    np.testing.assert_array_equal(expert.scores, legacy)


def test_reaction_seed_expert_matches_legacy_member_average_formula() -> None:
    rng = np.random.default_rng(11)
    members = [rng.normal(size=(10, 4)).astype(np.float32) for _ in range(3)]
    ids = [f"R{i}" for i in range(10)]
    rows = np.asarray([1, 6], dtype=np.int64)
    legacy = np.zeros(10, dtype=np.float32)
    for emb in members:
        legacy += (emb @ emb[rows].T).max(axis=1)
    legacy /= len(members)
    expert = reaction_seed_similarity_expert(members, ids, ["R1", "R6"])
    assert expert.available
    assert expert.name == "seed_reaction_similarity"
    np.testing.assert_array_equal(expert.scores, legacy)


def test_missing_context_is_not_zero_imputed() -> None:
    ids = ["a", "b"]
    expert = wrap_context_scores("neighbor", ids, None, source="test")
    assert not expert.available
    assert expert.scores is None
    assert expert.audit_record()["available"] is False
