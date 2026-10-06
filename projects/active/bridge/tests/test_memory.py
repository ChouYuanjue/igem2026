from __future__ import annotations

import numpy as np
import torch

from projects.active.bridge.runtime.memory import build_episodic_memory


def _fixture():
    candidates = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.2, 0.8],
            ],
            dtype=torch.float32,
        ),
        dim=1,
    )
    query = torch.nn.functional.normalize(
        torch.tensor([1.0, 0.1, 0.0], dtype=torch.float32),
        dim=0,
    )
    ids = ["P0", "P1", "P2", "P3"]
    index = {value: i for i, value in enumerate(ids)}
    return candidates, query, ids, index


def test_training_positive_is_not_reused_as_episodic_support():
    candidates, query, ids, index = _fixture()
    result = build_episodic_memory(
        query_embedding=query,
        candidate_embeddings=candidates,
        candidate_ids=ids,
        candidate_index=index,
        requested_support_ids=["P0"],
        training_positive_ids={"P0"},
        entity_seen_mask=np.asarray([True, True, False, False]),
        base_score=np.asarray([2.0, 1.0, 0.0, -1.0]),
        train_memory_weight=0.4,
    )
    assert result is None


def test_registered_runtime_positive_becomes_episodic_support():
    candidates, query, ids, index = _fixture()
    result = build_episodic_memory(
        query_embedding=query,
        candidate_embeddings=candidates,
        candidate_ids=ids,
        candidate_index=index,
        requested_support_ids=["P1"],
        training_positive_ids={"P0"},
        entity_seen_mask=np.asarray([True, True, False, False]),
        base_score=np.asarray([2.0, 1.0, 0.0, -1.0]),
        train_memory_weight=0.4,
    )
    assert result is not None
    assert result.effective_support_ids == ("P1",)
    assert result.ignored_training_ids == ()
    assert result.missing_support_ids == ()
    assert result.score.shape == (4,)
    assert np.isfinite(result.features).all()


def test_external_runtime_positive_can_condition_without_entering_candidate_universe():
    candidates, query, ids, index = _fixture()
    external = torch.nn.functional.normalize(
        torch.tensor([0.9, 0.1, 0.1], dtype=torch.float32),
        dim=0,
    )
    result = build_episodic_memory(
        query_embedding=query,
        candidate_embeddings=candidates,
        candidate_ids=ids,
        candidate_index=index,
        requested_support_ids=["USER:NEW"],
        training_positive_ids={"P0"},
        entity_seen_mask=np.asarray([True, True, False, False]),
        base_score=np.asarray([2.0, 1.0, 0.0, -1.0]),
        train_memory_weight=0.4,
        external_support_embeddings={"USER:NEW": external},
    )
    assert result is not None
    assert result.effective_support_ids == ("USER:NEW",)
    assert result.missing_support_ids == ()
    assert len(result.score) == len(ids)
    assert "USER:NEW" not in index
    # External entities were not present in the frozen relation graph.
    assert result.features[6] == 0.0


def test_missing_external_runtime_positive_falls_back_neutrally():
    candidates, query, ids, index = _fixture()
    result = build_episodic_memory(
        query_embedding=query,
        candidate_embeddings=candidates,
        candidate_ids=ids,
        candidate_index=index,
        requested_support_ids=["USER:MISSING"],
        training_positive_ids=set(),
        entity_seen_mask=np.asarray([True, True, False, False]),
        base_score=np.asarray([2.0, 1.0, 0.0, -1.0]),
        train_memory_weight=0.4,
    )
    assert result is None
