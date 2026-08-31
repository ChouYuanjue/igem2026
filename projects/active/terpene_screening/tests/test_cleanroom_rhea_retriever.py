from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch

from projects.active.terpene_screening.train_cleanroom_rhea_retriever import (
    build_candidate_batch,
    build_e2r_dev_reservoir,
    evaluate_e2r_dev,
    hard_proteins_for_reaction,
    multi_positive_topk_loss,
    split_double_cold,
    _reaction_replay_pool,
    _negative_curriculum_counts,
)


def test_double_cold_hash_split_has_zero_entity_overlap():
    pairs = pd.DataFrame(
        [
            {"protein_id": f"P{i}", "reaction_id": f"R{j}"}
            for i in range(30)
            for j in range(20)
            if (i + j) % 5 == 0
        ]
    )
    train, dev = split_double_cold(pairs, dev_fold=1, folds=5)
    assert len(train) > 0
    assert len(dev) > 0
    assert not (set(train.protein_id) & set(dev.protein_id))
    assert not (set(train.reaction_id) & set(dev.reaction_id))


def test_reaction_hard_negatives_never_include_known_positive():
    hard = hard_proteins_for_reaction(
        "R0",
        positives={"P_POS", "P_SHARED"},
        neighbors={"R0": [("R1", 0.9), ("R2", 0.8)]},
        proteins_by_reaction={
            "R1": {"P_POS", "P_HARD1", "P_SHARED"},
            "R2": {"P_HARD2", "P_SHARED"},
        },
        limit=10,
    )
    assert hard == ["P_HARD1", "P_HARD2"]


def test_candidate_union_marks_cross_query_known_positives_positive():
    positives = {"R1": {"P1", "P_SHARED"}, "R2": {"P2", "P_SHARED"}}
    candidates, mask = build_candidate_batch(
        ["R1", "R2"],
        direction="r2e",
        positives_by_query=positives,
        neighbors={"R1": [("RH1", 0.9)], "R2": [("RH2", 0.8)]},
        proteins_by_reaction={"RH1": {"P2", "N1"}, "RH2": {"P1", "N2"}},
        candidate_universe=["P1", "P2", "P_SHARED", "N1", "N2", "N3"],
        hard_negatives=2,
        random_negatives=1,
        rng=random.Random(7),
    )
    index = {value: i for i, value in enumerate(candidates)}
    assert mask[0, index["P1"]]
    assert mask[0, index["P_SHARED"]]
    assert mask[1, index["P2"]]
    assert mask[1, index["P_SHARED"]]
    # P2 may enter R1's candidate union as a hard negative, but is not a known
    # positive for R1; the full graph semantics are preserved query-by-query.
    assert not mask[0, index["P2"]]


def test_topk_loss_prefers_positive_above_hard_negatives():
    candidates = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]), dim=1
    )
    positive_mask = torch.tensor([[True, False, False]])
    good_query = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0]]), dim=1)
    bad_query = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0]]), dim=1)
    good, _ = multi_positive_topk_loss(
        good_query,
        candidates,
        positive_mask,
        temperature=0.1,
        topk=1,
        topk_weight=0.5,
        margin=0.05,
    )
    bad, _ = multi_positive_topk_loss(
        bad_query,
        candidates,
        positive_mask,
        temperature=0.1,
        topk=1,
        topk_weight=0.5,
        margin=0.05,
    )
    assert float(good) < float(bad)


def test_reaction_replay_pool_reweights_only_training_reactions():
    pool = _reaction_replay_pool(["R1", "R2", "R3"], ["R1", "R3"], repeat=2)
    assert pool.count("R1") == 3
    assert pool.count("R2") == 1
    assert pool.count("R3") == 3


def test_reaction_replay_pool_rejects_nontraining_reaction():
    import pytest
    with pytest.raises(ValueError, match="non-training"):
        _reaction_replay_pool(["R1", "R2"], ["R_TEST"], repeat=1)


def test_negative_curriculum_preserves_total_budget_and_reaches_target():
    assert _negative_curriculum_counts(epoch=1,target_hard=80,target_random=8,start_hard=16,ramp_epochs=4)==(16,72)
    assert _negative_curriculum_counts(epoch=2,target_hard=80,target_random=8,start_hard=16,ramp_epochs=4)==(37,51)
    assert _negative_curriculum_counts(epoch=4,target_hard=80,target_random=8,start_hard=16,ramp_epochs=4)==(80,8)
    assert _negative_curriculum_counts(epoch=8,target_hard=80,target_random=8,start_hard=16,ramp_epochs=4)==(80,8)


def test_negative_curriculum_disabled_preserves_old_behavior():
    assert _negative_curriculum_counts(epoch=1,target_hard=80,target_random=8,start_hard=0,ramp_epochs=0)==(80,8)


def test_e2r_dev_reservoir_uses_only_positive_reaction_neighbors():
    dev = pd.DataFrame(
        {
            "protein_id": ["P1", "P1", "P2"],
            "reaction_id": ["R_DEV1", "R_DEV2", "R_DEV3"],
        }
    )
    neighbors = {
        "R_DEV1": [("R_TRAIN_A", 0.9), ("R_TRAIN_B", 0.8)],
        "R_DEV2": [("R_TRAIN_B", 0.7), ("R_TRAIN_C", 0.6)],
        "R_DEV3": [("R_TRAIN_D", 0.95)],
    }
    reservoir = build_e2r_dev_reservoir(dev, neighbors=neighbors, neighbor_reactions=1)
    p1 = reservoir[reservoir.protein_id.eq("P1")].set_index("reaction_id")["label"].to_dict()
    assert p1 == {"R_DEV1": 1, "R_DEV2": 1, "R_TRAIN_A": 0, "R_TRAIN_B": 0}
    p2 = reservoir[reservoir.protein_id.eq("P2")].set_index("reaction_id")["label"].to_dict()
    assert p2 == {"R_DEV3": 1, "R_TRAIN_D": 0}


def test_e2r_dev_metrics_rank_positive_reaction_first():
    frame = pd.DataFrame(
        {
            "protein_id": ["P1", "P1", "P2", "P2"],
            "reaction_id": ["R1", "N1", "R2", "N2"],
            "label": [1, 0, 1, 0],
            "score": [0.9, 0.1, 0.8, 0.2],
        }
    )
    common, query = evaluate_e2r_dev(frame)
    assert common["mrr"] == 1.0
    assert common["map"] == 1.0
    assert common["hit_at_1"] == 1.0
    assert len(query) == 2
