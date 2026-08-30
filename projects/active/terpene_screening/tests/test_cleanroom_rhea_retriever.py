from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch

from projects.active.terpene_screening.train_cleanroom_rhea_retriever import (
    build_candidate_batch,
    hard_proteins_for_reaction,
    multi_positive_topk_loss,
    split_double_cold,
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
