from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from projects.active.terpene_screening.train_general_evidence_retriever import (
    _directional_full_candidate_loss,
    _query_positive_rows,
)


def test_query_positive_rows_builds_directional_multi_positive_targets():
    associations = pd.DataFrame([
        {"protein_id": "P1", "reaction_id": "R1"},
        {"protein_id": "P2", "reaction_id": "R1"},
        {"protein_id": "P1", "reaction_id": "R2"},
    ])
    q, positives = _query_positive_rows(
        associations, direction="r2e",
        query_index={"R1": 0, "R2": 1}, candidate_index={"P1": 0, "P2": 1},
    )
    mapping = dict(zip(q, positives))
    assert mapping["R1"].tolist() == [0, 1]
    assert mapping["R2"].tolist() == [0]


def test_directional_loss_rewards_positive_above_kth_negative():
    candidates = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]), dim=1)
    good = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0]]), dim=1)
    bad = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0]]), dim=1)
    good_loss, _ = _directional_full_candidate_loss(
        good, candidates, [np.asarray([0])], temperature=0.1,
        topk_k=1, topk_weight=0.1, topk_margin=0.0, all_positive_weight=0.05,
    )
    bad_loss, _ = _directional_full_candidate_loss(
        bad, candidates, [np.asarray([0])], temperature=0.1,
        topk_k=1, topk_weight=0.1, topk_margin=0.0, all_positive_weight=0.05,
    )
    assert float(good_loss) < float(bad_loss)
