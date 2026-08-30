from __future__ import annotations

import numpy as np

from projects.active.terpene_screening.evaluate_cage_neural_common_reservoir import (
    rank_metrics,
    reciprocal_rank_fusion,
    stable_seed,
)


def test_rrf_rewards_agreement_and_is_tie_deterministic():
    ids = np.asarray(["A", "B", "C"])
    first = np.asarray([3.0, 2.0, 1.0])
    second = np.asarray([2.0, 3.0, 1.0])
    fused = reciprocal_rank_fusion([first, second], ids, k=0)
    assert fused[0] == fused[1]
    assert fused[0] > fused[2]


def test_rank_metrics_reports_multi_positive_hit_and_recall():
    ids = np.asarray(["A", "B", "C", "D"])
    scores = np.asarray([0.1, 0.9, 0.8, 0.2])
    result = rank_metrics(ids, scores, {"B", "D"}, (1, 3))
    assert result["reciprocal_rank"] == 1.0
    assert result["hit_at_1"] == 1.0
    assert result["expected_hits_at_3"] == 2.0
    assert result["positive_recall_at_3"] == 1.0


def test_stable_seed_changes_with_protocol_coordinates():
    a = stable_seed(1, "reaction_to_enzyme", "R1", 1, 0)
    b = stable_seed(1, "reaction_to_enzyme", "R1", 2, 0)
    assert a != b
    assert a == stable_seed(1, "reaction_to_enzyme", "R1", 1, 0)
