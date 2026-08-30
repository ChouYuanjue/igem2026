from __future__ import annotations

import numpy as np

from projects.active.terpene_screening.evaluate_general_known_recovery import _metrics_from_scores, parse_budgets


def test_known_recovery_metrics_use_best_positive_and_positive_recall():
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    ids = ["A", "B", "C", "D", "E"]
    metrics = _metrics_from_scores(scores, ids, {"B", "D"}, (1, 3, 5))
    assert metrics["best_positive_rank"] == 2
    assert metrics["reciprocal_rank"] == 0.5
    assert metrics["hit_at_1"] == 0
    assert metrics["hit_at_3"] == 1
    assert metrics["positive_recall_at_3"] == 0.5
    assert metrics["positive_recall_at_5"] == 1.0


def test_known_recovery_ties_are_broken_by_candidate_id():
    scores = np.asarray([0.8, 0.8, 0.7], dtype=np.float32)
    ids = ["B", "A", "C"]
    metrics = _metrics_from_scores(scores, ids, {"B"}, (1, 2))
    assert metrics["best_positive_rank"] == 2
    assert metrics["hit_at_1"] == 0
    assert metrics["hit_at_2"] == 1


def test_parse_budgets_deduplicates_and_sorts():
    assert parse_budgets("20,3,10,3") == (3, 10, 20)
