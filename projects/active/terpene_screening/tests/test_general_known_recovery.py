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


def test_known_recovery_reports_precision_ndcg_map_and_rank_fraction():
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    ids = ["A", "B", "C", "D", "E"]
    metrics = _metrics_from_scores(scores, ids, {"B", "D"}, (3, 5))
    assert metrics["candidate_count"] == 5
    assert metrics["best_positive_rank_fraction"] == 0.4
    assert metrics["positive_hits_at_3"] == 1
    assert metrics["precision_at_3"] == 1 / 3
    assert 0 < metrics["ndcg_at_3"] < 1
    assert 0 < metrics["ap_at_3"] < 1
    assert metrics["positive_hits_at_5"] == 2
    assert metrics["positive_recall_at_5"] == 1.0


def test_known_recovery_query_exposure_buckets_are_explicit():
    from projects.active.terpene_screening.evaluate_general_known_recovery import (
        _degree_bucket,
        _positive_count_bucket,
    )
    assert _degree_bucket(0) == "query_unseen"
    assert _degree_bucket(1) == "query_seen_degree1"
    assert _degree_bucket(4) == "query_seen_degree2_5"
    assert _degree_bucket(9) == "query_seen_degree6plus"
    assert _positive_count_bucket(1) == "single_positive"
    assert _positive_count_bucket(3) == "positive_count2_3"
    assert _positive_count_bucket(7) == "positive_count4_10"
    assert _positive_count_bucket(11) == "positive_count11plus"


def test_stable_query_sampling_is_order_invariant_and_seeded():
    from projects.active.terpene_screening.evaluate_general_known_recovery import _stable_sample_query_ids
    values = [f"Q{i}" for i in range(40)]
    a = _stable_sample_query_ids(values, 8, 17)
    b = _stable_sample_query_ids(list(reversed(values)), 8, 17)
    c = _stable_sample_query_ids(values, 8, 18)
    assert a == b
    assert len(a) == 8 and len(set(a)) == 8
    assert a != c
    assert _stable_sample_query_ids(values, 0, 17) == sorted(values)
