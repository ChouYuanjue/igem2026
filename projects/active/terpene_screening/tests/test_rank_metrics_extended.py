import pytest

from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics


def test_rank_metrics_reports_multi_positive_map_and_ndcg():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    import numpy as np
    m = rank_metrics(np.asarray(scores), ["A", "B", "C", "D", "E"], {"B", "D"}, set(), (1, 3, 5))
    assert m["best_positive_rank"] == 2
    assert m["mean_positive_rank"] == 3.0
    assert m["mean_positive_reciprocal_rank"] == pytest.approx((1/2 + 1/4) / 2)
    assert m["average_precision"] == pytest.approx(((1/2) + (2/4)) / 2)
    assert m["ndcg_at_1"] == 0.0
    assert 0 < m["ndcg_at_3"] < 1
    assert m["positive_recall_at_5"] == 1.0


def test_rank_metrics_masked_known_candidates_do_not_consume_rank():
    import numpy as np
    m = rank_metrics(np.asarray([0.9, 0.8, 0.7]), ["MASK", "POS", "N"], {"POS"}, {"MASK"}, (1,))
    assert m["candidate_count"] == 2
    assert m["best_positive_rank"] == 1
    assert m["hit_at_1"] == 1
