from pathlib import Path

import pandas as pd
import pytest

from projects.active.terpene_screening.fair_benchmark import audit_exact_overlap, evaluate_ranking_frame


def test_common_metrics_match_rank_semantics() -> None:
    frame = pd.DataFrame({
        "query_id": ["R1"] * 5 + ["R2"] * 5,
        "candidate_id": [f"A{i}" for i in range(5)] + [f"B{i}" for i in range(5)],
        "score": [5, 4, 3, 2, 1, 5, 4, 3, 2, 1],
        "label": [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    })
    per_query, summary = evaluate_ranking_frame(frame)
    r1 = per_query.set_index("query_id").loc["R1"]
    assert r1["best_positive_rank"] == 2
    assert r1["hit_at_1"] == 0
    assert r1["hit_at_3"] == 1
    assert r1["positive_recall_at_3"] == 0.5
    assert summary["hit_at_1"] == 0.5
    assert summary["hit_at_3"] == 1.0
    assert summary["query_positive_coverage"] == 1.0
    assert summary["precision_at_1"] == 0.5
    assert summary["precision_at_3"] == pytest.approx(1.0 / 3.0)
    assert summary["micro_positive_recall_at_3"] == pytest.approx(2.0 / 3.0)
    assert summary["median_best_positive_rank"] == 1.5
    assert summary["macro_roc_auc"] is not None
    assert 0 < summary["map"] <= 1
    assert 0 < summary["ndcg_at_10"] <= 1


def test_external_percent_success_uses_author_floor_rule() -> None:
    frame = pd.DataFrame({
        "query_id": ["R"] * 100,
        "candidate_id": [f"P{i:03d}" for i in range(100)],
        "score": list(range(100, 0, -1)),
        "label": [0, 0, 1] + [0] * 97,
    })
    _, summary = evaluate_ranking_frame(frame)
    assert summary["success_at_0.01_fraction"] == 0.0
    assert summary["success_at_0.03_fraction"] == 1.0
    assert summary["success_at_0.05_fraction"] == 1.0


def test_exact_pair_audit_distinguishes_entity_seen_from_pair_seen() -> None:
    train = pd.DataFrame({"query_id": ["R1", "R2"], "candidate_id": ["P1", "P2"]})
    test = pd.DataFrame({"query_id": ["R1", "R1", "R3"], "candidate_id": ["P1", "P2", "P3"]})
    rows, summary = audit_exact_overlap(train, test)
    assert rows["exact_pair_seen"].tolist() == [True, False, False]
    assert rows["query_seen"].tolist() == [True, True, False]
    assert rows["candidate_seen"].tolist() == [True, True, False]
    assert summary["generalization_claim_safe_exact_pair"] is False
