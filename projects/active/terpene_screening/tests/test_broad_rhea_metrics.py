import numpy as np
import pandas as pd

from projects.active.terpene_screening.broad_rhea_metrics import (
    candidate_ranking_context,
    evaluate_full_candidate_scores,
    summarize_query_metrics,
)
from projects.active.terpene_screening.fair_benchmark import evaluate_ranking_frame


def test_full_candidate_metrics_match_materialized_evaluator() -> None:
    candidate_ids = ["P5", "P1", "P4", "P2", "P3", "P6"]
    scores = np.asarray([0.7, 0.9, 0.2, 0.7, 0.4, 0.1], dtype=float)
    positives = {"P2", "P3"}
    index, lexical = candidate_ranking_context(candidate_ids)

    fast = evaluate_full_candidate_scores(
        scores,
        candidate_ids,
        positives,
        candidate_index=index,
        lexical_order=lexical,
    )
    materialized = pd.DataFrame(
        {
            "query_id": ["Q"] * len(candidate_ids),
            "candidate_id": candidate_ids,
            "score": scores,
            "label": [int(value in positives) for value in candidate_ids],
        }
    )
    per_query, summary = evaluate_ranking_frame(materialized)
    row = per_query.iloc[0]

    exact_keys = [
        "candidate_count",
        "positive_count",
        "best_positive_rank",
        "positive_hits_at_1",
        "positive_hits_at_2",
        "positive_hits_at_3",
        "positive_hits_at_4",
        "positive_hits_at_5",
        "positive_hits_at_10",
        "positive_hits_at_20",
        "positive_hits_at_50",
    ]
    for key in exact_keys:
        assert fast[key] == row[key]

    float_keys = [
        "best_positive_rank_fraction",
        "mean_positive_rank",
        "mean_positive_reciprocal_rank",
        "reciprocal_rank",
        "average_precision",
        "roc_auc",
        "dcg_at_10",
        "ef_at_0.01_fraction",
        "ef_at_0.02_fraction",
    ]
    for key in float_keys:
        assert np.isclose(float(fast[key]), float(row[key]))

    for k in (1, 2, 3, 4, 5, 10, 20, 50):
        for prefix in ("hit_at_", "precision_at_", "positive_recall_at_", "ndcg_at_"):
            key = f"{prefix}{k}"
            assert np.isclose(float(fast[key]), float(row[key]))
    for percent in (0.01, 0.02, 0.03, 0.05):
        key = f"success_at_{percent:g}_fraction"
        assert np.isclose(float(fast[key]), float(row[key]))

    compact_summary = summarize_query_metrics(pd.DataFrame([fast]))
    assert np.isclose(compact_summary["mrr"], summary["mrr"])
    assert np.isclose(compact_summary["map"], summary["map"])
    assert np.isclose(compact_summary["macro_roc_auc"], summary["macro_roc_auc"])
    assert np.isclose(compact_summary["mean_positive_rank"], summary["mean_positive_rank"])
    assert np.isclose(
        compact_summary["mean_positive_reciprocal_rank"],
        summary["mean_positive_reciprocal_rank"],
    )


def test_ties_follow_candidate_id_order() -> None:
    candidate_ids = ["B", "A", "C"]
    scores = np.asarray([1.0, 1.0, 0.5], dtype=float)
    metrics = evaluate_full_candidate_scores(scores, candidate_ids, {"B"})
    # A and B tie; lexical candidate-id ordering puts A first and B second.
    assert metrics["best_positive_rank"] == 2
    assert metrics["hit_at_1"] == 0
    assert metrics["hit_at_2"] == 1
