import pandas as pd

from projects.active.terpene_screening.evaluate_pure_cage_full_support_v1 import evaluate


def test_evaluate_uses_raw_logit_and_skips_queries_without_positive():
    frame = pd.DataFrame(
        [
            {"reaction_id": "r1", "uniprot_id": "p1", "label": 0, "pred_logit": 9.0},
            {"reaction_id": "r1", "uniprot_id": "p2", "label": 1, "pred_logit": 8.0},
            {"reaction_id": "r1", "uniprot_id": "p3", "label": 0, "pred_logit": 7.0},
            {"reaction_id": "r2", "uniprot_id": "p1", "label": 0, "pred_logit": 2.0},
            {"reaction_id": "r2", "uniprot_id": "p2", "label": 0, "pred_logit": 1.0},
        ]
    )
    result = evaluate(frame)
    assert result["summary"]["n_evaluable_reactions"] == 1
    assert result["summary"]["mrr"] == 0.5
    assert result["summary"]["hit1"] == 0.0
    assert result["summary"]["hit3"] == 1.0
    assert result["summary"]["macro_positive_recall_at_1"] == 0.0
    assert result["summary"]["macro_positive_recall_at_3"] == 1.0
    assert result["summary"]["candidate_count_per_reaction_min"] == 3


def test_budget_metrics_are_truncated_consistently():
    frame = pd.DataFrame(
        [
            {"reaction_id": "r1", "uniprot_id": f"p{i}", "label": int(i == 4), "pred_logit": 20-i}
            for i in range(1, 7)
        ]
    )
    summary = evaluate(frame)["summary"]
    assert summary["hit3"] == 0.0
    assert summary["hit5"] == 1.0
    assert summary["mrr_at_3"] == 0.0
    assert summary["mrr_at_5"] == 0.25
    assert summary["expected_positive_hits_at_5"] == 1.0
    assert summary["macro_positive_recall_at_3"] == 0.0
    assert summary["macro_positive_recall_at_5"] == 1.0
