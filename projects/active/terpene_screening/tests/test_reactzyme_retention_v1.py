import json
from pathlib import Path

import pandas as pd

from projects.active.terpene_screening.build_reactzyme_safe_partitions_v1 import filter_training, norm_bag, norm_seq, split_rhea
from projects.active.terpene_screening.evaluate_reactzyme_retention_v1 import compare_frames, scope_pass, select_policy

ROOT = Path(__file__).resolve().parents[4]


def _frame(offset=0.0, rank=10.0):
    return pd.DataFrame({
        "query_id": ["q1", "q2"], "reciprocal_rank": [.1 + offset, .2 + offset],
        "average_precision": [.08 + offset, .18 + offset], "roc_auc": [.7 + offset, .8 + offset],
        "ndcg_at_10": [.1 + offset, .2 + offset], "hit_at_10": [.0 + offset, 1.0],
        "hit_at_20": [1.0, 1.0], "hit_at_50": [1.0, 1.0], "best_positive_rank": [rank, rank + 2],
    })


def test_protocol_frozen_before_safe_performance():
    p = json.loads((ROOT / "projects/active/terpene_screening/REACTZYME_NATIVE_SUPPORT_ADAPTATION_V1.json").read_text())
    assert p["status"] == "frozen_before_safe_model_training_or_retention_performance_materialization"
    assert p["data_isolation"]["external_test_labels_or_metrics_used_for_selection"] is False
    assert p["protection_semantics"]["selection_priority"] == ["union_safe_max", "enzyme_safe", "unchanged_current_model"]
    assert p["retention_evaluation"]["expected_pooled_support"] == {"r2e_all": 3226, "e2r_all": 23477, "r2e_lt0p3": 204, "e2r_no_hit": 5627}


def test_normalization_and_filter_semantics():
    assert norm_seq(" AC D* ") == "ACD"
    assert norm_bag("B.A.*") == "A.B.C"
    assert split_rhea("RHEA:1;RHEA:2;") == {"RHEA:1", "RHEA:2"}
    frame = pd.DataFrame({"protein_id": ["p1", "p2", "p3"], "reaction_id": ["r1", "r2", "r3"]})
    assert filter_training(frame, "enzyme_safe", {"p1"}, {"r2"}).protein_id.tolist() == ["p2", "p3"]
    assert filter_training(frame, "union_safe_max", {"p1"}, {"r2"}).protein_id.tolist() == ["p3"]


def test_query_support_must_match():
    base = _frame()
    cand = _frame()
    cand.loc[1, "query_id"] = "other"
    try:
        compare_frames(base, cand)
    except RuntimeError as exc:
        assert "Query support mismatch" in str(exc)
    else:
        raise AssertionError("support mismatch was accepted")


def test_scope_guard_and_priority_are_not_score_ranking():
    cmp = compare_frames(_frame(), _frame(offset=-0.001, rank=10.0))
    ok, why = scope_pass(cmp, {k: .002 for k in ["mrr", "map", "macro_roc_auc", "ndcg_at_10", "hit_at_10", "hit_at_20", "hit_at_50"]}, .05, 10.)
    assert ok and not why
    assert select_policy({"union_safe_max": {"pass": True}, "enzyme_safe": {"pass": True}}) == "union_safe_max"
    assert select_policy({"union_safe_max": {"pass": False}, "enzyme_safe": {"pass": True}}) == "enzyme_safe"
    assert select_policy({"union_safe_max": {"pass": False}, "enzyme_safe": {"pass": False}}) is None


def test_selector_uses_python_boolean_literal():
    source = (ROOT / "projects/active/terpene_screening/evaluate_reactzyme_retention_v1.py").read_text()
    assert '"external_metrics_used": false' not in source
    assert '"external_metrics_used": False' in source
    assert '"retuning_allowed": false' not in source
    assert '"retuning_allowed": False' in source
    assert ' false' not in source
