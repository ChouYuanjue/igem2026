import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PATH = ROOT / "projects/active/terpene_screening/CLEANROOM_R2E_REACTION_CENTER_FEATURE_V1_RESULT.json"


def test_center_v1_is_rejected_by_the_frozen_hard_slice_gate() -> None:
    r = json.loads(PATH.read_text(encoding="utf-8"))
    assert r["status"] == "rejected_internal_screening"
    assert r["decision"] == "reject_center_v1_without_retuning"
    assert r["fresh_confirmation_authorized"] is False
    assert r["outer_labels_used"] is False
    assert r["target_benchmark_labels_used"] is False
    assert set(r["failed_predeclared_checks"]) == {
        "lt0p3_ndcg10_no_regress", "lt0p3_hit10_no_regress"
    }


def test_center_v1_records_useful_but_insufficient_hard_slice_signal() -> None:
    r = json.loads(PATH.read_text(encoding="utf-8"))
    hard = r["pooled_lt0p3"]
    assert hard["candidate"]["mrr"] > hard["baseline"]["mrr"]
    assert hard["candidate"]["map"] > hard["baseline"]["map"]
    assert hard["candidate"]["macro_roc_auc"] > hard["baseline"]["macro_roc_auc"]
    assert hard["candidate"]["hit_at_50"] > hard["baseline"]["hit_at_50"]
    assert hard["candidate"]["ndcg_at_10"] < hard["baseline"]["ndcg_at_10"]
    assert hard["candidate"]["hit_at_10"] < hard["baseline"]["hit_at_10"]
    assert "Do not tune center radius" in r["post_result_policy"]


def test_center_v1_has_strong_aggregate_clean_gain_but_is_not_rebranded() -> None:
    r = json.loads(PATH.read_text(encoding="utf-8"))
    overall = r["pooled_all_r2e"]
    assert overall["candidate"]["mrr"] > overall["baseline"]["mrr"]
    assert overall["candidate"]["hit_at_10"] > overall["baseline"]["hit_at_10"]
    assert overall["candidate"]["hit_at_50"] > overall["baseline"]["hit_at_50"]
    assert r["feature_or_radius_sweep_performed"] is False
