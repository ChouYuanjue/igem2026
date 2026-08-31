import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PATH = ROOT / "projects/active/terpene_screening/CLEANROOM_R2E_RDKITPLUS_NOVELTY_HARD_SLICE_V1_RESULT.json"


def test_rejected_novelty_result_preserves_frozen_decision() -> None:
    result = json.loads(PATH.read_text(encoding="utf-8"))
    assert result["status"] == "rejected_internal_screening"
    assert result["decision"] == "reject_v1_without_retuning"
    assert result["fresh_confirmation_authorized"] is False
    assert result["outer_labels_used"] is False
    assert result["target_benchmark_labels_used"] is False
    assert result["threshold_repeat_sweep_performed"] is False
    assert set(result["failed_predeclared_checks"]) == {
        "lt0p3_macro_auc_no_regress", "all_hit10_guard", "all_hit50_guard"
    }


def test_rejected_novelty_result_cannot_be_rebranded_as_hard_slice_router() -> None:
    result = json.loads(PATH.read_text(encoding="utf-8"))
    assert result["pooled_support"]["reaction_similarity_lt0p3_queries"] == 204
    assert result["pooled_lt0p3"]["candidate"]["mrr"] > result["pooled_lt0p3"]["baseline"]["mrr"]
    assert result["pooled_lt0p3"]["candidate"]["macro_roc_auc"] < result["pooled_lt0p3"]["baseline"]["macro_roc_auc"]
    assert "Do not route this rejected expert" in result["post_result_policy"]
