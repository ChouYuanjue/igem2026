import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PATH = ROOT / "projects/active/terpene_screening/CLEANROOM_R2E_REACTION_CENTER_FEATURE_V1.json"


def test_center_protocol_is_frozen_single_candidate_clean_screen() -> None:
    p = json.loads(PATH.read_text(encoding="utf-8"))
    assert p["status"] == "frozen_before_feature_performance_execution"
    assert p["development_folds"] == [0, 1, 2]
    assert p["outer_labels_used"] is False
    assert p["target_benchmark_labels_used"] is False
    assert p["candidate"]["dimension"] == 4419
    assert p["screening_rule"]["candidate_count"] == 1
    assert p["screening_rule"]["no_feature_dimension_or_radius_sweep"] is True


def test_center_protocol_requires_hard_slice_and_broad_guards() -> None:
    p = json.loads(PATH.read_text(encoding="utf-8"))
    assert p["evaluation"]["expected_primary_slice_queries"] == 204
    assert "Hit@10 does not regress" in p["screening_rule"]["pooled_lt0p3_required"]
    assert "at least 2 of 3" in p["screening_rule"]["fold_stability"]
    assert "new deterministic salted strict double-cold" in p["screening_rule"]["if_pass"]
