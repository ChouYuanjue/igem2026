import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PATH = ROOT / "projects/active/terpene_screening/CLEANROOM_R2E_RDKITPLUS_NOVELTY_HARD_SLICE_V1.json"


def test_hard_slice_protocol_is_single_candidate_and_outer_free() -> None:
    p = json.loads(PATH.read_text(encoding="utf-8"))
    assert p["development_folds"] == [0, 1, 2]
    assert p["outer_labels_used"] is False
    assert p["target_benchmark_labels_used"] is False
    assert p["candidate_expert"]["reaction_novelty_threshold"] == 0.7
    assert p["candidate_expert"]["reaction_novelty_repeat"] == 1
    assert "no threshold/repeat sweep" in p["screening_rule"]["selection_scope"]


def test_hard_slice_protocol_requires_fresh_salted_confirmation() -> None:
    p = json.loads(PATH.read_text(encoding="utf-8"))
    assert p["evaluation"]["primary_slice"] == "reaction_similarity_lt0p3"
    assert p["evaluation"]["expected_primary_slice_queries_across_folds"] == 204
    assert "new deterministic salted strict double-cold" in p["screening_rule"]["if_pass"]
    assert "No already revealed outer benchmark" in p["outer_policy"]
