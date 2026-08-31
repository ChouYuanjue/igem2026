import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CONFIRM = ROOT / "projects/active/terpene_screening/CLEANROOM_R2E_TOP2000_DIFFICULTY_ROUTER_V1_CONFIRMATION.json"


def _record() -> dict[str, object]:
    return json.loads(CONFIRM.read_text(encoding="utf-8"))


def test_router_confirmation_passes_registered_multi_metric_gate() -> None:
    record = _record()
    coarse = record["coarse"]
    routed = record["routed"]
    for metric in ("mrr", "map", "ndcg_at_10", "hit_at_10"):
        assert routed[metric] > coarse[metric]
    for metric in ("hit_at_20", "hit_at_50"):
        assert routed[metric] >= coarse[metric]
    assert record["pass"] is True
    assert record["outer_labels_used"] is False
    assert record["target_benchmark_labels_used"] is False


def test_router_confirmation_does_not_authorize_revealed_outer_reuse() -> None:
    record = _record()
    policy = record["external_policy"]
    assert policy["fresh_external_evaluation_eligible"] is True
    assert policy["previously_revealed_outer_reuse_for_unbiased_claim"] is False
    assert "temporal_post2020_protein_cold" in policy["forbidden_revealed_outer_examples"]
    assert record["model_selection_allowed_after_confirmation"] is False


def test_low_similarity_region_is_explicit_coarse_fallback() -> None:
    record = _record()
    assert record["router"]["min_reaction_similarity"] == 0.9
    assert "otherwise exact coarse fallback" in record["router"]["decision"]
