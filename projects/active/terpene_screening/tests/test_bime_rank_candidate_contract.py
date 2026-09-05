from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
PROD = ROOT / "configs/production_routes/terpene_v1.yaml"
CAND = ROOT / "configs/production_routes/bime_rank_candidate_v1.yaml"
ADMISSION = ROOT / "projects/active/terpene_screening/BIME_RANK_EXPERT_ADMISSION_V1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _without(mapping: dict, key: str) -> dict:
    return {k: v for k, v in mapping.items() if k != key}


def test_candidate_and_production_manifests_match_promoted_bime_routes() -> None:
    prod = yaml.safe_load(PROD.read_text())
    cand = yaml.safe_load(CAND.read_text())
    assert cand["route_version"] == "bime-rank-production-candidate-v2"
    assert prod["route_version"] == "bime-rank-production-routes-v2"

    for objective in ("top3", "top10", "top20"):
        pr = prod["routes"]["reaction_to_enzyme"]["external"][objective]
        cr = cand["routes"]["reaction_to_enzyme"]["external"][objective]
        # Promotion copies the admitted route blocks exactly; only the top-level
        # route_version changes from candidate to production.
        assert cr == pr
        sx = cr["lambdarank_fusion"]["structure_expert"]
        assert sx["enabled"] is True
        assert sx["name"] == "CLIPZyme"
        assert "reciprocal" not in json.dumps(sx).lower()
        assert "cage" not in json.dumps(sx).lower()
        r2_seed = cr["lambdarank_fusion"]["seed_context"]
        assert r2_seed["enabled"] is True
        assert r2_seed["ranker_sha256"] == "b36c30e613b503974467979507d848e8a312fc16f45ff3c2d6b0adb982ec2907"
        assert _sha(ROOT / r2_seed["ranker_bundle"] / "ranker.json") == r2_seed["ranker_sha256"]

        pe = prod["routes"]["enzyme_to_reaction"]["external"][objective]
        ce = cand["routes"]["enzyme_to_reaction"]["external"][objective]
        assert ce == pe
        v4 = ce["anchored_lambdamart_v4"]
        assert v4["enabled"] is True
        assert v4["ranker_sha256"] == "95a8e0b3fdc444a14a9391b82d3be343c0ba2094d2c60063e00449f63f47139e"
        assert v4["protein_asset"] == "results/bime_rank_unified_v1/clipzyme_e2r_query_asset_v1"
        assert _sha(ROOT / v4["protein_asset"] / "manifest.json") == v4["protein_manifest_sha256"]
        assert _sha(ROOT / v4["reaction_asset"] / "manifest.json") == v4["reaction_manifest_sha256"]
        protein_manifest = json.loads((ROOT / v4["protein_asset"] / "manifest.json").read_text())
        assert protein_manifest["total_supported_count"] == 166212
        assert protein_manifest["selection_uses_labels"] is False
        assert protein_manifest["selection_uses_model_scores"] is False
        assert protein_manifest["extension_supported_count"] == 5
        e2_seed = v4["seed_context"]
        assert e2_seed["enabled"] is True
        assert e2_seed["ranker_sha256"] == "90a354702042eda0e5b5985f62fdb5620ac4f33edbd568870afc4666c35bdf7a"
        assert _sha(ROOT / e2_seed["ranker_bundle"] / "ranker.json") == e2_seed["ranker_sha256"]


def test_admission_manifest_matches_frozen_artifacts_and_rejections() -> None:
    data = json.loads(ADMISSION.read_text())
    experts = data["experts"]
    assert experts["r2e_clipzyme_structure"]["status"] == "promoted_production"
    assert experts["e2r_clipzyme_structure"]["status"] == "promoted_production"
    assert experts["seed_context"]["status"] == "promoted_production_conditional_context"
    assert experts["homology_context"]["status"] == "rejected_external_retention"
    assert experts["reciprocal_consistency"]["status"] == "rejected_external_retention"
    assert experts["enzymecage_top20_structure"]["status"] == "rejected_internal_oof"
    assert experts["seed_context"]["r2e"]["ranker_sha256"] == "b36c30e613b503974467979507d848e8a312fc16f45ff3c2d6b0adb982ec2907"
    assert experts["seed_context"]["e2r"]["ranker_sha256"] == "90a354702042eda0e5b5985f62fdb5620ac4f33edbd568870afc4666c35bdf7a"
    assert experts["homology_context"]["external_delta"]["mrr"] < 0

    for key in ("r2e_clipzyme_structure", "e2r_clipzyme_structure"):
        spec = experts[key]
        assert _sha(ROOT / spec["ranker"]) == spec["ranker_sha256"]

    assert data["staging"]["manifest_semantic_diff_status"] == "passed"
    assert data["staging"]["unexpected_manifest_diffs"] == 0
    assert data["staging"]["default_production_overwritten"] is True
    assert data["staging"]["post_promotion_smoke_status"] == "passed"
