from __future__ import annotations

from pathlib import Path

from scripts.starase_navigator.retrieval.backend_router import decide_expert


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    ada = tmp_path / "ada"; full = tmp_path / "full"
    (ada / "models").mkdir(parents=True, exist_ok=True); (full / "models").mkdir(parents=True, exist_ok=True)
    return ada, full


def _decide(tmp_path: Path, command: str, payload: dict):
    ada, full = _roots(tmp_path)
    return decide_expert(
        command,
        payload,
        adamerging_root=ada,
        full_root=full,
    )


def test_tps_universe_is_the_only_automatic_specialist_boundary(tmp_path: Path):
    assert _decide(tmp_path, "rank-enzymes", {"candidate_universe": "tps_specialized", "reaction_id": "RHEA:OTHER"}).expert == "tps_legacy"
    known_reaction = _decide(tmp_path, "rank-enzymes", {"candidate_universe": "general_merged", "reaction_id": "RHEA:TPS"})
    assert known_reaction.expert == "general_adamerging"
    known_protein = _decide(tmp_path, "rank-reactions", {"candidate_universe": "general_merged", "enzyme_id": "P_TPS"})
    assert known_protein.expert == "general_adamerging"

def test_general_zero_shot_uses_adamerging_for_top10_and_full_for_r2e_top20(tmp_path: Path):
    top10 = _decide(tmp_path, "rank-enzymes", {"candidate_universe": "general_merged", "reaction_id": "RHEA:NEW", "top_k": 10, "retrieval_mode": "auto"})
    assert top10.expert == "general_adamerging"
    assert top10.force_direct_zero_shot is True
    top20 = _decide(tmp_path, "rank-enzymes", {"candidate_universe": "general_merged", "reaction_id": "RHEA:NEW", "top_k": 20, "retrieval_mode": "auto"})
    assert top20.expert == "general_full_directional"
    e2r = _decide(tmp_path, "rank-reactions", {"candidate_universe": "general_merged", "enzyme_id": "P_NEW", "top_k": 20, "retrieval_mode": "auto"})
    assert e2r.expert == "general_adamerging"


def test_raw_r2e_uses_schema_compatible_external_production_route_and_e2r_keeps_general_expert(tmp_path: Path):
    decision = _decide(tmp_path, "rank-enzymes", {"candidate_universe": "general_merged", "reaction_smiles": "CCO>>CC=O", "top_k": 10})
    assert decision.expert == "production_external_r2e"
    assert decision.reason == "external_reaction_schema_compatible_production_route"
    assert decision.model_dir is None
    assert decision.force_direct_zero_shot is False
    manual = _decide(tmp_path, "rank-reactions", {"candidate_universe": "general_merged", "enzyme_sequence": "MKT", "retrieval_mode": "neighbor_hybrid"})
    assert manual.expert == "general_adamerging"
    assert manual.force_direct_zero_shot is False


def test_general_few_shot_keeps_general_expert_and_does_not_force_zero_shot(tmp_path: Path):
    seeded = _decide(tmp_path, "rank-enzymes", {"candidate_universe": "general_merged", "reaction_id": "RHEA:NEW", "known_enzyme_ids": ["P1"], "top_k": 10})
    assert seeded.expert == "general_adamerging"
    assert seeded.force_direct_zero_shot is False
    assert seeded.reason == "general_seed_guided_budget_route"
    top20 = _decide(tmp_path, "rank-enzymes", {"candidate_universe": "general_merged", "reaction_id": "RHEA:NEW", "known_enzyme_ids": ["P1"], "top_k": 20})
    assert top20.expert == "general_full_directional"
    assert top20.force_direct_zero_shot is False
    override = _decide(tmp_path, "rank-reactions", {"candidate_universe": "general_merged", "enzyme_id": "P_NEW", "model_dir": "/server/temporary"})
    assert override.expert == "internal_override"


def test_route_payload_does_not_inject_incompatible_model_root_for_raw_r2e(monkeypatch, tmp_path: Path):
    import scripts.starase_navigator.retrieval.backend_router as router

    ada, full = _roots(tmp_path)
    monkeypatch.setattr(router, "configured_expert_roots", lambda: (ada, full))
    payload = {"candidate_universe": "general_merged", "reaction_smiles": "CCO>>CC=O", "top_k": 10}
    routed, decision = router.route_payload("rank-enzymes", payload)
    assert decision.expert == "production_external_r2e"
    assert "model_dir" not in routed
    assert "internal_expert_override" not in routed
    assert routed == payload


def test_route_payload_marks_server_selected_model_as_internal_override(monkeypatch, tmp_path: Path):
    import scripts.starase_navigator.retrieval.backend_router as router

    ada, full = _roots(tmp_path)
    monkeypatch.setattr(router, "configured_expert_roots", lambda: (ada, full))
    routed, decision = router.route_payload(
        "rank-enzymes",
        {"candidate_universe": "general_merged", "reaction_id": "RHEA:NEW", "top_k": 10},
    )
    assert decision.expert == "general_adamerging"
    assert routed["internal_expert_override"] is True
    assert routed["model_dir"] == ada / "models"
