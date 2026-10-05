from __future__ import annotations

from scripts.starase_navigator.retrieval.backend_router import (
    decide_expert,
    route_payload,
)


def _decide(command: str, payload: dict):
    return decide_expert(
        command,
        payload,
        adamerging_root=None,
        full_root=None,
    )


def test_tps_universe_remains_explicit_legacy_specialist_boundary():
    decision = _decide(
        "rank-enzymes",
        {"candidate_universe": "tps_specialized", "reaction_id": "RHEA:OTHER"},
    )
    assert decision.expert == "tps_legacy"
    assert decision.reason == "tps_specialized_candidate_universe"


def test_general_universe_delegates_to_current_production_manifest_for_all_budgets():
    for command, payload in [
        (
            "rank-enzymes",
            {"candidate_universe": "general_merged", "reaction_id": "RHEA:NEW", "top_k": 10},
        ),
        (
            "rank-enzymes",
            {"candidate_universe": "general_merged", "reaction_id": "RHEA:NEW", "top_k": 20},
        ),
        (
            "rank-reactions",
            {"candidate_universe": "general_merged", "enzyme_id": "P_NEW", "top_k": 20},
        ),
        (
            "rank-enzymes",
            {
                "candidate_universe": "general_merged",
                "reaction_smiles": "CCO>>CC=O",
                "top_k": 10,
            },
        ),
        (
            "rank-reactions",
            {
                "candidate_universe": "general_merged",
                "enzyme_sequence": "MKT",
                "retrieval_mode": "neighbor_hybrid",
            },
        ),
    ]:
        decision = _decide(command, payload)
        assert decision.expert == "production_manifest"
        assert decision.model_dir is None
        assert decision.force_direct_zero_shot is False
        assert decision.reason == "general_candidate_universe_uses_current_production_manifest"


def test_general_few_shot_no_longer_injects_historical_model_directory():
    decision = _decide(
        "rank-enzymes",
        {
            "candidate_universe": "general_merged",
            "reaction_id": "RHEA:NEW",
            "known_enzyme_ids": ["P1"],
            "top_k": 10,
        },
    )
    assert decision.expert == "production_manifest"
    assert decision.model_dir is None


def test_preexisting_server_override_is_preserved():
    decision = _decide(
        "rank-reactions",
        {
            "candidate_universe": "general_merged",
            "enzyme_id": "P_NEW",
            "model_dir": "/server/temporary",
        },
    )
    assert decision.expert == "internal_override"


def test_route_payload_never_injects_stale_general_model_roots():
    payload = {
        "candidate_universe": "general_merged",
        "reaction_id": "RHEA:NEW",
        "top_k": 10,
    }
    routed, decision = route_payload("rank-enzymes", payload)
    assert decision.expert == "production_manifest"
    assert routed == payload
    assert "model_dir" not in routed
    assert "internal_expert_override" not in routed
