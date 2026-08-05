from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projects.active.terpene_screening.core.engine import payload_to_argv
from projects.active.terpene_screening.core.evidence import (
    apply_evidence_passport,
    compute_query_applicability,
    cycle_consistency_score,
)
from projects.active.terpene_screening.core.input_audit import audit_protein_sequence
from projects.active.terpene_screening.core.routing import resolve_route
from projects.active.terpene_screening.rank_open_world import build_parser


def test_engine_payload_uses_the_production_parser():
    argv = payload_to_argv(
        "rank-reactions",
        {
            "enzyme_id": "TEST",
            "top_k": 10,
            "known_reaction_ids": ["R1", "R2"],
        },
    )
    args = build_parser().parse_args(argv)
    assert args.command == "rank-reactions"
    assert args.enzyme_id == "TEST"
    assert args.top_k == 10
    assert args.known_reaction_ids == ["R1", "R2"]


def test_engine_rejects_file_and_model_overrides_by_default():
    with pytest.raises(ValueError):
        payload_to_argv("rank-enzymes", {"reaction_id": "R1", "model_dir": "/tmp/model"})


def test_route_manifest_resolves_locked_top20_auxiliary():
    route = resolve_route(
        direction="enzyme_to_reaction",
        objective="top20",
        is_current=False,
    )
    assert route.route_id == "e2r-external-top20-dual-kernel-rrf-v1"
    assert route.auxiliary_deployment is not None
    assert route.auxiliary_deployment.name == "marts_dual_kernel_e2r_top20"


def test_strict_protein_input_rejects_invalid_sequence():
    with pytest.raises(ValueError):
        audit_protein_sequence("MABC*?", policy="strict")


def test_applicability_proxy_is_bounded_and_transparent():
    evidence = compute_query_applicability(
        {
            "query_is_current_entity": False,
            "query_nearest_library_similarity": 0.75,
            "ensemble_top1_vote_fraction": 1.0,
            "ensemble_topk_jaccard": 0.8,
            "ensemble_topk_vote_mean": 0.9,
            "ensemble_top1_rank_std": 0.5,
            "ensemble_boundary_margin_z": 0.3,
        }
    )
    assert 0.0 <= evidence["score"] <= 1.0
    assert evidence["tier"] in {"in_domain", "near_domain"}
    assert evidence["interpretation"] == "diagnostic_applicability_not_activity_probability"
    assert set(evidence["components"]) == {
        "nearest_library_similarity",
        "ensemble_top1_consensus",
        "ensemble_topk_set_stability",
        "ensemble_topk_membership_support",
        "top1_rank_stability",
        "topk_boundary_separation",
    }


def test_evidence_passport_never_changes_score_or_rank():
    original = pd.DataFrame(
        [
            {
                "rank": 1,
                "candidate_id": "E1",
                "score": 0.9,
                "score_source": "direct",
                "query_is_current_entity": False,
                "query_nearest_library_similarity": 0.7,
                "ensemble_top1_vote_fraction": 1.0,
                "ensemble_top1_rank_std": 0.0,
                "ensemble_topk_jaccard": 0.8,
                "ensemble_topk_vote_mean": 1.0,
                "ensemble_boundary_margin_z": 0.2,
                "ensemble_topk_vote_fraction": 1.0,
                "ensemble_rank_std": 0.0,
                "empirical_reliability_score": 0.8,
                "empirical_reliability_status": "validated_external_double_cold",
                "empirical_reliability_tier": "higher_evidence",
                "is_external_candidate": False,
            },
            {
                "rank": 2,
                "candidate_id": "E2",
                "score": 0.8,
                "score_source": "direct",
                "query_is_current_entity": False,
                "query_nearest_library_similarity": 0.7,
                "ensemble_top1_vote_fraction": 1.0,
                "ensemble_top1_rank_std": 0.0,
                "ensemble_topk_jaccard": 0.8,
                "ensemble_topk_vote_mean": 1.0,
                "ensemble_boundary_margin_z": 0.2,
                "ensemble_topk_vote_fraction": 2 / 3,
                "ensemble_rank_std": 1.0,
                "empirical_reliability_score": 0.8,
                "empirical_reliability_status": "validated_external_double_cold",
                "empirical_reliability_tier": "higher_evidence",
                "is_external_candidate": True,
            },
        ]
    )
    annotated = apply_evidence_passport(original)
    assert annotated["candidate_id"].tolist() == original["candidate_id"].tolist()
    assert annotated["rank"].tolist() == original["rank"].tolist()
    assert annotated["score"].tolist() == original["score"].tolist()
    assert annotated.iloc[0]["candidate_evidence_score"] > annotated.iloc[1]["candidate_evidence_score"]
    assert annotated["evidence_passport_version"].nunique() == 1


def test_cycle_consistency_rewards_bidirectional_recovery():
    strong = cycle_consistency_score(1, 2)
    weak = cycle_consistency_score(1, 20)
    missing = cycle_consistency_score(1, None)
    assert 0.0 <= missing < weak < strong <= 1.0
    with pytest.raises(ValueError):
        cycle_consistency_score(0, 1)
