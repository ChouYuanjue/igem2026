from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projects.active.terpene_screening.core.conformal import (
    CONFORMAL_RETRIEVAL_VERSION,
    apply_conformal_retrieval_set,
    conformal_set_size,
    finite_sample_quantile,
    normalized_rank_score,
)
from projects.active.terpene_screening.core.engine import payload_to_argv
from projects.active.terpene_screening.core.evidence import (
    apply_evidence_passport,
    compute_query_applicability,
    cycle_consistency_score,
)
from projects.active.terpene_screening.core.input_audit import audit_protein_sequence
from projects.active.terpene_screening.core.routing import resolve_route
from projects.active.terpene_screening import rank_open_world
from projects.active.terpene_screening.rank_open_world import build_parser, candidate_subset_indices


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


def test_candidate_subset_indices_are_exact_deduplicated_and_library_ordered():
    keep, audit = candidate_subset_indices(["A", "B", "C"], ["C", "A", "X", "C"])
    assert keep == [0, 2]
    assert audit == {
        "applied": True,
        "requested_count": 3,
        "effective_count": 2,
        "missing_count": 1,
    }


def test_candidate_subset_empty_request_is_unrestricted_and_all_unknown_is_rejected():
    keep, audit = candidate_subset_indices(["A", "B"], [])
    assert keep == [0, 1]
    assert audit["applied"] is False
    with pytest.raises(ValueError, match="None of the requested candidate IDs"):
        candidate_subset_indices(["A", "B"], ["X", "Y"])


def test_engine_payload_accepts_exact_candidate_subset_for_both_directions():
    e2r_argv = payload_to_argv(
        "rank-reactions",
        {"enzyme_id": "TEST", "candidate_ids": ["R1", "R2"], "top_k": 2},
    )
    e2r = build_parser().parse_args(e2r_argv)
    assert e2r.candidate_ids == ["R1", "R2"]

    r2e_argv = payload_to_argv(
        "rank-enzymes",
        {"reaction_id": "RHEA:1", "candidate_ids": ["P1", "P2"], "top_k": 2},
    )
    r2e = build_parser().parse_args(r2e_argv)
    assert r2e.candidate_ids == ["P1", "P2"]


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


def test_conformal_rank_quantile_is_finite_sample_conservative():
    scores = [normalized_rank_score(rank, 101) for rank in [1, 10, 20, 40, 80]]
    qhat = finite_sample_quantile(scores, 0.20)
    assert qhat == scores[-1]
    assert conformal_set_size(qhat, 201) == 159
    with pytest.raises(ValueError):
        finite_sample_quantile([], 0.10)


def test_conformal_annotation_preserves_ranking_and_uses_validated_group(tmp_path: Path):
    calibrators = {
        "manifest_version": 1,
        "conformal_retrieval_version": CONFORMAL_RETRIEVAL_VERSION,
        "calibrators": {
            "reaction_to_enzyme_top10": {
                "production_candidate_count": 5,
                "compatibility": {
                    "route_id": "r2e-external-top10-v1",
                    "candidate_universe_hash": "universe-hash",
                    "model_bundle_version": "bundle-v1",
                },
                "alphas": {
                    "0.10": {
                        "global": {
                            "qhat": 0.50,
                            "validation": {"empirical_coverage": 0.91, "n_test": 100},
                        },
                        "groups": {
                            "moderate": {
                                "enabled": True,
                                "qhat": 0.25,
                                "validation": {"empirical_coverage": 0.93, "n_test": 40},
                            }
                        },
                    }
                },
            }
        },
    }
    path = tmp_path / "calibrators.json"
    path.write_text(__import__("json").dumps(calibrators), encoding="utf-8")
    original = pd.DataFrame(
        [
            {
                "rank": 1,
                "candidate_id": "E1",
                "score": 0.9,
                "direction": "reaction_to_enzyme",
                "ranking_objective": "top10",
                "route_id": "r2e-external-top10-v1",
                "candidate_universe_hash": "universe-hash",
                "candidate_universe_size": 5,
                "model_bundle_version": "bundle-v1",
                "query_is_current_entity": False,
                "query_applicability_tier": "near_domain",
            },
            {
                "rank": 2,
                "candidate_id": "E2",
                "score": 0.8,
                "direction": "reaction_to_enzyme",
                "ranking_objective": "top10",
                "route_id": "r2e-external-top10-v1",
                "candidate_universe_hash": "universe-hash",
                "candidate_universe_size": 5,
                "model_bundle_version": "bundle-v1",
                "query_is_current_entity": False,
                "query_applicability_tier": "near_domain",
            },
        ]
    )
    annotated = apply_conformal_retrieval_set(
        original, calibrators_path=path, alpha=0.10, mode="annotate"
    )
    assert annotated["candidate_id"].tolist() == original["candidate_id"].tolist()
    assert annotated["rank"].tolist() == original["rank"].tolist()
    assert annotated["score"].tolist() == original["score"].tolist()
    assert annotated.iloc[0]["conformal_group_source"] == "mondrian:moderate"
    assert int(annotated.iloc[0]["conformal_set_size"]) == 2
    assert annotated["conformal_set_member"].tolist() == [True, True]
    assert not bool(annotated.iloc[0]["conformal_set_truncated"])


def test_engine_payload_accepts_conformal_controls():
    argv = payload_to_argv(
        "rank-enzymes",
        {
            "reaction_id": "R1",
            "conformal_mode": "expand",
            "conformal_alpha": 0.05,
        },
    )
    args = build_parser().parse_args(argv)
    assert args.conformal_mode == "expand"
    assert args.conformal_alpha == 0.05


def test_esmc_asset_resolution_is_strictly_local_first(monkeypatch, tmp_path: Path):
    weight = tmp_path / "data/weights/esmc_600m_2024_12_v0.pth"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"cached")
    calls = []

    import huggingface_hub

    def fake_snapshot_download(**kwargs):
        calls.append(dict(kwargs))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    resolved = rank_open_world._resolve_cached_esmc_assets("esmc_600m")
    assert resolved is not None
    assert resolved[0] == weight
    assert calls == [{
        "repo_id": "EvolutionaryScale/esmc-600m-2024-12",
        "local_files_only": True,
    }]


def test_esmc_loader_prefers_cached_model_before_upstream(monkeypatch):
    sentinel = object()
    rank_open_world._ESMC_MODEL_CACHE.clear()
    rank_open_world._ESMC_MODEL_SOURCE.clear()
    monkeypatch.setattr(rank_open_world, "_load_esmc_from_local_cache", lambda *_args: sentinel)

    from esm.models.esmc import ESMC
    monkeypatch.setattr(ESMC, "from_pretrained", classmethod(lambda cls, *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("upstream loader should not run"))))

    loaded = rank_open_world.load_esmc_model_cached("esmc_600m", "cpu")
    assert loaded is sentinel
    assert rank_open_world._ESMC_MODEL_SOURCE[("esmc_600m", "cpu")] == "local_huggingface_cache"
    rank_open_world._ESMC_MODEL_CACHE.clear()
    rank_open_world._ESMC_MODEL_SOURCE.clear()
