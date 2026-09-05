from __future__ import annotations

from pathlib import Path

import numpy as np
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


def test_engine_payload_preserves_mask_semantics_for_both_directions():
    e2r = build_parser().parse_args(payload_to_argv(
        "rank-reactions",
        {"enzyme_id": "TEST", "mask_reaction_ids": ["R1"], "mask_semantics": "output_separation"},
    ))
    assert e2r.mask_reaction_ids == ["R1"]
    assert e2r.mask_semantics == "output_separation"

    r2e = build_parser().parse_args(payload_to_argv(
        "rank-enzymes",
        {"reaction_id": "RHEA:1", "mask_enzyme_ids": ["P1"], "mask_semantics": "novelty_filter"},
    ))
    assert r2e.mask_enzyme_ids == ["P1"]
    assert r2e.mask_semantics == "novelty_filter"


def test_internal_expert_override_is_server_only_and_parseable_when_overrides_are_enabled():
    with pytest.raises(ValueError):
        payload_to_argv("rank-enzymes", {"reaction_id": "R1", "internal_expert_override": True})
    argv = payload_to_argv(
        "rank-enzymes",
        {"reaction_id": "R1", "model_dir": "/tmp/model", "internal_expert_override": True},
        allow_overrides=True,
    )
    args = build_parser().parse_args(argv)
    assert args.internal_expert_override is True


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
    assert route.settings is not None
    assert route.settings["few_shot"] == {"retrieval": "hybrid", "direct_weight": 0.99}


def test_current_r2e_uses_route_specific_bundle_without_invalidating_external_bundle():
    current = resolve_route(direction="reaction_to_enzyme", objective="top10", is_current=True)
    external = resolve_route(direction="reaction_to_enzyme", objective="top10", is_current=False)
    assert current.deployment.name == "marts_adapted_drfp_pu_e2r"
    assert current.model_bundle_version == "terpene-r2e-current-crossdirection-20260830-v1"
    assert external.deployment.name == "r2e_center_bounded_cap0p1"
    assert external.secondary_deployment is not None
    assert external.secondary_deployment.name == "r2e_enzgfm_center_router_v1"
    assert external.model_bundle_version == "catalyst-r2e-lambdarank-fusion-v1"
    assert external.settings["similarity_model_router"]["threshold"] == 0.9
    assert external.settings["lambdarank_fusion"]["config_id"] == "cfg_07_392fe119"
    assert current.route_version == "bime-rank-production-routes-v1"


def test_exact_binary_drfp_router_similarity_matches_tanimoto(monkeypatch, tmp_path: Path):
    schema={"drfp_dimension":4}
    features=np.asarray([[1,1,0,0],[1,0,1,0],[1,1,1,0]],dtype=np.float32)
    ids=["TRAIN_A","TRAIN_B","QUERY"]
    index={value:i for i,value in enumerate(ids)}
    train_ids=["TRAIN_A","TRAIN_B"]
    train_binary=features[:2].copy(); train_counts=train_binary.sum(axis=1,dtype=np.float32)
    monkeypatch.setattr(
        rank_open_world,
        "_r2e_binary_drfp_router_assets",
        lambda *_args: (schema,features,ids,index,train_binary,train_counts,train_ids),
    )
    nearest,similarity=rank_open_world.exact_max_train_binary_drfp_tanimoto(
        reaction_id="QUERY",reaction_smiles=None,feature_dir=tmp_path/"features",training_pairs=tmp_path/"pairs.csv"
    )
    assert nearest == "TRAIN_A"
    assert similarity == pytest.approx(2.0/3.0)


def test_confirmed_r2e_similarity_router_selects_secondary_only_below_threshold(monkeypatch):
    args=build_parser().parse_args(payload_to_argv(
        "rank-enzymes",{"reaction_smiles":"CC>>CO","candidate_universe":"general_merged","top_k":10}
    ))
    route=resolve_route(direction="reaction_to_enzyme",objective="top10",is_current=False)
    monkeypatch.setattr(rank_open_world,"exact_max_train_binary_drfp_tanimoto",lambda **_kwargs:("RLOW",0.42))
    selected,protein_dir,audit=rank_open_world._resolve_r2e_similarity_model_route(args,route)
    assert selected.deployment.name == "r2e_enzgfm_center_router_v1"
    assert protein_dir.name == "general_merged_650m_mean_v1"
    assert audit["selected"] == "secondary"
    assert audit["max_train_drfp_tanimoto"] == 0.42

    monkeypatch.setattr(rank_open_world,"exact_max_train_binary_drfp_tanimoto",lambda **_kwargs:("RHIGH",0.9))
    selected,protein_dir,audit=rank_open_world._resolve_r2e_similarity_model_route(args,route)
    assert selected.deployment.name == "r2e_center_bounded_cap0p1"
    assert protein_dir.name == "proteins"
    assert audit["selected"] == "primary"


def test_confirmed_r2e_similarity_router_is_disabled_for_candidate_subset(monkeypatch):
    args=build_parser().parse_args(payload_to_argv(
        "rank-enzymes",{"reaction_smiles":"CC>>CO","candidate_universe":"general_merged","candidate_ids":["P1"]}
    ))
    route=resolve_route(direction="reaction_to_enzyme",objective="top20",is_current=False)
    def should_not_run(**_kwargs):
        raise AssertionError("similarity must not be computed for an ineligible routed scope")
    monkeypatch.setattr(rank_open_world,"exact_max_train_binary_drfp_tanimoto",should_not_run)
    selected,protein_dir,audit=rank_open_world._resolve_r2e_similarity_model_route(args,route)
    assert selected.deployment.name == "marts_adapted_drfp_pu_r2e_exact_residual"
    assert audit["status"] == "ineligible"
    assert audit["selected"] == "legacy_scope_fallback"
    assert protein_dir.name == "proteins"






def test_confirmed_r2e_similarity_router_preserves_tps_specialist_external_route(monkeypatch):
    args=build_parser().parse_args(payload_to_argv(
        "rank-enzymes",{"reaction_smiles":"CC>>CO","top_k":10}
    ))
    route=resolve_route(direction="reaction_to_enzyme",objective="top10",is_current=False)
    def should_not_run(**_kwargs):
        raise AssertionError("general similarity router must not run for TPS-specialized protein assets")
    monkeypatch.setattr(rank_open_world,"exact_max_train_binary_drfp_tanimoto",should_not_run)
    selected,protein_dir,audit=rank_open_world._resolve_r2e_similarity_model_route(args,route)
    assert selected.deployment.name == "marts_adapted_drfp_pu_r2e_exact_residual"
    assert audit["selected"] == "legacy_scope_fallback"
    assert protein_dir.name == "esmc600m_mean"


def test_runtime_reaction_encoder_composes_rdkitplus_and_center_extensions(monkeypatch):
    monkeypatch.setattr(
        rank_open_world.DrfpEncoder,
        "encode",
        lambda *_args, **_kwargs: [np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)],
    )
    monkeypatch.setattr(
        rank_open_world,
        "_encode_runtime_rdkitplus_extension",
        lambda _smiles: np.ones(1024, dtype=np.float32),
    )
    monkeypatch.setattr(
        rank_open_world,
        "_encode_runtime_reaction_center_extension",
        lambda _smiles, _fp, _token, _radius: np.full(6, 2.0, dtype=np.float32),
    )
    schema = {
        "drfp_dimension": 4,
        "feature_mode": "drfp_categorical",
        "precursor_classes": ["unknown"],
        "product_skeleton_classes": ["unknown"],
        "reaction_feature_dimension": 4 + 2 + 1024 + 6,
        "reaction_feature_mode_extension": "append_atom_mapped_reaction_center_v1",
        "reaction_center_extension": {
            "dimension": 6,
            "center_fp_size_each_side": 2,
            "token_dim": 2,
            "radius": 1,
        },
    }
    values, audit = rank_open_world.encode_reaction_with_audit(
        "CC>>CO", schema, cache_dir=None, failure_policy="strict"
    )
    assert values.shape == (1036,)
    assert np.array_equal(values[6:1030], np.ones(1024, dtype=np.float32))
    assert np.array_equal(values[1030:], np.full(6, 2.0, dtype=np.float32))
    assert audit.fallback_used is False


def test_runtime_reaction_encoder_zero_fallback_is_explicit(monkeypatch):
    monkeypatch.setattr(
        rank_open_world.DrfpEncoder,
        "encode",
        lambda *_args, **_kwargs: [np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)],
    )
    def fail_rdkit(_smiles):
        raise RuntimeError("rdkitplus unavailable")
    monkeypatch.setattr(rank_open_world, "_encode_runtime_rdkitplus_extension", fail_rdkit)
    schema = {
        "drfp_dimension": 4,
        "feature_mode": "drfp_categorical",
        "precursor_classes": ["unknown"],
        "product_skeleton_classes": ["unknown"],
        "reaction_feature_dimension": 1030,
        "reaction_feature_mode_extension": "append_horizyn_rdkitplus_struct_morgan1024",
    }
    values, audit = rank_open_world.encode_reaction_with_audit(
        "CC>>CO", schema, cache_dir=None, failure_policy="warn"
    )
    assert values.shape == (1030,)
    assert np.count_nonzero(values[6:]) == 0
    assert audit.fallback_used is True
    assert "rdkitplus_zero_fallback" in audit.warning
    with pytest.raises(ValueError, match="Runtime RDKit\\+ feature encoding failed"):
        rank_open_world.encode_reaction_with_audit(
            "CC>>CO", schema, cache_dir=None, failure_policy="strict"
        )


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


def test_candidate_universe_versions_follow_actual_assets():
    from projects.active.terpene_screening.core.candidate_universes import resolve_candidate_universe

    repo_root = Path(__file__).resolve().parents[4]
    general = resolve_candidate_universe(repo_root, "general_merged")
    tps = resolve_candidate_universe(repo_root, "tps_specialized")
    assert general.version == "general-merged-v2"
    assert tps.version.startswith("tps-specialized-")
    assert "1391" not in tps.version
    assert general.version != tps.version


def test_engine_reports_selected_candidate_universe_version(monkeypatch):
    import pandas as pd
    from projects.active.terpene_screening.core.engine import RetrievalEngine

    engine = RetrievalEngine()
    frame = pd.DataFrame({
        "query_id": ["Q"], "direction": ["reaction_to_enzyme"],
        "candidate_id": ["P"], "rank": [1], "score": [0.5],
        "candidate_universe_version": ["stale-route-value"],
    })
    monkeypatch.setattr(engine, "rank_frame", lambda command, payload: frame)
    general = engine.rank("rank-enzymes", {"reaction_id": "R", "candidate_universe": "general_merged"})
    tps = engine.rank("rank-enzymes", {"reaction_id": "R", "candidate_universe": "tps_specialized"})
    assert general["query"]["candidate_universe_version"] == "general-merged-v2"
    assert tps["query"]["candidate_universe_version"].startswith("tps-specialized-")
