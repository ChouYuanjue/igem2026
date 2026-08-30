from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from projects.active.terpene_screening.rank_open_world import (
    ExactResidualReactionDualTower,
    IdentityHiddenResidualReactionDualTower,
    SelfContainedResidualReactionDualTower,
    apply_empirical_reliability,
    apply_automatic_few_shot_policy,
    choose_retrieval_scores,
    encode_reaction,
    enforce_reliability_policy,
    ensemble_query_diagnostics,
    ensemble_similarity_members,
    load_models,
    reciprocal_rank_fusion_members,
    reciprocal_rank_fusion_scores,
    resolve_ranking_objective,
    sort_scores_with_cage_rescue,
)
from projects.active.terpene_screening.evaluate_architecture_auxiliary_reranking_double_cold import (
    architecture_class,
    architecture_score_matrices,
)
from projects.active.terpene_screening.train_dual_tower_cold import (
    ModelConfig,
    TerpeneDualTower,
    build_reaction_features,
    load_aligned_feature_augmentation,
    multi_positive_contrastive_loss,
    rank_metrics,
    scheduled_hard_negative_k,
    topk_hit_surrogate_loss,
)


def test_architecture_class_mapping_is_exact():
    assert architecture_class("PF01397;PF03936") == "plant_full"
    assert architecture_class("PF19086") == "bacterial_classI"
    assert architecture_class("PF13243;PF13249") == "osc_full"
    assert architecture_class("PF01397") == "plant_single"
    assert architecture_class("PF13243") == "classII_single"
    assert architecture_class("PF01397;PF19086") == "classI_hybrid"
    assert architecture_class("") == ""
    assert architecture_class("PF00348") == ""


def test_unknown_architecture_is_neutral():
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.05, 0.02, 0.02, 0.01],
            [0.1, 0.2, 0.6, 0.04, 0.03, 0.03],
        ],
        dtype=np.float32,
    )
    r2e, e2r = architecture_score_matrices(
        probabilities,
        ["plant_full", "", "osc_full"],
    )
    np.testing.assert_allclose(r2e[:, 0], probabilities[:, 0])
    np.testing.assert_allclose(r2e[:, 2], probabilities[:, 2])
    np.testing.assert_allclose(r2e[:, 1], probabilities.mean(axis=1))
    np.testing.assert_allclose(e2r[0], probabilities[:, 0])
    np.testing.assert_allclose(e2r[2], probabilities[:, 2])
    np.testing.assert_allclose(e2r[1], np.full(len(probabilities), 0.5))


def test_residual_reaction_tower_zero_gate_matches_base():
    config = ModelConfig(
        protein_input_dim=5,
        reaction_input_dim=7,
        hidden_dim=6,
        embedding_dim=4,
        dropout=0.0,
    )
    distiller_config = {
        "input_dim": 7,
        "hidden_dim": 5,
        "output_dim": 3,
        "dropout": 0.0,
        "residual_blocks": 1,
    }
    model = SelfContainedResidualReactionDualTower(
        config,
        aux_input_dim=3,
        aux_hidden_dim=5,
        gate_init=-100.0,
        vector_gate=False,
        distiller_config=distiller_config,
    ).eval()
    values = torch.randn(4, 7)
    with torch.no_grad():
        expected = model.base_reaction_tower(values)
        actual = model.encode_reactions(values)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_load_models_supports_packaged_residual_checkpoint(tmp_path):
    config = ModelConfig(
        protein_input_dim=5,
        reaction_input_dim=7,
        hidden_dim=6,
        embedding_dim=4,
        dropout=0.0,
    )
    distiller_config = {
        "input_dim": 7,
        "hidden_dim": 5,
        "output_dim": 3,
        "dropout": 0.0,
        "residual_blocks": 1,
    }
    source = SelfContainedResidualReactionDualTower(
        config,
        aux_input_dim=3,
        aux_hidden_dim=5,
        gate_init=-4.0,
        vector_gate=False,
        distiller_config=distiller_config,
    ).eval()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    residual_state = {
        name: value
        for name, value in source.state_dict().items()
        if not name.startswith("reaction_distiller.")
    }
    torch.save(
        {
            "model_type": "horizyn_reaction_residual",
            "model_state_dict": residual_state,
            "base_model_config": config.__dict__,
            "aux_input_dim": 3,
            "aux_hidden_dim": 5,
            "gate_init": -4.0,
            "vector_gate": False,
        },
        model_dir / "production_seed1.pt",
    )
    torch.save(
        {
            "model_state_dict": source.reaction_distiller.state_dict(),
            "model_config": distiller_config,
        },
        tmp_path / "reaction_feature_distiller.pt",
    )
    loaded = load_models(model_dir, "production", torch.device("cpu"))[0]
    proteins = torch.randn(3, 5)
    reactions = torch.randn(4, 7)
    with torch.no_grad():
        assert torch.allclose(
            loaded.encode_proteins(proteins),
            source.encode_proteins(proteins),
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.allclose(
            loaded.encode_reactions(reactions),
            source.encode_reactions(reactions),
            atol=1e-6,
            rtol=1e-6,
        )


def test_load_models_supports_exact_residual_checkpoint(tmp_path):
    config = ModelConfig(
        protein_input_dim=5,
        reaction_input_dim=7,
        hidden_dim=6,
        embedding_dim=4,
        dropout=0.0,
    )
    source = ExactResidualReactionDualTower(
        config,
        aux_input_dim=3,
        aux_hidden_dim=5,
        gate_init=-4.0,
        vector_gate=False,
    ).eval()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    torch.save(
        {
            "model_type": "horizyn_reaction_residual_exact",
            "model_state_dict": source.state_dict(),
            "base_model_config": config.__dict__,
            "aux_input_dim": 3,
            "aux_hidden_dim": 5,
            "gate_init": -4.0,
            "vector_gate": False,
        },
        model_dir / "production_seed1.pt",
    )
    loaded = load_models(model_dir, "production", torch.device("cpu"))[0]
    proteins = np.random.default_rng(1).normal(size=(3, 5)).astype(np.float32)
    reactions = np.random.default_rng(2).normal(size=(4, 7)).astype(np.float32)
    auxiliary = np.random.default_rng(3).normal(size=(4, 3)).astype(np.float32)
    expected = ensemble_similarity_members(
        [source], proteins, reactions, torch.device("cpu"), auxiliary
    )
    actual = ensemble_similarity_members(
        [loaded], proteins, reactions, torch.device("cpu"), auxiliary
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="requires auxiliary reaction features"):
        ensemble_similarity_members(
            [loaded], proteins, reactions, torch.device("cpu")
        )


def test_hard_negative_curriculum_activation():
    assert scheduled_hard_negative_k(128, 1, 26, 0) == 0
    assert scheduled_hard_negative_k(128, 25, 26, 0) == 0
    assert scheduled_hard_negative_k(128, 26, 26, 0) == 128
    assert scheduled_hard_negative_k(128, 50, 26, 0) == 128
    assert scheduled_hard_negative_k(128, 1, 1, 25) == 128
    assert scheduled_hard_negative_k(128, 25, 1, 25) == 128
    assert scheduled_hard_negative_k(128, 26, 1, 25) == 0
    with pytest.raises(ValueError):
        scheduled_hard_negative_k(128, 0, 1, 0)


def test_feature_augmentation_alignment_is_identifier_based(tmp_path):
    np.save(
        tmp_path / "embeddings.npy",
        np.asarray([[30.0, 31.0], [10.0, 11.0], [20.0, 21.0]], dtype=np.float32),
    )
    pd.DataFrame(
        {
            "row": [0, 1, 2],
            "reaction_id": ["R3", "R1", "R2"],
        }
    ).to_csv(tmp_path / "entries.csv", index=False)
    aligned = load_aligned_feature_augmentation(tmp_path, ["R1", "R2", "R3"])
    np.testing.assert_allclose(
        aligned,
        np.asarray([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="misses 1 identifiers"):
        load_aligned_feature_augmentation(tmp_path, ["R1", "R4"])


def test_external_reaction_encoding_matches_training_multiview():
    reaction = "CCOP(=O)(O)O>>CC=C"
    positives = pd.DataFrame(
        {
            "rhea_id": ["RHEA:test"],
            "Entry": ["PTEST1"],
            "smiles_seq": [reaction],
        }
    )
    matrix, reaction_ids, _, schema = build_reaction_features(positives, "multiview")

    external = encode_reaction(reaction, schema)

    assert reaction_ids == ["RHEA:test"]
    assert external.shape == matrix[0].shape
    np.testing.assert_allclose(external, matrix[0], rtol=0, atol=0)


def test_ensemble_uncertainty_diagnostics_reflect_member_disagreement():
    ids = ["a", "b", "c", "d"]
    unanimous = np.asarray(
        [
            [0.9, 0.8, 0.2, 0.1],
            [0.95, 0.7, 0.3, 0.0],
            [0.85, 0.75, 0.25, 0.05],
        ],
        dtype=np.float32,
    )
    unanimous_metrics = ensemble_query_diagnostics(unanimous, ids, set(), top_k=2)
    assert unanimous_metrics["ensemble_top1_vote_fraction"] == 1.0
    assert unanimous_metrics["ensemble_top1_rank_std"] == 0.0
    assert unanimous_metrics["ensemble_topk_jaccard"] == 1.0

    conflicting = np.asarray(
        [
            [0.9, 0.8, 0.2, 0.1],
            [0.1, 0.95, 0.8, 0.0],
            [0.2, 0.1, 0.99, 0.7],
        ],
        dtype=np.float32,
    )
    conflicting_metrics = ensemble_query_diagnostics(conflicting, ids, set(), top_k=2)
    assert conflicting_metrics["ensemble_top1_vote_fraction"] < 1.0
    assert conflicting_metrics["ensemble_top1_rank_std"] > 0.0
    assert conflicting_metrics["ensemble_topk_jaccard"] < 1.0


def test_empirical_reliability_and_policy_enforcement(tmp_path):
    calibrator_path = tmp_path / "calibrators.json"
    calibrator_path.write_text(
        json.dumps(
            {
                "enzyme_to_reaction_top10": {
                    "deployable": True,
                    "feature_columns": ["query_nearest_train_similarity"],
                    "imputer_statistics": [0.0],
                    "scaler_mean": [0.0],
                    "scaler_scale": [1.0],
                    "coefficient": [1.0],
                    "intercept": 0.0,
                    "thresholds": {"low": 0.4, "high": 0.6},
                }
            }
        ),
        encoding="utf-8",
    )
    result = pd.DataFrame(
        {
            "candidate_id": ["R1"],
            "query_nearest_library_similarity": [1.0],
        }
    )
    scored = apply_empirical_reliability(
        result,
        "enzyme_to_reaction",
        "top10",
        calibrator_path,
        applicable=True,
        not_applicable_reason="not_applicable",
    )
    assert scored.iloc[0]["empirical_reliability_tier"] == "higher_evidence"
    assert scored.iloc[0]["reliability_recommendation"] == "use_ranked_shortlist"
    enforce_reliability_policy(scored, "require_higher")

    uncalibrated = scored.copy()
    uncalibrated["empirical_reliability_status"] = "failed_double_cold_validation"
    uncalibrated["empirical_reliability_tier"] = "uncalibrated"
    with pytest.raises(RuntimeError):
        enforce_reliability_policy(uncalibrated, "require_calibrated")


def test_rank_metrics_masks_known_associations_but_not_hidden_positive():
    scores = np.asarray([0.99, 0.8, 0.7], dtype=np.float32)
    metrics = rank_metrics(
        scores,
        ["known", "hidden_positive", "other"],
        positive_ids={"hidden_positive"},
        masked_ids={"known"},
        budgets=(1, 3),
    )

    assert metrics["best_positive_rank"] == 1
    assert metrics["hit_at_1"] == 1
    assert metrics["hit_at_3"] == 1


def test_auto_retrieval_uses_seed_when_available_and_direct_otherwise():
    direct = np.asarray([0.9, 0.1, 0.2], dtype=np.float32)
    seed = np.asarray([0.1, 0.8, 0.2], dtype=np.float32)
    ids = ["a", "b", "c"]

    score, source = choose_retrieval_scores(direct, seed, ids, "auto")
    np.testing.assert_array_equal(score, seed)
    assert source == "seed"

    score, source = choose_retrieval_scores(direct, None, ids, "auto")
    np.testing.assert_array_equal(score, direct)
    assert source == "direct"

    neighbor = np.asarray([0.2, 0.3, 0.9], dtype=np.float32)
    score, source = choose_retrieval_scores(
        direct,
        None,
        ids,
        "auto",
        neighbor_scores=neighbor,
        hybrid_direct_weight=0.75,
    )
    assert source == "neighbor_hybrid_direct_0.75"
    assert score.shape == direct.shape
    assert np.isfinite(score).all()

    score, source = choose_retrieval_scores(direct, seed, ids, "hybrid")
    assert source == "hybrid_direct_0.5"
    assert score.shape == direct.shape
    assert np.isfinite(score).all()


def test_production_auto_few_shot_policy_is_manifest_driven_and_preserves_explicit_modes():
    seed = np.asarray([0.2, 0.8], dtype=np.float32)
    settings = {"few_shot": {"retrieval": "hybrid", "direct_weight": 0.99}}
    mode, weight = apply_automatic_few_shot_policy("auto", seed, settings, 0.5)
    assert mode == "hybrid"
    assert weight == 0.99
    assert apply_automatic_few_shot_policy("seed", seed, settings, 0.5) == ("seed", 0.5)
    assert apply_automatic_few_shot_policy("direct", seed, settings, 0.5) == ("direct", 0.5)
    assert apply_automatic_few_shot_policy("auto", None, settings, 0.5) == ("auto", 0.5)


def test_ranking_objective_auto_follows_requested_cutoff():
    assert resolve_ranking_objective(3, "auto") == "top3"
    assert resolve_ranking_objective(10, "auto") == "top10"
    assert resolve_ranking_objective(20, "auto") == "top20"
    assert resolve_ranking_objective(20, "top10") == "top10"


def test_cage_rescue_preserves_primary_prefix_and_uses_base_tiebreak(tmp_path):
    ids = [f"p{index:02d}" for index in range(25)]
    scores = np.arange(25, 0, -1, dtype=np.float32)
    cage_path = tmp_path / "cage.csv"
    pd.DataFrame(
        {
            "reaction_id": ["RHEA:test"] * 5,
            "uniprot_id": ["p20", "p21", "p22", "p23", "p24"],
            "cage_score": [1.0] * 5,
        }
    ).to_csv(cage_path, index=False)

    result = sort_scores_with_cage_rescue(
        ids,
        scores,
        masked_ids=set(),
        top_k=20,
        reaction_id="RHEA:test",
        cage_scores_path=cage_path,
        rescue_slots=5,
    )

    assert result.iloc[:15]["candidate_id"].tolist() == ids[:15]
    assert result.iloc[15:]["candidate_id"].tolist() == ["p20", "p21", "p22", "p23", "p24"]
    assert result.iloc[:15]["selection_source"].eq("primary").all()
    assert result.iloc[15:]["selection_source"].eq("cage_rescue").all()


def test_multi_positive_contrastive_loss_is_finite():
    reactions = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    proteins = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    positive_mask = torch.tensor(
        [
            [True, True, False, False],
            [False, True, True, False],
            [False, False, False, True],
        ]
    )

    loss, reaction_loss, protein_loss = multi_positive_contrastive_loss(
        reactions,
        proteins,
        positive_mask,
        temperature=0.07,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(reaction_loss)
    assert torch.isfinite(protein_loss)


def test_global_mlnce_and_hard_negative_loss_are_finite():
    reactions = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1).requires_grad_()
    proteins = torch.nn.functional.normalize(torch.randn(6, 16), dim=-1).requires_grad_()
    positive_mask = torch.tensor(
        [
            [True, False, True, False, False, False],
            [False, True, False, False, False, False],
            [False, False, False, True, True, False],
            [False, False, False, False, False, True],
        ]
    )
    loss, reaction_loss, protein_loss = multi_positive_contrastive_loss(
        reactions,
        proteins,
        positive_mask,
        temperature=0.1,
        loss_mode="global_mlnce",
        hard_negative_k=2,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(reaction_loss)
    assert torch.isfinite(protein_loss)
    assert reactions.grad is not None and torch.isfinite(reactions.grad).all()
    assert proteins.grad is not None and torch.isfinite(proteins.grad).all()


def test_topk_hit_surrogate_prefers_positive_above_kth_negative():
    positive_mask = torch.tensor([[True, False, False, False]])
    denominator = torch.ones_like(positive_mask)
    proteins = torch.eye(4, dtype=torch.float32)
    good_reaction = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    bad_reaction = torch.tensor([[0.0, 1.0, 0.0, 0.0]], requires_grad=True)
    good, _, _ = topk_hit_surrogate_loss(
        good_reaction,
        proteins,
        positive_mask,
        denominator,
        denominator,
        temperature=1.0,
        target_k=1,
        margin=0.0,
        reaction_loss_weight=1.0,
    )
    bad, _, _ = topk_hit_surrogate_loss(
        bad_reaction,
        proteins,
        positive_mask,
        denominator,
        denominator,
        temperature=1.0,
        target_k=1,
        margin=0.0,
        reaction_loss_weight=1.0,
    )
    good.backward()
    assert torch.isfinite(good)
    assert torch.isfinite(bad)
    assert good < bad
    assert good_reaction.grad is not None


def test_hard_negative_k_rejects_invalid_values():
    reactions = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
    proteins = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
    positive_mask = torch.eye(2, dtype=torch.bool)
    with pytest.raises(ValueError, match="hard_negative_k"):
        multi_positive_contrastive_loss(
            reactions,
            proteins,
            positive_mask,
            temperature=0.1,
            hard_negative_k=-1,
        )


def test_pu_denominator_masks_keep_loss_finite_and_reduce_false_negative_penalty():
    reactions = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    proteins = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]]),
        dim=-1,
    )
    positive_mask = torch.tensor([[True, False, False], [False, False, True]])
    full_loss, _, _ = multi_positive_contrastive_loss(
        reactions,
        proteins,
        positive_mask,
        temperature=0.1,
    )
    reaction_denominator = torch.ones_like(positive_mask)
    reaction_denominator[0, 1] = False
    masked_loss, _, _ = multi_positive_contrastive_loss(
        reactions,
        proteins,
        positive_mask,
        temperature=0.1,
        reaction_denominator_mask=reaction_denominator,
    )

    assert torch.isfinite(masked_loss)
    assert masked_loss <= full_loss


def test_directional_loss_weight_selects_expected_component():
    reactions = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    proteins = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    positive_mask = torch.tensor(
        [
            [True, True, False, False],
            [False, True, True, False],
            [False, False, False, True],
        ]
    )
    reaction_only, reaction_loss, _ = multi_positive_contrastive_loss(
        reactions,
        proteins,
        positive_mask,
        temperature=0.07,
        reaction_loss_weight=1.0,
    )
    protein_only, _, protein_loss = multi_positive_contrastive_loss(
        reactions,
        proteins,
        positive_mask,
        temperature=0.07,
        reaction_loss_weight=0.0,
    )

    assert torch.allclose(reaction_only, reaction_loss)
    assert torch.allclose(protein_only, protein_loss)


def test_reciprocal_rank_fusion_matches_locked_formula():
    candidate_ids = ["A", "B", "C", "D"]
    primary = np.asarray([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    secondary = np.asarray([1.0, 4.0, 3.0, 2.0], dtype=np.float32)
    fused = reciprocal_rank_fusion_scores(
        primary, secondary, candidate_ids, primary_weight=0.35, constant=60.0
    )
    expected = np.asarray(
        [
            0.35 / 61.0 + 0.65 / 64.0,
            0.35 / 62.0 + 0.65 / 61.0,
            0.35 / 63.0 + 0.65 / 62.0,
            0.35 / 64.0 + 0.65 / 63.0,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(fused, expected, rtol=1e-7, atol=1e-8)
    assert candidate_ids[int(np.argmax(fused))] == "B"


def test_reciprocal_rank_fusion_members_pairs_corresponding_seeds():
    candidate_ids = ["A", "B", "C"]
    primary = np.asarray([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]], dtype=np.float32)
    secondary = np.asarray([[1.0, 3.0, 2.0], [3.0, 2.0, 1.0]], dtype=np.float32)
    fused = reciprocal_rank_fusion_members(primary, secondary, candidate_ids)
    assert fused.shape == primary.shape
    np.testing.assert_allclose(
        fused[0], reciprocal_rank_fusion_scores(primary[0], secondary[0], candidate_ids)
    )
    np.testing.assert_allclose(
        fused[1], reciprocal_rank_fusion_scores(primary[1], secondary[1], candidate_ids)
    )
    with pytest.raises(ValueError, match="matching shapes"):
        reciprocal_rank_fusion_members(primary, secondary[:1], candidate_ids)


def test_ensemble_diagnostics_can_follow_rrf_consensus():
    candidate_ids = ["A", "B", "C"]
    members = np.asarray(
        [[3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0]], dtype=np.float32
    )
    consensus = np.asarray([0.1, 0.3, 0.2], dtype=np.float32)
    diagnostics = ensemble_query_diagnostics(
        members, candidate_ids, set(), 2, consensus_scores=consensus
    )
    assert diagnostics["ensemble_top1_vote_fraction"] == pytest.approx(1.0 / 3.0)
    assert diagnostics["ensemble_top1_rank_std"] > 0


def test_identity_hidden_residual_is_exact_at_zero_init():
    config = ModelConfig(protein_input_dim=5, reaction_input_dim=7, hidden_dim=6, embedding_dim=4, dropout=0.0)
    base = TerpeneDualTower(config).eval()
    expanded = IdentityHiddenResidualReactionDualTower(config, aux_input_dim=3).eval()
    expanded.load_base_state(base.state_dict())
    base_values = torch.randn(8, 7)
    aux_values = torch.randn(8, 3)
    with torch.no_grad():
        expected = base.encode_reactions(base_values)
        actual = expanded.encode_reactions(torch.cat([base_values, aux_values], dim=1))
    assert torch.equal(expanded.aux_to_hidden.weight, torch.zeros_like(expanded.aux_to_hidden.weight))
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)


def test_load_models_supports_identity_hidden_residual_checkpoint(tmp_path):
    config = ModelConfig(protein_input_dim=5, reaction_input_dim=7, hidden_dim=6, embedding_dim=4, dropout=0.0)
    base = TerpeneDualTower(config).eval()
    source = IdentityHiddenResidualReactionDualTower(config, aux_input_dim=3).eval()
    source.load_base_state(base.state_dict())
    with torch.no_grad():
        source.aux_to_hidden.weight.normal_(mean=0.0, std=0.01)
    model_dir = tmp_path / "models"; model_dir.mkdir()
    torch.save({
        "model_type": "rdkitplus_identity_hidden_residual",
        "model_state_dict": source.state_dict(),
        "base_model_config": config.__dict__,
        "aux_input_dim": 3,
    }, model_dir / "production_seed1.pt")
    loaded = load_models(model_dir, "production", torch.device("cpu"))[0]
    values = torch.randn(5, 10)
    with torch.no_grad():
        assert torch.allclose(loaded.encode_reactions(values), source.encode_reactions(values), atol=1e-7, rtol=1e-7)
