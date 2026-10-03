from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from reproducibility.bime_rank.support.evaluate_marts_multi_expert_rank_fusion import (
    FusionSpec,
    fuse_ranking,
)
from reproducibility.bime_rank.support.evaluate_multi_expert_protocol_comparison import (
    DirectionalMultiExpertDualTower,
    MultiExpertConfig,
    directional_multi_positive_loss,
    directional_topk_surrogate,
    gate_regularization,
)
from reproducibility.bime_rank.scripts.evaluate_fibre_atlas_tps_broad_universe_v1 import (
    canonical_alias_map,
    mapped_ids,
)
from projects.active.bridge.kernel.atlas import overlap_consistency_loss
from reproducibility.bime_rank.support.rank_current_library import (
    rank_current_library,
    resolve_budget,
)
from projects.active.terpene_screening.runtime.base_model import (
    paired_geometry_alignment_loss,
)


def test_paired_geometry_alignment_is_zero_for_matching_geometry() -> None:
    embeddings = torch.nn.functional.normalize(torch.randn(5, 7), dim=-1)
    positive_mask = torch.eye(5, dtype=torch.bool)
    loss = paired_geometry_alignment_loss(
        embeddings,
        embeddings.clone(),
        positive_mask,
        sample_size=5,
    )
    assert torch.isfinite(loss)
    assert float(loss) == pytest.approx(0.0, abs=1e-7)


def test_paired_geometry_alignment_detects_mismatched_geometry() -> None:
    reaction_embeddings = torch.nn.functional.normalize(torch.randn(5, 7), dim=-1)
    protein_embeddings = torch.nn.functional.normalize(torch.randn(5, 7), dim=-1)
    positive_mask = torch.eye(5, dtype=torch.bool)
    loss = paired_geometry_alignment_loss(
        reaction_embeddings,
        protein_embeddings,
        positive_mask,
        sample_size=5,
    )
    assert torch.isfinite(loss)
    assert float(loss) > 0


def test_multi_expert_gates_are_normalized_and_scores_are_directional() -> None:
    torch.manual_seed(7)
    config = MultiExpertConfig(
        protein_input_dim=5,
        reaction_input_dim=7,
        hidden_dim=11,
        global_dim=8,
        n_experts=4,
        expert_dim=3,
        dropout=0.0,
        gate_temperature=1.0,
        expert_mix_init=0.5,
    )
    model = DirectionalMultiExpertDualTower(config).eval()
    proteins = torch.randn(6, 5)
    reactions = torch.randn(4, 7)
    r2e, e2r, diagnostics = model.score_matrices(proteins, reactions)

    assert r2e.shape == (4, 6)
    assert e2r.shape == (4, 6)
    assert torch.isfinite(r2e).all()
    assert torch.isfinite(e2r).all()
    assert torch.allclose(
        diagnostics["protein_gates"].sum(dim=1), torch.ones(6), atol=1e-6
    )
    assert torch.allclose(
        diagnostics["reaction_gates"].sum(dim=1), torch.ones(4), atol=1e-6
    )
    assert torch.allclose(
        diagnostics["r2e_partition"].sum(dim=1), torch.ones(4), atol=1e-6
    )
    assert torch.allclose(
        diagnostics["e2r_partition"].sum(dim=1), torch.ones(6), atol=1e-6
    )
    assert diagnostics["chart_scores"].shape == (4, 6, 5)
    for gates, experts in (
        (diagnostics["protein_gates"], diagnostics["protein_experts"]),
        (diagnostics["reaction_gates"], diagnostics["reaction_experts"]),
    ):
        balance, entropy, diversity = gate_regularization(gates, experts)
        assert torch.isfinite(balance)
        assert torch.isfinite(entropy)
        assert torch.isfinite(diversity)


def test_multi_expert_scores_support_full_atlas_overlap_consistency() -> None:
    torch.manual_seed(11)
    config = MultiExpertConfig(
        protein_input_dim=5,
        reaction_input_dim=7,
        hidden_dim=11,
        global_dim=8,
        n_experts=4,
        expert_dim=3,
        dropout=0.0,
        gate_temperature=1.0,
        expert_mix_init=0.5,
    )
    model = DirectionalMultiExpertDualTower(config).eval()
    proteins = torch.randn(6, 5)
    reactions = torch.randn(4, 7)
    _, _, diagnostics = model.score_matrices(proteins, reactions)
    chart_scores = diagnostics["chart_scores"]
    n_reactions, n_proteins, n_charts = chart_scores.shape
    partition = (
        diagnostics["r2e_partition"][:, None, :]
        .expand(-1, n_proteins, -1)
        .reshape(-1, n_charts)
    )
    available = torch.ones_like(partition, dtype=torch.bool)
    loss = overlap_consistency_loss(
        chart_scores.reshape(-1, n_charts),
        partition,
        available,
    )
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_atlas_partition_is_exactly_the_historical_convex_mixture() -> None:
    torch.manual_seed(13)
    config = MultiExpertConfig(
        protein_input_dim=5,
        reaction_input_dim=7,
        hidden_dim=11,
        global_dim=8,
        n_experts=4,
        expert_dim=3,
        dropout=0.0,
        gate_temperature=1.0,
        expert_mix_init=0.35,
    )
    model = DirectionalMultiExpertDualTower(config).eval()
    proteins = torch.randn(6, 5)
    reactions = torch.randn(4, 7)
    r2e, e2r, diagnostics = model.score_matrices(proteins, reactions)

    global_scores = (
        diagnostics["reaction_global"] @ diagnostics["protein_global"].T
    )
    expert_scores = diagnostics["expert_scores"]
    r2e_expert = (
        expert_scores * diagnostics["reaction_gates"][:, None, :]
    ).sum(dim=-1)
    e2r_expert = (
        expert_scores * diagnostics["protein_gates"][None, :, :]
    ).sum(dim=-1)
    historical_r2e = (
        (1 - diagnostics["r2e_mix"]) * global_scores
        + diagnostics["r2e_mix"] * r2e_expert
    )
    historical_e2r = (
        (1 - diagnostics["e2r_mix"]) * global_scores
        + diagnostics["e2r_mix"] * e2r_expert
    )
    assert torch.allclose(r2e, historical_r2e, atol=1e-7, rtol=1e-7)
    assert torch.allclose(e2r, historical_e2r, atol=1e-7, rtol=1e-7)


def test_directional_ranking_losses_have_per_query_additive_gauge_freedom() -> None:
    logits = torch.tensor(
        [
            [1.2, 0.8, -0.1, -0.5],
            [0.2, 1.1, 0.7, -0.4],
        ],
        dtype=torch.float64,
    )
    positives = torch.tensor(
        [
            [True, False, False, False],
            [False, True, True, False],
        ]
    )
    denominator = torch.ones_like(positives)
    shifts = torch.tensor([[7.5], [-3.25]], dtype=logits.dtype)
    shifted = logits + shifts

    base_contrastive = directional_multi_positive_loss(
        logits,
        positives,
        denominator,
        hard_negative_k=0,
    )
    shifted_contrastive = directional_multi_positive_loss(
        shifted,
        positives,
        denominator,
        hard_negative_k=0,
    )
    torch.testing.assert_close(
        base_contrastive,
        shifted_contrastive,
        atol=1e-12,
        rtol=1e-12,
    )

    terms = ((3, 0.10), (10, 0.05), (20, 0.025))
    base_topk = directional_topk_surrogate(
        logits,
        positives,
        denominator,
        terms,
        margin=0.0,
    )
    shifted_topk = directional_topk_surrogate(
        shifted,
        positives,
        denominator,
        terms,
        margin=0.0,
    )
    torch.testing.assert_close(
        base_topk,
        shifted_topk,
        atol=1e-12,
        rtol=1e-12,
    )


def test_directional_ranking_losses_do_not_have_arbitrary_scale_gauge() -> None:
    logits = torch.tensor([[1.2, 0.8, -0.1, -0.5]], dtype=torch.float64)
    positives = torch.tensor([[True, False, False, False]])
    denominator = torch.ones_like(positives)
    base_loss = directional_multi_positive_loss(
        logits,
        positives,
        denominator,
        hard_negative_k=0,
    )
    scaled_loss = directional_multi_positive_loss(
        2.0 * logits,
        positives,
        denominator,
        hard_negative_k=0,
    )
    assert not torch.isclose(base_loss, scaled_loss)


def test_sparse_pair_scoring_matches_matrix_diagonal() -> None:
    torch.manual_seed(7)
    config = MultiExpertConfig(
        protein_input_dim=5,
        reaction_input_dim=7,
        hidden_dim=9,
        global_dim=4,
        n_experts=4,
        expert_dim=3,
        dropout=0.0,
        gate_temperature=1.0,
        expert_mix_init=0.5,
    )
    model = DirectionalMultiExpertDualTower(config).eval()
    proteins = torch.randn(6, 5)
    reactions = torch.randn(6, 7)
    matrix_r2e, matrix_e2r, _ = model.score_matrices(proteins, reactions)
    pair_r2e, pair_e2r, _ = model.score_pairs(proteins, reactions)
    torch.testing.assert_close(pair_r2e, matrix_r2e.diag())
    torch.testing.assert_close(pair_e2r, matrix_e2r.diag())


def test_query_adaptive_mix_initializes_to_historical_mix() -> None:
    torch.manual_seed(17)
    config = MultiExpertConfig(
        protein_input_dim=5,
        reaction_input_dim=7,
        hidden_dim=9,
        global_dim=4,
        n_experts=4,
        expert_dim=3,
        dropout=0.0,
        gate_temperature=1.0,
        expert_mix_init=0.35,
        query_adaptive_r2e_mix=True,
        query_adaptive_e2r_mix=True,
    )
    model = DirectionalMultiExpertDualTower(config).eval()
    proteins = torch.randn(6, 5)
    reactions = torch.randn(4, 7)
    _, _, diagnostics = model.score_matrices(proteins, reactions)
    assert diagnostics["r2e_mix"].shape == (4,)
    assert diagnostics["e2r_mix"].shape == (6,)
    torch.testing.assert_close(
        diagnostics["r2e_mix"],
        torch.full((4,), 0.35),
        atol=1e-7,
        rtol=1e-7,
    )
    torch.testing.assert_close(
        diagnostics["e2r_mix"],
        torch.full((6,), 0.35),
        atol=1e-7,
        rtol=1e-7,
    )
    torch.testing.assert_close(
        diagnostics["r2e_partition"].sum(dim=1),
        torch.ones(4),
        atol=1e-7,
        rtol=1e-7,
    )
    torch.testing.assert_close(
        diagnostics["e2r_partition"].sum(dim=1),
        torch.ones(6),
        atol=1e-7,
        rtol=1e-7,
    )


def test_query_adaptive_mix_can_vary_by_query_and_preserve_pair_matrix_agreement() -> None:
    torch.manual_seed(19)
    config = MultiExpertConfig(
        protein_input_dim=5,
        reaction_input_dim=7,
        hidden_dim=9,
        global_dim=4,
        n_experts=4,
        expert_dim=3,
        dropout=0.0,
        gate_temperature=1.0,
        expert_mix_init=0.5,
        query_adaptive_r2e_mix=True,
        query_adaptive_e2r_mix=True,
    )
    model = DirectionalMultiExpertDualTower(config).eval()
    assert model.reaction_tower.mix_gate is not None
    assert model.protein_tower.mix_gate is not None
    with torch.no_grad():
        model.reaction_tower.mix_gate.weight.fill_(0.25)
        model.protein_tower.mix_gate.weight.fill_(-0.20)
    proteins = torch.randn(6, 5)
    reactions = torch.randn(6, 7)
    matrix_r2e, matrix_e2r, diagnostics = model.score_matrices(
        proteins,
        reactions,
    )
    assert float(diagnostics["r2e_mix"].std()) > 0.0
    assert float(diagnostics["e2r_mix"].std()) > 0.0
    pair_r2e, pair_e2r, _ = model.score_pairs(proteins, reactions)
    torch.testing.assert_close(pair_r2e, matrix_r2e.diag())
    torch.testing.assert_close(pair_e2r, matrix_e2r.diag())


def test_e2r_only_mix_gate_cannot_change_r2e_scores() -> None:
    torch.manual_seed(23)
    base_config = MultiExpertConfig(
        protein_input_dim=5,
        reaction_input_dim=7,
        hidden_dim=9,
        global_dim=4,
        n_experts=4,
        expert_dim=3,
        dropout=0.0,
        gate_temperature=1.0,
        expert_mix_init=0.5,
    )
    base = DirectionalMultiExpertDualTower(base_config).eval()
    adaptive_config = MultiExpertConfig(
        **{
            **base_config.__dict__,
            "query_adaptive_e2r_mix": True,
        }
    )
    adaptive = DirectionalMultiExpertDualTower(adaptive_config).eval()
    incompatible = adaptive.load_state_dict(base.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {
        "protein_tower.mix_gate.weight",
        "protein_tower.mix_gate.bias",
    }
    assert not incompatible.unexpected_keys
    assert adaptive.protein_tower.mix_gate is not None
    with torch.no_grad():
        adaptive.protein_tower.mix_gate.weight.normal_()
        adaptive.protein_tower.mix_gate.bias.fill_(-0.7)
    proteins = torch.randn(6, 5)
    reactions = torch.randn(4, 7)
    base_r2e, base_e2r, _ = base.score_matrices(proteins, reactions)
    adaptive_r2e, adaptive_e2r, _ = adaptive.score_matrices(
        proteins,
        reactions,
    )
    torch.testing.assert_close(adaptive_r2e, base_r2e, atol=0.0, rtol=0.0)
    assert not torch.allclose(adaptive_e2r, base_e2r)


def test_broad_universe_alias_mapping_rejects_only_required_ambiguity(
    tmp_path: Path,
) -> None:
    metadata = pd.DataFrame(
        [
            {"protein_id": "A", "aliases": "A;OLD_A"},
            {"protein_id": "B", "aliases": "B;AMB"},
            {"protein_id": "C", "aliases": "C;AMB"},
        ]
    )
    path = tmp_path / "protein_metadata.csv"
    metadata.to_csv(path, index=False)
    alias_map, ambiguous = canonical_alias_map(path)

    assert ambiguous == {"AMB"}
    assert mapped_ids({"A", "OLD_A"}, alias_map) == {"A"}
    with pytest.raises(ValueError):
        mapped_ids({"AMB"}, alias_map)


def test_fusion_rescue_preserves_prefix_and_adds_novel_candidates() -> None:
    rankings = {
        "primary": ["A", "B", "C", "D", "E"],
        "rescue": ["C", "X", "A", "Y", "Z"],
    }
    spec = FusionSpec(
        name="rescue",
        kind="rescue",
        sources=("primary", "rescue"),
        rescue_slots=2,
    )
    result = fuse_ranking(rankings, spec, budget=5)
    assert result[:3] == ["A", "B", "C"]
    assert result[3:] == ["X", "Y"]
    assert len(result) == len(set(result)) == 5


def test_rrf_is_deterministic_and_unique() -> None:
    rankings = {
        "left": ["A", "B", "C", "D"],
        "right": ["B", "A", "D", "C"],
    }
    spec = FusionSpec(
        name="rrf",
        kind="rrf",
        sources=("left", "right"),
        weights=(0.5, 0.5),
        constant=10.0,
        power=1.0,
    )
    first = fuse_ranking(rankings, spec, budget=4)
    second = fuse_ranking(rankings, spec, budget=4)
    assert first == second
    assert len(first) == len(set(first)) == 4
    assert set(first) == {"A", "B", "C", "D"}


@pytest.mark.parametrize(
    ("requested", "resolved"),
    [(1, 3), (3, 3), (4, 5), (5, 5), (6, 10), (10, 10), (11, 20), (20, 20)],
)
def test_current_library_budget_resolution(requested: int, resolved: int) -> None:
    assert resolve_budget(requested) == resolved


def test_current_library_budget_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        resolve_budget(0)
    with pytest.raises(ValueError):
        resolve_budget(21)


def test_current_library_route_uses_selected_panel(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "scope": "all513",
                "B": 5,
                "method": "base_only",
                "hit_probability": 0.4,
                "expected_hits": 0.5,
                "delta_hit_vs_base": 0.0,
                "delta_expected_vs_base": 0.0,
            }
        ]
    ).to_csv(tmp_path / "best_methods.csv", index=False)
    rows = []
    for rank, candidate in enumerate(["P1", "P2", "P3", "P4", "P5"], start=1):
        rows.append(
            {
                "reaction_id": "RHEA:TEST",
                "method": "base_only",
                "B": 5,
                "rank": rank,
                "uniprot_id": candidate,
                "label": int(candidate == "P2"),
                "base_score": 1.0 - rank / 10,
                "base_rank": 1.0 - rank / 10,
                "cage_available": False,
                "cage_rank": 0.0,
                "calibrated_score": 0.1,
                "residual_gain": 0.0,
            }
        )
    pd.DataFrame(rows).to_csv(tmp_path / "panels.csv", index=False)

    result = rank_current_library("RHEA:TEST", 4, tmp_path)
    assert result["candidate_id"].tolist() == ["P1", "P2", "P3", "P4"]
    assert result["rank"].tolist() == [1, 2, 3, 4]
    assert result["candidate_scope"].eq("current_library_1391").all()
    assert not result["is_external_candidate"].any()


def test_current_library_route_uses_nested_fusion_when_available(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "budget": 5,
                "target_fold": 2,
                "reaction_id": "RHEA:NESTED",
                "selected_method": "rrf__base_0.5__dual_0.5",
                "selected_kind": "rrf",
                "selected_old_method": "base_only",
                "selected_dual_source": "baseline",
                "hit": 1,
                "hits": 1,
                "best_positive_rank_within_budget": 2,
                "reciprocal_rank_within_budget": 0.5,
                "ranking": "P5;P2;P3;P1;P4",
            }
        ]
    ).to_csv(tmp_path / "nested_query_metrics.csv", index=False)

    result = rank_current_library("RHEA:NESTED", 4, tmp_path)
    assert result["candidate_id"].tolist() == ["P5", "P2", "P3", "P1"]
    assert result["rank"].tolist() == [1, 2, 3, 4]
    assert result["score_source"].eq("nested_current_library_dual_fusion").all()
    assert result["selected_method"].eq("rrf__base_0.5__dual_0.5").all()
    assert result["selected_old_method"].eq("base_only").all()
    assert result["selected_dual_source"].eq("baseline").all()
    assert result["validation_fold"].eq(2).all()
    assert not result["is_external_candidate"].any()
