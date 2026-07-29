from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from projects.active.terpene_screening.evaluate_marts_multi_expert_rank_fusion import (
    FusionSpec,
    fuse_ranking,
)
from projects.active.terpene_screening.evaluate_multi_expert_protocol_comparison import (
    DirectionalMultiExpertDualTower,
    MultiExpertConfig,
    gate_regularization,
)
from projects.active.terpene_screening.rank_current_library import (
    rank_current_library,
    resolve_budget,
)
from projects.active.terpene_screening.train_dual_tower_cold import (
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
    for gates, experts in (
        (diagnostics["protein_gates"], diagnostics["protein_experts"]),
        (diagnostics["reaction_gates"], diagnostics["reaction_experts"]),
    ):
        balance, entropy, diversity = gate_regularization(gates, experts)
        assert torch.isfinite(balance)
        assert torch.isfinite(entropy)
        assert torch.isfinite(diversity)


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
