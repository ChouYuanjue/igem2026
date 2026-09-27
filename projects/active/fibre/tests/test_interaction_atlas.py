from __future__ import annotations

import torch

from projects.active.fibre.kernel.atlas import (
    bilinear_chart_score,
    glue_local_interactions,
    normalize_partition,
    partition_readout_discrepancy_bound,
)


def test_partition_of_unity_renormalizes_only_available_charts() -> None:
    logits = torch.tensor([[1.0, 3.0, -2.0], [2.0, 1.0, 4.0]])
    available = torch.tensor([[True, False, True], [True, True, False]])
    weights = normalize_partition(logits, available)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
    assert torch.equal(weights.masked_select(~available), torch.zeros(2))


def test_gluing_is_exact_single_chart_fallback_when_optional_views_missing() -> None:
    scores = torch.tensor([[0.2, 0.9, -0.4], [0.7, -0.1, 0.3]])
    available = torch.tensor([[True, False, False], [True, False, False]])
    weights = normalize_partition(torch.zeros_like(scores), available)
    actual = glue_local_interactions(scores, weights, available)
    torch.testing.assert_close(actual, scores[:, 0])


def test_bilinear_chart_is_coordinate_invariant_under_basis_change() -> None:
    torch.manual_seed(7)
    left = torch.randn(6, 3)
    right = torch.randn(6, 4)
    interaction = torch.randn(3, 4)
    a = torch.randn(3, 3)
    b = torch.randn(4, 4)
    while abs(float(torch.linalg.det(a))) < 0.1:
        a = torch.randn(3, 3)
    while abs(float(torch.linalg.det(b))) < 0.1:
        b = torch.randn(4, 4)

    original = bilinear_chart_score(left, interaction, right)
    left_prime = left @ a
    right_prime = right @ b
    interaction_prime = torch.linalg.inv(a) @ interaction @ torch.linalg.inv(b).T
    transformed = bilinear_chart_score(left_prime, interaction_prime, right_prime)
    torch.testing.assert_close(original, transformed, rtol=1e-5, atol=1e-5)


def test_gluing_uses_same_scalar_interaction_not_additive_residual() -> None:
    scores = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
    available = torch.ones_like(scores, dtype=torch.bool)
    partition = torch.tensor([[0.75, 0.25], [0.25, 0.75]])
    glued = glue_local_interactions(scores, partition, available)
    torch.testing.assert_close(glued, torch.tensor([0.65, 0.70]))


def test_overlap_consistency_ignores_nonoverlapping_charts() -> None:
    from projects.active.fibre.kernel.atlas import overlap_consistency_loss

    scores = torch.tensor([[0.2, 0.8], [0.3, 0.9]])
    available = torch.tensor([[True, False], [True, True]])
    partition = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    loss = overlap_consistency_loss(scores, partition, available)
    torch.testing.assert_close(loss, torch.tensor((0.3 - 0.9) ** 2))


def test_glued_interaction_lies_in_active_chart_convex_hull() -> None:
    scores = torch.tensor([[0.2, 0.9, -0.4], [0.7, -0.1, 0.3]])
    available = torch.tensor([[True, False, True], [True, True, False]])
    partition = normalize_partition(torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]]), available)
    glued = glue_local_interactions(scores, partition, available)
    active_min = torch.tensor([-0.4, -0.1])
    active_max = torch.tensor([0.2, 0.7])
    assert bool((glued >= active_min - 1e-7).all())
    assert bool((glued <= active_max + 1e-7).all())


def test_directional_partition_discrepancy_obeys_total_variation_bound() -> None:
    scores = torch.tensor([[0.1, 0.5, 0.9], [-0.4, 0.2, 0.7]])
    available = torch.tensor([[True, True, True], [True, True, False]])
    r2e = torch.tensor([[0.6, 0.3, 0.1], [0.8, 0.2, 0.0]])
    e2r = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.9, 0.0]])
    r2e_score = glue_local_interactions(scores, r2e, available)
    e2r_score = glue_local_interactions(scores, e2r, available)
    bound = partition_readout_discrepancy_bound(scores, r2e, e2r, available)
    assert bool(((r2e_score - e2r_score).abs() <= bound + 1e-7).all())


def test_directional_partition_discrepancy_vanishes_when_charts_agree() -> None:
    scores = torch.tensor([[0.4, 0.4, 0.4]])
    available = torch.ones_like(scores, dtype=torch.bool)
    left = torch.tensor([[1.0, 0.0, 0.0]])
    right = torch.tensor([[0.0, 0.0, 1.0]])
    bound = partition_readout_discrepancy_bound(scores, left, right, available)
    torch.testing.assert_close(bound, torch.zeros(1))
