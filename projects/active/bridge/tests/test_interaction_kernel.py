from __future__ import annotations

import torch

from projects.active.bridge.kernel.interaction import (
    BoundedBilinearInteraction,
    FiniteRankInteractionUpdate,
    PositivePairConditioner,
    conditioned_pair_scores,
    updated_bilinear_pair_scores,
)


def _unit(rows: int, dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, dim, generator=gen)
    return torch.nn.functional.normalize(x, dim=1)


def test_bilinear_correction_is_exact_identity_at_initialization() -> None:
    r = _unit(7, 12, 1)
    e = _unit(9, 12, 2)
    layer = BoundedBilinearInteraction(12, max_frobenius_norm=0.1)
    actual = layer.score_matrix(r, e)
    expected = r @ e.T
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_bilinear_correction_has_global_score_perturbation_bound() -> None:
    r = _unit(31, 16, 3)
    e = _unit(29, 16, 4)
    layer = BoundedBilinearInteraction(16, max_frobenius_norm=0.075)
    with torch.no_grad():
        layer.delta.normal_(mean=0.0, std=5.0)
    shift = layer.score_matrix(r, e) - r @ e.T
    assert float(shift.abs().max()) <= 0.075 + 1e-5
    assert layer.diagnostics()["bounded_frobenius_norm"] <= 0.075 + 1e-6


def test_positive_pair_conditioning_is_train_free_low_rank_update() -> None:
    observed_r = _unit(2, 8, 5)
    observed_e = _unit(2, 8, 6)
    conditioner = PositivePairConditioner.from_positive_pairs(
        observed_r,
        observed_e,
        weights=torch.tensor([1.0, 0.5]),
        max_frobenius_norm=0.1,
    )
    q_r = _unit(4, 8, 7)
    q_e = _unit(4, 8, 8)
    actual = conditioned_pair_scores(q_r, q_e, conditioner)
    expected = (q_r * q_e).sum(1) + (
        (q_r @ conditioner.delta) * q_e
    ).sum(1)
    torch.testing.assert_close(actual, expected)
    assert conditioner.pair_count == 2
    assert conditioner.bounded_frobenius_norm <= 0.1 + 1e-6


def test_chart_local_finite_rank_update_supports_rectangular_coordinates() -> None:
    observed_r = _unit(3, 5, 11)
    observed_e = _unit(3, 7, 12)
    update = FiniteRankInteractionUpdate.from_pair_coordinates(
        observed_r,
        observed_e,
        weights=torch.tensor([1.0, 0.5, 0.25]),
        max_frobenius_norm=0.08,
    )
    assert update.delta.shape == (5, 7)
    assert int(torch.linalg.matrix_rank(update.delta)) <= update.pair_count
    assert update.bounded_frobenius_norm <= 0.08 + 1e-6

    q_r = _unit(6, 5, 13)
    q_e = _unit(6, 7, 14)
    base = torch.randn(5, 7, generator=torch.Generator().manual_seed(15))
    actual = updated_bilinear_pair_scores(q_r, q_e, base, update)
    expected = ((q_r @ (base + update.delta)) * q_e).sum(dim=1)
    torch.testing.assert_close(actual, expected)


def test_chart_local_finite_rank_update_has_unit_coordinate_perturbation_bound() -> None:
    observed_r = _unit(4, 6, 16)
    observed_e = _unit(4, 9, 17)
    update = FiniteRankInteractionUpdate.from_pair_coordinates(
        observed_r, observed_e, max_frobenius_norm=0.06
    )
    q_r = _unit(20, 6, 18)
    q_e = _unit(20, 9, 19)
    zero = torch.zeros(6, 9)
    shift = updated_bilinear_pair_scores(q_r, q_e, zero, update)
    assert float(shift.abs().max()) <= 0.06 + 1e-5
