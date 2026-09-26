from __future__ import annotations

import torch

from projects.active.fibre.kernel.interaction import (
    BoundedBilinearInteraction,
    PositivePairConditioner,
    conditioned_pair_scores,
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
