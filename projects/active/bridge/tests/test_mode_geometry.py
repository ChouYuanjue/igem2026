from __future__ import annotations

import torch

from projects.active.bridge.kernel.mode_geometry import (
    condition_mode_weights,
    decompose_mode_and_model_variance,
    mode_information_gain,
    mode_simplex_natural_gradient,
    normalized_factorized_operator_scores,
    tempered_condition_mode_weights,
    weighted_mode_moments,
)


def test_factorized_operator_handles_unequal_input_dimensions() -> None:
    reaction_hidden = torch.tensor([[1.0, 2.0, -1.0], [0.5, -0.5, 1.0]])
    protein_hidden = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    reaction_factor = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, -1.0]])
    protein_factor = torch.tensor([[1.0, -1.0], [2.0, 1.0]])
    scores = normalized_factorized_operator_scores(
        reaction_hidden,
        protein_hidden,
        reaction_factor,
        protein_factor,
    )
    r = torch.nn.functional.normalize(reaction_hidden @ reaction_factor.T, dim=-1)
    p = torch.nn.functional.normalize(protein_hidden @ protein_factor.T, dim=-1)
    torch.testing.assert_close(scores, r @ p.T)
    assert int(torch.linalg.matrix_rank(reaction_factor.T @ protein_factor)) <= 2


def test_weighted_mode_moments() -> None:
    values = torch.tensor([[0.2, 0.5, -0.4]])
    weights = torch.tensor([[0.2, 0.3, 0.5]])
    mean, variance = weighted_mode_moments(values, weights)
    torch.testing.assert_close(mean, torch.tensor([-0.01]))
    torch.testing.assert_close(variance, torch.tensor([0.1629]))


def test_information_geometric_update_is_identity_for_zero_evidence() -> None:
    prior = torch.tensor([[0.1, 0.3, 0.6]])
    posterior = condition_mode_weights(prior, torch.zeros_like(prior))
    torch.testing.assert_close(posterior, prior)


def test_information_geometric_updates_compose_in_log_evidence() -> None:
    prior = torch.tensor([[0.2, 0.5, 0.3]])
    first = torch.tensor([[0.1, -0.2, 0.3]])
    second = torch.tensor([[-0.4, 0.2, 0.1]])
    sequential = condition_mode_weights(condition_mode_weights(prior, first), second)
    combined = condition_mode_weights(prior, first + second)
    torch.testing.assert_close(sequential, combined)


def test_mode_simplex_natural_gradient_is_tempered_update_derivative() -> None:
    prior = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float64)
    evidence = torch.tensor([[0.7, -0.4, 0.1]], dtype=torch.float64)
    tangent = mode_simplex_natural_gradient(prior, evidence)
    torch.testing.assert_close(
        tangent.sum(dim=-1),
        torch.zeros(1, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    epsilon = 1e-5
    posterior = tempered_condition_mode_weights(prior, evidence, epsilon)
    torch.testing.assert_close(
        (posterior - prior) / epsilon,
        tangent,
        atol=2e-6,
        rtol=2e-5,
    )


def test_small_evidence_kl_has_fisher_variance_curvature() -> None:
    prior = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float64)
    evidence = torch.tensor([[0.7, -0.4, 0.1]], dtype=torch.float64)
    mean = (prior * evidence).sum(dim=-1, keepdim=True)
    variance = (prior * (evidence - mean).square()).sum(dim=-1)
    eta = 1e-3
    posterior = tempered_condition_mode_weights(prior, evidence, eta)
    kl = mode_information_gain(posterior, prior)
    torch.testing.assert_close(
        kl,
        0.5 * eta * eta * variance,
        atol=2e-10,
        rtol=2e-3,
    )


def test_mode_information_gain_is_permutation_invariant() -> None:
    prior = torch.tensor([[0.2, 0.5, 0.3]])
    shifted = condition_mode_weights(prior, torch.tensor([[0.5, -0.1, 0.2]]))
    permutation = torch.tensor([2, 0, 1])
    torch.testing.assert_close(
        mode_information_gain(shifted, prior),
        mode_information_gain(shifted[:, permutation], prior[:, permutation]),
    )


def test_total_variance_separates_mode_and_model_components() -> None:
    member_means = torch.tensor([[1.0, 2.0], [3.0, 2.0], [2.0, 2.0]])
    member_mode_variances = torch.tensor([[0.2, 0.4], [0.4, 0.6], [0.3, 0.5]])
    within, between, total = decompose_mode_and_model_variance(
        member_means,
        member_mode_variances,
    )
    torch.testing.assert_close(total, within + between)
    torch.testing.assert_close(between[1], torch.tensor(0.0))
    assert float(between[0]) > 0
