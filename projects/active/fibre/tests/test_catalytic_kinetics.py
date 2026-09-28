from __future__ import annotations

import torch

from projects.active.fibre.kernel.catalytic_kinetics import (
    barrier_lowering_from_binding_free_energies,
    condition_mode_weights,
    gaussian_log_rate_mode_evidence,
    kinetic_log_mean_exp,
    kinetic_mode_responsibilities,
    log_rate_ratio,
    relative_activation_free_energy_j_per_mol,
    weighted_mode_moments,
)


def test_transition_state_preferential_binding_lowers_barrier() -> None:
    ground = torch.tensor([-20_000.0])
    transition = torch.tensor([-40_000.0])
    lowering = barrier_lowering_from_binding_free_energies(ground, transition)
    torch.testing.assert_close(lowering, torch.tensor([20_000.0]))


def test_transition_state_rate_ratio_has_energy_units_and_correct_sign() -> None:
    ratio = torch.tensor([10.0])
    delta = relative_activation_free_energy_j_per_mol(ratio, 298.15)
    assert float(delta) < 0
    reverse = relative_activation_free_energy_j_per_mol(1.0 / ratio, 298.15)
    torch.testing.assert_close(delta, -reverse)


def test_quantitative_rate_evidence_prefers_matching_mode() -> None:
    observed = log_rate_ratio(torch.tensor([10.0]), torch.tensor([1.0]))
    predicted = torch.tensor([[0.0, float(torch.log(torch.tensor(10.0))), 4.0]])
    evidence = gaussian_log_rate_mode_evidence(observed[:, None], predicted, sigma=0.5)
    assert int(evidence.argmax(dim=-1).item()) == 1
    posterior = condition_mode_weights(
        torch.full_like(evidence, 1.0 / evidence.shape[-1]),
        evidence,
    )
    assert int(posterior.argmax(dim=-1).item()) == 1


def test_kinetic_log_mean_exp_is_between_weighted_mean_and_max() -> None:
    utilities = torch.tensor([[0.2, 0.5, -0.4], [0.1, 0.1, 0.1]])
    weights = torch.tensor([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
    mean, _ = weighted_mode_moments(utilities, weights)
    exact = kinetic_log_mean_exp(utilities, weights)
    assert bool((exact >= mean - 1e-7).all())
    assert bool((exact <= utilities.max(dim=-1).values + 1e-7).all())
    torch.testing.assert_close(exact[1], torch.tensor(0.1), atol=1e-6, rtol=0)


def test_second_cumulant_approximates_small_mode_disagreement() -> None:
    utilities = torch.tensor([[0.01, -0.02, 0.03]])
    weights = torch.tensor([[0.2, 0.5, 0.3]])
    mean, variance = weighted_mode_moments(utilities, weights)
    exact = kinetic_log_mean_exp(utilities, weights)
    second_order = mean + 0.5 * variance
    torch.testing.assert_close(exact, second_order, atol=2e-6, rtol=0)


def test_mode_responsibilities_are_log_mixture_gradients() -> None:
    utilities = torch.tensor([[0.3, -0.2, 0.7]], requires_grad=True)
    weights = torch.tensor([[0.2, 0.5, 0.3]])
    exact = kinetic_log_mean_exp(utilities, weights)
    exact.sum().backward()
    responsibilities = kinetic_mode_responsibilities(utilities.detach(), weights)
    torch.testing.assert_close(utilities.grad, responsibilities)
    torch.testing.assert_close(
        responsibilities.sum(dim=-1),
        torch.ones(1),
    )


def test_information_geometric_update_is_identity_for_zero_evidence() -> None:
    prior = torch.tensor([[0.1, 0.3, 0.6]])
    posterior = condition_mode_weights(prior, torch.zeros_like(prior))
    torch.testing.assert_close(posterior, prior)


def test_information_geometric_updates_compose_in_log_evidence() -> None:
    prior = torch.tensor([[0.2, 0.5, 0.3]])
    first = torch.tensor([[0.1, -0.2, 0.3]])
    second = torch.tensor([[-0.4, 0.2, 0.1]])
    sequential = condition_mode_weights(
        condition_mode_weights(prior, first),
        second,
    )
    combined = condition_mode_weights(prior, first + second)
    torch.testing.assert_close(sequential, combined)
