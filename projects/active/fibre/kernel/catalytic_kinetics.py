from __future__ import annotations

import torch

R_GAS_J_PER_MOL_K = 8.31446261815324


def barrier_lowering_from_binding_free_energies(
    ground_state_binding_j_per_mol: torch.Tensor,
    transition_state_binding_j_per_mol: torch.Tensor,
) -> torch.Tensor:
    """Barrier lowering from differential ground/transition-state stabilization.

    Positive output means the transition state is stabilized more strongly
    than the ground state:

        delta_delta_G_stab = G_bind_GS - G_bind_TS.
    """
    if ground_state_binding_j_per_mol.shape != transition_state_binding_j_per_mol.shape:
        raise ValueError("ground- and transition-state binding tensors must align")
    return ground_state_binding_j_per_mol - transition_state_binding_j_per_mol


def relative_activation_free_energy_j_per_mol(
    rate_ratio: torch.Tensor,
    temperature_k: float,
) -> torch.Tensor:
    """Return ΔG2‡-ΔG1‡ implied by k2/k1 under transition-state theory.

    k2/k1 = exp(-(ΔG2‡-ΔG1‡)/(R T)), so
    ΔG2‡-ΔG1‡ = -R T log(k2/k1).

    This is a physical conversion for measured/assumed rate ratios. FIBRE
    ranking scores are not assigned joule units unless a separate kinetic
    calibration establishes the mapping from score to log-rate utility.
    """
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    if bool((rate_ratio <= 0).any()):
        raise ValueError("rate_ratio must be positive")
    return -R_GAS_J_PER_MOL_K * temperature_k * torch.log(rate_ratio)


def log_rate_ratio(
    rate: torch.Tensor,
    reference_rate: torch.Tensor,
) -> torch.Tensor:
    """Dimensionless log(rate/reference_rate) for like-unit positive rates."""
    if rate.shape != reference_rate.shape:
        raise ValueError("rate and reference_rate must align")
    if bool((rate <= 0).any()) or bool((reference_rate <= 0).any()):
        raise ValueError("rates must be positive")
    return torch.log(rate / reference_rate)


def gaussian_log_rate_mode_evidence(
    observed_log_rate_ratio: torch.Tensor,
    predicted_mode_log_rate_ratio: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Mode log-likelihood increments from quantitative kinetic evidence.

    The observed quantity is a dimensionless log rate ratio.  The prediction
    may carry an extra final mode dimension and is broadcast against the
    observation.  Constants shared by all modes are omitted because posterior
    mode weights depend only on relative log likelihoods.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    residual = predicted_mode_log_rate_ratio - observed_log_rate_ratio
    return -0.5 * residual.square() / (sigma * sigma)


def weighted_mode_moments(
    utilities: torch.Tensor,
    weights: torch.Tensor,
    *,
    dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted mean and variance of latent catalytic-mode utilities."""
    if utilities.shape != weights.shape:
        raise ValueError("utilities and weights must have identical shapes")
    if bool((weights < 0).any()):
        raise ValueError("weights must be non-negative")
    total = weights.sum(dim=dim, keepdim=True)
    if bool((total <= 0).any()):
        raise ValueError("weights must have positive total mass")
    normalized = weights / total
    mean = (normalized * utilities).sum(dim=dim)
    centered = utilities - mean.unsqueeze(dim)
    variance = (normalized * centered.square()).sum(dim=dim)
    return mean, variance


def kinetic_log_mean_exp(
    dimensionless_utilities: torch.Tensor,
    weights: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Exact parallel-mode log-rate utility.

    If u_k=-ΔG_k‡/(R T) and w_k are mode occupancies, parallel transition-state
    channels contribute rates proportional to w_k exp(u_k). Removing the common
    Eyring prefactor gives log(sum_k w_k exp(u_k)).
    """
    if dimensionless_utilities.shape != weights.shape:
        raise ValueError("dimensionless_utilities and weights must align")
    if bool((weights < 0).any()):
        raise ValueError("weights must be non-negative")
    total = weights.sum(dim=dim, keepdim=True)
    if bool((total <= 0).any()):
        raise ValueError("weights must have positive total mass")
    normalized = weights / total
    return torch.logsumexp(
        normalized.clamp_min(torch.finfo(normalized.dtype).tiny).log()
        + dimensionless_utilities,
        dim=dim,
    )


def kinetic_mode_responsibilities(
    dimensionless_utilities: torch.Tensor,
    prior_weights: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Pair-specific mode responsibilities under the parallel-rate model.

    pi_k ∝ w_k exp(u_k).  These responsibilities are also the derivative of
    the effective log-rate utility with respect to each mode utility.
    """
    if dimensionless_utilities.shape != prior_weights.shape:
        raise ValueError("dimensionless_utilities and prior_weights must align")
    if bool((prior_weights < 0).any()):
        raise ValueError("prior_weights must be non-negative")
    total = prior_weights.sum(dim=dim, keepdim=True)
    if bool((total <= 0).any()):
        raise ValueError("prior_weights must have positive total mass")
    prior = prior_weights / total
    logits = (
        prior.clamp_min(torch.finfo(prior.dtype).tiny).log()
        + dimensionless_utilities
    )
    return torch.softmax(logits, dim=dim)


def condition_mode_weights(
    prior_weights: torch.Tensor,
    log_likelihood_increment: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Information-geometric/Bayesian update of catalytic-mode weights.

    q_new = argmax_q <q, ell> - KL(q || q_old)
          ∝ q_old * exp(ell)

    Zero log-evidence is an exact identity update. Sequential evidence combines
    by addition in log-likelihood space, avoiding an additive correction to the
    pair-score matrix itself.
    """
    if prior_weights.shape != log_likelihood_increment.shape:
        raise ValueError("prior_weights and log_likelihood_increment must align")
    if bool((prior_weights < 0).any()):
        raise ValueError("prior_weights must be non-negative")
    total = prior_weights.sum(dim=dim, keepdim=True)
    if bool((total <= 0).any()):
        raise ValueError("prior_weights must have positive total mass")
    prior = prior_weights / total
    logits = prior.clamp_min(torch.finfo(prior.dtype).tiny).log()
    logits = logits + log_likelihood_increment
    return torch.softmax(logits, dim=dim)
