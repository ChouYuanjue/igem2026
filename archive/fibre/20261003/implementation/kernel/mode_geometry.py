from __future__ import annotations

import torch
import torch.nn.functional as F


def normalized_factorized_operator_scores(
    reaction_hidden: torch.Tensor,
    protein_hidden: torch.Tensor,
    reaction_factor: torch.Tensor,
    protein_factor: torch.Tensor,
) -> torch.Tensor:
    """Normalized finite-rank pairing between unequal molecular spaces."""
    if reaction_hidden.ndim != 2 or protein_hidden.ndim != 2:
        raise ValueError("hidden tensors must be two-dimensional")
    if reaction_factor.ndim != 2 or protein_factor.ndim != 2:
        raise ValueError("factor tensors must be two-dimensional")
    if reaction_hidden.shape[1] != reaction_factor.shape[1]:
        raise ValueError("reaction hidden/factor dimensions do not align")
    if protein_hidden.shape[1] != protein_factor.shape[1]:
        raise ValueError("protein hidden/factor dimensions do not align")
    if reaction_factor.shape[0] != protein_factor.shape[0]:
        raise ValueError("reaction/protein factors must share the same rank")
    reaction_local = F.normalize(reaction_hidden @ reaction_factor.T, p=2, dim=-1)
    protein_local = F.normalize(protein_hidden @ protein_factor.T, p=2, dim=-1)
    return reaction_local @ protein_local.T


def weighted_mode_moments(
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted mean and variance over latent modes."""
    if values.shape != weights.shape:
        raise ValueError("values and weights must have identical shapes")
    if bool((weights < 0).any()):
        raise ValueError("weights must be non-negative")
    total = weights.sum(dim=dim, keepdim=True)
    if bool((total <= 0).any()):
        raise ValueError("weights must have positive total mass")
    normalized = weights / total
    mean = (normalized * values).sum(dim=dim)
    centered = values - mean.unsqueeze(dim)
    variance = (normalized * centered.square()).sum(dim=dim)
    return mean, variance


def condition_mode_weights(
    prior_weights: torch.Tensor,
    log_evidence_increment: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """KL-minimal update q_new proportional to q_old * exp(log_evidence)."""
    if prior_weights.shape != log_evidence_increment.shape:
        raise ValueError("prior_weights and log_evidence_increment must align")
    if bool((prior_weights < 0).any()):
        raise ValueError("prior_weights must be non-negative")
    total = prior_weights.sum(dim=dim, keepdim=True)
    if bool((total <= 0).any()):
        raise ValueError("prior_weights must have positive total mass")
    prior = prior_weights / total
    logits = prior.clamp_min(torch.finfo(prior.dtype).tiny).log()
    return torch.softmax(logits + log_evidence_increment, dim=dim)


def tempered_condition_mode_weights(
    prior_weights: torch.Tensor,
    log_evidence_increment: torch.Tensor,
    strength: float | torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Tempered KL-minimal update on the mode simplex."""
    if prior_weights.shape != log_evidence_increment.shape:
        raise ValueError("prior_weights and log_evidence_increment must align")
    if bool((prior_weights < 0).any()):
        raise ValueError("prior_weights must be non-negative")
    total = prior_weights.sum(dim=dim, keepdim=True)
    if bool((total <= 0).any()):
        raise ValueError("prior_weights must have positive total mass")
    prior = prior_weights / total
    eta = torch.as_tensor(strength, dtype=prior.dtype, device=prior.device)
    if bool((eta < 0).any()):
        raise ValueError("strength must be non-negative")
    logits = prior.clamp_min(torch.finfo(prior.dtype).tiny).log()
    return torch.softmax(logits + eta * log_evidence_increment, dim=dim)


def mode_simplex_natural_gradient(
    prior_weights: torch.Tensor,
    log_evidence_increment: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Fisher-Rao tangent of the exponential mode update at zero strength."""
    if prior_weights.shape != log_evidence_increment.shape:
        raise ValueError("prior_weights and log_evidence_increment must align")
    if bool((prior_weights < 0).any()):
        raise ValueError("prior_weights must be non-negative")
    total = prior_weights.sum(dim=dim, keepdim=True)
    if bool((total <= 0).any()):
        raise ValueError("prior_weights must have positive total mass")
    prior = prior_weights / total
    mean_evidence = (prior * log_evidence_increment).sum(dim=dim, keepdim=True)
    return prior * (log_evidence_increment - mean_evidence)


def mode_information_gain(
    posterior_weights: torch.Tensor,
    prior_weights: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """KL(posterior || prior), invariant to a common mode permutation."""
    if posterior_weights.shape != prior_weights.shape:
        raise ValueError("posterior_weights and prior_weights must align")
    if bool((posterior_weights < 0).any()) or bool((prior_weights < 0).any()):
        raise ValueError("weights must be non-negative")
    post_total = posterior_weights.sum(dim=dim, keepdim=True)
    prior_total = prior_weights.sum(dim=dim, keepdim=True)
    if bool((post_total <= 0).any()) or bool((prior_total <= 0).any()):
        raise ValueError("weights must have positive total mass")
    posterior = posterior_weights / post_total
    prior = prior_weights / prior_total
    tiny = torch.finfo(posterior.dtype).tiny
    return (
        posterior.clamp_min(tiny)
        * (
            posterior.clamp_min(tiny).log()
            - prior.clamp_min(tiny).log()
        )
    ).sum(dim=dim)


def decompose_mode_and_model_variance(
    member_mode_means: torch.Tensor,
    member_mode_variances: torch.Tensor,
    *,
    member_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Law-of-total-variance split: within-mode vs between-model variance."""
    if member_mode_means.shape != member_mode_variances.shape:
        raise ValueError("member means and variances must align")
    if bool((member_mode_variances < 0).any()):
        raise ValueError("member mode variances must be non-negative")
    within_mode = member_mode_variances.mean(dim=member_dim)
    between_model = member_mode_means.var(dim=member_dim, unbiased=False)
    return within_mode, between_model, within_mode + between_model
