from __future__ import annotations

import torch


def sampled_average_ranks(score_samples: torch.Tensor) -> torch.Tensor:
    """Average-tie ranks for repeated score samples on a shortlist.

    score_samples has shape [samples, candidates]. Exact ties receive the
    midpoint rank. Complexity is quadratic in shortlist size, so this helper is
    intended for the reported shortlist rather than a 100k-scale full universe.
    """
    if score_samples.ndim != 2:
        raise ValueError("score_samples must have shape [samples, candidates]")
    if score_samples.shape[1] == 0:
        raise ValueError("at least one candidate is required")
    left = score_samples[:, None, :]
    right = score_samples[:, :, None]
    greater = (left > right).sum(dim=-1)
    tied = (left == right).sum(dim=-1) - 1
    return 1.0 + greater.to(score_samples.dtype) + 0.5 * tied.to(score_samples.dtype)


def topk_inclusion_probability(
    score_samples: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Empirical probability that each candidate remains inside the top-k."""
    if k <= 0:
        raise ValueError("k must be positive")
    ranks = sampled_average_ranks(score_samples)
    effective_k = min(k, score_samples.shape[1])
    return (ranks <= effective_k).to(score_samples.dtype).mean(dim=0)


def pairwise_dominance_probability(score_samples: torch.Tensor) -> torch.Tensor:
    """Empirical P(score_i > score_j), assigning half mass to exact ties."""
    if score_samples.ndim != 2:
        raise ValueError("score_samples must have shape [samples, candidates]")
    left = score_samples[:, :, None]
    right = score_samples[:, None, :]
    win = (left > right).to(score_samples.dtype)
    tie = (left == right).to(score_samples.dtype)
    return (win + 0.5 * tie).mean(dim=0)


def rank_quantile_interval(
    score_samples: torch.Tensor,
    *,
    lower: float = 0.025,
    upper: float = 0.975,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantile interval of perturbation ranks for each shortlist candidate."""
    if not 0 <= lower <= upper <= 1:
        raise ValueError("require 0 <= lower <= upper <= 1")
    ranks = sampled_average_ranks(score_samples)
    return (
        torch.quantile(ranks, lower, dim=0),
        torch.quantile(ranks, upper, dim=0),
    )
