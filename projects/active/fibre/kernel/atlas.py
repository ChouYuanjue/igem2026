from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalInteractionChart:
    """One local coordinate description of the same catalytic interaction.

    A chart is defined only where its molecular information is available and
    biologically applicable.  It returns a scalar interaction score.  Different
    charts may use unrelated latent dimensions or encoders.
    """

    name: str


def normalize_partition(
    logits: torch.Tensor,
    available: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Partition of unity over the charts available for each query/pair.

    Missing charts carry exactly zero mass.  Available chart weights are
    non-negative and sum to one.  At least one chart must be available.
    """
    if logits.shape != available.shape:
        raise ValueError("logits and available must have identical shape")
    if available.dtype is not torch.bool:
        raise ValueError("available must be boolean")
    if not bool(available.any(dim=dim).all()):
        raise ValueError("every item must have at least one available chart")
    masked = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
    weights = torch.softmax(masked, dim=dim)
    return weights.masked_fill(~available, 0.0)


def glue_local_interactions(
    scores: torch.Tensor,
    partition: torch.Tensor,
    available: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Glue local interaction estimates into one global interaction.

    This is not residual fusion: each chart is a local coordinate estimate of
    the same scalar catalytic interaction, and the partition of unity supplies
    the coordinate transition/gluing rule.
    """
    if scores.shape != partition.shape or scores.shape != available.shape:
        raise ValueError("scores, partition, and available must align")
    if available.dtype is not torch.bool:
        raise ValueError("available must be boolean")
    if bool((partition < 0).any()):
        raise ValueError("partition weights must be non-negative")
    if not torch.allclose(
        partition.sum(dim=dim),
        torch.ones_like(partition.sum(dim=dim)),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("partition weights must sum to one")
    if bool((partition.masked_select(~available) != 0).any()):
        raise ValueError("unavailable charts must have zero partition mass")
    return (scores * partition).sum(dim=dim)


def bilinear_chart_score(
    left: torch.Tensor,
    interaction: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Score paired coordinates under one chart-specific bilinear form."""
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("left and right coordinates must be 2D")
    if left.shape[0] != right.shape[0]:
        raise ValueError("paired chart coordinates must have equal row count")
    if interaction.ndim != 2:
        raise ValueError("interaction matrix must be 2D")
    if left.shape[1] != interaction.shape[0] or right.shape[1] != interaction.shape[1]:
        raise ValueError("interaction matrix dimensions do not match chart coordinates")
    return ((left @ interaction) * right).sum(dim=1)


def overlap_consistency_loss(
    scores: torch.Tensor,
    partition: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    """Agreement penalty for charts that are simultaneously applicable.

    For every pair of available charts alpha,beta this computes
    rho_alpha*rho_beta*(K_alpha-K_beta)^2 and averages across items.
    """
    if scores.shape != partition.shape or scores.shape != available.shape:
        raise ValueError("scores, partition, and available must align")
    if scores.ndim != 2:
        raise ValueError("scores must be [items, charts]")
    n_charts = scores.shape[1]
    if n_charts < 2:
        return scores.new_zeros(())
    total = scores.new_zeros(scores.shape[0])
    normalizer = scores.new_zeros(scores.shape[0])
    for left in range(n_charts):
        for right in range(left + 1, n_charts):
            overlap = available[:, left] & available[:, right]
            weight = partition[:, left] * partition[:, right] * overlap.to(scores.dtype)
            total = total + weight * (scores[:, left] - scores[:, right]).square()
            normalizer = normalizer + weight
    valid = normalizer > 0
    if not bool(valid.any()):
        return scores.new_zeros(())
    return (total[valid] / normalizer[valid]).mean()
