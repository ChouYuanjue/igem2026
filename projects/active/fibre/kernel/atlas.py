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


def partition_readout_discrepancy_bound(
    scores: torch.Tensor,
    left_partition: torch.Tensor,
    right_partition: torch.Tensor,
    available: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Bound disagreement between two partition-weighted readouts.

    For chart scores K_alpha and two valid partitions p and q over the same
    available charts,

        |sum p_alpha K_alpha - sum q_alpha K_alpha|
        <= 0.5 * ||p-q||_1 * (max K_alpha - min K_alpha).

    The bound is pointwise over every non-chart dimension. It separates two
    causes of directional disagreement: partition mismatch and disagreement
    among the active local interaction estimates.
    """
    if (
        scores.shape != left_partition.shape
        or scores.shape != right_partition.shape
        or scores.shape != available.shape
    ):
        raise ValueError("scores, partitions, and available must align")
    if available.dtype is not torch.bool:
        raise ValueError("available must be boolean")
    if not bool(available.any(dim=dim).all()):
        raise ValueError("every item must have at least one available chart")
    for name, partition in (
        ("left_partition", left_partition),
        ("right_partition", right_partition),
    ):
        if bool((partition < 0).any()):
            raise ValueError(f"{name} weights must be non-negative")
        total = partition.sum(dim=dim)
        if not torch.allclose(
            total, torch.ones_like(total), atol=1e-6, rtol=1e-6
        ):
            raise ValueError(f"{name} weights must sum to one")
        if bool((partition.masked_select(~available) != 0).any()):
            raise ValueError(f"{name} gives mass to unavailable charts")

    positive_inf = torch.full_like(scores, torch.inf)
    negative_inf = torch.full_like(scores, -torch.inf)
    active_min = torch.where(available, scores, positive_inf).amin(dim=dim)
    active_max = torch.where(available, scores, negative_inf).amax(dim=dim)
    total_variation = 0.5 * (left_partition - right_partition).abs().sum(dim=dim)
    return total_variation * (active_max - active_min)


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
