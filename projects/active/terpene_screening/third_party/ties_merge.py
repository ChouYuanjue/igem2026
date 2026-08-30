from __future__ import annotations

import torch


def topk_values_mask(matrix: torch.Tensor, keep_fraction: float = 0.2) -> torch.Tensor:
    """Keep the largest-magnitude fraction in each task vector.

    Adapted from TIES-Merging ``topk_values_mask``. ``keep_fraction=0.2`` matches
    the upstream ``topk20`` convention. The upstream inclusive kth-value threshold
    can retain one extra coordinate at a boundary/tie.
    """
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    original_shape = matrix.shape
    if matrix.dim() == 1:
        matrix = matrix.unsqueeze(0)
    if matrix.dim() != 2:
        raise ValueError("TIES task vectors must be a 1D or 2D tensor")
    _n, dim = matrix.shape
    if keep_fraction == 1.0:
        masked = matrix
    else:
        kth = max(1, dim - int(dim * keep_fraction))
        kth_values, _ = matrix.abs().kthvalue(kth, dim=1, keepdim=True)
        masked = matrix * (matrix.abs() >= kth_values)
    return masked.squeeze(0) if len(original_shape) == 1 else masked


def resolve_sign_mass(task_vectors: torch.Tensor) -> torch.Tensor:
    """Elect the sign with the largest signed parameter mass, as in TIES."""
    if task_vectors.dim() != 2:
        raise ValueError("task_vectors must be 2D")
    signs = torch.sign(task_vectors.sum(dim=0))
    majority = torch.sign(signs.sum())
    if majority == 0:
        majority = torch.tensor(1.0, device=signs.device, dtype=signs.dtype)
    signs = signs.clone()
    signs[signs == 0] = majority
    return signs


def disjoint_mean(task_vectors: torch.Tensor, elected_signs: torch.Tensor) -> torch.Tensor:
    """Average only non-zero task-vector entries agreeing with elected signs."""
    if task_vectors.dim() != 2 or elected_signs.dim() != 1:
        raise ValueError("expected [tasks, parameters] task_vectors and [parameters] signs")
    if task_vectors.shape[1] != elected_signs.shape[0]:
        raise ValueError("task vector/sign dimensions differ")
    keep = torch.where(elected_signs.unsqueeze(0) > 0, task_vectors > 0, task_vectors < 0)
    selected = task_vectors * keep
    counts = (selected != 0).sum(dim=0).float().clamp(min=1)
    return selected.sum(dim=0) / counts


def ties_merge(task_vectors: torch.Tensor, *, keep_fraction: float = 0.2) -> torch.Tensor:
    """TIES trim -> elect sign -> disjoint mean merge."""
    trimmed = topk_values_mask(task_vectors, keep_fraction=keep_fraction)
    signs = resolve_sign_mass(trimmed)
    return disjoint_mean(trimmed, signs)
