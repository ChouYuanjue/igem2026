"""Soft-target distillation primitive adapted from MIT-licensed Mammoth."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation(teacher_logits: torch.Tensor, student_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """Mammoth/ZSCL-style soft-target cross-entropy with T^2 scaling."""
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    return F.cross_entropy(student_logits / temperature, soft_targets, reduction='mean') * (temperature ** 2)


def bidirectional_distillation(teacher_logits: torch.Tensor, student_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """ZSCL-style distillation of a score matrix in both retrieval directions."""
    return distillation(teacher_logits, student_logits, temperature) + distillation(teacher_logits.T, student_logits.T, temperature)
