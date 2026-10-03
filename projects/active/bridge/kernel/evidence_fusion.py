from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


EvidenceKind = Literal[
    "molecular_view",
    "mechanistic",
    "structural",
    "experimental_context",
]


@dataclass(frozen=True)
class EvidenceChannelSpec:
    """A scientific evidence source that can modify a frozen retrieval score.

    Quality is defined so that a larger value means that the channel is
    expected to be more reliable for the current query. This lets the fusion
    layer use a monotone reliability function without prescribing the
    biological measurement.
    """

    name: str
    kind: EvidenceKind
    has_quality: bool = False


class AnchoredEvidenceFusion(nn.Module):
    """Small additive evidence pool on top of a frozen core ranking.

    The core coefficient is fixed to one. Each available evidence channel adds
    an independently weighted, query-calibrated score. Missing channels
    contribute exactly zero and never renormalise the core or other channels.

    This deliberately differs from a mixture-of-experts softmax: sequence,
    structure, reaction-centre and context evidence are complementary
    observations and can support the same candidate at once.
    """

    def __init__(
        self,
        specs: tuple[EvidenceChannelSpec, ...],
        *,
        initial_strength: float = 0.05,
        initial_quality_slope: float = 0.05,
    ) -> None:
        super().__init__()
        if not specs:
            raise ValueError("at least one evidence channel is required")
        if initial_strength <= 0 or initial_quality_slope <= 0:
            raise ValueError("initial strengths must be positive")
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError("evidence channel names must be unique")
        self.specs = specs
        n = len(specs)
        self.strength = nn.Parameter(torch.full((n,), float(initial_strength)))
        self.quality_slope = nn.Parameter(
            torch.full((n,), float(initial_quality_slope))
        )
        self.register_buffer(
            "quality_enabled",
            torch.tensor([spec.has_quality for spec in specs], dtype=torch.bool),
        )

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def channel_weights(self, quality: torch.Tensor) -> torch.Tensor:
        """Return non-negative per-example channel strengths."""
        if quality.shape[-1] != len(self.specs):
            raise ValueError("quality channel dimension does not match specs")
        q = quality.to(dtype=self.strength.dtype)
        slope = self.quality_slope * self.quality_enabled.to(dtype=self.strength.dtype)
        return self.strength + q * slope

    @torch.no_grad()
    def project_nonnegative_(self) -> None:
        """Project reliability parameters onto the non-negative orthant.

        With the core coefficient fixed at one and a linear quality term, the
        pairwise logistic objective is convex in these parameters. Projected
        gradient therefore gives a small, deterministic admission layer rather
        than another deep router.
        """
        self.strength.clamp_(min=0.0)
        self.quality_slope.clamp_(min=0.0)

    def forward(
        self,
        core_score: torch.Tensor,
        evidence_score: torch.Tensor,
        available: torch.Tensor,
        quality: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fuse calibrated evidence with a frozen core score."""
        if evidence_score.shape != available.shape:
            raise ValueError("evidence_score and available must have equal shape")
        if evidence_score.shape[:-1] != core_score.shape:
            raise ValueError("core_score must match evidence batch dimensions")
        if evidence_score.shape[-1] != len(self.specs):
            raise ValueError("evidence channel dimension does not match specs")
        if quality is None:
            quality = torch.zeros_like(evidence_score)
        if quality.shape != evidence_score.shape:
            raise ValueError("quality must match evidence_score shape")

        weight = self.channel_weights(quality)
        mask = available.to(dtype=evidence_score.dtype)
        contribution = evidence_score * weight * mask
        fused = core_score + contribution.sum(dim=-1)
        return fused, {
            "channel_weights": weight,
            "channel_contributions": contribution,
            "available": available,
        }


def query_standardize(
    score: torch.Tensor,
    available: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Standardize candidate scores within each query with missing-neutral output.

    The final dimension is the candidate dimension. Unsupported candidates
    receive exactly zero after standardisation, the neutral point of the
    additive evidence pool.
    """
    if available is None:
        available = torch.ones_like(score, dtype=torch.bool)
    if score.shape != available.shape:
        raise ValueError("score and available must have equal shape")
    mask = available.to(dtype=score.dtype)
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (score * mask).sum(dim=-1, keepdim=True) / count
    centered = (score - mean) * mask
    variance = (centered.square().sum(dim=-1, keepdim=True) / count).clamp_min(eps**2)
    z = centered / torch.sqrt(variance)
    return z * mask
