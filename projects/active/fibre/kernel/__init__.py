"""Factorized interaction-atlas primitives used by FIBRE."""

from .atlas import (
    LocalInteractionChart,
    bilinear_chart_score,
    glue_local_interactions,
    normalize_partition,
    overlap_consistency_loss,
)
from .interaction import (
    PositivePairConditioner,
    conditioned_pair_scores,
)

__all__ = [
    "LocalInteractionChart",
    "bilinear_chart_score",
    "glue_local_interactions",
    "normalize_partition",
    "overlap_consistency_loss",
    "PositivePairConditioner",
    "conditioned_pair_scores",
]
