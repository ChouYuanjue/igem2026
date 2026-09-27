"""Factorized interaction-atlas primitives used by FIBRE."""

from .atlas import (
    LocalInteractionChart,
    bilinear_chart_score,
    glue_local_interactions,
    normalize_partition,
    overlap_consistency_loss,
    partition_readout_discrepancy_bound,
)
from .interaction import (
    FiniteRankInteractionUpdate,
    PositivePairConditioner,
    conditioned_pair_scores,
    updated_bilinear_pair_scores,
)

__all__ = [
    "LocalInteractionChart",
    "bilinear_chart_score",
    "glue_local_interactions",
    "normalize_partition",
    "overlap_consistency_loss",
    "partition_readout_discrepancy_bound",
    "FiniteRankInteractionUpdate",
    "PositivePairConditioner",
    "conditioned_pair_scores",
    "updated_bilinear_pair_scores",
]
