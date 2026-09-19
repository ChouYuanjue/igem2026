from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import digamma, gammaln

from .correspondence import CorrespondenceSection


DEFAULT_LEVEL_ULPS = 64.0


def numerical_level_tolerance(
    values: np.ndarray,
    *,
    ulps: float = DEFAULT_LEVEL_ULPS,
) -> float:
    """Return the machine-scale tolerance used to define one FIBRE level set.

    The convention is inherited from the development stable-level-set audit:
    64 float64 machine eps at the scale of the complete query section.  It is
    numerical, not a tuned biochemical threshold.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = x[np.isfinite(x)]
    if not len(finite):
        raise ValueError("level-set values must contain a finite entry")
    if not np.isfinite(ulps) or ulps <= 0:
        raise ValueError("ulps must be finite and positive")
    scale = max(1.0, float(np.max(np.abs(finite))))
    return float(ulps * np.finfo(np.float64).eps * scale)


def stable_level_ids(
    values: np.ndarray,
    *,
    ulps: float = DEFAULT_LEVEL_ULPS,
) -> tuple[np.ndarray, float]:
    """Partition finite scalar values into deterministic numerical level sets.

    Sets are built in ascending value order.  A candidate joins the current
    level when it lies within the global machine tolerance of that level's first
    value.  Returned IDs are monotone with the scalar field but independent of
    candidate identifiers.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(x)):
        raise ValueError("stable_level_ids requires finite values")
    tol = numerical_level_tolerance(x, ulps=ulps)
    order = np.argsort(x, kind="stable")
    levels = np.empty(len(x), dtype=np.int64)
    level = -1
    ref = None
    for idx in order:
        value = float(x[idx])
        if ref is None or abs(value - ref) > tol:
            level += 1
            ref = value
        levels[idx] = level
    return levels, tol


@dataclass(frozen=True)
class TieAwareRank:
    """Rank uncertainty induced only by a FIBRE numerical level set."""

    best_positive_value: float
    tolerance: float
    better_count: int
    tie_size: int
    positive_count_in_tie: int
    optimistic_rank: int
    expected_rank: float
    pessimistic_rank: int
    expected_reciprocal_rank: float


def _expected_reciprocal_rank(
    better: int,
    tie_size: int,
    positives_in_tie: int,
) -> float:
    b = int(better)
    t = int(tie_size)
    m = int(positives_in_tie)
    if t <= 0 or m <= 0 or m > t:
        raise ValueError("invalid tie block")
    if m == t:
        return 1.0 / float(b + 1)
    if m == 1 and t > 4096:
        # Mean of 1/(b+k), k=1..t, evaluated without allocating t entries.
        return float((digamma(b + t + 1.0) - digamma(b + 1.0)) / t)
    k = np.arange(1, t - m + 2, dtype=np.float64)
    # P(first positive is k) = C(t-k,m-1) / C(t,m).
    log_p = (
        gammaln(t - k + 1.0)
        - gammaln(float(m))
        - gammaln(t - k - m + 2.0)
        - (
            gammaln(t + 1.0)
            - gammaln(m + 1.0)
            - gammaln(t - m + 1.0)
        )
    )
    p = np.exp(log_p)
    return float(np.sum(p / (float(b) + k)))


def tie_aware_best_positive_rank(
    values: np.ndarray,
    positive_indices: np.ndarray,
    *,
    ulps: float = DEFAULT_LEVEL_ULPS,
) -> TieAwareRank:
    """Return rank interval and expectation without arbitrary within-level order.

    Lower values are better, as for the FIBRE correspondence defect.  Within the
    best positive numerical level set, all permutations are treated as equally
    admissible.  The expected rank is the expected first-positive order statistic
    under that neutral permutation, not a candidate-ID tie break.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    pos = np.unique(np.asarray(positive_indices, dtype=np.int64).reshape(-1))
    if not len(pos):
        raise ValueError("at least one positive index is required")
    if np.any(pos < 0) or np.any(pos >= len(x)):
        raise ValueError("positive index out of range")
    if not np.all(np.isfinite(x)):
        raise ValueError("rank values must be finite")
    tol = numerical_level_tolerance(x, ulps=ulps)
    best = float(np.min(x[pos]))
    tie = np.abs(x - best) <= tol
    better = x < best - tol
    pos_in_tie = int(np.sum(tie[pos]))
    tie_size = int(np.sum(tie))
    better_count = int(np.sum(better))
    optimistic = better_count + 1
    pessimistic = better_count + (tie_size - pos_in_tie) + 1
    expected = better_count + (tie_size + 1.0) / (pos_in_tie + 1.0)
    expected_rr = _expected_reciprocal_rank(
        better_count, tie_size, pos_in_tie
    )
    return TieAwareRank(
        best_positive_value=best,
        tolerance=tol,
        better_count=better_count,
        tie_size=tie_size,
        positive_count_in_tie=pos_in_tie,
        optimistic_rank=int(optimistic),
        expected_rank=float(expected),
        pessimistic_rank=int(pessimistic),
        expected_reciprocal_rank=expected_rr,
    )


@dataclass(frozen=True)
class SectionApplicability:
    """Threshold-free geometric diagnostics for one FIBRE query section."""

    query_support_distance: float
    best_defect: float
    best_level_size: int
    best_level_fraction: float
    next_level_gap: float | None
    best_level_candidate_support_distance_min: float
    best_level_candidate_support_distance_median: float
    numerical_tolerance: float


def section_applicability(
    section: CorrespondenceSection,
    *,
    ulps: float = DEFAULT_LEVEL_ULPS,
) -> SectionApplicability:
    """Describe support and ambiguity directly from the FIBRE geometry.

    No OOD label, calibrated threshold, or learned confidence model is produced.
    The query marginal measures distance to observed support; the best numerical
    level size and next-level gap measure how strongly the field resolves the
    candidate fibre at machine-stable precision.
    """
    defect = np.asarray(section.defect, dtype=np.float64)
    marginal = np.asarray(section.candidate_marginal_sq, dtype=np.float64)
    tol = numerical_level_tolerance(defect, ulps=ulps)
    best = float(np.min(defect))
    top = np.abs(defect - best) <= tol
    above = defect > best + tol
    next_gap = (
        float(np.min(defect[above]) - best)
        if np.any(above)
        else None
    )
    top_support = np.sqrt(np.maximum(marginal[top], 0.0))
    return SectionApplicability(
        query_support_distance=float(math.sqrt(max(section.query_marginal_sq, 0.0))),
        best_defect=best,
        best_level_size=int(np.sum(top)),
        best_level_fraction=float(np.mean(top)),
        next_level_gap=next_gap,
        best_level_candidate_support_distance_min=float(np.min(top_support)),
        best_level_candidate_support_distance_median=float(np.median(top_support)),
        numerical_tolerance=tol,
    )
