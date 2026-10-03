from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

ExecutionMode = Literal[
    "global_cached",
    "query_conditional_cached",
    "shortlist_on_demand",
]
DecisionMode = Literal["full", "shortlist", "fallback"]


@dataclass(frozen=True)
class ExpertExecutionSpec:
    """Execution contract for one already-admitted candidate-level expert.

    This object does not decide whether an expert is scientifically valid. Expert
    admission remains a separate clean-development decision. The execution layer only
    decides where an admitted expert is economical to run given current asset state.
    """

    name: str
    mode: ExecutionMode
    fallback: str
    default_shortlist_k: int | None = None
    max_shortlist_k: int | None = None

    def __post_init__(self) -> None:
        if self.mode == "shortlist_on_demand":
            if not self.default_shortlist_k or self.default_shortlist_k <= 0:
                raise ValueError("shortlist_on_demand requires default_shortlist_k > 0")
            if self.max_shortlist_k is not None and self.max_shortlist_k < self.default_shortlist_k:
                raise ValueError("max_shortlist_k cannot be smaller than default_shortlist_k")
        elif self.default_shortlist_k is not None or self.max_shortlist_k is not None:
            raise ValueError("shortlist limits are only valid for shortlist_on_demand experts")


@dataclass(frozen=True)
class ExpertExecutionDecision:
    expert_name: str
    mode: DecisionMode
    candidate_rows: np.ndarray
    candidate_count: int
    candidate_fraction: float
    reason: str
    fallback: str


def stable_union_prefix(rank_orders: Sequence[np.ndarray], k: int, candidate_count: int) -> np.ndarray:
    """Return a deterministic immutable shortlist from several cheap rank orders.

    Rows are unique and sorted numerically only to make downstream feature materialization
    deterministic; candidate preference is still represented by the source ranks supplied
    to the downstream ranker. The function never widens beyond source Top-k prefixes.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    pieces: list[np.ndarray] = []
    for order in rank_orders:
        arr = np.asarray(order, dtype=np.int64)
        if arr.ndim != 1 or len(arr) != candidate_count:
            raise ValueError("rank order must be a full candidate permutation")
        head = arr[: min(k, candidate_count)]
        if np.any(head < 0) or np.any(head >= candidate_count):
            raise ValueError("rank order contains out-of-range candidate rows")
        pieces.append(head)
    if not pieces:
        raise ValueError("at least one cheap rank order is required")
    return np.unique(np.concatenate(pieces)).astype(np.int32, copy=False)


def plan_expert_execution(
    spec: ExpertExecutionSpec,
    *,
    candidate_count: int,
    query_supported: bool,
    candidate_features_cached: bool,
    shortlist_rows: np.ndarray | None = None,
) -> ExpertExecutionDecision:
    """Plan one expert without changing ranking semantics or scientific admission.

    - cached global experts score the full candidate universe;
    - cached conditional experts score the full universe only when query-side support exists;
    - on-demand experts never trigger full-universe feature materialization and may only
      consume an already-formed shortlist; otherwise the caller follows the declared fallback.
    """

    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    full = np.arange(candidate_count, dtype=np.int32)

    if spec.mode == "global_cached":
        if not candidate_features_cached:
            return ExpertExecutionDecision(
                spec.name, "fallback", np.empty(0, dtype=np.int32), 0, 0.0,
                "global expert candidate features are not cached", spec.fallback,
            )
        return ExpertExecutionDecision(
            spec.name, "full", full, candidate_count, 1.0,
            "candidate-side features are cached; full scoring has low marginal cost", spec.fallback,
        )

    if spec.mode == "query_conditional_cached":
        if not query_supported:
            return ExpertExecutionDecision(
                spec.name, "fallback", np.empty(0, dtype=np.int32), 0, 0.0,
                "query-side expert asset is unavailable", spec.fallback,
            )
        if not candidate_features_cached:
            return ExpertExecutionDecision(
                spec.name, "fallback", np.empty(0, dtype=np.int32), 0, 0.0,
                "candidate-side expert features are not cached", spec.fallback,
            )
        return ExpertExecutionDecision(
            spec.name, "full", full, candidate_count, 1.0,
            "query is supported and candidate-side features are cached", spec.fallback,
        )

    if spec.mode != "shortlist_on_demand":
        raise ValueError(f"unknown expert execution mode: {spec.mode}")
    if not query_supported:
        return ExpertExecutionDecision(
            spec.name, "fallback", np.empty(0, dtype=np.int32), 0, 0.0,
            "query is unsupported by the on-demand specialist", spec.fallback,
        )
    if candidate_features_cached:
        # If the supposedly expensive feature has already been materialized, using the full
        # universe is semantically safe and cheaper than forcing an unnecessary cascade.
        return ExpertExecutionDecision(
            spec.name, "full", full, candidate_count, 1.0,
            "candidate-side specialist features are already cached", spec.fallback,
        )
    if shortlist_rows is None or len(shortlist_rows) == 0:
        return ExpertExecutionDecision(
            spec.name, "fallback", np.empty(0, dtype=np.int32), 0, 0.0,
            "on-demand specialist requires a pre-existing shortlist", spec.fallback,
        )
    rows = np.unique(np.asarray(shortlist_rows, dtype=np.int64))
    if np.any(rows < 0) or np.any(rows >= candidate_count):
        raise ValueError("shortlist contains out-of-range candidate rows")
    if spec.max_shortlist_k is not None and len(rows) > int(spec.max_shortlist_k):
        raise ValueError("shortlist exceeds the expert execution contract")
    rows = rows.astype(np.int32, copy=False)
    return ExpertExecutionDecision(
        spec.name, "shortlist", rows, int(len(rows)), float(len(rows) / candidate_count),
        "candidate-side specialist features are uncached; materialize only the frozen shortlist", spec.fallback,
    )
