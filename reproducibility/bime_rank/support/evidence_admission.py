from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class EvidenceFit:
    coefficients: np.ndarray
    objective: float
    success: bool
    iterations: int


@dataclass(frozen=True)
class MeanBootstrapInterval:
    n: int
    mean: float
    median: float
    fraction_positive: float
    lower_95: float
    upper_95: float
    seed: int
    replicates: int


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int = 20260723,
    replicates: int = 4000,
) -> MeanBootstrapInterval:
    """Deterministic query-level bootstrap interval for an OOF improvement."""

    local = np.asarray(values, dtype=np.float64)
    if local.ndim != 1 or len(local) < 2:
        raise ValueError("bootstrap requires at least two query-level values")
    if not np.isfinite(local).all():
        raise ValueError("bootstrap values must be finite")
    if replicates < 200:
        raise ValueError("bootstrap requires at least 200 replicates")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    # Chunking avoids allocating replicates x queries for large local datasets.
    chunk = 256
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        indices = rng.integers(
            0,
            len(local),
            size=(stop - start, len(local)),
        )
        means[start:stop] = local[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return MeanBootstrapInterval(
        n=int(len(local)),
        mean=float(local.mean()),
        median=float(np.median(local)),
        fraction_positive=float(np.mean(local > 0.0)),
        lower_95=float(lower),
        upper_95=float(upper),
        seed=int(seed),
        replicates=int(replicates),
    )


def core_score_signature(frame: pd.DataFrame) -> str:
    """Hash only the frozen query/candidate/core-score relation.

    Evidence columns, labels, folds, row order and file formatting are excluded.
    Two independently prepared evidence tables therefore share a baseline ID
    exactly when they were fitted against the same core scores.
    """

    required = {"query_id", "candidate_id", "core_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"core score signature missing columns: {missing}")
    work = frame[["query_id", "candidate_id", "core_score"]].copy()
    work["query_id"] = work["query_id"].astype(str)
    work["candidate_id"] = work["candidate_id"].astype(str)
    work["core_score"] = pd.to_numeric(
        work["core_score"],
        errors="raise",
    ).astype(float)
    if work.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("core score signature contains duplicate query/candidate rows")
    if not np.isfinite(work["core_score"]).all():
        raise ValueError("core score signature requires finite scores")
    work = work.sort_values(
        ["query_id", "candidate_id"],
        kind="stable",
    )
    digest = hashlib.sha256()
    for row in work.itertuples(index=False):
        digest.update(str(row.query_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.candidate_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(float(row.core_score).hex().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def fit_nonnegative_pairwise_logistic(
    core_difference: np.ndarray,
    evidence_difference: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    l1: float = 0.0,
    l2: float = 0.0,
    initial: float = 0.1,
) -> EvidenceFit:
    """Fit non-negative additive evidence strengths with a convex ranking loss.

    The frozen core has coefficient one. core_difference is the
    positive-minus-negative core score for each training comparison and
    evidence_difference contains the corresponding difference for each
    scientific evidence channel. The only trainable quantities are non-negative
    evidence coefficients.
    """

    core = np.asarray(core_difference, dtype=np.float64)
    evidence = np.asarray(evidence_difference, dtype=np.float64)
    if core.ndim != 1:
        raise ValueError("core_difference must be one-dimensional")
    if evidence.ndim != 2 or evidence.shape[0] != len(core):
        raise ValueError("evidence_difference must have shape [pairs, channels]")
    if evidence.shape[1] == 0:
        raise ValueError("at least one evidence channel is required")
    if not np.isfinite(core).all() or not np.isfinite(evidence).all():
        raise ValueError("pairwise differences must be finite")
    if l1 < 0 or l2 < 0:
        raise ValueError("regularization strengths must be non-negative")

    if sample_weight is None:
        weight = np.full(len(core), 1.0 / max(len(core), 1), dtype=np.float64)
    else:
        weight = np.asarray(sample_weight, dtype=np.float64)
        if weight.shape != core.shape or np.any(weight < 0) or not np.isfinite(weight).all():
            raise ValueError("sample_weight must be finite, non-negative and pair-aligned")
        total = float(weight.sum())
        if total <= 0:
            raise ValueError("sample_weight must have positive total mass")
        weight = weight / total

    def value_and_grad(coef: np.ndarray) -> tuple[float, np.ndarray]:
        margin = core + evidence @ coef
        loss = np.logaddexp(0.0, -margin)
        objective = float(
            np.sum(weight * loss)
            + l1 * coef.sum()
            + 0.5 * l2 * np.dot(coef, coef)
        )
        sigmoid_negative_margin = 1.0 / (
            1.0 + np.exp(np.clip(margin, -60.0, 60.0))
        )
        grad = (
            -(evidence.T @ (weight * sigmoid_negative_margin))
            + l1
            + l2 * coef
        )
        return objective, np.asarray(grad, dtype=np.float64)

    result = minimize(
        lambda x: value_and_grad(x)[0],
        x0=np.full(evidence.shape[1], float(initial), dtype=np.float64),
        jac=lambda x: value_and_grad(x)[1],
        bounds=[(0.0, None)] * evidence.shape[1],
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(str(result.message))
    return EvidenceFit(
        coefficients=np.asarray(result.x, dtype=np.float64),
        objective=float(result.fun),
        success=bool(result.success),
        iterations=int(result.nit),
    )
