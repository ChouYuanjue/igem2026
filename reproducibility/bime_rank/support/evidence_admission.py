from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class EvidenceFit:
    coefficients: np.ndarray
    objective: float
    success: bool
    iterations: int


def fit_nonnegative_pairwise_logistic(
    core_difference: np.ndarray,
    evidence_difference: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
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
    if l2 < 0:
        raise ValueError("l2 must be non-negative")

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
        objective = float(np.sum(weight * loss) + 0.5 * l2 * np.dot(coef, coef))
        sigmoid_negative_margin = 1.0 / (
            1.0 + np.exp(np.clip(margin, -60.0, 60.0))
        )
        grad = -(evidence.T @ (weight * sigmoid_negative_margin)) + l2 * coef
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
