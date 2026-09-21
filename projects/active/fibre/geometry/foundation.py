from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ProductSupportDiagnostics:
    """Exact decomposition of one zero-temperature FIBRE candidate.

    For product metric d_M^2=d_R^2+d_E^2 and accepted correspondence Omega,

        J_Omega = Delta_Omega + m_R + m_E.

    J_Omega is the squared distance to the nearest accepted *joint* precedent.
    Delta removes marginal novelty so it should never be interpreted without
    m_R/m_E when making an extrapolation/applicability statement.
    """

    defect: float
    reaction_support_sq: float
    protein_support_sq: float
    joint_precedent_sq: float
    joint_precedent_distance: float

    def to_dict(self) -> dict[str,float]:
        return {
            "defect":float(self.defect),
            "reaction_support_sq":float(self.reaction_support_sq),
            "protein_support_sq":float(self.protein_support_sq),
            "joint_precedent_sq":float(self.joint_precedent_sq),
            "joint_precedent_distance":float(self.joint_precedent_distance),
        }


def product_support_diagnostics(
    defect: float,
    reaction_support_sq: float,
    protein_support_sq: float,
    *,
    tolerance: float = 1e-12,
) -> ProductSupportDiagnostics:
    """Reconstruct exact joint-precedent distance from the FIBRE decomposition."""

    values=np.asarray(
        [defect,reaction_support_sq,protein_support_sq],dtype=np.float64
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("FIBRE support decomposition requires finite values")
    if np.any(values < -abs(float(tolerance))):
        raise ValueError("FIBRE defect/support terms must be non-negative")
    values=np.maximum(values,0.0)
    joint=float(np.sum(values))
    return ProductSupportDiagnostics(
        defect=float(values[0]),
        reaction_support_sq=float(values[1]),
        protein_support_sq=float(values[2]),
        joint_precedent_sq=joint,
        joint_precedent_distance=float(math.sqrt(joint)),
    )


def local_fiber_distance_upper_bound(
    defect: float,
    reaction_support_sq: float,
    protein_support_sq: float,
    *,
    hausdorff_lipschitz_constant: float,
) -> float:
    """Conditional local-generalization bound for a set-valued catalytic fiber.

    Assume the true relation Gamma is locally represented by a closed-valued
    set-valued map G: M_R -> 2^{M_E} satisfying

        d_H(G(r),G(r')) <= L d_R(r,r').

    Because accepted positives Omega are a subset of Gamma, for z=(r,e)

        d_E(e,G(r))
          <= sqrt(1+L^2) d_M(z,Gamma)
          <= sqrt(1+L^2) d_M(z,Omega)
          =  sqrt(1+L^2) sqrt(Delta + m_R + m_E).

    This is a geometric bound under the stated local regularity assumption,
    not a calibrated probability or an empirically estimated error bar.
    """

    L=float(hausdorff_lipschitz_constant)
    if not math.isfinite(L) or L < 0:
        raise ValueError("hausdorff_lipschitz_constant must be finite and non-negative")
    diag=product_support_diagnostics(
        defect,reaction_support_sq,protein_support_sq
    )
    return float(math.sqrt(1.0+L*L)*diag.joint_precedent_distance)


def relation_sampling_distance_bounds(
    distance_to_true_relation: float,
    *,
    one_sided_coverage_radius: float,
) -> tuple[float,float]:
    """Bounds distance to accepted positives from coverage of the true relation.

    Let Omega subset Gamma and suppose every point of the covered part of Gamma
    is within h of Omega. Then for any z in that covered neighborhood,

        d(z,Gamma) <= d(z,Omega) <= d(z,Gamma)+h.

    The returned values are the corresponding lower/upper distances. Squaring
    them gives bounds for J_Omega.
    """

    d=float(distance_to_true_relation)
    h=float(one_sided_coverage_radius)
    if not (math.isfinite(d) and math.isfinite(h)) or d < 0 or h < 0:
        raise ValueError("distances and coverage radius must be finite and non-negative")
    return d,d+h


def defect_stability_bound(
    max_reaction_squared_distance_error: float,
    max_protein_squared_distance_error: float,
) -> float:
    """Uniform perturbation bound for the zero-temperature defect.

    If every squared reaction distance used by the operator is perturbed by at
    most eps_R and every squared protein distance by at most eps_E, then minima
    are 1-Lipschitz in the sup norm:

        |J_hat-J| <= eps_R+eps_E,
        |mR_hat-mR| <= eps_R,
        |mE_hat-mE| <= eps_E,

    hence

        |Delta_hat-Delta| <= 2(eps_R+eps_E).

    This connects graph-geodesic approximation quality directly to the FIBRE
    operator without assuming a learned scalar score.
    """

    er=float(max_reaction_squared_distance_error)
    ee=float(max_protein_squared_distance_error)
    if not (math.isfinite(er) and math.isfinite(ee)) or er < 0 or ee < 0:
        raise ValueError("distance errors must be finite and non-negative")
    return float(2.0*(er+ee))
