from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .correspondence import CorrespondenceState, add_positive_seed


@dataclass(frozen=True)
class SeedInfluence:
    """Exact change induced by adding one verified positive to Omega."""

    reaction_index: int
    protein_index: int
    seed_product_isolation_sq: float
    seed_reaction_novelty_sq: float
    seed_protein_novelty_sq: float
    joint_changed_count: int
    reaction_marginal_changed_count: int
    protein_marginal_changed_count: int
    defect_decreased_count: int
    defect_unchanged_count: int
    defect_increased_count: int
    defect_delta_mean: float
    defect_delta_abs_max: float
    affected_pair_fraction: float


def seed_support_novelty(
    reaction_distance_sq: np.ndarray,
    protein_distance_sq: np.ndarray,
    positive_pairs: np.ndarray,
    reaction_index: int,
    protein_index: int,
) -> tuple[float, float, float]:
    """Intrinsic novelty of a candidate seed relative to the existing support.

    The three values are nearest old-support squared distance in the reaction
    factor, protein factor, and Cartesian product.  They are descriptive
    geometry, not thresholds or learned confidence scores.
    """
    dr=np.asarray(reaction_distance_sq,dtype=np.float64)
    de=np.asarray(protein_distance_sq,dtype=np.float64)
    pairs=np.asarray(positive_pairs,dtype=np.int64)
    rr,ee=int(reaction_index),int(protein_index)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or len(pairs)==0:
        raise ValueError("positive_pairs must be a non-empty [n,2] array")
    if not (0<=rr<dr.shape[0] and 0<=ee<de.shape[0]):
        raise ValueError("seed index out of range")
    r_support=np.unique(pairs[:,0]); e_support=np.unique(pairs[:,1])
    rnov=float(np.min(dr[rr,r_support]))
    enov=float(np.min(de[ee,e_support]))
    pnov=float(np.min(dr[rr,pairs[:,0]]+de[ee,pairs[:,1]]))
    return rnov,enov,pnov


def seed_influence(
    state: CorrespondenceState,
    reaction_distance_sq: np.ndarray,
    protein_distance_sq: np.ndarray,
    positive_pairs: np.ndarray,
    reaction_index: int,
    protein_index: int,
    *,
    atol: float = 1e-12,
) -> SeedInfluence:
    """Decompose the global field change from one exact positive-seed update.

    A new seed can lower the joint transform and the two marginal transforms at
    different spatial rates.  Consequently the defect itself can decrease,
    remain unchanged, or increase.  This function exposes that exact geometry
    instead of assuming every verified seed must monotonically improve every
    unrelated ranking.
    """
    if atol < 0 or not np.isfinite(atol):
        raise ValueError("atol must be finite and non-negative")
    new=add_positive_seed(
        state,reaction_distance_sq,protein_distance_sq,
        int(reaction_index),int(protein_index),
    )
    old_joint=np.asarray(state.joint_sq,dtype=np.float64)
    new_joint=np.asarray(new.joint_sq,dtype=np.float64)
    old_r=np.asarray(state.reaction_marginal_sq,dtype=np.float64)
    new_r=np.asarray(new.reaction_marginal_sq,dtype=np.float64)
    old_e=np.asarray(state.protein_marginal_sq,dtype=np.float64)
    new_e=np.asarray(new.protein_marginal_sq,dtype=np.float64)
    delta=np.asarray(new.defect-state.defect,dtype=np.float64)
    decreased=delta < -atol
    increased=delta > atol
    unchanged=~(decreased|increased)
    rnov,enov,pnov=seed_support_novelty(
        reaction_distance_sq,protein_distance_sq,positive_pairs,
        reaction_index,protein_index,
    )
    affected=decreased|increased
    return SeedInfluence(
        reaction_index=int(reaction_index),
        protein_index=int(protein_index),
        seed_product_isolation_sq=pnov,
        seed_reaction_novelty_sq=rnov,
        seed_protein_novelty_sq=enov,
        joint_changed_count=int(np.sum(new_joint < old_joint-atol)),
        reaction_marginal_changed_count=int(np.sum(new_r < old_r-atol)),
        protein_marginal_changed_count=int(np.sum(new_e < old_e-atol)),
        defect_decreased_count=int(np.sum(decreased)),
        defect_unchanged_count=int(np.sum(unchanged)),
        defect_increased_count=int(np.sum(increased)),
        defect_delta_mean=float(np.mean(delta)),
        defect_delta_abs_max=float(np.max(np.abs(delta))),
        affected_pair_fraction=float(np.mean(affected)),
    )
