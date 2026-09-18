from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CorrespondenceState:
    """Distance state induced by a sparse positive correspondence Omega."""
    joint_sq: np.ndarray
    reaction_marginal_sq: np.ndarray
    protein_marginal_sq: np.ndarray

    @property
    def defect(self) -> np.ndarray:
        """Non-negative excess cost of a shared biochemical precedent."""
        out = (
            np.asarray(self.joint_sq, dtype=np.float64)
            - np.asarray(self.reaction_marginal_sq, dtype=np.float64)[:, None]
            - np.asarray(self.protein_marginal_sq, dtype=np.float64)[None, :]
        )
        if not np.all(np.isfinite(out)):
            raise ValueError("correspondence defect requires finite connected factor distances")
        return np.maximum(out, 0.0)

    @property
    def field(self) -> np.ndarray:
        """Compatibility field: larger is better, with zero attained on Omega."""
        return -self.defect


def correspondence_state(
    reaction_distance_sq: np.ndarray,
    protein_distance_sq: np.ndarray,
    positive_pairs: np.ndarray,
) -> CorrespondenceState:
    """Construct the exact zero-temperature product correspondence field.

    For Omega subset M_R x M_E,
      J(r,e)=min_(ri,ei in Omega) d_R(r,ri)^2+d_E(e,ei)^2
      m_R(r)=min_(ri in Omega_R) d_R(r,ri)^2
      m_E(e)=min_(ei in Omega_E) d_E(e,ei)^2
      Delta_Omega=J-m_R-m_E.

    Delta is the additional geometric price of requiring r and e to share the
    same known biochemical precedent rather than merely being independently
    close to the observed reaction and protein marginals.
    """
    dr = np.asarray(reaction_distance_sq, dtype=np.float64)
    de = np.asarray(protein_distance_sq, dtype=np.float64)
    pairs = np.asarray(positive_pairs, dtype=np.int64)
    if dr.ndim != 2 or dr.shape[0] != dr.shape[1]:
        raise ValueError("reaction_distance_sq must be square")
    if de.ndim != 2 or de.shape[0] != de.shape[1]:
        raise ValueError("protein_distance_sq must be square")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
        raise ValueError("positive_pairs must be a non-empty [n,2] array")
    if not (np.all(np.isfinite(dr)) and np.all(np.isfinite(de))):
        raise ValueError("factor distances must be finite; use connected manifolds")
    if np.any(dr < 0) or np.any(de < 0):
        raise ValueError("squared distances must be non-negative")

    nr, ne = dr.shape[0], de.shape[0]
    if np.any(pairs[:, 0] < 0) or np.any(pairs[:, 0] >= nr):
        raise ValueError("reaction positive index out of range")
    if np.any(pairs[:, 1] < 0) or np.any(pairs[:, 1] >= ne):
        raise ValueError("protein positive index out of range")

    joint = np.full((nr, ne), np.inf, dtype=np.float64)
    for rr, ee in pairs:
        joint = np.minimum(joint, dr[:, rr, None] + de[:, ee][None, :])
    r_support = np.unique(pairs[:, 0])
    e_support = np.unique(pairs[:, 1])
    mr = np.min(dr[:, r_support], axis=1)
    me = np.min(de[:, e_support], axis=1)
    return CorrespondenceState(joint, mr, me)


def add_positive_seed(
    state: CorrespondenceState,
    reaction_distance_sq: np.ndarray,
    protein_distance_sq: np.ndarray,
    reaction_index: int,
    protein_index: int,
) -> CorrespondenceState:
    """Exact incremental update for Omega <- Omega union {(r*,e*)}.

    No parameter or model is retrained. The three distance transforms are
    updated by pointwise minima, exactly matching a full reconstruction.
    """
    dr = np.asarray(reaction_distance_sq, dtype=np.float64)
    de = np.asarray(protein_distance_sq, dtype=np.float64)
    rr, ee = int(reaction_index), int(protein_index)
    if not (0 <= rr < dr.shape[0] and 0 <= ee < de.shape[0]):
        raise ValueError("seed index out of range")
    joint = np.minimum(
        np.asarray(state.joint_sq, dtype=np.float64),
        dr[:, rr, None] + de[:, ee][None, :],
    )
    mr = np.minimum(np.asarray(state.reaction_marginal_sq, dtype=np.float64), dr[:, rr])
    me = np.minimum(np.asarray(state.protein_marginal_sq, dtype=np.float64), de[:, ee])
    return CorrespondenceState(joint, mr, me)

def correspondence_witness(
    reaction_distance_sq: np.ndarray,
    protein_distance_sq: np.ndarray,
    positive_pairs: np.ndarray,
    reaction_index: int,
    protein_index: int,
) -> dict[str, int | float]:
    """Exact geometric witness for one candidate pair.

    Returns the shared positive precedent that realizes J_Omega, the separately
    nearest observed reaction/protein marginal precedents, and the exact defect
    decomposition. This is part of the field definition, not a post-hoc explainer.
    """
    dr = np.asarray(reaction_distance_sq, dtype=np.float64)
    de = np.asarray(protein_distance_sq, dtype=np.float64)
    pairs = np.asarray(positive_pairs, dtype=np.int64)
    r, e = int(reaction_index), int(protein_index)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
        raise ValueError("positive_pairs must be a non-empty [n,2] array")
    costs = dr[r, pairs[:, 0]] + de[e, pairs[:, 1]]
    # Stable deterministic tie handling affects only which equally optimal
    # witness is displayed, never the scalar field.
    best = int(np.lexsort((pairs[:, 1], pairs[:, 0], costs))[0])
    rr, ee = map(int, pairs[best])
    r_support = np.unique(pairs[:, 0])
    e_support = np.unique(pairs[:, 1])
    r_costs = dr[r, r_support]
    e_costs = de[e, e_support]
    rb = int(np.lexsort((r_support, r_costs))[0])
    eb = int(np.lexsort((e_support, e_costs))[0])
    mr = float(r_costs[rb]); me = float(e_costs[eb]); joint = float(costs[best])
    return {
        "shared_reaction_index": rr,
        "shared_protein_index": ee,
        "marginal_reaction_index": int(r_support[rb]),
        "marginal_protein_index": int(e_support[eb]),
        "shared_reaction_cost": float(dr[r, rr]),
        "shared_protein_cost": float(de[e, ee]),
        "joint_cost": joint,
        "reaction_marginal_cost": mr,
        "protein_marginal_cost": me,
        "correspondence_defect": float(max(joint - mr - me, 0.0)),
    }
