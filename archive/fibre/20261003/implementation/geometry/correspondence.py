from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp

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



@dataclass(frozen=True)
class CorrespondenceSection:
    """One exact row or column of the zero-temperature FIBRE field.

    The section stores the same three min-plus transforms as CorrespondenceState
    without materializing the full reaction by protein matrix.
    """
    direction: str
    query_index: int
    candidate_indices: np.ndarray
    joint_sq: np.ndarray
    query_marginal_sq: float
    candidate_marginal_sq: np.ndarray

    @property
    def defect(self) -> np.ndarray:
        out = (
            np.asarray(self.joint_sq, dtype=np.float64)
            - float(self.query_marginal_sq)
            - np.asarray(self.candidate_marginal_sq, dtype=np.float64)
        )
        if not np.all(np.isfinite(out)):
            raise ValueError("correspondence section requires finite connected factor distances")
        return np.maximum(out, 0.0)

    @property
    def field(self) -> np.ndarray:
        return -self.defect


@dataclass(frozen=True)
class ThermalCorrespondenceSection:
    """Finite-temperature FIBRE section with the defect as tropical limit.

    For tau > 0, Gibbs partition functions over Omega and its two factor
    marginals define soft-min free costs C_tau = -tau log Z.  The excess
    C_joint - C_query - C_candidate converges to the canonical correspondence
    defect as tau approaches zero.  Positive temperature is a mathematical
    bridge to kernel or heat realizations, not a promoted replacement for the
    zero-temperature ranking.
    """
    direction: str
    query_index: int
    candidate_indices: np.ndarray
    temperature: float
    joint_free_cost: np.ndarray
    query_marginal_free_cost: float
    candidate_marginal_free_cost: np.ndarray

    @property
    def excess_free_energy(self) -> np.ndarray:
        return (
            np.asarray(self.joint_free_cost, dtype=np.float64)
            - float(self.query_marginal_free_cost)
            - np.asarray(self.candidate_marginal_free_cost, dtype=np.float64)
        )

    @property
    def field(self) -> np.ndarray:
        return -self.excess_free_energy


def _section_inputs(
    reaction_distance_sq: np.ndarray,
    protein_distance_sq: np.ndarray,
    positive_pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    if np.any(pairs[:, 0] < 0) or np.any(pairs[:, 0] >= dr.shape[0]):
        raise ValueError("reaction positive index out of range")
    if np.any(pairs[:, 1] < 0) or np.any(pairs[:, 1] >= de.shape[0]):
        raise ValueError("protein positive index out of range")
    return dr, de, np.unique(pairs, axis=0)


def _section_candidates(size: int, candidates: np.ndarray | None) -> np.ndarray:
    if candidates is None:
        return np.arange(size, dtype=np.int64)
    out = np.asarray(candidates, dtype=np.int64).reshape(-1)
    if np.any(out < 0) or np.any(out >= size):
        raise ValueError("candidate index out of range")
    return out


def _chunk_bounds(size: int, chunk_size: int):
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    for start in range(0, size, chunk_size):
        yield start, min(size, start + chunk_size)


def reaction_to_protein_section(
    reaction_distance_sq: np.ndarray,
    protein_distance_sq: np.ndarray,
    positive_pairs: np.ndarray,
    reaction_index: int,
    *,
    candidate_protein_indices: np.ndarray | None = None,
    candidate_batch_size: int = 4096,
    anchor_batch_size: int = 4096,
) -> CorrespondenceSection:
    """Compute one exact R2E section with bounded peak memory."""
    dr, de, pairs = _section_inputs(
        reaction_distance_sq, protein_distance_sq, positive_pairs
    )
    r = int(reaction_index)
    if not 0 <= r < dr.shape[0]:
        raise ValueError("reaction query index out of range")
    candidates = _section_candidates(de.shape[0], candidate_protein_indices)
    r_support = np.unique(pairs[:, 0])
    p_support, inverse = np.unique(pairs[:, 1], return_inverse=True)
    anchor_cost = np.full(len(p_support), np.inf, dtype=np.float64)
    np.minimum.at(anchor_cost, inverse, dr[r, pairs[:, 0]])
    query_marginal = float(np.min(dr[r, r_support]))
    joint = np.full(len(candidates), np.inf, dtype=np.float64)
    marginal = np.full(len(candidates), np.inf, dtype=np.float64)
    for c0, c1 in _chunk_bounds(len(candidates), int(candidate_batch_size)):
        ids = candidates[c0:c1]
        jb = np.full(len(ids), np.inf, dtype=np.float64)
        mb = np.full(len(ids), np.inf, dtype=np.float64)
        for a0, a1 in _chunk_bounds(len(p_support), int(anchor_batch_size)):
            support = p_support[a0:a1]
            d = de[np.ix_(ids, support)]
            mb = np.minimum(mb, np.min(d, axis=1))
            jb = np.minimum(jb, np.min(d + anchor_cost[a0:a1][None, :], axis=1))
        joint[c0:c1] = jb
        marginal[c0:c1] = mb
    return CorrespondenceSection(
        direction="reaction_to_protein",
        query_index=r,
        candidate_indices=candidates,
        joint_sq=joint,
        query_marginal_sq=query_marginal,
        candidate_marginal_sq=marginal,
    )


def protein_to_reaction_section(
    reaction_distance_sq: np.ndarray,
    protein_distance_sq: np.ndarray,
    positive_pairs: np.ndarray,
    protein_index: int,
    *,
    candidate_reaction_indices: np.ndarray | None = None,
    candidate_batch_size: int = 4096,
    anchor_batch_size: int = 4096,
) -> CorrespondenceSection:
    """Compute one exact E2R section with bounded peak memory."""
    dr, de, pairs = _section_inputs(
        reaction_distance_sq, protein_distance_sq, positive_pairs
    )
    e = int(protein_index)
    if not 0 <= e < de.shape[0]:
        raise ValueError("protein query index out of range")
    candidates = _section_candidates(dr.shape[0], candidate_reaction_indices)
    e_support = np.unique(pairs[:, 1])
    r_support, inverse = np.unique(pairs[:, 0], return_inverse=True)
    anchor_cost = np.full(len(r_support), np.inf, dtype=np.float64)
    np.minimum.at(anchor_cost, inverse, de[e, pairs[:, 1]])
    query_marginal = float(np.min(de[e, e_support]))
    joint = np.full(len(candidates), np.inf, dtype=np.float64)
    marginal = np.full(len(candidates), np.inf, dtype=np.float64)
    for c0, c1 in _chunk_bounds(len(candidates), int(candidate_batch_size)):
        ids = candidates[c0:c1]
        jb = np.full(len(ids), np.inf, dtype=np.float64)
        mb = np.full(len(ids), np.inf, dtype=np.float64)
        for a0, a1 in _chunk_bounds(len(r_support), int(anchor_batch_size)):
            support = r_support[a0:a1]
            d = dr[np.ix_(ids, support)]
            mb = np.minimum(mb, np.min(d, axis=1))
            jb = np.minimum(jb, np.min(d + anchor_cost[a0:a1][None, :], axis=1))
        joint[c0:c1] = jb
        marginal[c0:c1] = mb
    return CorrespondenceSection(
        direction="protein_to_reaction",
        query_index=e,
        candidate_indices=candidates,
        joint_sq=joint,
        query_marginal_sq=query_marginal,
        candidate_marginal_sq=marginal,
    )



def reaction_to_protein_section_from_support_distances(
    query_to_reaction_support_sq: np.ndarray,
    candidate_to_protein_support_sq: np.ndarray,
    positive_pair_support_indices: np.ndarray,
    *,
    query_index: int = -1,
    candidate_indices: np.ndarray | None = None,
) -> CorrespondenceSection:
    """Exact R2E section from distances only to positive marginal supports.

    This is the broad-domain primitive.  It does not require a reaction x
    reaction or protein x protein all-pairs matrix.  The caller may stream any
    candidate batch, provided columns are aligned to the unique positive protein
    support and pair indices address the supplied reaction/protein support axes.
    """
    qr = np.asarray(query_to_reaction_support_sq, dtype=np.float64).reshape(-1)
    cp = np.asarray(candidate_to_protein_support_sq, dtype=np.float64)
    pp = np.asarray(positive_pair_support_indices, dtype=np.int64)
    if cp.ndim != 2:
        raise ValueError("candidate_to_protein_support_sq must be [candidate,support]")
    if pp.ndim != 2 or pp.shape[1] != 2 or len(pp) == 0:
        raise ValueError("positive_pair_support_indices must be non-empty [n,2]")
    if np.any(pp[:, 0] < 0) or np.any(pp[:, 0] >= len(qr)):
        raise ValueError("reaction support index out of range")
    if np.any(pp[:, 1] < 0) or np.any(pp[:, 1] >= cp.shape[1]):
        raise ValueError("protein support index out of range")
    if not (np.all(np.isfinite(qr)) and np.all(np.isfinite(cp))):
        raise ValueError("support distances must be finite")
    if np.any(qr < 0) or np.any(cp < 0):
        raise ValueError("support distances must be non-negative")
    pp = np.unique(pp, axis=0)
    anchor_cost = np.full(cp.shape[1], np.inf, dtype=np.float64)
    np.minimum.at(anchor_cost, pp[:, 1], qr[pp[:, 0]])
    active = np.isfinite(anchor_cost)
    if not np.any(active):
        raise ValueError("positive pairs provide no protein support")
    joint = np.min(cp[:, active] + anchor_cost[active][None, :], axis=1)
    query_marginal = float(np.min(qr[np.unique(pp[:, 0])]))
    candidate_marginal = np.min(cp[:, np.unique(pp[:, 1])], axis=1)
    ids = (
        np.arange(len(cp), dtype=np.int64)
        if candidate_indices is None
        else np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    )
    if len(ids) != len(cp):
        raise ValueError("candidate_indices length must match candidate rows")
    return CorrespondenceSection(
        direction="reaction_to_protein",
        query_index=int(query_index),
        candidate_indices=ids,
        joint_sq=joint,
        query_marginal_sq=query_marginal,
        candidate_marginal_sq=candidate_marginal,
    )


def protein_to_reaction_section_from_support_distances(
    query_to_protein_support_sq: np.ndarray,
    candidate_to_reaction_support_sq: np.ndarray,
    positive_pair_support_indices: np.ndarray,
    *,
    query_index: int = -1,
    candidate_indices: np.ndarray | None = None,
) -> CorrespondenceSection:
    """Exact E2R section from distances only to positive marginal supports."""
    qe = np.asarray(query_to_protein_support_sq, dtype=np.float64).reshape(-1)
    cr = np.asarray(candidate_to_reaction_support_sq, dtype=np.float64)
    pp = np.asarray(positive_pair_support_indices, dtype=np.int64)
    if cr.ndim != 2:
        raise ValueError("candidate_to_reaction_support_sq must be [candidate,support]")
    if pp.ndim != 2 or pp.shape[1] != 2 or len(pp) == 0:
        raise ValueError("positive_pair_support_indices must be non-empty [n,2]")
    if np.any(pp[:, 1] < 0) or np.any(pp[:, 1] >= len(qe)):
        raise ValueError("protein support index out of range")
    if np.any(pp[:, 0] < 0) or np.any(pp[:, 0] >= cr.shape[1]):
        raise ValueError("reaction support index out of range")
    if not (np.all(np.isfinite(qe)) and np.all(np.isfinite(cr))):
        raise ValueError("support distances must be finite")
    if np.any(qe < 0) or np.any(cr < 0):
        raise ValueError("support distances must be non-negative")
    pp = np.unique(pp, axis=0)
    anchor_cost = np.full(cr.shape[1], np.inf, dtype=np.float64)
    np.minimum.at(anchor_cost, pp[:, 0], qe[pp[:, 1]])
    active = np.isfinite(anchor_cost)
    if not np.any(active):
        raise ValueError("positive pairs provide no reaction support")
    joint = np.min(cr[:, active] + anchor_cost[active][None, :], axis=1)
    query_marginal = float(np.min(qe[np.unique(pp[:, 1])]))
    candidate_marginal = np.min(cr[:, np.unique(pp[:, 0])], axis=1)
    ids = (
        np.arange(len(cr), dtype=np.int64)
        if candidate_indices is None
        else np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    )
    if len(ids) != len(cr):
        raise ValueError("candidate_indices length must match candidate rows")
    return CorrespondenceSection(
        direction="protein_to_reaction",
        query_index=int(query_index),
        candidate_indices=ids,
        joint_sq=joint,
        query_marginal_sq=query_marginal,
        candidate_marginal_sq=candidate_marginal,
    )


def _thermal_section(
    dr: np.ndarray,
    de: np.ndarray,
    pairs: np.ndarray,
    *,
    direction: str,
    query_index: int,
    candidates: np.ndarray,
    temperature: float,
    candidate_batch_size: int,
    anchor_batch_size: int,
) -> ThermalCorrespondenceSection:
    tau = float(temperature)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("temperature must be finite and positive")
    r_support = np.unique(pairs[:, 0])
    e_support = np.unique(pairs[:, 1])
    joint = np.empty(len(candidates), dtype=np.float64)
    marginal = np.empty(len(candidates), dtype=np.float64)

    if direction == "reaction_to_protein":
        q = int(query_index)
        query_free = float(-tau * logsumexp(-dr[q, r_support] / tau))
        for c0, c1 in _chunk_bounds(len(candidates), int(candidate_batch_size)):
            ids = candidates[c0:c1]
            log_joint = np.full(len(ids), -np.inf, dtype=np.float64)
            log_marginal = np.full(len(ids), -np.inf, dtype=np.float64)
            for a0, a1 in _chunk_bounds(len(pairs), int(anchor_batch_size)):
                pp = pairs[a0:a1]
                energy = dr[q, pp[:, 0]][None, :] + de[np.ix_(ids, pp[:, 1])]
                log_joint = np.logaddexp(
                    log_joint, logsumexp(-energy / tau, axis=1)
                )
            for a0, a1 in _chunk_bounds(len(e_support), int(anchor_batch_size)):
                support = e_support[a0:a1]
                log_marginal = np.logaddexp(
                    log_marginal,
                    logsumexp(-de[np.ix_(ids, support)] / tau, axis=1),
                )
            joint[c0:c1] = -tau * log_joint
            marginal[c0:c1] = -tau * log_marginal
    elif direction == "protein_to_reaction":
        q = int(query_index)
        query_free = float(-tau * logsumexp(-de[q, e_support] / tau))
        for c0, c1 in _chunk_bounds(len(candidates), int(candidate_batch_size)):
            ids = candidates[c0:c1]
            log_joint = np.full(len(ids), -np.inf, dtype=np.float64)
            log_marginal = np.full(len(ids), -np.inf, dtype=np.float64)
            for a0, a1 in _chunk_bounds(len(pairs), int(anchor_batch_size)):
                pp = pairs[a0:a1]
                energy = dr[np.ix_(ids, pp[:, 0])] + de[q, pp[:, 1]][None, :]
                log_joint = np.logaddexp(
                    log_joint, logsumexp(-energy / tau, axis=1)
                )
            for a0, a1 in _chunk_bounds(len(r_support), int(anchor_batch_size)):
                support = r_support[a0:a1]
                log_marginal = np.logaddexp(
                    log_marginal,
                    logsumexp(-dr[np.ix_(ids, support)] / tau, axis=1),
                )
            joint[c0:c1] = -tau * log_joint
            marginal[c0:c1] = -tau * log_marginal
    else:
        raise ValueError(f"unsupported direction: {direction}")

    return ThermalCorrespondenceSection(
        direction=direction,
        query_index=int(query_index),
        candidate_indices=candidates,
        temperature=tau,
        joint_free_cost=joint,
        query_marginal_free_cost=query_free,
        candidate_marginal_free_cost=marginal,
    )


def thermal_reaction_to_protein_section(
    reaction_distance_sq: np.ndarray,
    protein_distance_sq: np.ndarray,
    positive_pairs: np.ndarray,
    reaction_index: int,
    *,
    temperature: float,
    candidate_protein_indices: np.ndarray | None = None,
    candidate_batch_size: int = 4096,
    anchor_batch_size: int = 4096,
) -> ThermalCorrespondenceSection:
    """Finite-temperature R2E section for the FIBRE kernel/min-plus family."""
    dr, de, pairs = _section_inputs(
        reaction_distance_sq, protein_distance_sq, positive_pairs
    )
    r = int(reaction_index)
    if not 0 <= r < dr.shape[0]:
        raise ValueError("reaction query index out of range")
    candidates = _section_candidates(de.shape[0], candidate_protein_indices)
    return _thermal_section(
        dr, de, pairs,
        direction="reaction_to_protein",
        query_index=r,
        candidates=candidates,
        temperature=temperature,
        candidate_batch_size=candidate_batch_size,
        anchor_batch_size=anchor_batch_size,
    )


def thermal_protein_to_reaction_section(
    reaction_distance_sq: np.ndarray,
    protein_distance_sq: np.ndarray,
    positive_pairs: np.ndarray,
    protein_index: int,
    *,
    temperature: float,
    candidate_reaction_indices: np.ndarray | None = None,
    candidate_batch_size: int = 4096,
    anchor_batch_size: int = 4096,
) -> ThermalCorrespondenceSection:
    """Finite-temperature E2R section for the FIBRE kernel/min-plus family."""
    dr, de, pairs = _section_inputs(
        reaction_distance_sq, protein_distance_sq, positive_pairs
    )
    e = int(protein_index)
    if not 0 <= e < de.shape[0]:
        raise ValueError("protein query index out of range")
    candidates = _section_candidates(dr.shape[0], candidate_reaction_indices)
    return _thermal_section(
        dr, de, pairs,
        direction="protein_to_reaction",
        query_index=e,
        candidates=candidates,
        temperature=temperature,
        candidate_batch_size=candidate_batch_size,
        anchor_batch_size=anchor_batch_size,
    )


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
