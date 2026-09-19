from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

import numpy as np


@dataclass(frozen=True)
class PositiveSupportGroups:
    """Exact grouping of reactions with identical positive protein support.

    reaction_to_group maps each reaction-support row to a group. protein_groups
    stores the unique protein-support indices for each group. Grouping is exact:
    reactions in one group induce the same protein-side nearest-set distance.
    """

    reaction_to_group: np.ndarray
    protein_groups: tuple[np.ndarray, ...]

    @property
    def group_count(self) -> int:
        return len(self.protein_groups)


def group_reactions_by_protein_support(
    positive_pairs: np.ndarray,
    *,
    reaction_support_size: int,
) -> PositiveSupportGroups:
    pairs=np.asarray(positive_pairs,dtype=np.int64)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or len(pairs)==0:
        raise ValueError("positive_pairs must be non-empty [n,2]")
    if np.any(pairs[:,0]<0) or np.any(pairs[:,0]>=reaction_support_size):
        raise ValueError("reaction support index out of range")
    by_reaction: dict[int,set[int]]=defaultdict(set)
    for r,e in pairs:
        by_reaction[int(r)].add(int(e))
    if len(by_reaction)!=reaction_support_size:
        missing=sorted(set(range(reaction_support_size))-set(by_reaction))
        raise ValueError(f"every reaction support row must have positives; missing={missing[:8]}")

    signature_to_group: dict[tuple[int,...],int]={}
    groups: list[np.ndarray]=[]
    reaction_to_group=np.empty(reaction_support_size,dtype=np.int64)
    for r in range(reaction_support_size):
        signature=tuple(sorted(by_reaction[r]))
        g=signature_to_group.get(signature)
        if g is None:
            g=len(groups)
            signature_to_group[signature]=g
            groups.append(np.asarray(signature,dtype=np.int64))
        reaction_to_group[r]=g
    return PositiveSupportGroups(
        reaction_to_group=reaction_to_group,
        protein_groups=tuple(groups),
    )


def collapse_reaction_anchor_costs(
    query_to_reaction_support_sq: np.ndarray,
    reaction_to_group: np.ndarray,
    group_count: int,
) -> np.ndarray:
    q=np.asarray(query_to_reaction_support_sq,dtype=np.float64).reshape(-1)
    mapping=np.asarray(reaction_to_group,dtype=np.int64).reshape(-1)
    if len(q)!=len(mapping):
        raise ValueError("reaction anchor/mapping length mismatch")
    if np.any(mapping<0) or np.any(mapping>=int(group_count)):
        raise ValueError("reaction group index out of range")
    out=np.full(int(group_count),np.inf,dtype=np.float64)
    np.minimum.at(out,mapping,q)
    if not np.all(np.isfinite(out)):
        raise ValueError("every support group must receive a finite reaction anchor")
    return out


def section_from_group_distances(
    query_to_reaction_support_sq: np.ndarray,
    reaction_to_group: np.ndarray,
    candidate_to_group_protein_sq: np.ndarray,
) -> tuple[np.ndarray,np.ndarray,float]:
    """Exact zero-temperature R2E section from grouped protein-set distances.

    Returns (defect, joint_cost, query_marginal). candidate_to_group_protein_sq
    contains g_G(e)=min_{p in G} d_E(e,p)^2 for each exact protein-support
    signature G.
    """
    g=np.asarray(candidate_to_group_protein_sq,dtype=np.float64)
    if g.ndim!=2:
        raise ValueError("candidate_to_group_protein_sq must be [candidate,group]")
    anchors=collapse_reaction_anchor_costs(
        query_to_reaction_support_sq,reaction_to_group,g.shape[1]
    )
    joint=np.min(g+anchors[None,:],axis=1)
    query_marginal=float(np.min(np.asarray(query_to_reaction_support_sq,dtype=np.float64)))
    candidate_marginal=np.min(g,axis=1)
    defect=np.maximum(joint-query_marginal-candidate_marginal,0.0)
    return defect,joint,query_marginal
