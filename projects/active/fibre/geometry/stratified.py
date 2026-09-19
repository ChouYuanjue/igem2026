from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .levelset import numerical_level_tolerance, stable_level_ids


@dataclass(frozen=True)
class StratifiedResolution:
    """Coarse-to-fine resolution of one FIBRE query fibre.

    coarse_level is defined by the canonical global correspondence defect.
    local_sublevel is defined by a catalytic-local correspondence defect only
    inside a coarse level for which every member has that local observation.
    A value of -1 means the coarse level remains unresolved locally.

    The object is therefore a refinement of an equivalence relation, not a
    weighted reranking score.
    """

    coarse_defect: np.ndarray
    local_defect: np.ndarray
    coarse_level: np.ndarray
    local_sublevel: np.ndarray
    coarse_tolerance: float
    local_tolerances: dict[int, float]
    locally_observed_coarse_levels: tuple[int, ...]
    locally_refined_coarse_levels: tuple[int, ...]

    @property
    def scientific_keys(self) -> list[tuple[int, int]]:
        return [
            (int(c), int(l))
            for c, l in zip(self.coarse_level, self.local_sublevel)
        ]

    def display_order(self) -> np.ndarray:
        """Deterministic coarse-to-fine order.

        Cross-coarse-level order can never change. Within an unresolved coarse
        level, original index order is preserved. Within a resolved level,
        local sublevels are ordered and candidate index is only a final display
        tie break.
        """
        baseline=np.argsort(self.coarse_defect,kind="stable")
        out=[]
        refined=set(self.locally_refined_coarse_levels)
        for level in range(int(np.max(self.coarse_level))+1):
            members=[int(i) for i in baseline if int(self.coarse_level[int(i)])==level]
            if level in refined:
                sub=np.asarray([self.local_sublevel[i] for i in members],dtype=np.int64)
                order=np.argsort(sub,kind="stable")
                members=[members[int(k)] for k in order]
            out.extend(members)
        return np.asarray(out,dtype=np.int64)


def stratified_resolution(
    coarse_defect: np.ndarray,
    local_defect: np.ndarray,
    local_available: np.ndarray,
    order_authority: np.ndarray | None = None,
) -> StratifiedResolution:
    """Refine canonical FIBRE numerical levels with catalytic-local geometry.

    Missing local observation is neutral by construction: if any member of one
    coarse numerical level lacks the local coordinate, that whole coarse level
    remains unresolved locally. This prevents differential missingness from
    becoming an implicit penalty.
    """
    coarse=np.asarray(coarse_defect,dtype=np.float64).reshape(-1)
    local=np.asarray(local_defect,dtype=np.float64).reshape(-1)
    available=np.asarray(local_available,dtype=bool).reshape(-1)
    authority=(available.copy() if order_authority is None else np.asarray(order_authority,dtype=bool).reshape(-1))
    if not (len(coarse)==len(local)==len(available)==len(authority)):
        raise ValueError("stratified arrays must have equal length")
    authority &= available
    if not np.all(np.isfinite(coarse)):
        raise ValueError("coarse defect must be finite")

    coarse_level,coarse_tol=stable_level_ids(coarse)
    local_sublevel=np.full(len(coarse),-1,dtype=np.int64)
    local_tolerances={}
    observed=[]
    refined=[]

    for level in range(int(coarse_level.max())+1):
        members=np.flatnonzero(coarse_level==level)
        if len(members)<=1:
            continue
        if not np.all(available[members]):
            continue
        values=local[members]
        if not np.all(np.isfinite(values)):
            continue
        sub,tol=stable_level_ids(values)
        local_sublevel[members]=sub
        local_tolerances[int(level)]=float(tol)
        observed.append(int(level))
        if np.all(authority[members]):
            refined.append(int(level))

    return StratifiedResolution(
        coarse_defect=coarse.copy(),
        local_defect=local.copy(),
        coarse_level=coarse_level,
        local_sublevel=local_sublevel,
        coarse_tolerance=float(coarse_tol),
        local_tolerances=local_tolerances,
        locally_observed_coarse_levels=tuple(observed),
        locally_refined_coarse_levels=tuple(refined),
    )


@dataclass(frozen=True)
class ConsensusStratifiedResolution:
    """Coarse levels refined by Pareto consensus of local FIBRE coordinates.

    Each row of local_defects is one biologically meaningful local realization
    of the same correspondence operator. A candidate dominates another only if
    it is no worse in every jointly observed local coordinate and strictly
    better in at least one. Incomparable candidates remain in the same or an
    overlapping unresolved stratum; no scalar modality weight is introduced.
    """

    coarse_defect: np.ndarray
    coarse_level: np.ndarray
    catalytic_stratum: np.ndarray
    coarse_tolerance: float
    refined_coarse_levels: tuple[int, ...]

    def display_order(self) -> np.ndarray:
        baseline=np.argsort(self.coarse_defect,kind="stable")
        out=[]
        refined=set(self.refined_coarse_levels)
        for level in range(int(np.max(self.coarse_level))+1):
            members=[int(i) for i in baseline if int(self.coarse_level[int(i)])==level]
            if level in refined:
                strata=np.asarray([self.catalytic_stratum[i] for i in members],dtype=np.int64)
                order=np.argsort(strata,kind="stable")
                members=[members[int(k)] for k in order]
            out.extend(members)
        return np.asarray(out,dtype=np.int64)


def consensus_stratified_resolution(
    coarse_defect: np.ndarray,
    local_defects: np.ndarray,
    local_available: np.ndarray,
) -> ConsensusStratifiedResolution:
    """Pareto-consensus refinement of coarse numerical levels.

    local_defects has shape (n_local_coordinates, n_candidates). A coarse level
    is eligible only when every candidate has every local coordinate. Within
    that level, Pareto fronts are computed under coordinate-wise minimization.
    Missingness is therefore exactly neutral and disagreement never requires a
    hand-set fusion weight.
    """
    coarse=np.asarray(coarse_defect,dtype=np.float64).reshape(-1)
    local=np.asarray(local_defects,dtype=np.float64)
    avail=np.asarray(local_available,dtype=bool)
    if local.ndim!=2 or avail.shape!=local.shape or local.shape[1]!=len(coarse):
        raise ValueError("local defect/availability shape mismatch")
    if not np.all(np.isfinite(coarse)):
        raise ValueError("coarse defect must be finite")
    coarse_level,tol=stable_level_ids(coarse)
    strata=np.full(len(coarse),-1,dtype=np.int64)
    refined=[]

    for level in range(int(coarse_level.max())+1):
        members=np.flatnonzero(coarse_level==level)
        if len(members)<=1:
            continue
        if not np.all(avail[:,members]) or not np.all(np.isfinite(local[:,members])):
            continue
        values=local[:,members]
        m=len(members)
        dominates=np.zeros((m,m),dtype=bool)
        for i in range(m):
            for j in range(m):
                if i==j:
                    continue
                no_worse=True
                any_strict=False
                for c in range(values.shape[0]):
                    pair_tol=numerical_level_tolerance(
                        np.asarray([values[c,i],values[c,j]],dtype=np.float64)
                    )
                    if values[c,i] > values[c,j] + pair_tol:
                        no_worse=False
                        break
                    if values[c,i] < values[c,j] - pair_tol:
                        any_strict=True
                dominates[i,j]=no_worse and any_strict

        remaining=set(range(m))
        front=0
        while remaining:
            nondominated=[
                i for i in sorted(remaining)
                if not any(dominates[j,i] for j in remaining if j!=i)
            ]
            if not nondominated:
                raise RuntimeError("Pareto dominance produced a cycle")
            for i in nondominated:
                strata[members[i]]=front
                remaining.remove(i)
            front+=1
        if front>1:
            refined.append(int(level))
        else:
            # One Pareto front means local biology did not resolve this level.
            strata[members]=-1

    return ConsensusStratifiedResolution(
        coarse_defect=coarse.copy(),
        coarse_level=coarse_level,
        catalytic_stratum=strata,
        coarse_tolerance=float(tol),
        refined_coarse_levels=tuple(refined),
    )
