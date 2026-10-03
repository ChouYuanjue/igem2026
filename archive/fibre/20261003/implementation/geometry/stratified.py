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
    observed_coarse_levels: tuple[int, ...]
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
    observed=[]
    refined=[]

    for level in range(int(coarse_level.max())+1):
        members=np.flatnonzero(coarse_level==level)
        if len(members)<=1:
            continue
        if not np.all(avail[:,members]) or not np.all(np.isfinite(local[:,members])):
            continue
        observed.append(int(level))
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
        observed_coarse_levels=tuple(observed),
        refined_coarse_levels=tuple(refined),
    )


@dataclass(frozen=True)
class MechanisticChartResolution:
    """Third FIBRE resolution: mechanism charts inside observed catalytic cells.

    mechanistic_chart is an applicability bit mask over family-aware motif
    coordinates. Different masks are different local charts and are therefore
    incomparable. mechanistic_stratum contains Pareto-front indices only when
    all coordinates applicable to one chart are actually observed for the
    whole compared group. Missing observations never become negative evidence.

    This object deliberately has no total-order method: a mechanistic chart is
    a finer scientific relation, not a scalar reranker.
    """

    coarse_level: np.ndarray
    catalytic_stratum: np.ndarray
    mechanistic_chart: np.ndarray
    mechanistic_stratum: np.ndarray
    observed_parent_charts: tuple[tuple[int, int, int], ...]
    refined_parent_charts: tuple[tuple[int, int, int], ...]

    @property
    def refined_candidate_count(self) -> int:
        return int(np.sum(self.mechanistic_stratum >= 0))


def _pareto_fronts(values: np.ndarray) -> np.ndarray:
    """Return Pareto-front indices under coordinate-wise minimization."""
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=2:
        raise ValueError("Pareto values must be [coordinate,candidate]")
    m=x.shape[1]
    dominates=np.zeros((m,m),dtype=bool)
    for i in range(m):
        for j in range(m):
            if i==j:
                continue
            no_worse=True
            any_strict=False
            for c in range(x.shape[0]):
                pair_tol=numerical_level_tolerance(
                    np.asarray([x[c,i],x[c,j]],dtype=np.float64)
                )
                if x[c,i] > x[c,j] + pair_tol:
                    no_worse=False
                    break
                if x[c,i] < x[c,j] - pair_tol:
                    any_strict=True
            dominates[i,j]=no_worse and any_strict

    strata=np.full(m,-1,dtype=np.int64)
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
            strata[i]=front
            remaining.remove(i)
        front+=1
    return strata


def mechanistic_chart_resolution(
    coarse_level: np.ndarray,
    catalytic_stratum: np.ndarray,
    catalytic_observed_coarse_levels: tuple[int, ...] | list[int] | np.ndarray,
    mechanistic_defects: np.ndarray,
    mechanistic_applicable: np.ndarray,
    mechanistic_available: np.ndarray,
) -> MechanisticChartResolution:
    """Resolve family-aware mechanism charts inside catalytic FIBRE cells.

    The protein-side motif family determines which coordinates are applicable.
    A bit mask over applicable coordinates defines the chart. Candidates in
    different charts are incomparable rather than forced onto one numerical
    scale. Within one observed catalytic parent cell and one chart, all
    applicable coordinates must be observed for all members before Pareto
    fronts are defined.

    catalytic_stratum == -1 is allowed only for a coarse level listed in
    catalytic_observed_coarse_levels. It represents an observed catalytic layer
    that produced one unresolved front, so the finer mechanistic layer may
    still add resolution without bypassing a missing catalytic observation.
    """
    coarse=np.asarray(coarse_level,dtype=np.int64).reshape(-1)
    catalytic=np.asarray(catalytic_stratum,dtype=np.int64).reshape(-1)
    defect=np.asarray(mechanistic_defects,dtype=np.float64)
    applicable=np.asarray(mechanistic_applicable,dtype=bool)
    available=np.asarray(mechanistic_available,dtype=bool)
    if len(coarse)!=len(catalytic):
        raise ValueError("coarse/catalytic arrays must have equal length")
    if defect.ndim!=2 or applicable.shape!=defect.shape or available.shape!=defect.shape:
        raise ValueError("mechanistic defect/applicability/availability shape mismatch")
    if defect.shape[1]!=len(coarse):
        raise ValueError("mechanistic candidate axis mismatch")
    if defect.shape[0] > 62:
        raise ValueError("mechanistic chart bit mask supports at most 62 coordinates")

    observed_levels={int(x) for x in catalytic_observed_coarse_levels}
    chart=np.zeros(len(coarse),dtype=np.int64)
    for c in range(defect.shape[0]):
        chart |= applicable[c].astype(np.int64) << c
    strata=np.full(len(coarse),-1,dtype=np.int64)
    observed_parent_charts=[]
    refined_parent_charts=[]

    for level in sorted(observed_levels):
        members=np.flatnonzero(coarse==level)
        if len(members)<=1:
            continue
        nonnegative=np.unique(catalytic[members][catalytic[members]>=0])
        if len(nonnegative):
            parent_groups=[
                (int(parent),members[catalytic[members]==parent])
                for parent in nonnegative
            ]
        else:
            parent_groups=[(-1,members)]

        for parent,parent_members in parent_groups:
            for mask in sorted(int(x) for x in np.unique(chart[parent_members]) if int(x)>0):
                group=parent_members[chart[parent_members]==mask]
                if len(group)<=1:
                    continue
                coords=np.asarray([
                    c for c in range(defect.shape[0]) if mask & (1 << c)
                ],dtype=np.int64)
                if not np.all(available[np.ix_(coords,group)]):
                    continue
                values=defect[np.ix_(coords,group)]
                if not np.all(np.isfinite(values)):
                    continue
                key=(int(level),int(parent),int(mask))
                observed_parent_charts.append(key)
                local=_pareto_fronts(values)
                if int(local.max())<=0:
                    continue
                strata[group]=local
                refined_parent_charts.append(key)

    return MechanisticChartResolution(
        coarse_level=coarse.copy(),
        catalytic_stratum=catalytic.copy(),
        mechanistic_chart=chart,
        mechanistic_stratum=strata,
        observed_parent_charts=tuple(observed_parent_charts),
        refined_parent_charts=tuple(refined_parent_charts),
    )
