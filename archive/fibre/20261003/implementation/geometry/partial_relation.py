from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .levelset import numerical_level_tolerance


@dataclass(frozen=True)
class PartialCorrespondenceRelation:
    """Pareto relation over a fixed FIBRE coordinate family.

    Coordinates are defect-like quantities under minimization. They may be
    global molecular correspondence, catalytic-pocket correspondence, or any
    later validated condition/mechanism coordinate.

    There is no lexicographic priority and no scalar fusion weight. A
    candidate participates in this relation only when every coordinate in the
    declared family is observed. Missingness therefore creates absence of an
    ordering statement, not a penalty.

    front is a diagnostic Pareto decomposition of the complete-case subset.
    It is not a replacement total ranking. Incomplete candidates receive -1.
    """

    defects: np.ndarray
    available: np.ndarray
    complete: np.ndarray
    dominance: np.ndarray
    front: np.ndarray
    coordinate_names: tuple[str, ...]

    @property
    def candidate_count(self) -> int:
        return int(self.defects.shape[1])

    @property
    def coordinate_count(self) -> int:
        return int(self.defects.shape[0])

    @property
    def complete_count(self) -> int:
        return int(np.sum(self.complete))

    @property
    def front_count(self) -> int:
        f=self.front[self.front >= 0]
        return 0 if len(f) == 0 else int(np.max(f)) + 1

    @property
    def front_sizes(self) -> tuple[int, ...]:
        return tuple(
            int(np.sum(self.front == k))
            for k in range(self.front_count)
        )

    def dominates(self, a: int, b: int) -> bool:
        return bool(self.dominance[int(a),int(b)])

    def relation(self, a: int, b: int) -> str:
        """Return the scientific relation between two candidates."""
        a=int(a); b=int(b)
        if a == b:
            return "same_candidate"
        if not (self.complete[a] and self.complete[b]):
            return "unresolved_missing_coordinate"
        if self.dominance[a,b]:
            return "dominates"
        if self.dominance[b,a]:
            return "dominated_by"
        equal=True
        for c in range(self.coordinate_count):
            x=float(self.defects[c,a]); y=float(self.defects[c,b])
            tol=numerical_level_tolerance(np.asarray([x,y],dtype=np.float64))
            if abs(x-y) > tol:
                equal=False
                break
        return "numerically_equal" if equal else "incomparable_tradeoff"


def partial_correspondence_relation(
    defects: np.ndarray,
    available: np.ndarray,
    *,
    coordinate_names: tuple[str, ...] | list[str] | None = None,
) -> PartialCorrespondenceRelation:
    """Construct a weight-free Pareto relation across FIBRE coordinates.

    defects and available have shape [coordinate, candidate]. Dominance is
    evaluated only among candidates complete on the entire declared coordinate
    family. This fixed-domain rule is intentional: pair-dependent coordinate
    intersections can make dominance non-transitive and allow missingness to
    change the comparison rule.
    """
    x=np.asarray(defects,dtype=np.float64)
    a=np.asarray(available,dtype=bool)
    if x.ndim != 2 or a.shape != x.shape:
        raise ValueError("defects/available must be [coordinate,candidate]")
    if x.shape[0] == 0:
        raise ValueError("at least one coordinate is required")
    if coordinate_names is None:
        names=tuple(f"coordinate_{i}" for i in range(x.shape[0]))
    else:
        names=tuple(str(v) for v in coordinate_names)
        if len(names) != x.shape[0]:
            raise ValueError("coordinate_names length mismatch")

    finite=np.isfinite(x)
    complete=np.all(a & finite,axis=0)
    n=x.shape[1]
    dominance=np.zeros((n,n),dtype=bool)
    members=np.flatnonzero(complete)
    m=len(members)

    if m:
        values=x[:,members]
        no_worse=np.ones((m,m),dtype=bool)
        any_strict=np.zeros((m,m),dtype=bool)
        eps64=np.finfo(np.float64).eps
        for c in range(values.shape[0]):
            v=values[c]
            # Exact vectorization of numerical_level_tolerance([vi,vj]).
            scale=np.maximum(
                1.0,
                np.maximum(np.abs(v)[:,None],np.abs(v)[None,:]),
            )
            tol=64.0*eps64*scale
            diff=v[:,None]-v[None,:]
            no_worse &= diff <= tol
            any_strict |= diff < -tol
        local_dom=no_worse & any_strict
        np.fill_diagonal(local_dom,False)
        dominance[np.ix_(members,members)]=local_dom

    front=np.full(n,-1,dtype=np.int64)
    remaining=np.ones(m,dtype=bool)
    level=0
    while np.any(remaining):
        active=np.flatnonzero(remaining)
        sub=dominance[np.ix_(members[active],members[active])]
        incoming=np.any(sub,axis=0)
        nondominated_local=active[~incoming]
        if not len(nondominated_local):
            raise RuntimeError("Pareto dominance produced a cycle")
        front[members[nondominated_local]]=level
        remaining[nondominated_local]=False
        level+=1

    return PartialCorrespondenceRelation(
        defects=x.copy(),
        available=a.copy(),
        complete=complete,
        dominance=dominance,
        front=front,
        coordinate_names=names,
    )
