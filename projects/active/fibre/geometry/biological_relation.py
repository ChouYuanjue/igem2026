from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from projects.active.fibre.evidence.assay_context import ContextAssessment
from projects.active.fibre.geometry.partial_relation import PartialCorrespondenceRelation


_CONTEXT_VALUES={"supported","contradicted","conflicting","unresolved"}


@dataclass(frozen=True)
class BiologicalCandidateState:
    """Scientifically interpretable state for one candidate.

    No extra scalar score is introduced. The molecular partial relation,
    absolute support distance, and assay-context constraint remain separate
    because they answer different biological questions.
    """

    candidate_index: int
    molecular_complete: bool
    molecular_front: int
    support_distance: float | None
    context_status: str
    context_relevant_observations: int
    catalytic_state_components: tuple[str,...]

    def to_dict(self) -> dict[str,Any]:
        return {
            "candidate_index":int(self.candidate_index),
            "molecular_complete":bool(self.molecular_complete),
            "molecular_front":int(self.molecular_front),
            "support_distance":(
                None if self.support_distance is None else float(self.support_distance)
            ),
            "support_interpretation":"distance_to_accepted_support_not_activity_probability",
            "context_status":self.context_status,
            "context_relevant_observations":int(self.context_relevant_observations),
            "catalytic_state_components":list(self.catalytic_state_components),
        }


@dataclass(frozen=True)
class BiologicalCorrespondenceRelation:
    """Composite scientific relation without a fused biological score.

    Ordering authority is deliberately narrow:
      * an explicitly matched pair-specific inactive/below-detection assay can
        exclude a candidate under that requested context;
      * otherwise the molecular FIBRE partial relation is retained;
      * contextual support does not automatically beat missing context evidence;
      * absolute support distance is reported as applicability, not used as a
        ranking coordinate.

    This avoids both extremes: pretending assay context is irrelevant, and
    inventing a dense weighted activity model from sparse heterogeneous records.
    """

    molecular: PartialCorrespondenceRelation
    support_distance: np.ndarray
    context: tuple[ContextAssessment,...]
    catalytic_state_components: tuple[tuple[str,...],...]

    def __post_init__(self) -> None:
        d=np.asarray(self.support_distance,dtype=np.float64)
        if d.shape!=(self.molecular.candidate_count,):
            raise ValueError("support_distance must align with molecular candidates")
        if len(self.context)!=self.molecular.candidate_count:
            raise ValueError("context assessments must align with molecular candidates")
        if len(self.catalytic_state_components)!=self.molecular.candidate_count:
            raise ValueError("catalytic-state components must align with molecular candidates")
        for row in self.context:
            if row.status not in _CONTEXT_VALUES:
                raise ValueError(f"unsupported context status: {row.status}")

    @property
    def candidate_count(self) -> int:
        return self.molecular.candidate_count

    def candidate_state(self,index: int) -> BiologicalCandidateState:
        i=int(index)
        value=float(np.asarray(self.support_distance,dtype=np.float64)[i])
        return BiologicalCandidateState(
            candidate_index=i,
            molecular_complete=bool(self.molecular.complete[i]),
            molecular_front=int(self.molecular.front[i]),
            support_distance=value if np.isfinite(value) else None,
            context_status=self.context[i].status,
            context_relevant_observations=int(
                self.context[i].relevant_observation_count
            ),
            catalytic_state_components=tuple(self.catalytic_state_components[i]),
        )

    def relation(self,a: int,b: int) -> str:
        """Return a context-aware scientific relation, not a total rank."""

        a=int(a); b=int(b)
        if a==b:
            return "same_candidate"
        ca=self.context[a].status
        cb=self.context[b].status

        # Explicitly matched negative assay evidence is the only context state
        # that receives exclusion authority. "Supported" never automatically
        # dominates "unresolved", because missing assays are not negative evidence.
        if ca=="contradicted" and cb!="contradicted":
            return "a_excluded_by_matched_assay"
        if cb=="contradicted" and ca!="contradicted":
            return "b_excluded_by_matched_assay"
        if ca=="conflicting" or cb=="conflicting":
            return "unresolved_conflicting_assay_evidence"
        if ca=="contradicted" and cb=="contradicted":
            return "both_excluded_by_matched_assay"

        return self.molecular.relation(a,b)

    def summary(self) -> dict[str,Any]:
        statuses=[x.status for x in self.context]
        d=np.asarray(self.support_distance,dtype=np.float64)
        finite=d[np.isfinite(d)]
        return {
            "candidate_count":self.candidate_count,
            "molecular_coordinate_names":list(self.molecular.coordinate_names),
            "molecular_complete_count":self.molecular.complete_count,
            "context_status_counts":{
                key:int(sum(value==key for value in statuses))
                for key in sorted(_CONTEXT_VALUES)
            },
            "support_distance_finite_count":int(len(finite)),
            "support_distance_min":None if not len(finite) else float(np.min(finite)),
            "support_distance_median":None if not len(finite) else float(np.median(finite)),
            "catalytic_state_component_counts":{
                key:int(sum(key in row for row in self.catalytic_state_components))
                for key in sorted({x for row in self.catalytic_state_components for x in row})
            },
            "ranking_policy":(
                "no fused biological score; explicit matched assay contradiction is a "
                "context constraint, molecular FIBRE remains a partial relation, "
                "support distance is applicability only, and catalytic-state evidence "
                "is descriptive until a separately validated coordinate receives ordering authority"
            ),
        }


def biological_correspondence_relation(
    molecular: PartialCorrespondenceRelation,
    support_distance: np.ndarray,
    context: Iterable[ContextAssessment] | None = None,
    catalytic_state_components: Iterable[Iterable[str]] | None = None,
) -> BiologicalCorrespondenceRelation:
    rows=(
        tuple(context)
        if context is not None
        else tuple(
            ContextAssessment(
                status="unresolved",
                matched_support_observations=(),
                matched_contradiction_observations=(),
                relevant_observation_count=0,
                requested_dimensions=(),
                interpretation="no assay context requested/materialized",
            )
            for _ in range(molecular.candidate_count)
        )
    )
    components=(
        tuple(tuple(str(x) for x in row) for row in catalytic_state_components)
        if catalytic_state_components is not None
        else tuple(() for _ in range(molecular.candidate_count))
    )
    return BiologicalCorrespondenceRelation(
        molecular=molecular,
        support_distance=np.asarray(support_distance,dtype=np.float64).copy(),
        context=rows,
        catalytic_state_components=components,
    )
