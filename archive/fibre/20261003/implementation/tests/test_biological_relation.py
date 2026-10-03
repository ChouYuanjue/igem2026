import numpy as np

from projects.active.fibre.evidence.assay_context import ContextAssessment
from projects.active.fibre.geometry.biological_relation import (
    biological_correspondence_relation,
    context_admissible_mask,
)
from projects.active.fibre.geometry.partial_relation import partial_correspondence_relation


def _context(status: str) -> ContextAssessment:
    return ContextAssessment(
        status=status,
        matched_support_observations=("support",) if status=="supported" else (),
        matched_contradiction_observations=("negative",) if status=="contradicted" else (),
        relevant_observation_count=1 if status!="unresolved" else 0,
        requested_dimensions=("ph",),
        interpretation="test",
    )


def test_biological_relation_keeps_support_distance_non_order_bearing():
    molecular=partial_correspondence_relation(
        np.asarray([[0.2,0.3],[0.2,0.1]],dtype=float),
        np.ones((2,2),dtype=bool),
        coordinate_names=("global","catalytic"),
    )
    rel=biological_correspondence_relation(
        molecular,
        np.asarray([100.0,0.1]),
        [_context("unresolved"),_context("unresolved")],
    )
    assert rel.relation(0,1)=="incomparable_tradeoff"
    assert rel.candidate_state(0).support_distance==100.0
    assert "not_activity_probability" in rel.candidate_state(0).to_dict()["support_interpretation"]


def test_explicit_matched_assay_contradiction_has_constraint_authority():
    molecular=partial_correspondence_relation(
        np.asarray([[0.1,0.2]],dtype=float),
        np.ones((1,2),dtype=bool),
        coordinate_names=("global",),
    )
    rel=biological_correspondence_relation(
        molecular,
        np.asarray([0.0,0.0]),
        [_context("contradicted"),_context("unresolved")],
    )
    assert molecular.relation(0,1)=="dominates"
    assert rel.relation(0,1)=="a_excluded_by_matched_assay"


def test_supported_does_not_automatically_dominate_unresolved():
    molecular=partial_correspondence_relation(
        np.asarray([[0.3,0.1]],dtype=float),
        np.ones((1,2),dtype=bool),
        coordinate_names=("global",),
    )
    rel=biological_correspondence_relation(
        molecular,
        np.asarray([0.0,0.0]),
        [_context("supported"),_context("unresolved")],
    )
    assert rel.relation(0,1)=="dominated_by"


def test_context_restriction_is_domain_only_and_composes_with_existing_eligibility():
    contexts=(
        _context("contradicted"),
        _context("supported"),
        _context("conflicting"),
        _context("unresolved"),
    )
    assert context_admissible_mask(contexts).tolist()==[False,True,True,True]
    assert context_admissible_mask(
        contexts,np.asarray([True,True,False,True],dtype=bool)
    ).tolist()==[False,True,False,True]

    molecular=partial_correspondence_relation(
        np.asarray([[0.4,0.3,0.2,0.1]],dtype=float),
        np.ones((1,4),dtype=bool),
        coordinate_names=("global",),
    )
    rel=biological_correspondence_relation(
        molecular,np.zeros(4,dtype=float),contexts
    )
    assert rel.admissible_mask().tolist()==[False,True,True,True]
    assert rel.summary()["context_admissible_count"]==3
    assert rel.summary()["context_excluded_count"]==1
    # Context restriction does not reorder admissible molecular comparisons.
    assert rel.relation(1,3)==molecular.relation(1,3)


def test_conflicting_assay_evidence_stays_unresolved():
    molecular=partial_correspondence_relation(
        np.asarray([[0.1,0.2]],dtype=float),
        np.ones((1,2),dtype=bool),
        coordinate_names=("global",),
    )
    rel=biological_correspondence_relation(
        molecular,
        np.asarray([0.0,0.0]),
        [_context("conflicting"),_context("supported")],
    )
    assert rel.relation(0,1)=="unresolved_conflicting_assay_evidence"
