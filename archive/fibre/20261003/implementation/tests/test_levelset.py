import numpy as np
import pytest

from projects.active.fibre.geometry.correspondence import CorrespondenceSection
from projects.active.fibre.geometry.levelset import (
    numerical_level_tolerance,
    stable_level_ids,
    tie_aware_best_positive_rank,
    section_applicability,
)


def test_stable_level_ids_group_machine_scale_perturbations_only():
    base=np.array([0.0,0.0,1.0,1.0,2.0],dtype=float)
    tol=numerical_level_tolerance(base)
    x=base.copy()
    x[1]+=0.5*tol
    x[3]+=0.5*tol
    ids,got_tol=stable_level_ids(x)
    assert got_tol == pytest.approx(numerical_level_tolerance(x))
    assert ids[0] == ids[1]
    assert ids[2] == ids[3]
    assert ids[0] != ids[2] != ids[4]


def test_tie_aware_rank_uses_neutral_expected_order_not_candidate_id():
    values=np.array([0.0,0.0,0.0,1.0])
    one=tie_aware_best_positive_rank(values,np.array([1]))
    assert one.optimistic_rank == 1
    assert one.expected_rank == pytest.approx(2.0)
    assert one.pessimistic_rank == 3
    assert one.expected_reciprocal_rank == pytest.approx((1+1/2+1/3)/3)

    two=tie_aware_best_positive_rank(values,np.array([1,2]))
    assert two.optimistic_rank == 1
    assert two.expected_rank == pytest.approx(4/3)
    assert two.pessimistic_rank == 2
    assert two.expected_reciprocal_rank == pytest.approx(5/6)


def test_tie_aware_rank_is_invariant_to_candidate_permutation():
    values=np.array([0.0,0.0,0.2,0.2,1.0])
    pos=np.array([1,3])
    a=tie_aware_best_positive_rank(values,pos)
    perm=np.array([4,2,1,0,3])
    inv=np.empty(len(perm),dtype=int);inv[perm]=np.arange(len(perm))
    b=tie_aware_best_positive_rank(values[perm],inv[pos])
    assert a == b


def test_section_applicability_reports_geometry_without_threshold_tier():
    # defect = joint - query marginal - candidate marginal = [0,0,1]
    sec=CorrespondenceSection(
        direction="reaction_to_protein",
        query_index=7,
        candidate_indices=np.array([10,11,12]),
        joint_sq=np.array([5.0,13.0,9.0]),
        query_marginal_sq=4.0,
        candidate_marginal_sq=np.array([1.0,9.0,4.0]),
    )
    x=section_applicability(sec)
    assert x.query_support_distance == pytest.approx(2.0)
    assert x.best_level_size == 2
    assert x.best_level_fraction == pytest.approx(2/3)
    assert x.next_level_gap == pytest.approx(1.0)
    assert x.best_level_candidate_support_distance_min == pytest.approx(1.0)
    assert x.best_level_candidate_support_distance_median == pytest.approx(2.0)
