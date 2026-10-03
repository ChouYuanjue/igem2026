import numpy as np
import pytest

from projects.active.fibre.geometry.correspondence import correspondence_state
from projects.active.fibre.geometry.stability import (
    seed_support_novelty,
    seed_influence,
    section_update_influence,
)


def line_sq(n):
    x=np.arange(n,dtype=float)
    return (x[:,None]-x[None,:])**2


def test_seed_support_novelty_is_intrinsic_product_distance():
    dr=line_sq(5); de=line_sq(6)
    pairs=np.array([[0,0],[4,5]])
    rnov,enov,pnov=seed_support_novelty(dr,de,pairs,2,3)
    assert rnov == pytest.approx(4.0)
    assert enov == pytest.approx(4.0)
    assert pnov == pytest.approx(8.0)


def test_seed_influence_exactly_matches_rebuilt_field_change_counts():
    dr=line_sq(5); de=line_sq(6)
    pairs=np.array([[0,0],[4,5]])
    state=correspondence_state(dr,de,pairs)
    out=seed_influence(state,dr,de,pairs,2,3)

    rebuilt=correspondence_state(
        dr,de,np.vstack([pairs,np.array([[2,3]])])
    )
    delta=rebuilt.defect-state.defect
    assert out.joint_changed_count == int(np.sum(rebuilt.joint_sq < state.joint_sq-1e-12))
    assert out.reaction_marginal_changed_count == int(
        np.sum(rebuilt.reaction_marginal_sq < state.reaction_marginal_sq-1e-12)
    )
    assert out.protein_marginal_changed_count == int(
        np.sum(rebuilt.protein_marginal_sq < state.protein_marginal_sq-1e-12)
    )
    assert out.defect_decreased_count == int(np.sum(delta < -1e-12))
    assert out.defect_increased_count == int(np.sum(delta > 1e-12))
    assert out.defect_unchanged_count == int(np.sum(np.abs(delta) <= 1e-12))
    assert out.affected_pair_fraction == pytest.approx(
        np.mean(np.abs(delta) > 1e-12)
    )


def test_section_update_influence_exactly_describes_before_after_defect():
    before=np.array([0.0,1.0,2.0,3.0,4.0])
    after=np.array([0.0,0.5,2.0,3.5,4.0])
    out=section_update_influence(before,after)
    assert out.candidate_count == 5
    assert out.defect_decreased_count == 1
    assert out.defect_unchanged_count == 3
    assert out.defect_increased_count == 1
    assert out.defect_delta_mean == pytest.approx(0.0)
    assert out.defect_delta_abs_max == pytest.approx(0.5)
    assert out.affected_candidate_fraction == pytest.approx(0.4)


def test_section_update_influence_rejects_shape_or_nonfinite_mismatch():
    with pytest.raises(ValueError):
        section_update_influence(np.array([0.0]),np.array([0.0,1.0]))
    with pytest.raises(ValueError):
        section_update_influence(np.array([0.0,np.inf]),np.array([0.0,1.0]))


def test_seed_can_increase_some_defects_without_violating_exact_update():
    # This is the key stability fact: joint and marginals are all monotone
    # transforms, but their difference need not be monotone everywhere.
    dr=line_sq(4); de=line_sq(4)
    pairs=np.array([[0,0],[3,3]])
    state=correspondence_state(dr,de,pairs)

    found=False
    for r in range(4):
        for e in range(4):
            if any((pairs==[r,e]).all(axis=1)):
                continue
            out=seed_influence(state,dr,de,pairs,r,e)
            if out.defect_increased_count:
                found=True
                break
        if found:
            break
    assert found
