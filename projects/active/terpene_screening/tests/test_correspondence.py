import numpy as np
import pytest
from projects.active.terpene_screening.geometry.correspondence import (
    correspondence_state, add_positive_seed,
)

def line_sq(n):
    x=np.arange(n,dtype=float)
    return (x[:,None]-x[None,:])**2

def test_defect_is_nonnegative_and_zero_on_observed_pairs():
    dr=line_sq(4); de=line_sq(5); pairs=np.array([[0,1],[3,4]])
    s=correspondence_state(dr,de,pairs)
    assert np.min(s.defect) >= 0
    assert s.defect[0,1] == pytest.approx(0)
    assert s.defect[3,4] == pytest.approx(0)

def test_factor_swap_equivariance():
    dr=line_sq(4); de=line_sq(5); pairs=np.array([[0,1],[3,4],[2,0]])
    a=correspondence_state(dr,de,pairs).defect
    b=correspondence_state(de,dr,pairs[:,::-1]).defect
    np.testing.assert_allclose(a,b.T,atol=1e-12)

def test_incremental_seed_matches_full_reconstruction_exactly():
    dr=line_sq(5); de=line_sq(6)
    old=np.array([[0,0],[4,5]])
    seed=(2,3)
    s0=correspondence_state(dr,de,old)
    inc=add_positive_seed(s0,dr,de,*seed)
    full=correspondence_state(dr,de,np.vstack([old,np.array(seed)]))
    np.testing.assert_allclose(inc.joint_sq,full.joint_sq,atol=0)
    np.testing.assert_allclose(inc.reaction_marginal_sq,full.reaction_marginal_sq,atol=0)
    np.testing.assert_allclose(inc.protein_marginal_sq,full.protein_marginal_sq,atol=0)
    np.testing.assert_allclose(inc.field,full.field,atol=0)
    assert inc.defect[seed] == pytest.approx(0)

def test_disconnected_infinite_geometry_is_rejected():
    dr=line_sq(3); de=line_sq(3); dr[2,:]=np.inf; dr[:,2]=np.inf; dr[2,2]=0
    with pytest.raises(ValueError):
        correspondence_state(dr,de,np.array([[0,0]]))

def test_correspondence_witness_exactly_decomposes_field():
    from projects.active.terpene_screening.geometry.correspondence import correspondence_witness
    dr=line_sq(5); de=line_sq(6); pairs=np.array([[0,0],[2,4],[4,5]])
    state=correspondence_state(dr,de,pairs)
    r,e=1,3
    w=correspondence_witness(dr,de,pairs,r,e)
    assert w["joint_cost"] == pytest.approx(state.joint_sq[r,e])
    assert w["reaction_marginal_cost"] == pytest.approx(state.reaction_marginal_sq[r])
    assert w["protein_marginal_cost"] == pytest.approx(state.protein_marginal_sq[e])
    assert w["correspondence_defect"] == pytest.approx(state.defect[r,e])
    assert w["shared_reaction_cost"] + w["shared_protein_cost"] == pytest.approx(w["joint_cost"])
