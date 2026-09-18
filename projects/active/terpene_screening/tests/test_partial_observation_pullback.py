import numpy as np

from projects.active.terpene_screening.multiscale_geometry import (
    partial_observation_pullback_affinity,
)


def _line_distance(x):
    x=np.asarray(x,dtype=float).reshape(-1,1)
    return np.abs(x-x.T)


def test_missing_optional_view_leaves_base_geometry_exactly_unchanged():
    base=_line_distance([0.0,0.2,0.7,1.5,3.0])
    optional=_line_distance([4.0,3.0,2.0,1.0,0.0])
    absent=np.zeros(5,dtype=bool)
    a,_=partial_observation_pullback_affinity([base],k=4)
    b,_=partial_observation_pullback_affinity([base,optional],[np.ones(5,bool),absent],k=4)
    np.testing.assert_allclose(a.toarray(),b.toarray(),rtol=0,atol=0)


def test_duplicate_identical_view_has_no_availability_bonus():
    base=_line_distance([0.0,0.3,1.0,2.0,4.0,8.0])
    mask=np.ones(6,dtype=bool)
    a,_=partial_observation_pullback_affinity([base],[mask],k=5)
    b,_=partial_observation_pullback_affinity([base,base],[mask,mask],k=5)
    np.testing.assert_allclose(a.toarray(),b.toarray(),rtol=1e-6,atol=1e-7)


def test_positive_rescaling_of_one_view_is_intrinsically_cancelled():
    a=_line_distance([0.0,0.2,0.8,1.1,2.5,5.0])
    b=_line_distance([3.0,1.0,4.0,2.0,8.0,7.0])
    mask=np.ones(6,dtype=bool)
    x,_=partial_observation_pullback_affinity([a,b],[mask,mask],k=5)
    y,_=partial_observation_pullback_affinity([a,17.0*b],[mask,mask],k=5)
    np.testing.assert_allclose(x.toarray(),y.toarray(),rtol=1e-6,atol=1e-7)


def test_optional_view_only_changes_pairs_with_both_endpoints_observed():
    base=_line_distance([0.0,0.4,1.2,2.2,4.0])
    optional=_line_distance([10.0,0.0,5.0,9.0,2.0])
    avail=np.array([False,True,True,False,False])
    a,_=partial_observation_pullback_affinity([base],k=4)
    b,_=partial_observation_pullback_affinity([base,optional],[np.ones(5,bool),avail],k=4)
    aa=a.toarray(); bb=b.toarray()
    # only the jointly observed pair (1,2) can have a changed effective energy.
    mask=np.ones((5,5),dtype=bool)
    mask[1,2]=mask[2,1]=False
    np.testing.assert_allclose(aa[mask],bb[mask],rtol=0,atol=0)


def test_output_is_symmetric_nonnegative_and_missing_pairs_are_absent():
    d=_line_distance([0.0,1.0,2.0,3.0])
    avail=np.array([True,True,False,False])
    a,info=partial_observation_pullback_affinity([d],[avail],k=3)
    x=a.toarray()
    np.testing.assert_allclose(x,x.T)
    assert np.all(x>=0)
    np.testing.assert_allclose(np.diag(x),0)
    assert np.all(x[2:]==0) and np.all(x[:,2:]==0)
    assert info['view_available_counts']==[2]


def test_extra_agreeing_view_can_refine_observed_pair_without_affecting_missing_pairs():
    base=_line_distance([0.0,1.0,2.0,4.0,8.0])
    local=_line_distance([0.0,0.05,4.0,9.0,12.0])
    avail=np.array([True,True,False,False,False])
    a,_=partial_observation_pullback_affinity([base],k=4)
    b,info=partial_observation_pullback_affinity([base,local],[np.ones(5,bool),avail],k=4)
    # The observed local coordinate is actually used rather than ignored.
    assert not np.isclose(a.toarray()[0,1],b.toarray()[0,1])
    assert info['view_available_counts']==[5,2]
