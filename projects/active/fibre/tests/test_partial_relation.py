import numpy as np

from projects.active.fibre.geometry.partial_relation import (
    partial_correspondence_relation,
)


def test_local_coordinate_can_make_global_order_incomparable():
    x=np.array([[0.0,1.0],[5.0,0.0]])
    r=partial_correspondence_relation(x,np.ones_like(x,dtype=bool))
    assert r.relation(0,1)=="incomparable_tradeoff"
    assert r.front[0]==r.front[1]==0


def test_no_coordinate_has_lexicographic_priority():
    x=np.array([
        [0.0,1.0,2.0],
        [3.0,1.0,0.0],
    ])
    r=partial_correspondence_relation(
        x,np.ones_like(x,dtype=bool),
        coordinate_names=("global","pocket"),
    )
    assert r.relation(0,1)=="incomparable_tradeoff"
    assert r.relation(1,2)=="incomparable_tradeoff"
    assert tuple(r.front)==(0,0,0)


def test_dominance_requires_no_worse_in_every_coordinate():
    x=np.array([
        [0.0,1.0,2.0],
        [0.0,1.0,0.5],
    ])
    r=partial_correspondence_relation(x,np.ones_like(x,dtype=bool))
    assert r.dominates(0,1)
    assert r.dominates(0,2)
    assert r.front[0]==0
    assert r.front[1]>0
    assert r.front[2]>0


def test_missing_coordinate_is_unordered_not_penalized():
    x=np.array([[0.0,1.0],[0.0,np.nan]])
    a=np.array([[True,True],[True,False]])
    r=partial_correspondence_relation(x,a)
    assert tuple(r.complete)==(True,False)
    assert r.front[1]==-1
    assert r.relation(0,1)=="unresolved_missing_coordinate"
    assert not r.dominates(0,1)
    assert not r.dominates(1,0)


def test_positive_rescaling_does_not_change_pareto_relation():
    x=np.array([
        [0.0,1.0,2.0,3.0],
        [3.0,1.0,2.0,4.0],
    ])
    a=np.ones_like(x,dtype=bool)
    r1=partial_correspondence_relation(x,a)
    y=x.copy()
    y[0]=7.3*y[0]+11.0
    y[1]=0.002*y[1]-5.0
    r2=partial_correspondence_relation(y,a)
    np.testing.assert_array_equal(r1.dominance,r2.dominance)
    np.testing.assert_array_equal(r1.front,r2.front)


def test_fixed_complete_domain_keeps_dominance_transitive():
    x=np.array([
        [0.0,1.0,2.0],
        [0.0,1.0,2.0],
        [0.0,1.0,2.0],
    ])
    r=partial_correspondence_relation(x,np.ones_like(x,dtype=bool))
    assert r.dominates(0,1)
    assert r.dominates(1,2)
    assert r.dominates(0,2)
    assert tuple(r.front)==(0,1,2)
