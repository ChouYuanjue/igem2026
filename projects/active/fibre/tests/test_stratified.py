import numpy as np

from projects.active.fibre.geometry.stratified import stratified_resolution, consensus_stratified_resolution


def test_stratified_resolution_never_reverses_coarse_levels():
    coarse=np.array([0.0,0.0,1.0,1.0,2.0])
    local=np.array([9.0,0.0,9.0,0.0,0.0])
    available=np.ones(5,bool)
    r=stratified_resolution(coarse,local,available)
    order=r.display_order()
    keys=[r.scientific_keys[int(i)] for i in order]
    assert [k[0] for k in keys] == sorted(k[0] for k in keys)


def test_missing_local_observation_keeps_whole_coarse_level_unresolved():
    coarse=np.array([0.0,0.0,1.0])
    local=np.array([0.0,9.0,0.0])
    available=np.array([True,False,True])
    r=stratified_resolution(coarse,local,available)
    assert r.local_sublevel[0] == -1
    assert r.local_sublevel[1] == -1
    assert 0 not in r.locally_refined_coarse_levels
    np.testing.assert_array_equal(r.display_order(),np.array([0,1,2]))


def test_local_geometry_can_split_only_an_unresolved_coarse_level():
    coarse=np.array([0.0,0.0,0.0,1.0])
    local=np.array([2.0,0.0,1.0,99.0])
    available=np.ones(4,bool)
    r=stratified_resolution(coarse,local,available)
    assert tuple(r.local_sublevel[:3]) == (2,0,1)
    np.testing.assert_array_equal(r.display_order(),np.array([1,2,0,3]))


def test_local_machine_ties_remain_scientifically_tied():
    coarse=np.array([0.0,0.0,1.0])
    local=np.array([1.0,1.0,5.0])
    available=np.ones(3,bool)
    r=stratified_resolution(coarse,local,available)
    assert r.local_sublevel[0] == r.local_sublevel[1]


def test_observed_local_sublevel_can_exist_without_order_authority():
    coarse=np.array([0.0,0.0,1.0])
    local=np.array([0.0,2.0,0.0])
    available=np.ones(3,bool)
    authority=np.array([False,False,True])
    r=stratified_resolution(coarse,local,available,authority)
    assert 0 in r.locally_observed_coarse_levels
    assert 0 not in r.locally_refined_coarse_levels
    assert tuple(r.local_sublevel[:2]) == (0,1)
    np.testing.assert_array_equal(r.display_order(),np.array([0,1,2]))


def test_consensus_strata_require_pareto_agreement_without_weights():
    coarse=np.array([0.0,0.0,0.0,1.0])
    local=np.array([
        [0.0,1.0,2.0,0.0],
        [0.0,2.0,1.0,0.0],
    ])
    avail=np.ones_like(local,dtype=bool)
    r=consensus_stratified_resolution(coarse,local,avail)
    # Candidate 0 dominates both 1 and 2; 1 and 2 disagree and remain same front.
    assert r.catalytic_stratum[0] == 0
    assert r.catalytic_stratum[1] == r.catalytic_stratum[2] == 1
    np.testing.assert_array_equal(r.display_order(),np.array([0,1,2,3]))


def test_consensus_missing_coordinate_keeps_whole_coarse_level_unresolved():
    coarse=np.array([0.0,0.0,1.0])
    local=np.array([[0.0,1.0,0.0],[0.0,2.0,0.0]])
    avail=np.array([[True,True,True],[True,False,True]])
    r=consensus_stratified_resolution(coarse,local,avail)
    assert r.catalytic_stratum[0] == -1
    assert r.catalytic_stratum[1] == -1
    np.testing.assert_array_equal(r.display_order(),np.array([0,1,2]))


def test_consensus_cross_level_order_is_immutable():
    coarse=np.array([0.0,0.0,1.0,1.0])
    local=np.array([[9.0,0.0,9.0,0.0],[8.0,0.0,8.0,0.0]])
    avail=np.ones_like(local,dtype=bool)
    r=consensus_stratified_resolution(coarse,local,avail)
    order=r.display_order()
    assert [int(r.coarse_level[i]) for i in order] == [0,0,1,1]
