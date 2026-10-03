import numpy as np
import pytest
from projects.active.fibre.geometry.correspondence import (
    correspondence_state, add_positive_seed,
    reaction_to_protein_section, protein_to_reaction_section,
    thermal_reaction_to_protein_section, thermal_protein_to_reaction_section,
    reaction_to_protein_section_from_support_distances,
    protein_to_reaction_section_from_support_distances,
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
    from projects.active.fibre.geometry.correspondence import correspondence_witness
    dr=line_sq(5); de=line_sq(6); pairs=np.array([[0,0],[2,4],[4,5]])
    state=correspondence_state(dr,de,pairs)
    r,e=1,3
    w=correspondence_witness(dr,de,pairs,r,e)
    assert w["joint_cost"] == pytest.approx(state.joint_sq[r,e])
    assert w["reaction_marginal_cost"] == pytest.approx(state.reaction_marginal_sq[r])
    assert w["protein_marginal_cost"] == pytest.approx(state.protein_marginal_sq[e])
    assert w["correspondence_defect"] == pytest.approx(state.defect[r,e])
    assert w["shared_reaction_cost"] + w["shared_protein_cost"] == pytest.approx(w["joint_cost"])


def test_exact_sections_match_full_field_with_small_batches():
    dr=line_sq(6); de=line_sq(7)
    pairs=np.array([[0,1],[1,1],[2,4],[5,6],[4,2]])
    full=correspondence_state(dr,de,pairs)

    for r in range(len(dr)):
        sec=reaction_to_protein_section(
            dr,de,pairs,r,candidate_batch_size=2,anchor_batch_size=2,
        )
        np.testing.assert_allclose(sec.joint_sq,full.joint_sq[r],atol=0)
        np.testing.assert_allclose(sec.defect,full.defect[r],atol=0)
        np.testing.assert_allclose(sec.field,full.field[r],atol=0)

    for e in range(len(de)):
        sec=protein_to_reaction_section(
            dr,de,pairs,e,candidate_batch_size=2,anchor_batch_size=2,
        )
        np.testing.assert_allclose(sec.joint_sq,full.joint_sq[:,e],atol=0)
        np.testing.assert_allclose(sec.defect,full.defect[:,e],atol=0)
        np.testing.assert_allclose(sec.field,full.field[:,e],atol=0)


def test_exact_sections_preserve_candidate_subset_order():
    dr=line_sq(5); de=line_sq(6); pairs=np.array([[0,0],[2,3],[4,5]])
    full=correspondence_state(dr,de,pairs)
    pids=np.array([5,1,4,0])
    rsec=reaction_to_protein_section(
        dr,de,pairs,2,candidate_protein_indices=pids,
        candidate_batch_size=1,anchor_batch_size=1,
    )
    np.testing.assert_array_equal(rsec.candidate_indices,pids)
    np.testing.assert_allclose(rsec.field,full.field[2,pids],atol=0)

    rids=np.array([4,0,3])
    esec=protein_to_reaction_section(
        dr,de,pairs,3,candidate_reaction_indices=rids,
        candidate_batch_size=1,anchor_batch_size=1,
    )
    np.testing.assert_array_equal(esec.candidate_indices,rids)
    np.testing.assert_allclose(esec.field,full.field[rids,3],atol=0)


def test_thermal_family_converges_to_zero_temperature_sections():
    dr=line_sq(6); de=line_sq(7)
    pairs=np.array([[0,1],[1,1],[2,4],[5,6],[4,2]])
    exact_r=reaction_to_protein_section(dr,de,pairs,3)
    exact_e=protein_to_reaction_section(dr,de,pairs,5)

    thermal_r=thermal_reaction_to_protein_section(
        dr,de,pairs,3,temperature=1e-6,
        candidate_batch_size=2,anchor_batch_size=2,
    )
    thermal_e=thermal_protein_to_reaction_section(
        dr,de,pairs,5,temperature=1e-6,
        candidate_batch_size=2,anchor_batch_size=2,
    )
    np.testing.assert_allclose(
        thermal_r.excess_free_energy,exact_r.defect,atol=5e-6,rtol=0,
    )
    np.testing.assert_allclose(
        thermal_e.excess_free_energy,exact_e.defect,atol=5e-6,rtol=0,
    )


def test_thermal_family_treats_duplicate_positive_rows_as_one_observation():
    dr=line_sq(5); de=line_sq(6)
    unique=np.array([[0,0],[2,3],[4,5]])
    duplicate=np.vstack([unique,unique[[1]],unique[[1]]])
    a=thermal_reaction_to_protein_section(dr,de,unique,1,temperature=0.5)
    b=thermal_reaction_to_protein_section(dr,de,duplicate,1,temperature=0.5)
    np.testing.assert_allclose(a.excess_free_energy,b.excess_free_energy,atol=1e-12)


def test_support_distance_sections_match_full_reference():
    dr=line_sq(7); de=line_sq(8)
    pairs=np.array([[0,1],[1,1],[2,4],[5,6],[6,7],[4,2]])
    full=correspondence_state(dr,de,pairs)

    r_support=np.unique(pairs[:,0])
    p_support=np.unique(pairs[:,1])
    rpos={x:i for i,x in enumerate(r_support)}
    ppos={x:i for i,x in enumerate(p_support)}
    support_pairs=np.asarray(
        [(rpos[int(r)],ppos[int(e)]) for r,e in pairs],dtype=int
    )

    r=3
    pids=np.array([7,0,5,2])
    sec_r=reaction_to_protein_section_from_support_distances(
        dr[r,r_support],
        de[np.ix_(pids,p_support)],
        support_pairs,
        query_index=r,
        candidate_indices=pids,
    )
    np.testing.assert_allclose(sec_r.field,full.field[r,pids],atol=0)

    e=5
    rids=np.array([6,0,4,1])
    sec_e=protein_to_reaction_section_from_support_distances(
        de[e,p_support],
        dr[np.ix_(rids,r_support)],
        support_pairs,
        query_index=e,
        candidate_indices=rids,
    )
    np.testing.assert_allclose(sec_e.field,full.field[rids,e],atol=0)


def test_support_distance_sections_can_be_streamed_in_candidate_batches():
    dr=line_sq(6); de=line_sq(9)
    pairs=np.array([[0,0],[2,3],[4,5],[5,8]])
    full=correspondence_state(dr,de,pairs)
    r_support=np.unique(pairs[:,0]); p_support=np.unique(pairs[:,1])
    rpos={x:i for i,x in enumerate(r_support)}
    ppos={x:i for i,x in enumerate(p_support)}
    support_pairs=np.asarray([(rpos[int(r)],ppos[int(e)]) for r,e in pairs],dtype=int)

    pieces=[]
    for ids in (np.arange(0,3),np.arange(3,7),np.arange(7,9)):
        part=reaction_to_protein_section_from_support_distances(
            dr[1,r_support],de[np.ix_(ids,p_support)],support_pairs,
            query_index=1,candidate_indices=ids,
        )
        pieces.append(part.field)
    np.testing.assert_allclose(np.concatenate(pieces),full.field[1],atol=0)
