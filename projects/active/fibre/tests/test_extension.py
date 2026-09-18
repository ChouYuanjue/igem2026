import numpy as np
from scipy.sparse import csr_matrix
from projects.active.fibre.geometry.extension import (
    attach_query_to_reference, query_geodesic_to_reference,
)

def test_missing_view_is_neutral_for_query_attachment():
    d1=np.array([0.1,0.2,1.0,1.2])
    d2=np.array([10.0,0.1,0.1,10.0])
    avail=np.ones(4,dtype=bool)
    scale=np.ones(4)
    a=attach_query_to_reference([d1],[True],[avail],[scale],k=2)
    b=attach_query_to_reference([d1,d2],[True,False],[avail,avail],[scale,scale],k=2)
    np.testing.assert_array_equal(a.reference_indices,b.reference_indices)
    np.testing.assert_allclose(a.edge_lengths,b.edge_lengths)

def test_additional_observed_view_can_refine_query_neighbours_without_rebuilding_reference():
    d1=np.array([0.1,0.2,1.0,1.2])
    d2=np.array([2.0,2.0,0.05,0.06])
    avail=np.ones(4,dtype=bool)
    scale=np.ones(4)
    base=attach_query_to_reference([d1],[True],[avail],[scale],k=2)
    refined=attach_query_to_reference([d1,d2],[True,True],[avail,avail],[scale,scale],k=2)
    assert set(base.reference_indices.tolist()) != set(refined.reference_indices.tolist())

def test_query_geodesic_matches_explicit_augmented_graph():
    # Reference path 0--1--2 with lengths 1, 1. Query attaches to 0 at .5 and 2 at 3.
    ref=csr_matrix(np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=float))
    from projects.active.fibre.geometry.extension import QueryAttachment
    att=QueryAttachment(
        reference_indices=np.array([0,2]),
        edge_lengths=np.array([0.5,3.0]),
        query_view_scales=(1.0,),
        observed_view_count=np.array([1,1]),
        graph_k=2,
    )
    got=query_geodesic_to_reference(ref,att)
    np.testing.assert_allclose(got,[0.5,1.5,2.5])

def test_scale_invariance_when_query_and_reference_coordinate_scale_together():
    d=np.array([0.2,0.4,1.0,2.0])
    avail=np.ones(4,dtype=bool)
    rscale=np.array([0.5,0.5,0.7,1.0])
    a=attach_query_to_reference([d],[True],[avail],[rscale],k=3)
    c=7.0
    b=attach_query_to_reference([c*d],[True],[avail],[c*rscale],k=3)
    np.testing.assert_array_equal(a.reference_indices,b.reference_indices)
    np.testing.assert_allclose(a.edge_lengths,b.edge_lengths,rtol=1e-12,atol=1e-12)
