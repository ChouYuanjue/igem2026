import numpy as np

from projects.active.fibre.geometry.multiscale import (
    binary_jaccard_cross_distance,
    binary_jaccard_distance,
    multiview_neighbour_union,
    self_tuning_product_affinity,
)


def test_binary_jaccard_marks_zero_rows_missing():
    x = np.asarray([[1, 0, 1], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
    d, available = binary_jaccard_distance(x)
    assert available.tolist() == [True, True, False]
    assert np.isclose(d[0, 1], 2.0 / 3.0)
    assert d[0, 0] == 0.0


def test_product_kernel_is_availability_neutral():
    d1 = np.asarray([[0.0, 0.2, 0.8], [0.2, 0.0, 0.7], [0.8, 0.7, 0.0]])
    d2 = np.asarray([[0.0, 0.1, 0.9], [0.1, 0.0, 0.9], [0.9, 0.9, 0.0]])
    a1 = np.ones(3, dtype=bool)
    a2 = np.asarray([True, True, False])
    graph, diag = self_tuning_product_affinity([d1, d2], [a1, a2], k=2)
    assert diag.view_count == 2
    assert graph.shape == (3, 3)
    assert np.allclose(graph.toarray(), graph.toarray().T)
    assert graph[0, 2] > 0  # view 2 missing at node 2 does not erase view-1 geometry


def test_multiview_neighbour_union_uses_coordinate_atlas():
    d1 = np.asarray([0.01, 0.02, 0.8, 0.9])
    d2 = np.asarray([0.9, 0.8, 0.02, 0.01])
    selected, counts = multiview_neighbour_union(
        [d1, d2], [True, True], [np.ones(4, bool), np.ones(4, bool)]
    )
    # sqrt(4)=2 from each coordinate view: both chemical neighbourhoods survive.
    assert selected.tolist() == [0, 1, 2, 3]
    assert counts == [2, 2]


def test_cross_jaccard_missing_query_is_explicit():
    d, q_ok, r_ok = binary_jaccard_cross_distance(
        np.zeros(3), np.asarray([[1, 0, 0], [0, 0, 0]], dtype=np.float32)
    )
    assert not q_ok
    assert r_ok.tolist() == [True, False]
    assert d.shape == (2,)


def test_self_tuning_product_neighbours_balances_view_scales():
    from projects.active.fibre.geometry.multiscale import self_tuning_product_neighbours

    # First coordinate lives near 1e-2, second near 1.0. Local conformal scaling
    # should prevent the raw numerical scale from deciding the chart.
    d1 = np.asarray([0.01, 0.02, 0.50, 0.60])
    d2 = np.asarray([0.90, 0.80, 0.02, 0.01])
    chosen, info = self_tuning_product_neighbours(
        [d1, d2], [True, True], [np.ones(4, bool), np.ones(4, bool)], k=2
    )
    assert len(chosen) == 2
    assert len(info["view_scales"]) == 2
    assert set(chosen).issubset({0, 1, 2, 3})


def test_information_product_neighbours_prefers_sharp_coordinate():
    from projects.active.fibre.geometry.multiscale import information_product_neighbours
    # View 1 has one distinctly close neighbour; view 2 is almost flat. The
    # adaptive tensor should assign more information to view 1 without labels.
    d1 = np.asarray([0.01, 0.7, 0.8, 0.9, 0.95])
    d2 = np.asarray([0.50, 0.51, 0.49, 0.50, 0.52])
    chosen, info = information_product_neighbours(
        [d1, d2], [True, True], [np.ones(5, bool), np.ones(5, bool)], k=2
    )
    assert info["view_information"][0] > info["view_information"][1]
    assert 0 in chosen


def test_information_product_affinity_is_symmetric_and_finite():
    from projects.active.fibre.geometry.multiscale import information_product_affinity
    d1 = np.asarray([[0,.1,.8],[.1,0,.7],[.8,.7,0]], float)
    d2 = np.asarray([[0,.5,.51],[.5,0,.49],[.51,.49,0]], float)
    g, info = information_product_affinity([d1,d2], [np.ones(3,bool),np.ones(3,bool)], k=2)
    a=g.toarray()
    assert np.all(np.isfinite(a))
    assert np.allclose(a,a.T)
    assert info["undirected_edges"] > 0


def test_perplexity_contraction_grows_with_resolution():
    from projects.active.fibre.geometry.multiscale import _perplexity_contraction_from_energy
    flat = _perplexity_contraction_from_energy(np.zeros(8))
    sharp = _perplexity_contraction_from_energy(np.asarray([0,9,9,9,9,9,9,9],float))
    assert np.isclose(flat, 0.0)
    assert sharp > 5.0


def test_resolution_product_neighbours_preserves_sharp_fine_view():
    from projects.active.fibre.geometry.multiscale import resolution_product_neighbours
    fine=np.asarray([0.01,.8,.85,.9,.95,.97,.98,.99])
    coarse=np.asarray([.45,.44,.43,.42,.41,.40,.39,.38])
    chosen,info=resolution_product_neighbours([coarse,fine],[True,True],[np.ones(8,bool),np.ones(8,bool)],k=2)
    assert info['view_resolution'][1] > info['view_resolution'][0]
    assert 0 in chosen


def test_one_step_diffusion_distance_is_metric_like_and_symmetric():
    from scipy.sparse import csr_matrix
    from projects.active.fibre.geometry.multiscale import one_step_diffusion_distance
    a=csr_matrix(np.asarray([[0,1,0],[1,0,1],[0,1,0]],float))
    d=one_step_diffusion_distance(a)
    assert np.allclose(d,d.T)
    assert np.allclose(np.diag(d),0)
    assert np.all(d>=0)
    # Endpoints have identical one-step transition distributions on a 3-chain.
    assert np.isclose(d[0,2],0.0)


def test_one_step_diffusion_affinity_returns_intrinsic_graph():
    from scipy.sparse import csr_matrix
    from projects.active.fibre.geometry.multiscale import one_step_diffusion_affinity
    a=csr_matrix(np.asarray([[0,1,.1,0],[1,0,1,.1],[.1,1,0,1],[0,.1,1,0]],float))
    g,info=one_step_diffusion_affinity(a,k=2)
    assert g.shape==(4,4)
    assert np.allclose(g.toarray(),g.toarray().T)
    assert info['edges']>0


def test_diffusion_conformal_affinity_preserves_edge_set():
    from scipy.sparse import csr_matrix
    from projects.active.fibre.geometry.multiscale import diffusion_conformal_affinity
    a=csr_matrix(np.asarray([[0,1,.2,0],[1,0,.8,0],[.2,.8,0,1],[0,0,1,0]],float))
    g,info=diffusion_conformal_affinity(a)
    assert g.shape==a.shape
    assert np.allclose(g.toarray(),g.toarray().T)
    assert np.all((g.toarray()>0) <= (a.toarray()>0))
    assert info['edges']==a.nnz//2
    assert 0.0 < info['mean_diffusion_correction'] <= 1.0


def test_resolution_product_partition_conserves_each_atom_and_ignores_missing_view():
    from projects.active.fibre.geometry.multiscale import resolution_product_partition

    ref1=np.asarray([[0,.2,.8],[.2,0,.7],[.8,.7,0]],float)
    ref2=np.asarray([[0,.1,.9],[.1,0,.8],[.9,.8,0]],float)
    cross1=np.asarray([[.1,.3,.9],[.8,.6,.1]],float)
    cross2=np.asarray([[.05,.2,.95],[0.,0.,0.]],float)
    q1=np.ones(2,bool); q2=np.asarray([True,False])
    r1=np.ones(3,bool); r2=np.asarray([True,True,False])
    w,info=resolution_product_partition([cross1,cross2],[ref1,ref2],[q1,q2],[r1,r2])
    assert w.shape==(2,3)
    assert np.all(w>=0)
    assert np.allclose(w.sum(axis=1),1.0)
    # Query 1 has no second-view observation; changing its unavailable values
    # cannot alter the partition.
    cross2b=cross2.copy(); cross2b[1]=[99,17,42]
    wb,_=resolution_product_partition([cross1,cross2b],[ref1,ref2],[q1,q2],[r1,r2])
    assert np.allclose(w[1],wb[1])
    assert len(info['mean_query_view_resolution'])==2


def test_resolution_product_partition_reduces_to_one_view_when_optional_view_missing():
    from projects.active.fibre.geometry.multiscale import resolution_product_partition
    ref=np.asarray([[0,.2,.7],[.2,0,.6],[.7,.6,0]],float)
    cross=np.asarray([[.1,.3,.8]],float)
    zeros=np.zeros_like(ref); zcross=np.zeros_like(cross)
    all3=np.ones(3,bool); q=np.ones(1,bool)
    w1,_=resolution_product_partition([cross],[ref],[q],[all3])
    w2,_=resolution_product_partition([cross,zcross],[ref,zeros],[q,np.zeros(1,bool)],[all3,np.zeros(3,bool)])
    assert np.allclose(w1,w2)


def test_intrinsic_view_information_downweights_flat_view():
    from projects.active.fibre.geometry.multiscale import intrinsic_view_information
    sharp=np.asarray([
        [0,.05,.8,.9],
        [.05,0,.85,.95],
        [.8,.85,0,.1],
        [.9,.95,.1,0],
    ],float)
    flat=np.asarray([
        [0,.5,.51,.49],
        [.5,0,.50,.52],
        [.51,.50,0,.48],
        [.49,.52,.48,0],
    ],float)
    a=np.ones(4,bool)
    qs,_=intrinsic_view_information(sharp,a)
    qf,_=intrinsic_view_information(flat,a)
    assert qs.mean() > qf.mean()


def test_fixed_topology_information_refinement_preserves_edges():
    from scipy.sparse import csr_matrix
    from projects.active.fibre.geometry.multiscale import (
        fixed_topology_information_refinement_affinity,
    )
    base=csr_matrix(np.asarray([
        [0,.9,0,.3],
        [.9,0,.8,0],
        [0,.8,0,.7],
        [.3,0,.7,0],
    ],float))
    local=np.asarray([
        [0,.1,.9,.8],
        [.1,0,.8,.9],
        [.9,.8,0,.1],
        [.8,.9,.1,0],
    ],float)
    refined,info=fixed_topology_information_refinement_affinity(
        base,[local],[np.ones(4,bool)]
    )
    assert info["topology_preserved"]
    assert np.array_equal(refined.toarray()>0,base.toarray()>0)
    assert np.all(refined.toarray()[base.toarray()>0] <= base.toarray()[base.toarray()>0]+1e-7)


def test_fixed_topology_missing_refinement_is_exactly_neutral():
    from scipy.sparse import csr_matrix
    from projects.active.fibre.geometry.multiscale import (
        fixed_topology_information_refinement_affinity,
    )
    base=csr_matrix(np.asarray([
        [0,.9,.2],
        [.9,0,.8],
        [.2,.8,0],
    ],float))
    local=np.zeros((3,3),float)
    refined,info=fixed_topology_information_refinement_affinity(
        base,[local],[np.zeros(3,bool)]
    )
    np.testing.assert_allclose(refined.toarray(),base.toarray(),rtol=1e-6,atol=1e-7)
    assert info["refined_undirected_edges"] == 0


def test_fixed_topology_duplicate_refinement_is_idempotent():
    from scipy.sparse import csr_matrix
    from projects.active.fibre.geometry.multiscale import (
        fixed_topology_information_refinement_affinity,
    )
    base=csr_matrix(np.asarray([
        [0,.9,.2,.1],
        [.9,0,.8,.2],
        [.2,.8,0,.7],
        [.1,.2,.7,0],
    ],float))
    local=np.asarray([
        [0,.1,.8,.9],
        [.1,0,.7,.8],
        [.8,.7,0,.1],
        [.9,.8,.1,0],
    ],float)
    a=np.ones(4,bool)
    one,_=fixed_topology_information_refinement_affinity(base,[local],[a])
    two,_=fixed_topology_information_refinement_affinity(base,[local,local],[a,a])
    np.testing.assert_allclose(one.toarray(),two.toarray(),rtol=1e-6,atol=1e-7)
