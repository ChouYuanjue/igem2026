import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from projects.active.fibre.geometry.correspondence import (
    protein_to_reaction_section,
    reaction_to_protein_section,
)
from projects.active.fibre.geometry.geodesic_section import (
    protein_to_reaction_geodesic_section,
    reaction_to_protein_geodesic_section,
    support_geodesic_marginal,
    weighted_multisource_geodesic,
)


def _path_graph(weights):
    n = len(weights) + 1
    a = np.zeros((n, n), dtype=float)
    for i, w in enumerate(weights):
        a[i, i + 1] = a[i + 1, i] = float(w)
    return csr_matrix(a)


def test_weighted_multisource_matches_dense_lower_envelope():
    g = _path_graph([0.3, 0.7, 0.2, 1.1])
    dense = np.asarray(shortest_path(g, directed=False), dtype=float)
    sources = np.asarray([0, 2, 2, 4])
    costs = np.asarray([0.8, 0.4, 0.1, 0.6])
    expected = np.min(dense[:, sources] + costs[None, :], axis=1)
    actual = weighted_multisource_geodesic(g, sources, costs)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_geodesic_sections_equal_dense_correspondence_sections():
    rg = _path_graph([0.4, 0.9, 0.2])
    pg = _path_graph([0.5, 0.1, 0.8, 0.3])
    dr = np.asarray(shortest_path(rg, directed=False), dtype=float)
    de = np.asarray(shortest_path(pg, directed=False), dtype=float)
    pairs = np.asarray([[0, 0], [1, 2], [1, 2], [3, 4]], dtype=np.int64)

    pm = support_geodesic_marginal(pg, np.unique(pairs[:, 1]))
    rm = support_geodesic_marginal(rg, np.unique(pairs[:, 0]))

    for q in range(rg.shape[0]):
        dense = reaction_to_protein_section(dr, de, pairs, q)
        sparse = reaction_to_protein_geodesic_section(rg, pg, pairs, q, protein_marginal_cost=pm)
        np.testing.assert_array_equal(sparse.candidate_indices, dense.candidate_indices)
        np.testing.assert_allclose(sparse.joint_sq, dense.joint_sq, rtol=0, atol=1e-12)
        np.testing.assert_allclose(sparse.candidate_marginal_sq, dense.candidate_marginal_sq, rtol=0, atol=1e-12)
        assert sparse.query_marginal_sq == dense.query_marginal_sq
        np.testing.assert_allclose(sparse.defect, dense.defect, rtol=0, atol=1e-12)

    for q in range(pg.shape[0]):
        dense = protein_to_reaction_section(dr, de, pairs, q)
        sparse = protein_to_reaction_geodesic_section(rg, pg, pairs, q, reaction_marginal_cost=rm)
        np.testing.assert_array_equal(sparse.candidate_indices, dense.candidate_indices)
        np.testing.assert_allclose(sparse.joint_sq, dense.joint_sq, rtol=0, atol=1e-12)
        np.testing.assert_allclose(sparse.candidate_marginal_sq, dense.candidate_marginal_sq, rtol=0, atol=1e-12)
        assert sparse.query_marginal_sq == dense.query_marginal_sq
        np.testing.assert_allclose(sparse.defect, dense.defect, rtol=0, atol=1e-12)


def test_geodesic_sections_preserve_requested_candidate_order():
    rg = _path_graph([0.4, 0.9, 0.2])
    pg = _path_graph([0.5, 0.1, 0.8, 0.3])
    pairs = np.asarray([[0, 0], [1, 2], [3, 4]], dtype=np.int64)
    rsel = np.asarray([4, 1, 3], dtype=np.int64)
    esel = np.asarray([3, 0, 2], dtype=np.int64)

    r2e = reaction_to_protein_geodesic_section(rg, pg, pairs, 2, candidate_protein_indices=rsel)
    e2r = protein_to_reaction_geodesic_section(rg, pg, pairs, 1, candidate_reaction_indices=esel)
    np.testing.assert_array_equal(r2e.candidate_indices, rsel)
    np.testing.assert_array_equal(e2r.candidate_indices, esel)
