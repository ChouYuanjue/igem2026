from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from projects.active.terpene_screening.geometry.product_field import (
    _scalar_graph_flux,
    anisotropic_bregman_product_field,
    anisotropic_product_field,
    availability_weighted_similarity,
    axis_transport_diagnostics,
    degree_normalized_affinity,
    masked_symmetric_knn_operator,
    smooth_product_field,
    smooth_product_field_multiview,
    self_tuning_cosine_knn_affinity,
    symmetric_knn_affinity,
    symmetric_knn_operator,
)


def test_degree_normalized_affinity_preserves_missing_vertices_and_symmetry() -> None:
    raw = csr_matrix(
        np.asarray(
            [[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
    )
    normalized = degree_normalized_affinity(raw).toarray()
    assert np.allclose(normalized, normalized.T)
    assert np.all(np.isfinite(normalized))
    assert np.all(normalized >= 0)
    isolated = degree_normalized_affinity(
        csr_matrix(np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    ).toarray()
    assert np.allclose(isolated[2], 0.0)
    assert np.allclose(isolated[:, 2], 0.0)


def test_self_tuning_cosine_affinity_respects_missing_observations() -> None:
    vectors = np.asarray(
        [[1.0, 0.0], [0.98, 0.2], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = vectors @ vectors.T
    graph = self_tuning_cosine_knn_affinity(
        similarity, k=1, available=np.asarray([True, True, False, True])
    )
    dense = graph.toarray()
    assert np.allclose(dense, dense.T)
    assert np.all(dense >= 0)
    assert np.allclose(dense[2], 0.0)
    assert np.allclose(dense[:, 2], 0.0)
    assert dense[0, 1] > 0


def test_zero_steps_is_exact_fallback() -> None:
    similarity = np.eye(3, dtype=np.float32)
    similarity[0, 1] = similarity[1, 0] = 0.8
    similarity[1, 2] = similarity[2, 1] = 0.7
    operator = symmetric_knn_operator(similarity, k=1, temperature=0.1)
    prior = np.arange(9, dtype=np.float32).reshape(3, 3)
    result = smooth_product_field(
        prior,
        operator,
        operator,
        reaction_lambda=1.0,
        protein_lambda=1.0,
        steps=0,
    )
    np.testing.assert_array_equal(result, prior)


def test_product_field_moves_support_to_both_axes() -> None:
    similarity = np.asarray(
        [
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ],
        dtype=np.float32,
    )
    operator = symmetric_knn_operator(similarity, k=1, temperature=0.1)
    prior = np.zeros((3, 3), dtype=np.float32)
    prior[0, 0] = 1.0
    result = smooth_product_field(
        prior,
        operator,
        operator,
        reaction_lambda=0.5,
        protein_lambda=0.5,
        steps=2,
    )
    assert result[1, 0] > 0
    assert result[0, 1] > 0
    assert result[2, 2] >= 0


def test_axis_disagreement_is_finite_and_bounded() -> None:
    similarity = np.asarray([[1.0, 0.8], [0.8, 1.0]], dtype=np.float32)
    operator = symmetric_knn_operator(similarity, k=1, temperature=0.1)
    field = np.asarray([[1.0, 0.0], [0.2, 0.1]], dtype=np.float32)
    _, _, disagreement = axis_transport_diagnostics(field, operator, operator)
    assert np.isfinite(disagreement).all()
    assert float(disagreement.min()) >= 0
    assert float(disagreement.max()) <= 1.0 + 1e-6


def test_missing_optional_view_preserves_base_for_unavailable_pairs() -> None:
    base = np.asarray(
        [[1.0, 0.2, 0.3], [0.2, 1.0, 0.4], [0.3, 0.4, 1.0]], dtype=np.float32
    )
    structure = np.asarray(
        [[1.0, 0.9, 0.8], [0.9, 1.0, 0.7], [0.8, 0.7, 1.0]], dtype=np.float32
    )
    available = np.asarray([True, True, False])
    combined = availability_weighted_similarity(base, [(structure, available, 2.0)])
    assert combined[0, 1] > base[0, 1]
    np.testing.assert_allclose(combined[0, 2], base[0, 2])
    np.testing.assert_allclose(combined[2, 1], base[2, 1])


def test_masked_operator_has_zero_rows_for_missing_structure() -> None:
    similarity = np.asarray(
        [[1.0, 0.8, 0.7], [0.8, 1.0, 0.6], [0.7, 0.6, 1.0]], dtype=np.float32
    )
    operator = masked_symmetric_knn_operator(
        similarity, np.asarray([True, True, False]), k=1, temperature=0.1
    )
    np.testing.assert_allclose(operator.getrow(2).toarray(), 0.0)
    np.testing.assert_allclose(operator.getcol(2).toarray(), 0.0)


def test_multiview_zero_structure_weight_matches_single_view_solver() -> None:
    similarity = np.asarray(
        [[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]], dtype=np.float32
    )
    operator = symmetric_knn_operator(similarity, k=1, temperature=0.1)
    prior = np.arange(9, dtype=np.float32).reshape(3, 3) / 8.0
    expected = smooth_product_field(
        prior,
        operator,
        operator,
        reaction_lambda=0.5,
        protein_lambda=0.25,
        steps=3,
    )
    result = smooth_product_field_multiview(
        prior,
        operator,
        [
            (operator, 0.25, np.ones(3, dtype=np.float32)),
            (operator, 0.0, np.asarray([1.0, 1.0, 0.0], dtype=np.float32)),
        ],
        reaction_lambda=0.5,
        steps=3,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)


def test_anisotropic_zero_strength_is_exact_prior() -> None:
    similarity = np.asarray(
        [[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]], dtype=np.float32
    )
    graph = symmetric_knn_affinity(similarity, k=1, temperature=0.1)
    prior = np.arange(9, dtype=np.float32).reshape(3, 3) / 8.0
    result, _ = anisotropic_product_field(
        prior, graph, [graph], strength=0.0, max_steps=4
    )
    np.testing.assert_array_equal(result, prior)


def test_missing_structure_graph_cannot_modify_unobserved_column_directly() -> None:
    similarity = np.asarray(
        [[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]], dtype=np.float32
    )
    reaction_graph = csr_matrix((3, 3), dtype=np.float32)
    structure_graph = symmetric_knn_affinity(
        similarity,
        k=1,
        temperature=0.1,
        available=np.asarray([True, True, False]),
    )
    prior = np.asarray(
        [[1.0, 0.1, 0.3], [0.2, 0.0, 0.4], [0.0, 0.0, 0.5]], dtype=np.float32
    )
    with_structure, _ = anisotropic_product_field(
        prior,
        reaction_graph,
        [structure_graph],
        strength=0.5,
        max_steps=2,
    )
    zero_structure = csr_matrix(structure_graph.shape, dtype=np.float32)
    without_structure, _ = anisotropic_product_field(
        prior,
        reaction_graph,
        [zero_structure],
        strength=0.5,
        max_steps=2,
    )
    # With every other geometric term removed, the unobserved structural node has
    # no incident edge and therefore cannot move away from the prior.
    np.testing.assert_allclose(with_structure[:, 2], without_structure[:, 2], atol=1e-6)
    np.testing.assert_allclose(with_structure[:, 2], prior[:, 2], atol=1e-6)


def test_anisotropic_irls_does_not_increase_fixed_robust_energy() -> None:
    reaction_similarity = np.asarray(
        [[1.0, 0.9, 0.2], [0.9, 1.0, 0.4], [0.2, 0.4, 1.0]], dtype=np.float32
    )
    protein_similarity = np.asarray(
        [[1.0, 0.7, 0.1], [0.7, 1.0, 0.5], [0.1, 0.5, 1.0]], dtype=np.float32
    )
    reaction_graph = symmetric_knn_affinity(reaction_similarity, k=2, temperature=0.1)
    protein_graph = symmetric_knn_affinity(protein_similarity, k=2, temperature=0.1)
    prior = np.asarray(
        [[0.9, 0.2, 0.0], [0.6, 0.1, 0.4], [0.0, 0.3, 0.8]], dtype=np.float32
    )
    _, diagnostics = anisotropic_product_field(
        prior,
        reaction_graph,
        [protein_graph],
        strength=0.3,
        max_steps=12,
        tolerance=1e-10,
    )
    energy = np.asarray(diagnostics["energy_history"], dtype=np.float64)
    assert len(energy) >= 2
    assert np.all(np.diff(energy) <= 1e-8 * np.maximum(1.0, np.abs(energy[:-1])))


def test_positive_anchor_is_same_energy_not_a_router() -> None:
    prior = np.zeros((2, 2), dtype=np.float32)
    zero = csr_matrix((2, 2), dtype=np.float32)
    weights = np.zeros_like(prior)
    targets = np.zeros_like(prior)
    weights[0, 1] = 1.0
    targets[0, 1] = 1.0
    field, diagnostics = anisotropic_product_field(
        prior,
        zero,
        [zero],
        strength=0.0,
        anchor_weights=weights,
        anchor_targets=targets,
        anchor_strength=3.0,
        max_steps=1,
    )
    # With no geometry, the observation term has the closed-form minimizer
    # (F0 + mu*w*y)/(1 + mu*w). The stable step reaches it in one update.
    assert np.isclose(field[0, 1], 0.75, atol=1e-7)
    np.testing.assert_allclose(field[[0, 1, 1], [0, 0, 1]], 0.0, atol=1e-8)
    assert diagnostics["final_energy_terms"]["anchors"] >= 0.0


def test_positive_measure_adds_only_positive_source_without_negative_labels() -> None:
    prior = np.zeros((2, 3), dtype=np.float32)
    zero_r = csr_matrix((2, 2), dtype=np.float32)
    zero_p = csr_matrix((3, 3), dtype=np.float32)
    relation = np.zeros_like(prior)
    relation[0, 0] = 1.0
    relation[0, 2] = 1.0
    field, diagnostics = anisotropic_product_field(
        prior,
        zero_r,
        [zero_p],
        strength=0.0,
        positive_measure=relation,
        positive_strength=1.0,
        max_steps=1,
    )
    assert field[0, 0] > 0.0
    assert field[0, 2] > 0.0
    assert np.isclose(field[0, 0], field[0, 2])
    assert np.isclose(field[0, 1], 0.0)
    assert np.allclose(field[1], 0.0)
    assert diagnostics["final_energy_terms"]["positive_source"] < 0.0


def test_bregman_field_is_exact_prior_without_positive_observation() -> None:
    similarity = np.asarray(
        [[1.0, 0.9, 0.2], [0.9, 1.0, 0.4], [0.2, 0.4, 1.0]], dtype=np.float32
    )
    graph = degree_normalized_affinity(
        symmetric_knn_affinity(similarity, k=2, temperature=0.1)
    )
    prior = np.asarray(
        [[1.2, 0.1, -0.3], [0.4, 0.8, 0.2], [-0.2, 0.3, 1.0]], dtype=np.float32
    )
    field, diagnostics = anisotropic_bregman_product_field(
        prior, graph, [graph], strength=1.0, positive_measure=np.zeros_like(prior)
    )
    np.testing.assert_array_equal(field, prior)
    assert diagnostics["steps"] == 0
    assert np.isclose(diagnostics["final_energy_terms"]["total"], 0.0, atol=1e-10)


def test_bregman_positive_measure_deforms_and_propagates_with_nonincreasing_energy() -> None:
    similarity = np.asarray([[1.0, 0.9], [0.9, 1.0]], dtype=np.float32)
    graph = degree_normalized_affinity(
        symmetric_knn_affinity(similarity, k=1, temperature=0.1)
    )
    prior = np.asarray([[0.0, 0.5], [0.2, -0.1]], dtype=np.float32)
    measure = np.zeros_like(prior)
    measure[1, 0] = 1.0
    field, diagnostics = anisotropic_bregman_product_field(
        prior,
        graph,
        [graph],
        strength=1.0,
        positive_measure=measure,
        positive_strength=1.0,
        max_steps=32,
        tolerance=1e-12,
    )
    assert field[1, 0] > prior[1, 0]
    assert not np.isclose(field[0, 0], prior[0, 0])
    energy = np.asarray(diagnostics["energy_history"], dtype=np.float64)
    assert np.all(np.diff(energy) <= 1e-8 * np.maximum(1.0, np.abs(energy[:-1])))


def test_anchor_influence_propagates_over_product_geometry() -> None:
    similarity = np.asarray([[1.0, 0.9], [0.9, 1.0]], dtype=np.float32)
    graph = symmetric_knn_affinity(similarity, k=1, temperature=0.1)
    prior = np.zeros((2, 2), dtype=np.float32)
    weights = np.zeros_like(prior)
    targets = np.zeros_like(prior)
    weights[0, 0] = 1.0
    targets[0, 0] = 1.0
    field, diagnostics = anisotropic_product_field(
        prior,
        graph,
        [graph],
        strength=0.5,
        anchor_weights=weights,
        anchor_targets=targets,
        anchor_strength=2.0,
        max_steps=8,
        tolerance=1e-12,
    )
    assert field[0, 0] > 0.0
    assert field[0, 1] > 0.0
    assert field[1, 0] > 0.0
    energy = np.asarray(diagnostics["energy_history"], dtype=np.float64)
    assert np.all(np.diff(energy) <= 1e-8 * np.maximum(1.0, np.abs(energy[:-1])))


def test_scalar_product_edges_have_pair_specific_conductance() -> None:
    graph = csr_matrix(np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    field = np.asarray([[0.0, 0.0], [0.01, 1.0]], dtype=np.float64)
    flux = _scalar_graph_flux(field, graph, axis=0, scale=0.1)
    small_jump_conductance = abs(flux[0, 0] / (field[0, 0] - field[1, 0]))
    large_jump_conductance = abs(flux[0, 1] / (field[0, 1] - field[1, 1]))
    assert small_jump_conductance > 0.9
    assert large_jump_conductance < 0.11
    assert small_jump_conductance > 8.0 * large_jump_conductance
