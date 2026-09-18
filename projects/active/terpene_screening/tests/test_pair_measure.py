from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from projects.active.terpene_screening.geometry.pair_measure import (
    anisotropic_protein_pushforward,
    anisotropic_reaction_pushforward,
    binary_relation_from_indices,
    conditional_measure,
    normalized_information_support,
    positive_joint_intensity,
)


def test_anisotropic_reaction_pushforward_respects_compatibility_boundary() -> None:
    relation = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    graph = csr_matrix(np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    field = np.asarray([[0.0, 0.0], [0.0, 10.0]], dtype=np.float64)
    transported = anisotropic_reaction_pushforward(
        field, relation, graph, scale=1.0, include_identity=False
    )
    # The compatible coordinate crosses essentially unhindered; the large field
    # discontinuity suppresses transport on the second coordinate continuously.
    assert np.isclose(transported[0, 0], 1.0, atol=1e-7)
    assert 0.0 < transported[0, 1] < 0.2
    assert np.all(transported >= 0)


def test_anisotropic_protein_pushforward_respects_compatibility_boundary() -> None:
    relation = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    graph = csr_matrix(np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    field = np.asarray([[0.0, 0.0], [0.0, 10.0]], dtype=np.float64)
    transported = anisotropic_protein_pushforward(
        field, relation, graph, scale=1.0, include_identity=False
    )
    assert np.isclose(transported[0, 1], 1.0, atol=1e-7)
    assert 0.0 < transported[1, 1] < 0.2
    assert np.all(transported >= 0)


def test_mass_conserving_reaction_pushforward_preserves_total_mass() -> None:
    relation = np.asarray([[0.2, 0.0], [0.3, 0.5]], dtype=np.float64)
    graph = csr_matrix(np.asarray([[0.0, 0.8], [0.8, 0.0]], dtype=np.float32))
    field = np.asarray([[0.0, 0.0], [0.2, 2.0]], dtype=np.float64)
    transported = anisotropic_reaction_pushforward(
        field,
        relation,
        graph,
        scale=0.5,
        include_identity=True,
        mass_conserving=True,
    )
    assert np.all(transported >= 0)
    assert np.isclose(transported.sum(), relation.sum(), atol=1e-12)
    np.testing.assert_allclose(transported.sum(axis=0), relation.sum(axis=0), atol=1e-12)


def test_mass_conserving_protein_pushforward_preserves_total_mass() -> None:
    relation = np.asarray([[0.2, 0.3], [0.0, 0.5]], dtype=np.float64)
    graph = csr_matrix(np.asarray([[0.0, 0.7], [0.7, 0.0]], dtype=np.float32))
    field = np.asarray([[0.0, 0.1], [0.0, 3.0]], dtype=np.float64)
    transported = anisotropic_protein_pushforward(
        field,
        relation,
        graph,
        scale=0.5,
        include_identity=True,
        mass_conserving=True,
    )
    assert np.all(transported >= 0)
    assert np.isclose(transported.sum(), relation.sum(), atol=1e-12)
    np.testing.assert_allclose(transported.sum(axis=1), relation.sum(axis=1), atol=1e-12)


def test_one_joint_measure_gives_both_conditionals() -> None:
    relation = binary_relation_from_indices(
        np.asarray([0, 0, 1]), np.asarray([0, 1, 1]), shape=(2, 3)
    )
    identity_r = csr_matrix(np.eye(2, dtype=np.float32))
    identity_e = csr_matrix(np.eye(3, dtype=np.float32))
    joint = positive_joint_intensity(identity_r, relation, identity_e)
    measure = conditional_measure(joint)
    np.testing.assert_allclose(measure.r2e[0], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(measure.r2e[1], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(measure.e2r[:, 0], [1.0, 0.0])
    np.testing.assert_allclose(measure.e2r[:, 1], [0.5, 0.5])


def test_transport_moves_positive_mass_on_both_axes() -> None:
    relation = binary_relation_from_indices(np.asarray([0]), np.asarray([0]), shape=(2, 2))
    transport = csr_matrix(np.asarray([[0.75, 0.25], [0.25, 0.75]], dtype=np.float32))
    joint = positive_joint_intensity(transport, relation, transport)
    assert joint[0, 1] > 0
    assert joint[1, 0] > 0
    assert joint[1, 1] > 0
    assert np.isclose(joint.sum(), 1.0)


def test_unsupported_conditionals_remain_zero_not_negative() -> None:
    joint = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    measure = conditional_measure(joint)
    np.testing.assert_array_equal(measure.r2e[1], 0.0)
    np.testing.assert_array_equal(measure.e2r[:, 1], 0.0)
    np.testing.assert_array_equal(measure.e2r[:, 2], 0.0)


def test_information_support_is_continuous_and_intrinsic() -> None:
    uniform = np.full((1, 4), 0.25, dtype=np.float64)
    concentrated = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    middle = np.asarray([[0.7, 0.1, 0.1, 0.1]], dtype=np.float64)
    u = normalized_information_support(uniform, axis=0)[0]
    m = normalized_information_support(middle, axis=0)[0]
    c = normalized_information_support(concentrated, axis=0)[0]
    assert np.isclose(u, 0.0, atol=1e-7)
    assert 0.0 < m < 1.0
    assert np.isclose(c, 1.0, atol=1e-7)


def test_product_pushforward_preserves_mass_and_has_no_axis_order():
    from scipy.sparse import csr_matrix
    from projects.active.terpene_screening.geometry.pair_measure import anisotropic_product_pushforward

    field = np.asarray([[0.2, -0.1, 0.4], [0.3, 0.0, -0.2]], dtype=float)
    mass = np.asarray([[0.0, 2.0, 0.0], [1.0, 0.0, 3.0]], dtype=float)
    rg = csr_matrix(np.asarray([[0.0, 0.8], [0.8, 0.0]], dtype=float))
    pg = csr_matrix(np.asarray([[0.0, 0.5, 0.0], [0.5, 0.0, 0.7], [0.0, 0.7, 0.0]], dtype=float))
    out = anisotropic_product_pushforward(field, mass, rg, pg, scale=0.5)
    assert np.all(out >= 0)
    assert np.isclose(out.sum(), mass.sum())

    # Swapping the two product factors and their graphs must only transpose the result.
    swapped = anisotropic_product_pushforward(field.T, mass.T, pg, rg, scale=0.5)
    assert np.allclose(out, swapped.T, atol=1e-12)


def test_product_pushforward_reduces_to_single_axis_pushforward():
    from scipy.sparse import csr_matrix
    from projects.active.terpene_screening.geometry.pair_measure import (
        anisotropic_product_pushforward,
        anisotropic_reaction_pushforward,
    )

    field = np.asarray([[0.1, 0.3], [0.2, -0.1]], dtype=float)
    mass = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=float)
    rg = csr_matrix(np.asarray([[0.0, 0.9], [0.9, 0.0]], dtype=float))
    pg = csr_matrix((2, 2), dtype=float)
    joint = anisotropic_product_pushforward(field, mass, rg, pg, scale=0.7)
    axis = anisotropic_reaction_pushforward(
        field, mass, rg, scale=0.7, include_identity=True, mass_conserving=True
    )
    assert np.allclose(joint, axis, atol=1e-12)


def test_symmetrized_product_pushforward_is_mass_conserving_and_axis_equivariant():
    from scipy.sparse import csr_matrix
    from projects.active.terpene_screening.geometry.pair_measure import symmetrized_anisotropic_product_pushforward

    field=np.asarray([[.2,-.1,.4],[.3,0.,-.2]],float)
    mass=np.asarray([[0.,2.,0.],[1.,0.,3.]],float)
    rg=csr_matrix(np.asarray([[0.,.8],[.8,0.]],float))
    pg=csr_matrix(np.asarray([[0.,.5,0.],[.5,0.,.7],[0.,.7,0.]],float))
    out=symmetrized_anisotropic_product_pushforward(field,mass,rg,pg,scale=.5)
    swapped=symmetrized_anisotropic_product_pushforward(field.T,mass.T,pg,rg,scale=.5)
    assert np.all(out>=0)
    assert np.isclose(out.sum(),mass.sum())
    assert np.allclose(out,swapped.T,atol=1e-12)


def test_second_order_product_pushforward_is_mass_conserving_and_axis_equivariant():
    from scipy.sparse import csr_matrix
    from projects.active.terpene_screening.geometry.pair_measure import second_order_anisotropic_product_pushforward

    field=np.asarray([[.2,-.1,.4],[.3,0.,-.2]],float)
    mass=np.asarray([[0.,2.,0.],[1.,0.,3.]],float)
    rg=csr_matrix(np.asarray([[0.,.8],[.8,0.]],float))
    pg=csr_matrix(np.asarray([[0.,.5,0.],[.5,0.,.7],[0.,.7,0.]],float))
    out=second_order_anisotropic_product_pushforward(field,mass,rg,pg,scale=.5)
    swapped=second_order_anisotropic_product_pushforward(field.T,mass.T,pg,rg,scale=.5)
    assert np.all(out>=0)
    assert np.isclose(out.sum(),mass.sum())
    assert np.allclose(out,swapped.T,atol=1e-12)


def test_product_heat_pushforward_is_mass_conserving_and_axis_equivariant():
    from scipy.sparse import csr_matrix
    from projects.active.terpene_screening.geometry.pair_measure import anisotropic_product_heat_pushforward

    field = np.asarray([[0.2, -0.1, 0.4], [0.3, 0.0, -0.2]], dtype=float)
    mass = np.asarray([[0.0, 2.0, 0.0], [1.0, 0.0, 3.0]], dtype=float)
    rg = csr_matrix(np.asarray([[0.0, 0.8], [0.8, 0.0]], dtype=float))
    pg = csr_matrix(np.asarray([[0.0, 0.5, 0.0], [0.5, 0.0, 0.7], [0.0, 0.7, 0.0]], dtype=float))
    out = anisotropic_product_heat_pushforward(field, mass, rg, pg, scale=0.5)
    swapped = anisotropic_product_heat_pushforward(field.T, mass.T, pg, rg, scale=0.5)
    assert np.all(out >= 0)
    assert np.isclose(out.sum(), mass.sum(), atol=1e-11)
    assert np.allclose(out, swapped.T, atol=1e-11)


def test_product_heat_pushforward_reaches_two_axis_diagonal():
    from scipy.sparse import csr_matrix
    from projects.active.terpene_screening.geometry.pair_measure import anisotropic_product_heat_pushforward

    # From (0,0), a single Cartesian edge cannot reach (1,1), but continuous
    # product heat has positive two-jump probability through either axis order.
    field = np.zeros((2, 2), dtype=float)
    mass = np.zeros((2, 2), dtype=float)
    mass[0, 0] = 1.0
    edge = csr_matrix(np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float))
    out = anisotropic_product_heat_pushforward(field, mass, edge, edge, scale=1.0)
    assert out[1, 1] > 0
    assert np.isclose(out.sum(), 1.0, atol=1e-11)


def test_factor_sum_heat_is_mass_conserving_and_axis_equivariant():
    from scipy.sparse import csr_matrix
    from projects.active.terpene_screening.geometry.pair_measure import anisotropic_factor_sum_heat_pushforward

    field = np.asarray([[0.2, -0.1, 0.4], [0.3, 0.0, -0.2]], dtype=float)
    mass = np.asarray([[0.0, 2.0, 0.0], [1.0, 0.0, 3.0]], dtype=float)
    rg = csr_matrix(np.asarray([[0.0, 0.8], [0.8, 0.0]], dtype=float))
    pg = csr_matrix(np.asarray([[0.0, 0.5, 0.0], [0.5, 0.0, 0.7], [0.0, 0.7, 0.0]], dtype=float))
    out = anisotropic_factor_sum_heat_pushforward(field, mass, rg, pg, scale=0.5)
    swapped = anisotropic_factor_sum_heat_pushforward(field.T, mass.T, pg, rg, scale=0.5)
    assert np.all(out >= 0)
    assert np.isclose(out.sum(), mass.sum(), atol=1e-11)
    assert np.allclose(out, swapped.T, atol=1e-11)


def test_factor_sum_heat_reaches_diagonal_with_one_unit_time_per_factor():
    from scipy.sparse import csr_matrix
    from projects.active.terpene_screening.geometry.pair_measure import anisotropic_factor_sum_heat_pushforward

    field = np.zeros((2, 2), dtype=float)
    mass = np.zeros((2, 2), dtype=float)
    mass[0, 0] = 1.0
    edge = csr_matrix(np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float))
    out = anisotropic_factor_sum_heat_pushforward(field, mass, edge, edge, scale=1.0)
    assert out[1, 1] > 0
    assert np.isclose(out.sum(), 1.0, atol=1e-11)
