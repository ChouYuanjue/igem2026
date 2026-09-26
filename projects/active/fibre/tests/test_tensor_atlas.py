import numpy as np
import pytest

from projects.active.fibre.geometry.tensor_atlas import TensorAtlasField, restrict_positive_relation


def test_tensor_restriction_conserves_unique_positive_mass() -> None:
    pr = np.asarray([[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]])
    pe = np.asarray([[0.6, 0.4], [0.0, 1.0], [1.0, 0.0]])
    pairs = np.asarray([[0, 0], [1, 1], [1, 1], [2, 2]])
    coarse = restrict_positive_relation(pr, pe, pairs)
    assert np.isclose(coarse.sum(), 3.0)
    probability = restrict_positive_relation(pr, pe, pairs, normalize=True)
    assert np.isclose(probability.sum(), 1.0)


def test_tensor_field_sections_are_exactly_one_scalar_field() -> None:
    pr = np.asarray([[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]])
    pe = np.asarray([[0.6, 0.4], [0.0, 1.0], [1.0, 0.0]])
    delta = np.asarray([[0.2, -0.3], [0.7, 0.5]])
    field = TensorAtlasField(pr, pe, delta)
    full = pr @ delta @ pe.T
    for r in range(len(pr)):
        np.testing.assert_allclose(field.reaction_to_protein_section(r), full[r])
    for e in range(len(pe)):
        np.testing.assert_allclose(field.protein_to_reaction_section(e), full[:, e])
    pairs = np.asarray([(r, e) for r in range(len(pr)) for e in range(len(pe))])
    np.testing.assert_allclose(field.overlap_defect(pairs), 0.0, atol=1e-14)


def test_tensor_field_subset_contractions_match_point_values() -> None:
    rng = np.random.default_rng(7)
    pr = rng.random((7, 3)); pr /= pr.sum(axis=1, keepdims=True)
    pe = rng.random((9, 4)); pe /= pe.sum(axis=1, keepdims=True)
    delta = rng.normal(size=(3, 4))
    field = TensorAtlasField(pr, pe, delta)
    eids = np.asarray([1, 4, 8]); rids = np.asarray([0, 3, 6])
    np.testing.assert_allclose(field.reaction_to_protein_section(2, eids), [field.value(2, int(e)) for e in eids])
    np.testing.assert_allclose(field.protein_to_reaction_section(5, rids), [field.value(int(r), 5) for r in rids])


def test_tensor_field_rejects_non_partition_inputs() -> None:
    with pytest.raises(ValueError):
        TensorAtlasField(np.asarray([[0.2, 0.2]]), np.asarray([[1.0]]), np.zeros((2, 1)))
