import numpy as np
from projects.active.terpene_screening.extract_enzymecage_esmc_minimal import (
    author_exact_from_full_embedding, build_batches, infer_sequence_indices,
    parse_pocket_residue_ids, residue_aligned_from_full_embedding,
)


def test_parse_pocket_residue_ids_tolerates_serialization_noise() -> None:
    assert parse_pocket_residue_ids("[1, '2', 3.0]") == [1, 2, 3]


def test_offset_one_matches_author_mapping() -> None:
    seq = "ACDEFG"
    records = [(1, "A"), (2, "C"), (3, "D"), (4, "E")]
    indices, audit = infer_sequence_indices(seq, records, [1, 3, 4])
    assert indices == [0, 2, 3]
    assert audit["offset"] == 1
    assert audit["mismatches"] == 0


def test_author_exact_intentionally_keeps_original_bos_index_semantics() -> None:
    full = np.arange(5 * 1152, dtype=np.float32).reshape(5, 1152)  # L=3 plus BOS/EOS
    mean, pocket = author_exact_from_full_embedding(full, 3, [0, 2])
    np.testing.assert_allclose(mean, full.mean(axis=0))
    np.testing.assert_array_equal(pocket, full[[0, 2]])
    corrected_mean, corrected_pocket = residue_aligned_from_full_embedding(full, 3, [0, 2])
    np.testing.assert_allclose(corrected_mean, full[1:-1].mean(axis=0))
    np.testing.assert_array_equal(corrected_pocket, full[[1, 3]])


def test_length_batches_obey_token_and_size_limits() -> None:
    batches = build_batches([("a", "A" * 10), ("b", "A" * 11), ("c", "A" * 50)], 40, 2)
    assert [[x[0] for x in batch] for batch in batches] == [["a", "b"], ["c"]]


def test_infer_sequence_indices_rejects_incompatible_structure_sequence() -> None:
    seq = "AAAA"
    records = [(10, "C"), (11, "D")]
    try:
        infer_sequence_indices(seq, records, [10, 11])
    except ValueError:
        pass
    else:
        raise AssertionError("incompatible structure/sequence mapping must fail")
