from __future__ import annotations

import unittest

from scripts.catalyst_finder.open_world_inputs import (
    detect_direct_open_world_inputs,
    extract_protein_sequences,
    extract_reaction_smiles,
    stable_protein_query_id,
    stable_reaction_query_id,
    strip_structured_payloads,
)


class OpenWorldInputParserTests(unittest.TestCase):
    def test_bare_reaction_smiles_gets_stable_external_id(self) -> None:
        value = "CCO.O=C=O>>CCOC(=O)O"
        row = extract_reaction_smiles(value)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.reaction_smiles, value)
        self.assertEqual(row.query_id, stable_reaction_query_id(value))
        self.assertTrue(row.query_id.startswith("EXT-RXN-"))

    def test_labeled_reaction_smiles_is_extracted_from_prose(self) -> None:
        row = extract_reaction_smiles("Please score this. Reaction SMILES: CCO>>CC=O")
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.reaction_smiles, "CCO>>CC=O")

    def test_natural_language_arrow_is_not_reaction_smiles(self) -> None:
        self.assertIsNone(extract_reaction_smiles("convert glucose -> pyruvate"))
        self.assertIsNone(extract_reaction_smiles("A → B"))

    def test_fasta_is_extracted_with_header_and_stable_id(self) -> None:
        text = ">my_new_enzyme\nMKTIIALSYIFCLVFADYKDDDDAAAAGGGVVVV\n"
        rows = extract_protein_sequences(text)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.header, "my_new_enzyme")
        self.assertEqual(row.query_id, stable_protein_query_id(row.sequence))
        self.assertTrue(row.query_id.startswith("EXT-PROT-"))

    def test_labeled_multiline_sequence_is_extracted(self) -> None:
        text = "protein sequence:\nMKTIIALSYIFCLVFADYKDDDD\nAAAAGGGVVVVVVVVVVVVVV\nuse this enzyme"
        rows = extract_protein_sequences(text)
        self.assertEqual(len(rows), 1)
        self.assertGreater(len(rows[0].sequence), 20)

    def test_bare_amino_acid_sequence_is_supported(self) -> None:
        sequence = "MKTIIALSYIFCLVFADYKDDDDK"
        rows = extract_protein_sequences(sequence)
        self.assertEqual([row.sequence for row in rows], [sequence])

    def test_ordinary_english_is_not_misclassified_as_protein_sequence(self) -> None:
        self.assertEqual(extract_protein_sequences("Please find possible reactions for this enzyme"), ())

    def test_combined_reaction_and_positive_fasta_are_both_preserved(self) -> None:
        text = "Reaction SMILES: CCO>>CC=O\nKnown active enzyme FASTA:\n>positive\nMKTIIALSYIFCLVFADYKDDDDK"
        parsed = detect_direct_open_world_inputs(text)
        self.assertIsNotNone(parsed.reaction)
        self.assertEqual(len(parsed.protein_sequences), 1)
        self.assertEqual(parsed.protein_sequences[0].header, "positive")

    def test_structured_payload_stripping_preserves_task_prose(self) -> None:
        text = (
            "Find enzymes for converting alcohol to aldehyde.\n"
            "Known active enzyme FASTA:\n>positive\nMKTIIALSYIFCLVFADYKDDDDK\n"
            "Return 10 additional candidates."
        )
        residual = strip_structured_payloads(text)
        self.assertIn("Find enzymes", residual)
        self.assertIn("Return 10 additional candidates", residual)
        self.assertNotIn("MKTIIAL", residual)
        self.assertNotIn(">positive", residual)


if __name__ == "__main__":
    unittest.main()
