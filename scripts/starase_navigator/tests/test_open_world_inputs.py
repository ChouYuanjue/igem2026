from __future__ import annotations

import unittest

from scripts.starase_navigator.open_world_inputs import (
    detect_direct_open_world_inputs,
    extract_protein_sequences,
    explicit_positive_protein_query_ids,
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

    def test_labeled_reaction_smiles_stops_before_attached_natural_language(self) -> None:
        bare = extract_reaction_smiles("CCO>>CC=O")
        chinese = extract_reaction_smiles("针对 Reaction SMILES CCO>>CC=O，给我 10 个候选酶。")
        english = extract_reaction_smiles("Reaction SMILES: CCO>>CC=O, please rank candidates")
        self.assertIsNotNone(bare)
        self.assertIsNotNone(chinese)
        self.assertIsNotNone(english)
        assert bare is not None and chinese is not None and english is not None
        self.assertEqual(chinese.reaction_smiles, "CCO>>CC=O")
        self.assertEqual(english.reaction_smiles, "CCO>>CC=O")
        self.assertEqual(chinese.query_id, bare.query_id)
        self.assertEqual(english.query_id, bare.query_id)

    def test_unlabeled_reaction_smiles_with_chinese_prose_keeps_stable_identity(self) -> None:
        bare = extract_reaction_smiles("CCO>>CC=O")
        wrapped = extract_reaction_smiles("CCO>>CC=O 这个反应数据库里有哪些已记录催化酶？")
        self.assertIsNotNone(bare)
        self.assertIsNotNone(wrapped)
        assert bare is not None and wrapped is not None
        self.assertEqual(wrapped.reaction_smiles, bare.reaction_smiles)
        self.assertEqual(wrapped.query_id, bare.query_id)

    def test_smarts_comma_inside_atom_brackets_is_not_treated_as_prose(self) -> None:
        row = extract_reaction_smiles("Reaction SMARTS: [C,N:1]>>[C:1]")
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.reaction_smiles, "[C,N:1]>>[C:1]")

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

    def test_long_amino_acid_literal_embedded_in_task_prose_is_preserved(self) -> None:
        sequence = "MKTIIALSYIFCLVFADYKDDDDKAAAAGGGVVVVVVVVVVVVVV"
        rows = extract_protein_sequences(sequence + " 这个蛋白可能适合催化什么反应？")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].sequence, sequence)
        self.assertEqual(rows[0].query_id, stable_protein_query_id(sequence))

    def test_multiline_embedded_sequence_can_carry_surrounding_metadata_prose(self) -> None:
        part1 = "MATKAVCVLKGDGPVQGIINFEQKESNGPVKVWGSIKGLTEGL"
        part2 = "HGFHVHEFGDNTAGCTSAGPHFNPLSRKHGGPKDEERHVGDL"
        rows = extract_protein_sequences(
            "new sequence:\n" + part1 + "\n" + part2 + "\nhuman; compare it with the previous target"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].sequence, part1 + part2)

    def test_short_amino_acid_like_words_in_prose_do_not_form_a_sequence(self) -> None:
        self.assertEqual(
            extract_protein_sequences("THIS PROTEIN MAY ACT IN CELL METABOLISM"),
            (),
        )

    def test_numbered_single_residue_layout_is_parsed_as_literal_sequence(self) -> None:
        residues = "MKTIIALSYIFCLVFADYKDDDDKAAAAGGGVVVVVVVVV"
        lines = []
        for start in range(0, len(residues), 10):
            block = residues[start:start + 10]
            lines.append(str(start + len(block)))
            lines.extend(block)
        text = "\n".join(lines) + "\n这是一段从网页复制出的蛋白序列，继续研究它。"
        rows = extract_protein_sequences(text)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].sequence, residues)
        self.assertEqual(rows[0].query_id, stable_protein_query_id(residues))

    def test_ordinary_numbered_prose_is_not_parsed_as_protein_sequence(self) -> None:
        text = "10\nA\n20\nB\n30\nC\n这是普通编号列表"
        self.assertEqual(extract_protein_sequences(text), ())

    def test_combined_reaction_and_positive_fasta_are_both_preserved(self) -> None:
        text = "Reaction SMILES: CCO>>CC=O\nKnown active enzyme FASTA:\n>positive\nMKTIIALSYIFCLVFADYKDDDDK"
        parsed = detect_direct_open_world_inputs(text)
        self.assertIsNotNone(parsed.reaction)
        self.assertEqual(len(parsed.protein_sequences), 1)
        self.assertEqual(parsed.protein_sequences[0].header, "positive")
        self.assertEqual(
            explicit_positive_protein_query_ids(text, parsed.protein_sequences),
            (parsed.protein_sequences[0].query_id,),
        )

    def test_plain_or_hypothetical_sequence_is_not_promoted_to_positive(self) -> None:
        sequence = "MKTIIALSYIFCLVFADYKDDDDKAAAAGGGVVVV"
        neutral = detect_direct_open_world_inputs(sequence + " 这个序列可能催化什么？")
        hypothetical = detect_direct_open_world_inputs(
            "Could this be an active enzyme?\n>" + "query" + "\n" + sequence
        )
        self.assertEqual(
            explicit_positive_protein_query_ids(
                sequence + " 这个序列可能催化什么？",
                neutral.protein_sequences,
            ),
            (),
        )
        self.assertEqual(
            explicit_positive_protein_query_ids(
                "Could this be an active enzyme?\n>query\n" + sequence,
                hypothetical.protein_sequences,
            ),
            (),
        )

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
