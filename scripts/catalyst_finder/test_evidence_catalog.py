from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from scripts.catalyst_finder.evidence_catalog import IntegratedEvidenceCatalog


def test_merged_evidence_catalog_resolves_aliases_bidirectionally(tmp_path: Path):
    merged = tmp_path / "data/catalyst_candidate_universes/general_merged"
    merged.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "protein_id": "P_CANON",
                "canonical_accession": "P_CANON",
                "aliases": "P_CANON;P_ALIAS",
                "source_layer": "project_current",
                "evidence_scope": "candidate",
            }
        ]
    ).to_csv(merged / "protein_metadata.csv", index=False)
    pd.DataFrame(
        [
            {
                "protein_id": "P_CANON",
                "reaction_id": "RHEA:12345",
                "source": "uniprot_rhea_cached",
                "evidence_type": "recorded_association",
            }
        ]
    ).to_csv(merged / "associations.csv", index=False)

    catalog = IntegratedEvidenceCatalog(tmp_path)
    assert catalog.canonical_protein_id("p_alias") == "P_CANON"
    assert {row.protein_id for row in catalog.known_proteins("12345")} == {"P_CANON"}
    assert {row.reaction_id for row in catalog.known_reactions("P_ALIAS")} == {"RHEA:12345"}
    assert catalog.summary()["recorded_associations"] == 1


def test_evidence_catalog_falls_back_to_cached_uniprot_rhea_without_merged_artifact(tmp_path: Path):
    source = tmp_path / "data/external/reactzyme"
    source.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "Entry": "P11111",
                "EC number": "1.1.1.1",
                "Rhea ID": "RHEA:12345;RHEA:54321",
                "Date of creation": "20200101",
                "Sequence": "MPEPTIDE",
            }
        ]
    ).to_csv(source / "cleaned_uniprot_rhea.tsv", sep="\t", index=False)

    catalog = IntegratedEvidenceCatalog(tmp_path)
    assert {row.reaction_id for row in catalog.known_reactions("p11111")} == {
        "RHEA:12345",
        "RHEA:54321",
    }
    assert {row.protein_id for row in catalog.known_proteins("RHEA:54321")} == {"P11111"}


def test_model_readiness_is_not_part_of_evidence_identity(tmp_path: Path):
    merged = tmp_path / "data/catalyst_candidate_universes/general_merged"
    merged.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "protein_id": "P_EXTERNAL",
                "canonical_accession": "P_EXTERNAL",
                "aliases": "P_EXTERNAL",
                "source_layer": "uniprot_rhea_general",
                "evidence_scope": "database_record",
                "model_ready": "False",
            }
        ]
    ).to_csv(merged / "protein_metadata.csv", index=False)
    pd.DataFrame(
        [
            {
                "protein_id": "P_EXTERNAL",
                "reaction_id": "RHEA:22222",
                "source": "database",
                "evidence_type": "recorded_association",
            }
        ]
    ).to_csv(merged / "associations.csv", index=False)

    catalog = IntegratedEvidenceCatalog(tmp_path)
    rows = catalog.known_proteins("RHEA:22222")
    assert len(rows) == 1
    assert rows[0].protein_id == "P_EXTERNAL"


def test_raw_sequence_and_reaction_smiles_can_reconnect_to_existing_candidates(tmp_path: Path):
    merged = tmp_path / "data/catalyst_candidate_universes/general_merged"
    merged.mkdir(parents=True)
    sequence = "MKTIIALSYIFCLVFADYKDDDDK"
    sequence_sha = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    pd.DataFrame(
        [
            {
                "protein_id": "P_EXISTING",
                "canonical_accession": "P_EXISTING",
                "aliases": "P_EXISTING;P_ALIAS",
                "source_layer": "general",
                "evidence_scope": "candidate",
                "sequence_sha256": sequence_sha,
            }
        ]
    ).to_csv(merged / "protein_metadata.csv", index=False)
    pd.DataFrame(
        [
            {
                "reaction_id": "RHEA:12345",
                "reaction_smiles": "CCO>>CC=O",
                "source_layer": "general",
            }
        ]
    ).to_csv(merged / "reactions.csv", index=False)
    pd.DataFrame(columns=["protein_id", "reaction_id", "source", "evidence_type"]).to_csv(
        merged / "associations.csv", index=False
    )

    catalog = IntegratedEvidenceCatalog(tmp_path)
    assert catalog.candidate_protein_for_sequence(sequence) == "P_EXISTING"
    assert catalog.candidate_protein_for_sequence("M" * 30) is None
    assert catalog.candidate_reactions_for_smiles(" CCO >> CC=O ") == ["RHEA:12345"]
    assert catalog.candidate_reactions_for_smiles("CCC>>CC") == []
