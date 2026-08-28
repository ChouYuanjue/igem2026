from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts.catalyst_finder.evidence_catalog import IntegratedEvidenceCatalog
from scripts.catalyst_finder.evidence_query_service import AssociationEvidenceQueryService


class _NoNetworkProteins:
    class _UniProt:
        def exact(self, accession: str):
            raise AssertionError("reverse evidence lookup must not require UniProt HTTP")
    uniprot = _UniProt()


class _NoNetworkRhea:
    def exact(self, reaction_id: str):
        raise AssertionError("reverse evidence lookup must not require Rhea HTTP")


def test_lookup_protein_reactions_is_local_first(tmp_path: Path) -> None:
    merged = tmp_path / "data/catalyst_candidate_universes/general_merged"
    merged.mkdir(parents=True)
    pd.DataFrame([
        {
            "protein_id": "P_TEST",
            "canonical_accession": "P_TEST",
            "aliases": "P_TEST;P_ALIAS",
            "source_layer": "test",
            "evidence_scope": "candidate",
        }
    ]).to_csv(merged / "protein_metadata.csv", index=False)
    pd.DataFrame([
        {"reaction_id": "RHEA:12345", "reaction_smiles": "CCO>>CC=O", "source_layer": "test"},
        {"reaction_id": "RHEA:54321", "reaction_smiles": "CCN>>CC=N", "source_layer": "test"},
    ]).to_csv(merged / "reactions.csv", index=False)
    pd.DataFrame([
        {"protein_id": "P_TEST", "reaction_id": "RHEA:12345", "source": "database_a", "evidence_type": "recorded_association"},
        {"protein_id": "P_TEST", "reaction_id": "RHEA:54321", "source": "database_b", "evidence_type": "recorded_association"},
    ]).to_csv(merged / "associations.csv", index=False)

    evidence = IntegratedEvidenceCatalog(tmp_path)
    service = AssociationEvidenceQueryService(
        evidence=evidence,
        families=SimpleNamespace(),
        proteins=_NoNetworkProteins(),
        rhea=_NoNetworkRhea(),
        deepseek=SimpleNamespace(),
        catalog=SimpleNamespace(protein_by_id={}),
    )
    result = service.lookup_protein_reactions("P_ALIAS", ui_language="en")
    assert result["direction"] == "enzyme_to_reaction"
    assert result["answer_mode"] == "recorded_protein_reaction_lookup"
    assert result["protein"]["id"] == "P_TEST"
    assert result["known_associations"]["count"] == 2
    assert [row["candidate_id"] for row in result["known_associations"]["items"]] == ["RHEA:12345", "RHEA:54321"]
    assert result["known_associations"]["items"][0]["name"] == "CCO>>CC=O"


def test_lookup_protein_reactions_zero_evidence_is_scoped_not_negative_truth(tmp_path: Path) -> None:
    merged = tmp_path / "data/catalyst_candidate_universes/general_merged"
    merged.mkdir(parents=True)
    pd.DataFrame([
        {"protein_id": "P_EMPTY", "canonical_accession": "P_EMPTY", "aliases": "P_EMPTY"}
    ]).to_csv(merged / "protein_metadata.csv", index=False)
    pd.DataFrame(columns=["reaction_id", "reaction_smiles", "source_layer"]).to_csv(merged / "reactions.csv", index=False)
    pd.DataFrame(columns=["protein_id", "reaction_id", "source", "evidence_type"]).to_csv(merged / "associations.csv", index=False)
    evidence = IntegratedEvidenceCatalog(tmp_path)
    service = AssociationEvidenceQueryService(
        evidence=evidence,
        families=SimpleNamespace(),
        proteins=_NoNetworkProteins(),
        rhea=_NoNetworkRhea(),
        deepseek=SimpleNamespace(),
        catalog=SimpleNamespace(protein_by_id={}),
    )
    result = service.lookup_protein_reactions("P_EMPTY", ui_language="en")
    assert result["known_associations"]["count"] == 0
    assert "not proof" in result["known_associations"]["note"].lower()
