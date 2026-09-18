from __future__ import annotations

from types import SimpleNamespace

from scripts.catalyst_finder.protein_family_service import ProteinFamilyEvidenceService


class _Evidence:
    def known_reactions(self, protein_id: str):
        return [SimpleNamespace(reaction_id="RHEA:12345", source="integrated")]

    def reaction_metadata(self, reaction_id: str):
        return {"reaction_smiles": "CCO>>CC=O"}


class _NoNetworkRhea:
    def exact(self, reaction_id: str):
        raise AssertionError("local reaction metadata should avoid live Rhea enrichment")


def test_family_summary_uses_local_reaction_metadata_before_rhea_http() -> None:
    service = ProteinFamilyEvidenceService(
        families=SimpleNamespace(),
        evidence=_Evidence(),
        rhea=_NoNetworkRhea(),
        proteins=SimpleNamespace(),
    )
    result = service._summarize_scope(
        scope={"scope_type": "functional_class", "scope_id": "CLASS-X", "label": "Example class"},
        member_ids=["P1"],
        ui_language="en",
    )
    item = result["known_associations"]["items"][0]
    assert item["candidate_id"] == "RHEA:12345"
    assert item["name"] == "CCO>>CC=O"
