from __future__ import annotations

from types import SimpleNamespace

from scripts.catalyst_finder.protein_resolution import ProteinResolver


class _FakeUniProt:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, **kwargs):
        self.calls += 1
        return [
            {
                "accession": f"P{i:05d}",
                "name": f"Class enzyme {i}",
                "organism": "Test organism",
                "gene_names": [],
                "reviewed": True,
                "length": 300,
            }
            for i in range(40)
        ]


def test_class_member_search_stops_at_limit_and_reuses_cache() -> None:
    resolver = ProteinResolver(SimpleNamespace(proteins=[], protein_by_id={}), user_agent="test")
    fake = _FakeUniProt()
    resolver.uniprot = fake
    first = resolver.search_class_members(protein_terms=["broad enzyme class", "synonym"], limit=40)
    assert len(first) == 40
    assert fake.calls == 1
    second = resolver.search_class_members(protein_terms=["broad enzyme class", "synonym"], limit=40)
    assert [row.identifier for row in second] == [row.identifier for row in first]
    assert fake.calls == 1


def test_detail_for_enriches_local_record_once():
    from types import SimpleNamespace
    from scripts.catalyst_finder.protein_resolution import ProteinResolver

    catalog = SimpleNamespace(
        protein_by_id={"A0A000": {"uniprot_id": "A0A000", "name": "A0A000", "species": "", "sequence_length": 123}},
        protein_alias_to_id={"A0A000": "A0A000"},
    )
    resolver = ProteinResolver.__new__(ProteinResolver)
    resolver.catalog = catalog
    resolver._class_member_cache = {}
    resolver.uniprot = SimpleNamespace(exact=lambda accession: {
        "accession": accession,
        "name": "Remote protein name",
        "organism": "Example species",
        "gene_names": ["EXAMPLE"],
        "length": 123,
    })
    resolver.canonical_local_id = lambda identifier: "A0A000" if str(identifier).upper() == "A0A000" else None
    row = resolver.detail_for("A0A000")
    assert row is not None
    assert row.identifier == "A0A000"
    assert row.name == "Remote protein name"
    assert row.organism == "Example species"
    assert row.model_ready is True
    assert row.source == "uniprot+model_catalog"
