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
