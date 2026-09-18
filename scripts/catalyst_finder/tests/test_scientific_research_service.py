from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import requests
from types import SimpleNamespace
from typing import Any

from scripts.catalyst_finder.scientific_research_service import ScientificResearchService


class _Evidence:
    def __init__(self) -> None:
        self._candidate_proteins = {"P001", "P002", "P003", "P004"}
        self._candidate_reactions = {"RHEA:11111", "RHEA:22222", "RHEA:33333", "RHEA:44444"}

    def canonical_protein_id(self, value: str) -> str:
        return str(value)

    def is_candidate_protein(self, value: str) -> bool:
        return str(value) in self._candidate_proteins

    def is_candidate_reaction(self, value: str) -> bool:
        return str(value) in self._candidate_reactions

    def reaction_metadata(self, value: str) -> dict[str, str]:
        return {"name": f"reaction {value}", "substrate_name": "A", "product_name": "B"}

    def protein_metadata(self, value: str) -> dict[str, str]:
        return {"name": f"protein {value}", "species": "Example species", "uniprot_id": value}


class _Queries:
    @staticmethod
    def lookup_protein_reactions(_protein_id: str, *, ui_language: str = "en") -> dict[str, Any]:
        return {
            "known_associations": {
                "count": 2,
                "items": [
                    {"candidate_id": "RHEA:11111", "rhea_url": "https://rhea/11111"},
                    {"candidate_id": "RHEA:22222", "rhea_url": "https://rhea/22222"},
                ],
            }
        }

    @staticmethod
    def lookup_reaction_proteins(_reaction_id: str, *, ui_language: str = "en") -> dict[str, Any]:
        return {
            "known_associations": {
                "count": 2,
                "items": [
                    {"candidate_id": "P001", "uniprot_url": "https://uniprot/P001"},
                    {"candidate_id": "P002", "uniprot_url": "https://uniprot/P002"},
                ],
            }
        }


class _Proteins:
    uniprot = SimpleNamespace(exact=lambda accession: {
        "accession": accession,
        "name": "Example enzyme",
        "organism": "Example species",
        "sequence": "M" * 120,
    })


class _Gateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def rank(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((command, dict(payload)))
        if command == "rank-reactions":
            return {
                "query": {"route_id": "e2r-test"},
                "candidates": [
                    {"candidate_id": "RHEA:11111", "rank": 2, "score": 0.91},
                    {"candidate_id": "RHEA:33333", "rank": 3, "score": 0.88},
                    {"candidate_id": "RHEA:44444", "rank": 4, "score": 0.82},
                ],
            }
        if command == "rank-enzymes":
            return {
                "query": {"route_id": "r2e-test"},
                "candidates": [
                    {"candidate_id": "P001", "rank": 1, "score": 0.93},
                    {"candidate_id": "P003", "rank": 2, "score": 0.89},
                    {"candidate_id": "P004", "rank": 3, "score": 0.84},
                ],
            }
        raise AssertionError(command)


class ScientificResearchModelLensTests(unittest.TestCase):
    def build(self, *, cache_root: Path | None = None) -> ScientificResearchService:
        return ScientificResearchService(
            evidence=_Evidence(),
            evidence_queries=_Queries(),
            proteins=_Proteins(),
            rhea=SimpleNamespace(reaction_smiles=lambda *_a, **_k: {"reaction_smiles": "A>>B"}),
            route_designer=SimpleNamespace(),
            model_gateway=_Gateway(),
            catalog=SimpleNamespace(
                reaction_by_id={},
                protein_by_id={
                    "P003": {"name": "novel protein", "species": "Species C", "uniprot_id": "P003"},
                    "P004": {"name": "novel protein 2", "species": "Species D", "uniprot_id": "P004"},
                },
            ),
            user_agent="test",
            cache_root=cache_root,
        )

    def test_protein_lens_recovers_known_and_frontier_excludes_recorded(self) -> None:
        service = self.build()
        known = _Queries.lookup_protein_reactions("P001")
        lens = service._model_lens_protein("P001", known_result=known)
        self.assertEqual(lens["status"], "ok")
        self.assertFalse(lens["evidence_conditioned"])
        self.assertEqual(lens["mode"], "production_frontier_with_separate_recovery_audit")
        self.assertEqual(lens["seed_ids"], ["RHEA:11111"])
        audit_command, audit_payload = service.model_gateway.calls[0]
        frontier_command, frontier_payload = service.model_gateway.calls[-1]
        self.assertEqual(audit_command, "rank-reactions")
        self.assertEqual(audit_payload["known_reaction_ids"], ["RHEA:11111"])
        self.assertEqual(audit_payload["retrieval_mode"], "hybrid")
        self.assertEqual(audit_payload["hybrid_direct_weight"], 0.5)
        self.assertEqual(frontier_command, "rank-reactions")
        self.assertNotIn("known_reaction_ids", frontier_payload)
        self.assertNotIn("retrieval_mode", frontier_payload)
        self.assertEqual(frontier_payload["top_k"], 10)
        self.assertEqual(frontier_payload["ranking_objective"], "top10")
        self.assertEqual(lens["recorded_recovery"]["mode"], "leave_one_out")
        self.assertEqual(lens["recorded_recovery"]["holdout_id"], "RHEA:22222")
        self.assertEqual(lens["recorded_recovery"]["eligible_recorded"], 1)
        self.assertEqual(lens["recorded_recovery"]["recovered"], 0)
        self.assertEqual(lens["recorded_recovery"]["rate"], 0.0)
        frontier_ids = [row["candidate_id"] for row in lens["frontier"]]
        self.assertEqual(frontier_ids, ["RHEA:33333", "RHEA:44444"])
        self.assertNotIn("RHEA:11111", frontier_ids)
        self.assertNotIn("RHEA:22222", frontier_ids)

    def test_reaction_lens_recovers_known_and_frontier_excludes_recorded(self) -> None:
        service = self.build()
        known = _Queries.lookup_reaction_proteins("RHEA:11111")
        lens = service._model_lens_reaction("RHEA:11111", known_result=known)
        self.assertFalse(lens["evidence_conditioned"])
        self.assertEqual(lens["mode"], "production_frontier_with_separate_recovery_audit")
        self.assertEqual(lens["seed_ids"], ["P001"])
        audit_command, audit_payload = service.model_gateway.calls[0]
        frontier_command, frontier_payload = service.model_gateway.calls[-1]
        self.assertEqual(audit_command, "rank-enzymes")
        self.assertEqual(audit_payload["known_enzyme_ids"], ["P001"])
        self.assertEqual(audit_payload["retrieval_mode"], "hybrid")
        self.assertEqual(audit_payload["hybrid_direct_weight"], 0.5)
        self.assertEqual(frontier_command, "rank-enzymes")
        self.assertNotIn("known_enzyme_ids", frontier_payload)
        self.assertNotIn("retrieval_mode", frontier_payload)
        self.assertEqual(frontier_payload["top_k"], 10)
        self.assertEqual(frontier_payload["ranking_objective"], "top10")
        self.assertEqual(lens["recorded_recovery"]["mode"], "leave_one_out")
        self.assertEqual(lens["recorded_recovery"]["holdout_id"], "P002")
        self.assertEqual(lens["recorded_recovery"]["eligible_recorded"], 1)
        self.assertEqual(lens["recorded_recovery"]["recovered"], 0)
        self.assertEqual([row["candidate_id"] for row in lens["frontier"]], ["P003", "P004"])

    def test_conditioning_plan_uses_single_known_as_anchor_without_fake_holdout(self) -> None:
        seeds, holdout = ScientificResearchService._conditioning_plan(["RHEA:11111"])
        self.assertEqual(seeds, ["RHEA:11111"])
        self.assertIsNone(holdout)
        recovery = ScientificResearchService._recovery_payload(
            ranked={}, holdout_id=holdout, seed_ids=seeds,
        )
        self.assertEqual(recovery["mode"], "seeded_no_holdout")
        self.assertEqual(recovery["eligible_recorded"], 0)
        self.assertEqual(recovery["seed_count"], 1)

    def test_conditioning_plan_holds_out_one_known_when_multiple_exist(self) -> None:
        seeds, holdout = ScientificResearchService._conditioning_plan(
            ["P001", "P002", "P003", "P004"], max_seeds=2,
        )
        self.assertEqual(seeds, ["P001", "P002"])
        self.assertEqual(holdout, "P004")
        recovery = ScientificResearchService._recovery_payload(
            ranked={"P004": {"rank": 3, "score": 0.7}},
            holdout_id=holdout, seed_ids=seeds,
        )
        self.assertEqual(recovery["mode"], "leave_one_out")
        self.assertEqual(recovery["recovered"], 1)
        self.assertEqual(recovery["items"][0]["rank"], 3)

    def test_uniprot_workspace_keeps_complete_finite_annotation_lists(self) -> None:
        service = self.build()

        class Response:
            status_code = 200
            def raise_for_status(self):
                return None
            def json(self):
                refs = []
                for index in range(65):
                    refs.append({
                        "citation": {
                            "title": f"Paper {index}", "authors": ["A"], "journal": "J", "publicationDate": "2026",
                            "citationCrossReferences": [{"database": "PubMed", "id": str(10000 + index)}],
                        },
                        "referencePositions": ["FUNCTION"],
                    })
                comments = [
                    {"commentType": "CATALYTIC ACTIVITY", "reaction": {"name": f"R{i}", "ecNumber": "1.1.1.1", "reactionCrossReferences": []}}
                    for i in range(15)
                ]
                comments.append({"commentType": "COFACTOR", "cofactors": [{"name": f"C{i}"} for i in range(14)]})
                comments.extend({"commentType": "FUNCTION", "texts": [{"value": f"Function {i}"}]} for i in range(11))
                return {
                    "primaryAccession": "P001", "uniProtkbId": "P001",
                    "entryType": "UniProtKB reviewed (Swiss-Prot)",
                    "annotationScore": 5.0,
                    "proteinExistence": "Evidence at protein level",
                    "proteinDescription": {"recommendedName": {"fullName": {"value": "Example enzyme"}}},
                    "organism": {"scientificName": "Example species"},
                    "genes": [{"geneName": {"value": f"G{i}"}} for i in range(12)],
                    "comments": comments,
                    "references": refs,
                    "uniProtKBCrossReferences": [
                        {"database": "PDB", "id": f"PDB{i:03d}"} for i in range(25)
                    ] + [{"database": "GO", "id": f"GO:{i:07d}"} for i in range(30)],
                }

        service.session.get = lambda *_a, **_k: Response()
        panel = service._uniprot_panel("P001")
        self.assertEqual(len(panel["catalytic_activities"]), 15)
        self.assertEqual(len(panel["cofactors"]), 14)
        self.assertEqual(len(panel["annotations"]["FUNCTION"]), 11)
        self.assertEqual(len(panel["cross_references"]["PDB"]), 25)
        self.assertEqual(len(panel["cross_reference_items"]), 55)
        self.assertEqual(len(panel["publication_ids"]), 65)
        self.assertIn("G11", next(row["value"] for row in panel["facts"] if row["label"] == "Gene"))
        panel_zh = service._uniprot_panel("P001", ui_language="zh")
        zh_labels = [row["label"] for row in panel_zh["facts"]]
        self.assertEqual(zh_labels, ["蛋白名称", "物种", "基因", "条目类型", "注释评分", "蛋白存在证据"])
        self.assertNotIn("Protein", zh_labels)

    def test_research_panel_product_labels_follow_ui_language(self) -> None:
        service = self.build()
        zh_structures = service._structure_panel("P001", uniprot_panel={"cross_references": {}}, ui_language="zh")
        en_structures = service._structure_panel("P001", uniprot_panel={"cross_references": {}}, ui_language="en")
        self.assertEqual(zh_structures["title"], "结构")
        self.assertEqual([row["label"] for row in zh_structures["facts"]], ["实验结构", "预测结构"])
        self.assertEqual(en_structures["title"], "Structures")
        self.assertEqual([row["label"] for row in en_structures["facts"]], ["Experimental structures", "Predicted models"])
        zh_literature = service._literature_panel_for_pmids([], limit=10, curated_by="UniProtKB", ui_language="zh")
        en_literature = service._literature_panel_for_pmids([], limit=10, curated_by="UniProtKB", ui_language="en")
        self.assertEqual(zh_literature["title"], "数据库直接关联文献 · Europe PMC")
        self.assertEqual(en_literature["title"], "Database-linked references · Europe PMC")

    def test_structure_panel_keeps_all_pdb_identities_but_prefetches_one_page(self) -> None:
        service = self.build()
        calls = []

        class Response:
            def __init__(self, url):
                self.url = url
                self.status_code = 200
            def raise_for_status(self):
                return None
            def json(self):
                if "alphafold" in self.url:
                    return [{"modelEntityId": "AF-P001-F1", "latestVersion": 4, "globalMetricValue": 91.0, "toolUsed": "AlphaFold"}]
                pdb = self.url.rsplit("/", 1)[-1]
                return {"struct": {"title": f"Structure {pdb}"}, "rcsb_entry_info": {"resolution_combined": [2.0]}, "exptl": [{"method": "X-RAY"}]}

        def fake_get(url, **_kwargs):
            calls.append(url)
            return Response(url)
        service.session.get = fake_get
        panel = service._structure_panel("P001", uniprot_panel={
            "cross_references": {"PDB": [f"P{i:03d}" for i in range(25)], "AlphaFoldDB": ["P001"]}
        })
        pdb_items = [row for row in panel["items"] if row.get("type") == "experimental_structure"]
        self.assertEqual(len(pdb_items), 25)
        self.assertEqual(panel["facts"][0]["value"], 25)
        self.assertEqual(sum("data.rcsb.org" in url for url in calls), 10)
        self.assertEqual(sum("alphafold" in url for url in calls), 1)
        self.assertEqual(pdb_items[-1]["id"], "P024")
        self.assertEqual(pdb_items[-1]["detail_status"], "identity_only")

    def test_interpro_panel_follows_all_remote_pages(self) -> None:
        service = self.build()
        calls = []

        class Response:
            def __init__(self, payload): self.payload = payload
            def raise_for_status(self): return None
            def json(self): return self.payload

        def fake_get(url, params=None, **_kwargs):
            calls.append((url, params))
            if len(calls) == 1:
                return Response({"count": 3, "results": [
                    {"metadata": {"accession": "IPR1", "name": "One", "type": "domain", "member_databases": {"pfam": {"PF1": {}}}}},
                    {"metadata": {"accession": "IPR2", "name": "Two", "type": "family", "member_databases": {}}},
                ], "next": "https://next.example/page2"})
            return Response({"count": 3, "results": [
                {"metadata": {"accession": "IPR3", "name": "Three", "type": "domain", "member_databases": {"cathgene3d": {"C1": {}, "C2": {}}}}},
            ], "next": None})
        service.session.get = fake_get
        panel = service._interpro_panel("P001")
        self.assertEqual(panel["count"], 3)
        self.assertEqual([row["id"] for row in panel["items"]], ["IPR1", "IPR2", "IPR3"])
        self.assertEqual(panel["items"][-1]["member_entries"], ["cathgene3d:C1", "cathgene3d:C2"])
        self.assertEqual(panel["pagination"]["mode"], "local")
        self.assertEqual(len(calls), 2)

    def test_curated_literature_batches_without_finite_reference_cap(self) -> None:
        service = self.build()
        batch_sizes = []
        def fake_panel(query: str, *, limit: int, cursor_mark: str = "*") -> dict[str, Any]:
            ids = [part.split(":", 1)[1] for part in query.split(" OR ")]
            batch_sizes.append(len(ids))
            return {"items": [{"id": pid, "pmid": pid, "source": "MED", "title": f"Paper {pid}"} for pid in ids]}
        service._literature_panel = fake_panel
        ids = [str(20000 + i) for i in range(165)]
        panel = service._literature_panel_for_pmids(ids, limit=10, curated_by="UnitTest")
        self.assertEqual(batch_sizes, [80, 80, 5])
        self.assertEqual(panel["curated_reference_count"], 165)
        self.assertEqual(panel["count"], 165)
        self.assertEqual(len(panel["items"]), 165)
        self.assertEqual([row["pmid"] for row in panel["items"]], ids)
        self.assertEqual(panel["missing_reference_ids"], [])

    def test_common_uniprot_cross_references_have_direct_external_links(self) -> None:
        self.assertEqual(
            ScientificResearchService._xref_url("PDB", "1ABC"),
            "https://www.rcsb.org/structure/1ABC",
        )
        self.assertIn("alphafold.ebi.ac.uk", ScientificResearchService._xref_url("AlphaFoldDB", "P001") or "")
        self.assertIn("brenda-enzymes.org", ScientificResearchService._xref_url("BRENDA", "1.1.1.27") or "")
        self.assertIsNone(ScientificResearchService._xref_url("UNKNOWN", "X"))

    def test_protein_workspace_survives_one_external_source_failure(self) -> None:
        service = self.build()
        service._uniprot_panel = lambda accession, **_kwargs: {
            "id": "uniprot", "title": "UniProtKB", "status": "ok",
            "record": {"accession": accession, "name": "Example enzyme", "organism": "Example species"},
        }
        service._interpro_panel = lambda _accession: (_ for _ in ()).throw(RuntimeError("InterPro down"))
        service._literature_panel = lambda query, limit: {
            "id": "literature", "title": "Europe PMC", "status": "ok",
            "query": query, "count": 4, "items": [],
        }
        result = service.protein_workspace("P001", ui_language="zh", sections=["annotations", "literature", "model"])
        self.assertEqual(result["answer_mode"], "research_workspace")
        panels = {row["id"]: row for row in result["source_panels"]}
        self.assertEqual(panels["uniprot"]["status"], "ok")
        self.assertEqual(panels["literature_europe_pmc"]["status"], "ok")
        self.assertIn("literature_openalex", panels)
        self.assertEqual(panels["interpro"]["status"], "unavailable")
        self.assertEqual(result["model_lens"]["status"], "ok")
        self.assertNotIn("RHEA:11111", [row["candidate_id"] for row in result["model_lens"]["frontier"]])

    def test_relations_and_model_do_not_fetch_unrequested_external_modules(self) -> None:
        service = self.build()
        service._uniprot_panel = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("UniProt should not be fetched"))
        service._interpro_panel = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("InterPro should not be fetched"))
        service._structure_panel = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("Structures should not be fetched"))
        service._literature_panel = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("Literature should not be fetched"))
        result = service.protein_workspace("P001", sections=["recorded_relations", "model"])
        self.assertEqual(result["selected_sections"], ["recorded_relations", "model"])
        self.assertEqual(result["source_panels"], [])
        self.assertEqual(result["known_associations"]["count"], 2)
        self.assertEqual(result["model_lens"]["status"], "ok")
        self.assertEqual(len(service.model_gateway.calls), 2)
        self.assertIn("known_reaction_ids", service.model_gateway.calls[0][1])
        self.assertNotIn("known_reaction_ids", service.model_gateway.calls[1][1])

    def test_literature_and_structures_do_not_run_model_interpro_or_relation_lookup(self) -> None:
        service = self.build()
        service.evidence_queries = SimpleNamespace(
            lookup_protein_reactions=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("relations should not be queried"))
        )
        service._uniprot_panel = lambda accession, **_kwargs: {
            "id": "uniprot", "title": "UniProtKB", "status": "ok",
            "publication_ids": ["123"], "curated_reference_metadata": {},
            "record": {"accession": accession, "name": "Example enzyme", "organism": "Example species"},
        }
        service._interpro_panel = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("InterPro should not be fetched"))
        service._model_lens_protein = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("model should not run"))
        service._structure_panel = lambda accession, **_k: {"id": "structures", "title": "Structures", "status": "ok", "items": [{"id": accession}]}
        service._literature_panel_for_pmids = lambda *_a, **_k: {"id": "literature", "title": "Europe PMC", "status": "ok", "count": 1, "items": [{"id": "123"}]}
        result = service.protein_workspace("P001", sections=["literature", "structures"], primary_section="literature")
        self.assertEqual(result["selected_sections"], ["literature", "structures"])
        self.assertEqual(result["primary_section"], "literature")
        self.assertIsNone(result["known_associations"])
        self.assertIsNone(result["model_lens"])
        self.assertEqual({row["section"] for row in result["source_panels"]}, {"literature", "structures"})
        self.assertEqual(service.model_gateway.calls, [])

    def test_annotations_only_do_not_fetch_literature_structures_or_model(self) -> None:
        service = self.build()
        service.evidence_queries = SimpleNamespace(
            lookup_protein_reactions=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("relations should not be queried"))
        )
        service._uniprot_panel = lambda accession, **_kwargs: {"id": "uniprot", "title": "UniProtKB", "status": "ok", "record": {"accession": accession, "name": "Example"}}
        service._interpro_panel = lambda _accession: {"id": "interpro", "title": "InterPro", "status": "ok", "items": []}
        service._structure_panel = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("structures should not run"))
        service._literature_panel = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("literature should not run"))
        service._model_lens_protein = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("model should not run"))
        result = service.protein_workspace("P001", sections=["annotations"])
        self.assertEqual(result["selected_sections"], ["annotations"])
        self.assertEqual([row["id"] for row in result["source_panels"]], ["uniprot", "interpro"])
        self.assertIsNone(result["known_associations"])
        self.assertIsNone(result["model_lens"])

    def test_protein_detail_exposes_bounded_substantive_uniprot_evidence(self) -> None:
        service = self.build()
        service._uniprot_panel = lambda accession, **_kwargs: {
            "record": {"accession": accession, "name": "Example enzyme", "organism": "Example species", "genes": ["EX1"]},
            "facts": [{"label": "Protein", "value": "Example enzyme"}] * 20,
            "catalytic_activities": [{"reaction": "A = B"}] * 20,
            "cofactors": [f"C{i}" for i in range(20)],
            "annotations": {"FUNCTION": [f"F{i}" for i in range(8)]},
            "cross_references": {"PDB": [f"P{i}" for i in range(20)]},
            "url": "https://uniprot/example",
        }
        detail = service.protein_detail("P001")
        self.assertEqual(detail["record"]["accession"], "P001")
        self.assertEqual(len(detail["facts"]), 12)
        self.assertEqual(len(detail["catalytic_activities"]), 12)
        self.assertEqual(len(detail["cofactors"]), 12)
        self.assertEqual(len(detail["annotations"]["FUNCTION"]), 5)
        self.assertEqual(len(detail["cross_references"]["PDB"]), 12)

    def test_next_steps_section_uses_dynamic_contextual_generator(self) -> None:
        service = self.build()
        captured = {}

        def suggest_next_steps(**kwargs):
            captured.update(kwargs)
            return [{
                "prompt": "Inspect the recorded relation RHEA:11111 in more detail.",
                "title": "Inspect a returned relation",
                "reason": "It is present in the current recorded evidence.",
                "priority": "high",
            }]

        service.deepseek = SimpleNamespace(suggest_next_steps=suggest_next_steps)
        result = service.protein_workspace("P001", sections=["next_steps"], ui_language="en")
        self.assertEqual(result["selected_sections"], ["next_steps"])
        self.assertEqual(result["opportunities"][0]["kind"], "model_generated_next_step")
        self.assertEqual(result["opportunities"][0]["prompt"], "Inspect the recorded relation RHEA:11111 in more detail.")
        self.assertEqual(captured["result_context"]["entity"]["id"], "P001")
        self.assertEqual(captured["result_context"]["recorded_association_count"], 2)
        self.assertIn("1 contextual suggestions", result["route_view"]["nodes"][-1]["metric"])

    def test_remote_source_snapshot_recovers_only_transient_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self.build(cache_root=Path(tmp))
            fresh = {"id": "source", "status": "ok", "count": 3, "items": [{"id": "A"}]}
            first = service._with_stale_snapshot(
                "unit", "key", max_stale_seconds=3600, fetch=lambda: dict(fresh),
            )
            self.assertEqual(first["source_freshness"], "fresh")

            stale = service._with_stale_snapshot(
                "unit", "key", max_stale_seconds=3600,
                fetch=lambda: (_ for _ in ()).throw(requests.Timeout("temporary")),
            )
            self.assertEqual(stale["source_freshness"], "stale_cache")
            self.assertEqual(stale["count"], 3)
            self.assertEqual(stale["live_fetch_error"], "Timeout")
            self.assertGreaterEqual(stale["stale_cache_age_seconds"], 0)

            response = requests.Response(); response.status_code = 404
            with self.assertRaises(requests.HTTPError):
                service._with_stale_snapshot(
                    "unit", "key", max_stale_seconds=3600,
                    fetch=lambda: (_ for _ in ()).throw(requests.HTTPError("not found", response=response)),
                )

    def test_uniprot_snapshot_preserves_curated_literature_on_transient_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self.build(cache_root=Path(tmp))
            fresh_panel = {
                "id": "uniprot", "title": "UniProtKB", "status": "ok",
                "record": {"accession": "P001", "name": "Example enzyme", "organism": "Species"},
                "publication_ids": ["111", "222"], "curated_reference_metadata": {},
                "cross_references": {},
            }
            service._fetch_uniprot_panel = lambda *_a, **_k: dict(fresh_panel)
            first = service._uniprot_panel("P001", ui_language="en")
            self.assertEqual(first["source_freshness"], "fresh")
            self.assertEqual(first["publication_ids"], ["111", "222"])

            service._fetch_uniprot_panel = lambda *_a, **_k: (_ for _ in ()).throw(requests.Timeout("UniProt temporary timeout"))
            recovered = service._uniprot_panel("P001", ui_language="en")
            self.assertEqual(recovered["source_freshness"], "stale_cache")
            self.assertEqual(recovered["publication_ids"], ["111", "222"])

            service._literature_panel_for_pmids = lambda ids, **_k: {
                "id": "literature_curated", "provider": "europe_pmc", "entity_kind": "literature",
                "status": "ok", "count": len(ids), "items": [{"pmid": value} for value in ids],
            }
            service._literature_panel = lambda *_a, **_k: {
                "id": "literature_europe_pmc", "provider": "europe_pmc", "entity_kind": "literature",
                "status": "ok", "count": 0, "items": [],
            }
            service._openalex_panel = lambda *_a, **_k: {
                "id": "literature_openalex", "provider": "openalex", "entity_kind": "literature",
                "status": "ok", "count": 0, "items": [],
            }
            workspace = service.protein_workspace("P001", sections=["literature"])
            panels = {row["id"]: row for row in workspace["source_panels"]}
            self.assertEqual(panels["literature_curated"]["count"], 2)
            self.assertEqual([row["pmid"] for row in panels["literature_curated"]["items"]], ["111", "222"])

    def test_europe_pmc_and_openalex_wrappers_reuse_success_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = self.build(cache_root=Path(tmp))
            service._fetch_literature_panel = lambda *_a, **_k: {
                "id": "literature_europe_pmc", "provider": "europe_pmc", "status": "ok",
                "count": 4, "items": [{"pmid": "123"}], "pagination": {"mode": "remote"},
            }
            epmc = service._literature_panel("enzyme", limit=10)
            self.assertEqual(epmc["source_freshness"], "fresh")
            service._fetch_literature_panel = lambda *_a, **_k: (_ for _ in ()).throw(requests.ConnectionError("down"))
            epmc_stale = service._literature_panel("enzyme", limit=10)
            self.assertEqual(epmc_stale["source_freshness"], "stale_cache")
            self.assertEqual(epmc_stale["items"][0]["pmid"], "123")

            service._fetch_openalex_panel = lambda *_a, **_k: {
                "id": "literature_openalex", "provider": "openalex", "status": "ok",
                "count": 5, "items": [{"id": "OPENALEX:W1"}], "pagination": {"mode": "remote"},
            }
            openalex = service._openalex_panel("enzyme", page_size=10)
            self.assertEqual(openalex["source_freshness"], "fresh")
            service._fetch_openalex_panel = lambda *_a, **_k: (_ for _ in ()).throw(requests.Timeout("down"))
            openalex_stale = service._openalex_panel("enzyme", page_size=10)
            self.assertEqual(openalex_stale["source_freshness"], "stale_cache")
            self.assertEqual(openalex_stale["items"][0]["id"], "OPENALEX:W1")

    def test_europe_pmc_search_retries_one_transient_timeout(self) -> None:
        service = self.build()
        calls = []

        class Response:
            status_code = 200
            def raise_for_status(self): return None
            def json(self):
                return {
                    "hitCount": 1,
                    "nextCursorMark": "",
                    "resultList": {"result": [{
                        "source": "MED", "pmid": "12345", "title": "Recovered paper",
                        "authorString": "A. Author", "journalTitle": "Journal", "pubYear": "2026",
                    }]},
                }

        def fake_get(url, **kwargs):
            calls.append((url, kwargs))
            if len(calls) == 1:
                raise requests.Timeout("temporary Europe PMC timeout")
            return Response()

        service.session.get = fake_get
        panel = service._literature_panel("EXT_ID:12345", limit=10)
        self.assertEqual(len(calls), 2)
        self.assertEqual(panel["items"][0]["pmid"], "12345")

    def test_external_retry_does_not_retry_nontransient_client_error(self) -> None:
        service = self.build()
        calls = []

        class Response:
            status_code = 404
            def raise_for_status(self):
                response = requests.Response()
                response.status_code = 404
                raise requests.HTTPError("not found", response=response)

        def fake_get(url, **kwargs):
            calls.append((url, kwargs))
            return Response()

        service.session.get = fake_get
        with self.assertRaises(requests.HTTPError):
            response = service._get_with_transient_retry("https://example.invalid/missing")
            response.raise_for_status()
        self.assertEqual(len(calls), 1)

    def test_curated_literature_fetches_complete_reference_set_before_local_pagination(self) -> None:
        service = self.build()
        seen = {}
        def fake_panel(query: str, *, limit: int, cursor_mark: str = "*") -> dict[str, Any]:
            seen["limit"] = limit
            ids = [part.split(":", 1)[1].strip() for part in query.split(" OR ")]
            return {
                "id": "literature", "title": "Europe PMC", "status": "ok",
                "count": len(ids), "items": [{"id": pid, "pmid": pid, "source": "MED", "title": f"Paper {pid}"} for pid in ids],
                "pagination": {"mode": "remote", "page_size": limit, "has_more": False},
            }
        service._literature_panel = fake_panel
        pmids = [str(10000 + index) for index in range(28)]
        panel = service._literature_panel_for_pmids(pmids, limit=6, curated_by="UnitTest")
        self.assertEqual(seen["limit"], 28)
        self.assertEqual(panel["count"], 28)
        self.assertEqual(len(panel["items"]), 28)
        self.assertEqual([row["pmid"] for row in panel["items"]], pmids)
        self.assertEqual(panel["pagination"], {"mode": "local", "provider": "europe_pmc", "page_size": 10, "has_more": False})

    def test_literature_detail_follows_erratum_relation_without_treating_notice_as_article(self) -> None:
        service = self.build()
        notice = {
            "id": "200", "pmid": "200", "source": "MED", "title": "Correction notice",
            "publication_types": ["Published Erratum"],
            "corrections": [{"id": "100", "source": "MED", "type": "Erratum for", "reference": "Original article"}],
            "abstract": "",
        }
        original = {
            "id": "100", "pmid": "100", "source": "MED", "title": "Original research",
            "publication_types": ["Journal Article"], "abstract": "The original study reports a verified mechanism.",
        }
        def fake_panel(query: str, *, limit: int, cursor_mark: str = "*") -> dict[str, Any]:
            row = notice if "200" in query else original
            return {"id": "literature", "status": "ok", "items": [dict(row)], "count": 1}
        service._literature_panel = fake_panel
        service._literature_full_text_sections = lambda _pmcid: []
        detail = service.literature_detail(notice)
        self.assertEqual(detail["publication_types"], ["Published Erratum"])
        self.assertEqual(detail["content_basis"], "bibliographic_relation+linked_article_abstract")
        self.assertEqual(detail["related_publications"][0]["id"], "100")
        self.assertEqual(detail["related_publications"][0]["type"], "Erratum for")
        self.assertIn("verified mechanism", detail["related_publications"][0]["abstract"])

    def test_openalex_panel_normalizes_identifiers_abstract_and_cursor(self) -> None:
        service = self.build()
        captured = {}
        class Response:
            def raise_for_status(self): return None
            def json(self):
                return {
                    "meta": {"count": 42, "next_cursor": "next-oa"},
                    "results": [{
                        "id": "https://openalex.org/W123",
                        "doi": "https://doi.org/10.1000/example",
                        "display_name": "OpenAlex paper",
                        "publication_year": 2026,
                        "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/12345678"},
                        "authorships": [{"author": {"display_name": "A. Author"}}],
                        "primary_location": {"landing_page_url": "https://doi.org/10.1000/example", "source": {"display_name": "Journal X"}},
                        "abstract_inverted_index": {"Alpha": [0], "beta": [1], "result": [2]},
                        "cited_by_count": 9,
                        "type": "article",
                        "indexed_in": ["crossref", "pubmed"],
                        "open_access": {"is_oa": True},
                    }],
                }
        def fake_get(url, params=None, **_kwargs):
            captured.update({"url": url, "params": dict(params or {})})
            return Response()
        service.session.get = fake_get
        panel = service._openalex_panel("P00338", page_size=7, cursor="cursor-1")
        self.assertEqual(captured["url"], "https://api.openalex.org/works")
        self.assertEqual(captured["params"]["cursor"], "cursor-1")
        self.assertEqual(captured["params"]["per-page"], 7)
        self.assertEqual(panel["provider"], "openalex")
        self.assertEqual(panel["count"], 42)
        self.assertEqual(panel["pagination"]["next_cursor"], "next-oa")
        row = panel["items"][0]
        self.assertEqual(row["id"], "OPENALEX:W123")
        self.assertEqual(row["pmid"], "12345678")
        self.assertEqual(row["doi"], "10.1000/example")
        self.assertEqual(row["abstract"], "Alpha beta result")
        self.assertEqual(row["indexed_in"], ["crossref", "pubmed"])

    def test_literature_page_dispatches_by_provider(self) -> None:
        service = self.build()
        seen = {}
        service._openalex_panel = lambda query, *, page_size, cursor: seen.update({"query": query, "page_size": page_size, "cursor": cursor}) or {"provider": "openalex", "items": []}
        result = service.literature_page("enzyme", cursor_mark="oa2", page_size=9, provider="openalex")
        self.assertEqual(result["provider"], "openalex")
        self.assertEqual(seen, {"query": "enzyme", "page_size": 9, "cursor": "oa2"})
        with self.assertRaises(ValueError):
            service.literature_page("enzyme", provider="unknown")

    def test_protein_literature_keeps_curated_and_runs_broad_providers(self) -> None:
        service = self.build()
        service._uniprot_panel = lambda accession, **_kwargs: {
            "id": "uniprot", "status": "ok", "record": {"accession": accession, "name": "Example enzyme", "organism": "Species"},
            "publication_ids": ["111", "222"], "curated_reference_metadata": {},
        }
        calls = []
        service._literature_panel_for_pmids = lambda ids, **_kwargs: {
            "id": "literature_curated", "provider": "europe_pmc", "entity_kind": "literature", "status": "ok", "items": [{"pmid": "111"}], "count": len(ids),
        }
        service._literature_panel = lambda query, **_kwargs: calls.append(("europe_pmc", query)) or {
            "id": "literature_europe_pmc", "provider": "europe_pmc", "entity_kind": "literature", "status": "ok", "query": query, "items": [{"pmid": "333"}], "count": 8,
        }
        service._openalex_panel = lambda query, **_kwargs: calls.append(("openalex", query)) or {
            "id": "literature_openalex", "provider": "openalex", "entity_kind": "literature", "status": "ok", "query": query, "items": [{"id": "OPENALEX:W1", "doi": "10.1/x"}], "count": 10,
        }
        result = service.protein_workspace("P001", sections=["literature"])
        panels = result["source_panels"]
        self.assertEqual([row["id"] for row in panels], ["literature_curated", "literature_europe_pmc", "literature_openalex"])
        self.assertEqual([row["provider"] for row in panels], ["europe_pmc", "europe_pmc", "openalex"])
        self.assertEqual([kind for kind, _query in calls], ["europe_pmc", "openalex"])

    def test_resolve_literature_accepts_identifier_or_free_title_query(self) -> None:
        service = self.build()
        queries = []
        def fake_panel(query: str, *, limit: int, cursor_mark: str = "*") -> dict[str, Any]:
            queries.append(query)
            return {"id": "literature", "status": "ok", "items": [{"id": "12345", "pmid": "12345", "source": "MED", "title": "Matched"}], "count": 1}
        service._literature_panel = fake_panel
        self.assertEqual(service.resolve_literature("MED:12345")[0]["pmid"], "12345")
        self.assertEqual(queries[-1], "EXT_ID:12345")
        service.resolve_literature("DOI 10.1038/s41594-021-00633-2。")
        self.assertEqual(queries[-1], 'DOI:"10.1038/s41594-021-00633-2"')
        service.resolve_literature("A paper title about enzyme regulation")
        self.assertEqual(queries[-1], "A paper title about enzyme regulation")

    def test_model_domain_distinguishes_project_aligned_and_expanded_universe(self) -> None:
        service = self.build()
        service.catalog.reaction_by_id = {"RHEA:11111": {}}
        service.catalog.protein_by_id["P001"] = {"name": "project protein"}
        service.proteins = SimpleNamespace(
            uniprot=_Proteins.uniprot,
            canonical_local_id=lambda value: "P001" if value == "P001" else None,
        )
        project_reaction = service._model_domain("reaction", "RHEA:11111", precomputed=True)
        external_reaction = service._model_domain("reaction", "RHEA:33333", precomputed=True)
        project_protein = service._model_domain("protein", "P001", precomputed=True)
        external_protein = service._model_domain("protein", "P003", precomputed=True)
        self.assertEqual(project_reaction["status"], "project_aligned")
        self.assertEqual(project_protein["status"], "project_aligned")
        self.assertGreater(project_reaction["retrospective_audit"]["queries"], 0)
        self.assertGreater(project_protein["retrospective_audit"]["queries"], 0)
        self.assertTrue(project_reaction["retrospective_audit"]["context"]["project_pairs_excluded"])
        self.assertEqual(external_reaction["status"], "expanded_universe_exploratory")
        self.assertEqual(external_protein["status"], "expanded_universe_exploratory")
        self.assertNotIn("retrospective_audit", external_reaction)
        self.assertNotIn("retrospective_audit", external_protein)
        self.assertIn("扩展候选域", external_reaction["interpretation_zh"])

    def test_route_view_makes_evidence_and_model_one_workflow(self) -> None:
        view = ScientificResearchService._workspace_route_view(
            entity_kind="protein",
            entity_id="P001",
            selected_sections=["annotations", "recorded_relations", "model", "next_steps"],
            source_panels=[
                {"title": "UniProtKB", "status": "ok"},
                {"title": "InterPro", "status": "ok"},
                {"title": "Europe PMC", "status": "unavailable"},
            ],
            known_count=4,
            model_lens={
                "status": "ok",
                "top_k": 20,
                "recorded_recovery": {"eligible_recorded": 3, "recovered": 2},
                "frontier": [{"candidate_id": "RHEA:33333"}],
            },
            opportunities_count=2,
            ui_language="zh",
        )
        self.assertEqual(view["route_id"], "research-workspace-v2")
        self.assertEqual(
            [row["id"] for row in view["nodes"]],
            ["research-entity", "research-annotations", "research-recorded_relations", "research-model", "research-next_steps"],
        )
        self.assertIn("2/3", view["nodes"][3]["metric"])
        self.assertIn("2", view["nodes"][4]["metric"])
        self.assertIn("动态建议", view["nodes"][4]["metric"])


if __name__ == "__main__":
    unittest.main()
