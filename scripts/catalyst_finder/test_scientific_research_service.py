from __future__ import annotations

import unittest
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
    def build(self) -> ScientificResearchService:
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
        )

    def test_protein_lens_recovers_known_and_frontier_excludes_recorded(self) -> None:
        service = self.build()
        known = _Queries.lookup_protein_reactions("P001")
        lens = service._model_lens_protein("P001", known_result=known)
        self.assertEqual(lens["status"], "ok")
        self.assertTrue(lens["evidence_conditioned"])
        self.assertEqual(lens["seed_ids"], ["RHEA:11111"])
        command, payload = service.model_gateway.calls[-1]
        self.assertEqual(command, "rank-reactions")
        self.assertEqual(payload["known_reaction_ids"], ["RHEA:11111"])
        self.assertEqual(payload["retrieval_mode"], "hybrid")
        self.assertEqual(payload["hybrid_direct_weight"], 0.5)
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
        self.assertTrue(lens["evidence_conditioned"])
        self.assertEqual(lens["seed_ids"], ["P001"])
        command, payload = service.model_gateway.calls[-1]
        self.assertEqual(command, "rank-enzymes")
        self.assertEqual(payload["known_enzyme_ids"], ["P001"])
        self.assertEqual(payload["retrieval_mode"], "hybrid")
        self.assertEqual(payload["hybrid_direct_weight"], 0.5)
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
        service._uniprot_panel = lambda accession: {
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
        self.assertEqual(panels["literature"]["status"], "ok")
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
        self.assertEqual(len(service.model_gateway.calls), 1)

    def test_literature_and_structures_do_not_run_model_interpro_or_relation_lookup(self) -> None:
        service = self.build()
        service.evidence_queries = SimpleNamespace(
            lookup_protein_reactions=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("relations should not be queried"))
        )
        service._uniprot_panel = lambda accession: {
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
        service._uniprot_panel = lambda accession: {"id": "uniprot", "title": "UniProtKB", "status": "ok", "record": {"accession": accession, "name": "Example"}}
        service._interpro_panel = lambda _accession: {"id": "interpro", "title": "InterPro", "status": "ok", "items": []}
        service._structure_panel = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("structures should not run"))
        service._literature_panel = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("literature should not run"))
        service._model_lens_protein = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("model should not run"))
        result = service.protein_workspace("P001", sections=["annotations"])
        self.assertEqual(result["selected_sections"], ["annotations"])
        self.assertEqual([row["id"] for row in result["source_panels"]], ["uniprot", "interpro"])
        self.assertIsNone(result["known_associations"])
        self.assertIsNone(result["model_lens"])

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
