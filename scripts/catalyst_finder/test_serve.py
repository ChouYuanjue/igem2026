from __future__ import annotations

import http.client
import json
import stat
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.catalyst_finder.serve import (
    CatalystFinderRuntime,
    Handler,
    ProductionHTTPServer,
    _candidate_match,
    _explicit_uniprot_accession,
    _fallback_queries,
    canonical_rhea_id,
)
from scripts.catalyst_finder.protein_resolution import ProteinResolver
from scripts.catalyst_finder.model_gateway import ModelGateway
from projects.active.terpene_screening.core.candidate_universes import TPS_SPECIALIZED_UNIVERSE


class CatalystFinderUnitTests(unittest.TestCase):
    def test_canonical_rhea_id(self) -> None:
        self.assertEqual(canonical_rhea_id("RHEA:33983"), "RHEA:33983")
        self.assertEqual(canonical_rhea_id("33983"), "RHEA:33983")

    def test_candidate_match_prefers_requested_orientation(self) -> None:
        equation = "(+)-copalyl diphosphate = miltiradiene + diphosphate"
        score, orientation = _candidate_match(
            equation,
            ["(+)-copalyl diphosphate"],
            ["miltiradiene"],
        )
        self.assertGreater(score, 6.0)
        self.assertEqual(orientation, "forward")

    def test_candidate_match_can_detect_reverse_orientation(self) -> None:
        equation = "miltiradiene + diphosphate = (+)-copalyl diphosphate"
        _, orientation = _candidate_match(
            equation,
            ["(+)-copalyl diphosphate"],
            ["miltiradiene"],
        )
        self.assertEqual(orientation, "reverse")

    def test_fallback_queries_include_both_sides(self) -> None:
        queries = _fallback_queries(["(+)-copalyl diphosphate"], ["miltiradiene"])
        self.assertTrue(any("AND" in query for query in queries))
        self.assertTrue(any("miltiradiene" in query for query in queries))

    def test_explicit_uniprot_accession_is_extracted_from_natural_language(self) -> None:
        self.assertEqual(_explicit_uniprot_accession("查看 UniProt P00338 的 3 个优先反应"), "P00338")
        self.assertEqual(_explicit_uniprot_accession("查看这个酶的优先反应"), "")

    def test_resolve_protein_exact_id_does_not_require_language_context(self) -> None:
        runtime = CatalystFinderRuntime()
        row = type("ProteinRow", (), {
            "name": "test enzyme",
            "identifier": "PTEST1",
            "as_dict": lambda self: {"id": "PTEST1", "name": "test enzyme"},
        })()
        runtime.proteins.exact_or_search = lambda text, limit=8: [row]
        resolved = runtime.resolve_protein("PTEST1")
        self.assertEqual(resolved["mode"], "protein_id")
        self.assertEqual(resolved["recommended_id"], "PTEST1")

    def test_seen_bonus_does_not_create_unrelated_local_match(self) -> None:
        resolver = ProteinResolver.__new__(ProteinResolver)
        row = {
            "id": "E8W6C7",
            "uniprot_id": "E8W6C7",
            "genbank_id": None,
            "name": "germacradienol synthase",
            "species": "Streptomyces pratensis",
            "seen": True,
        }
        score = resolver._local_score(
            row, protein_terms=[], organism_terms=[], gene_terms=[], accession_terms=["P00338"]
        )
        self.assertEqual(score, 0.0)

    def test_feedback_is_persisted_as_jsonl(self) -> None:
        runtime = CatalystFinderRuntime()
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime.feedback_path = Path(tmpdir) / "feedback.jsonl"
            result = runtime.submit_feedback({
                "rating": "helpful",
                "category": "results",
                "message": "候选结果很有帮助",
                "contact": "",
                "context": {"direction": "reaction_to_enzyme", "route_id": "r2e-current-top10-v1"},
            })
            self.assertTrue(result["ok"])
            rows = runtime.feedback_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            payload = json.loads(rows[0])
            self.assertEqual(payload["rating"], "helpful")
            self.assertEqual(payload["category"], "results")
            self.assertEqual(payload["context"]["direction"], "reaction_to_enzyme")
            self.assertEqual(stat.S_IMODE(runtime.feedback_path.stat().st_mode), 0o600)

    def test_run_event_is_persisted_with_prompt_and_private_mode(self) -> None:
        runtime = CatalystFinderRuntime()
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime.run_events_path = Path(tmpdir) / "run_events.jsonl"
            result = runtime.record_run_event(
                event_type="candidate_ranking",
                session_id="sess_test",
                run_id="run_test",
                step_id="step_test",
                input_data={"final_user_prompt": "请找催化 A 到 B 的酶"},
                output_data={"candidates": [{"candidate_id": "P123"}]},
                metadata={"card_id": "reaction_to_enzyme", "prompt_source": "shortcut_card"},
            )
            self.assertTrue(result["ok"])
            payload = json.loads(runtime.run_events_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], "run_test")
            self.assertEqual(payload["input"]["final_user_prompt"], "请找催化 A 到 B 的酶")
            self.assertEqual(payload["output"]["candidates"][0]["candidate_id"], "P123")
            self.assertEqual(stat.S_IMODE(runtime.run_events_path.stat().st_mode), 0o600)

    def test_pending_steps_are_grouped_for_one_run(self) -> None:
        runtime = CatalystFinderRuntime()
        runtime.hold_run_step("run_test", {"step_type": "intent_and_entity_resolution"})
        runtime.hold_run_step("run_test", {"step_type": "candidate_ranking"})
        steps = runtime.take_run_steps("run_test")
        self.assertEqual([step["step_type"] for step in steps], ["intent_and_entity_resolution", "candidate_ranking"])
        self.assertEqual(runtime.take_run_steps("run_test"), [])

    def test_stale_pending_run_steps_are_pruned(self) -> None:
        runtime = CatalystFinderRuntime()
        runtime.hold_run_step("stale", {"step_type": "intent_and_entity_resolution"})
        runtime.runtime_store._pending_run_started["stale"] = time.time() - 3700
        runtime.hold_run_step("fresh", {"step_type": "intent_and_entity_resolution"})
        self.assertNotIn("stale", runtime.runtime_store._pending_run_steps)
        self.assertNotIn("stale", runtime.runtime_store._pending_run_started)
        self.assertIn("fresh", runtime.runtime_store._pending_run_steps)

    def test_pending_run_step_cache_is_bounded(self) -> None:
        runtime = CatalystFinderRuntime()
        for index in range(300):
            runtime.hold_run_step(f"run_{index}", {"step_type": "intent", "index": index})
        self.assertLessEqual(len(runtime.runtime_store._pending_run_steps), 256)
        self.assertLessEqual(len(runtime.runtime_store._pending_run_started), 256)
        for index in range(20):
            runtime.hold_run_step("same_run", {"step_type": "intent", "index": index})
        self.assertEqual(len(runtime.runtime_store._pending_run_steps["same_run"]), 8)
        self.assertEqual(runtime.runtime_store._pending_run_steps["same_run"][-1]["index"], 19)

    def test_missing_key_does_not_block_exact_rhea_mode(self) -> None:
        runtime = CatalystFinderRuntime()
        exact = type(
            "RheaRow",
            (),
            {
                "rhea_id": "RHEA:33983",
                "equation": "A = B",
                "as_dict": lambda self, *, model_ready: {
                    "rhea_id": self.rhea_id,
                    "equation": self.equation,
                    "model_ready": model_ready,
                },
            },
        )()
        runtime.rhea.exact = lambda _value: exact
        runtime.deepseek.parse = lambda *_args, **_kwargs: self.fail(
            "Exact Rhea resolution must not call DeepSeek."
        )
        payload = runtime.resolve("RHEA:33983")
        self.assertEqual(payload["recommended_id"], "RHEA:33983")
        self.assertEqual(payload["candidates"][0]["equation"], "A = B")

    def test_status_reports_general_product_universe_separately_from_project_catalog(self) -> None:
        runtime = CatalystFinderRuntime()
        payload = runtime.status()
        self.assertTrue(payload["build_revision"])
        self.assertGreater(payload["process_id"], 0)
        self.assertGreaterEqual(payload["uptime_seconds"], 0)
        self.assertEqual(payload["candidate_universe"], "general_merged")
        self.assertEqual(payload["candidate_enzymes"], 185918)
        self.assertEqual(payload["candidate_reactions"], 11081)
        self.assertEqual(payload["model_reactions"], 11081)
        self.assertEqual(payload["project_catalog"]["proteins"], 2085)
        self.assertEqual(payload["project_catalog"]["reactions"], 753)
        self.assertGreater(payload["recorded_associations"], 200000)

    def test_production_http_supports_head_without_python_fingerprint(self) -> None:
        class FakeRuntime:
            _route_catalog = {"counts": {}}

            @staticmethod
            def status():
                return {"status": "ready", "build_revision": "test-revision"}

        original_runtime = Handler.runtime
        Handler.runtime = FakeRuntime()
        server = ProductionHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            connection.request("HEAD", "/")
            response = connection.getresponse()
            body = response.read()
            self.assertEqual(response.status, 200)
            self.assertEqual(body, b"")
            self.assertGreater(int(response.getheader("Content-Length") or "0"), 0)
            self.assertEqual(response.getheader("Server"), "CatalystFinder")
            self.assertNotIn("Python", response.getheader("Server") or "")
            connection.close()

            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            connection.request("HEAD", "/api/status")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b"")
            self.assertEqual(response.getheader("Content-Type"), "application/json; charset=utf-8")
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            Handler.runtime = original_runtime

    def test_production_systemd_unit_and_manager_contract(self) -> None:
        root = Path(__file__).resolve().parents[2]
        unit = (root / "scripts/catalyst_finder/catalyst-finder.service").read_text(encoding="utf-8")
        manager = (root / "scripts/catalyst_finder/manage.sh").read_text(encoding="utf-8")
        self.assertIn("--host 127.0.0.1 --port 8791", unit)
        self.assertIn("Restart=on-failure", unit)
        self.assertIn("EnvironmentFile=-%h/igem2026/results/catalyst_finder_runtime/deepseek.env", unit)
        self.assertIn("Environment=CATALYST_PROTEIN_ENCODER_PREWARM=auto", unit)
        self.assertIn("NoNewPrivileges=true", unit)
        self.assertIn('HOST="${CATALYST_FINDER_HOST:-127.0.0.1}"', manager)
        self.assertIn("install-service", manager)
        self.assertIn('systemctl --user start "${SYSTEMD_UNIT_NAME}"', manager)

    def test_mixed_reaction_request_preserves_provided_fasta_as_positive_seed(self) -> None:
        runtime = CatalystFinderRuntime()
        sequence = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"
        text = (
            "For RHEA:32883, use the following experimentally active enzyme as a positive reference.\n"
            ">my_positive_seed\n"
            f"{sequence}"
        )
        runtime.deepseek.interpret_agent_request = lambda *_args, **_kwargs: {
            "direction": "reaction_to_enzyme",
            "confidence": 0.99,
            "alternative_direction": "",
            "ambiguity": False,
            "summary": "Use the supplied active enzyme as a positive reference.",
            "reaction": {"raw_text": "RHEA:32883", "substrate_terms": [], "product_terms": []},
            "enzyme": {"raw_text": "", "protein_terms": [], "organism_terms": [], "gene_terms": [], "accession_terms": []},
            "positive_enzymes": [],
        }
        runtime.resolve = lambda _value: {
            "mode": "rhea_id",
            "interpreted_reaction": "A = B",
            "assumptions": [],
            "candidates": [{"rhea_id": "RHEA:32883", "equation": "A = B"}],
            "recommended_id": "RHEA:32883",
        }
        payload = runtime.agent_resolve(text, direction_hint="auto", ui_language="en")
        self.assertEqual(payload["direction"], "reaction_to_enzyme")
        self.assertEqual(payload["reaction_resolution"]["recommended_id"], "RHEA:32883")
        self.assertEqual(len(payload["positive_enzyme_resolutions"]), 1)
        seed = payload["positive_enzyme_resolutions"][0]["candidates"][0]
        self.assertTrue(seed["id"].startswith("EXT-PROT-"))
        self.assertEqual(seed["sequence"], sequence)
        self.assertEqual(seed["input_mode"], "raw_protein_sequence")

    def test_ubia_family_query_resolves_as_family_instead_of_single_protein(self) -> None:
        runtime = CatalystFinderRuntime()
        runtime.deepseek.interpret_agent_request = lambda *_args, **_kwargs: {
            "direction": "enzyme_to_reaction",
            "confidence": 1.0,
            "alternative_direction": "",
            "ambiguity": False,
            "summary": "用户询问 UbiA 型萜环化酶可以催化哪些反应。",
            "reaction": {"raw_text": "", "substrate_terms": [], "product_terms": []},
            "enzyme": {
                "raw_text": "ubiA型萜环化酶",
                "protein_terms": ["ubiA-type terpene cyclase"],
                "organism_terms": [],
                "gene_terms": [],
                "accession_terms": [],
            },
            "positive_enzymes": [],
        }
        payload = runtime.agent_resolve(
            "ubiA型萜环化酶能催化什么反应",
            direction_hint="auto",
            ui_language="zh",
        )
        protein = payload["protein_resolution"]
        self.assertEqual(payload["direction"], "enzyme_to_reaction")
        self.assertEqual(protein["mode"], "protein_family")
        self.assertEqual(protein["recommended_id"], "PF01040")
        self.assertEqual(protein["family"]["member_count"], 24)
        self.assertIn("broader", protein["family"]["caution"])
        self.assertIsNone(runtime.families.resolve("UBIAD1 protein"))

    def test_pf01040_family_evidence_is_aggregated_without_fictitious_neural_query(self) -> None:
        runtime = CatalystFinderRuntime()

        class FakeRhea:
            def __init__(self, reaction_id: str) -> None:
                self.equation = f"equation for {reaction_id}"
                self.url = f"https://www.rhea-db.org/rhea/{reaction_id.split(':')[-1]}"

        runtime.rhea.exact = lambda reaction_id: FakeRhea(reaction_id)
        payload = runtime.rank_family_reactions("PF01040", ui_language="zh")
        self.assertEqual(payload["protein"]["input_mode"], "protein_family")
        self.assertEqual(payload["family"]["member_count"], 24)
        self.assertEqual(payload["family"]["evidence_member_count"], 13)
        self.assertEqual(payload["family"]["recorded_reaction_count"], 6)
        self.assertEqual(payload["known_associations"]["count"], 6)
        self.assertEqual(payload["candidates"], [])
        self.assertEqual(payload["ranking"]["route_id"], "e2r-family-evidence-v1")
        top = payload["known_associations"]["items"][0]
        self.assertEqual(top["candidate_id"], "RHEA:49632")
        self.assertEqual(top["family_support_count"], 8)
        self.assertEqual(top["family_member_count"], 24)
        self.assertIn("不会把整个家族虚构成一条平均蛋白序列", payload["score_note"])

    def test_explicit_uniprot_accession_precedes_family_description(self) -> None:
        runtime = CatalystFinderRuntime()
        row = type(
            "ProteinRow",
            (),
            {
                "name": "specific protein",
                "identifier": "P00338",
                "as_dict": lambda self: {"id": "P00338", "name": "specific protein"},
            },
        )()
        runtime.proteins.exact_or_search = lambda query, limit=8: [row] if query == "P00338" else []
        runtime.deepseek.interpret_agent_request = lambda *_args, **_kwargs: {
            "direction": "enzyme_to_reaction",
            "confidence": 1.0,
            "alternative_direction": "",
            "ambiguity": False,
            "summary": "Analyze the specified protein.",
            "reaction": {"raw_text": "", "substrate_terms": [], "product_terms": []},
            "enzyme": {
                "raw_text": "P00338 triterpene cyclase family",
                "protein_terms": ["triterpene cyclase family"],
                "organism_terms": [],
                "gene_terms": [],
                "accession_terms": ["P00338"],
            },
            "positive_enzymes": [],
        }
        payload = runtime.agent_resolve(
            "P00338 triterpene cyclase family",
            direction_hint="auto",
            ui_language="en",
        )
        protein = payload["protein_resolution"]
        self.assertEqual(protein["mode"], "protein_id")
        self.assertEqual(protein["recommended_id"], "P00338")
        self.assertNotIn("family", protein)

    def test_user_provided_external_sequence_materializes_as_temporary_seed(self) -> None:
        runtime = CatalystFinderRuntime()
        sequence = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "scripts.catalyst_finder.serve.RUNTIME_ROOT", Path(tmpdir)
        ):
            ids, path, metadata = runtime._prepare_seed_inputs(
                [],
                [{"id": "EXT-PROT-USERCONFIRMED", "sequence": sequence, "header": "active enzyme"}],
            )
            self.assertEqual(ids, ["EXT-PROT-USERCONFIRMED"])
            self.assertIsNotNone(path)
            assert path is not None
            self.assertTrue(path.is_file())
            rows = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(rows[0], "enzyme_id,sequence")
            self.assertIn(f"EXT-PROT-USERCONFIRMED,{sequence}", rows[1])
            self.assertEqual(metadata[0]["source"], "user_provided_sequence")

    def test_raw_protein_encoder_warmup_is_fixed_and_reported_in_status(self) -> None:
        runtime = CatalystFinderRuntime()

        class FakeEngine:
            def prewarm_protein_encoder(self):
                return {"model": "esmc_600m", "device": "cuda", "status": "ready"}

        runtime.model_gateway._engine = FakeEngine()
        result = runtime.prewarm_protein_encoder(background=False)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["model"], "esmc_600m")
        self.assertEqual(runtime.status()["open_world_protein_encoder"]["status"], "ready")

    def test_startup_encoder_prewarm_auto_defers_without_cuda(self) -> None:
        gateway = ModelGateway()
        result = gateway.startup_prewarm_protein_encoder(
            mode="auto",
            cuda_available=False,
        )
        self.assertEqual(result["status"], "deferred")
        self.assertEqual(result["reason"], "cuda_unavailable")
        self.assertEqual(result["policy"], "auto")

    def test_startup_encoder_prewarm_can_be_disabled_explicitly(self) -> None:
        gateway = ModelGateway()
        result = gateway.startup_prewarm_protein_encoder(mode="off")
        self.assertEqual(result["status"], "disabled")
        self.assertEqual(result["policy"], "off")

    def test_e2r_tps_specialized_scope_keeps_general_only_query_external(self) -> None:
        runtime = CatalystFinderRuntime()
        captured: dict[str, object] = {}
        sequence = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"

        runtime.e2r_planner.plan = lambda **_kwargs: {
            "top_k": 10,
            "ranking_objective": "top10",
            "known_association_policy": "allow_known",
            "known_reaction_ids": [],
            "mask_reaction_ids": [],
            "candidate_universe": TPS_SPECIALIZED_UNIVERSE,
            "planned_route_id": "e2r-external-top10-v1",
            "warnings": [],
        }
        runtime.route_designer.known_rhea_ids = lambda _accession: []
        runtime.proteins.uniprot.exact = lambda accession: {
            "accession": accession,
            "sequence": sequence,
            "name": "test protein",
            "organism": "test organism",
        }

        def fake_rank(command: str, payload: dict[str, object]) -> dict[str, object]:
            captured["command"] = command
            captured["payload"] = dict(payload)
            return {
                "query": {
                    "route_id": "e2r-external-top10-v1",
                    "scope": "external",
                    "shot_mode": "zero_shot",
                    "ranking_objective": "top10",
                    "score_source": "test",
                    "candidate_universe": TPS_SPECIALIZED_UNIVERSE,
                    "candidate_universe_size": 753,
                    "empirical_reliability_status": "test",
                },
                "candidates": [],
            }

        runtime.model_gateway.rank = fake_rank
        result = runtime.rank_reactions(
            "P00338",
            user_text="Restrict this query to the TPS-specialized candidate library.",
            route_mode="intelligent",
            ui_language="en",
        )
        payload = captured["payload"]
        assert isinstance(payload, dict)
        self.assertEqual(captured["command"], "rank-reactions")
        self.assertEqual(payload["candidate_universe"], TPS_SPECIALIZED_UNIVERSE)
        self.assertEqual(payload["enzyme_sequence"], sequence)
        self.assertNotIn("enzyme_id", payload)
        self.assertEqual(result["ranking"]["candidate_universe"], TPS_SPECIALIZED_UNIVERSE)

    def test_frontend_prewarms_only_when_verification_contains_raw_sequence(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn("function maybePrewarmProteinEncoder(resolution)", js)
        self.assertIn('candidate?.input_mode === "raw_protein_sequence"', js)
        self.assertIn('api("/api/warmup/protein-encoder", {})', js)
        self.assertIn("maybePrewarmProteinEncoder(resolution);", js)

    def test_result_scope_ui_is_prompt_only_not_stateful_selector(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn("scope-prompt-hints", html)
        self.assertIn('data-policy-prompt-en="Show recorded associations together with unrecorded candidates."', html)
        self.assertIn('data-policy-prompt-zh="同时查看数据库已知关联和新关联候选。"', html)
        self.assertIn('data-policy-prompt-en="Show only database-recorded associations."', html)
        self.assertIn('data-policy-prompt-zh="只看数据库已经记录的反应–酶关联。"', html)
        self.assertIn('data-policy-prompt-en="Show only unrecorded candidate associations."', html)
        self.assertIn('data-policy-prompt-zh="只看尚未记录的新关联候选。"', html)
        self.assertNotIn("data-result-scope", html)
        self.assertNotIn("resultScopeOverride", js)
        self.assertNotIn("known_association_policy:", js)
        start = js.index("function wirePolicyPromptButtons")
        end = js.index("function resetConversation", start)
        prompt_handler = js[start:end]
        self.assertIn("input.value", prompt_handler)
        self.assertNotIn("sendPrompt(", prompt_handler)
        self.assertNotIn("setRouteMode(", prompt_handler)

    def test_results_separate_database_evidence_from_unrecorded_candidate_ranking(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        css = (frontend / "styles.css").read_text(encoding="utf-8")
        self.assertIn('const known = result.known_associations', js)
        self.assertIn('const discoveryRows = mode.knownOnly ? [] : (result.candidates || [])', js)
        self.assertIn('tr("Known enzymes", "已知酶")', js)
        self.assertIn('tr("Retrieval score", "检索分数")', js)
        self.assertIn('tr("Unrecorded candidates", "新关联候选酶")', js)
        self.assertNotIn('Recorded database evidence; not a model prediction', js)
        self.assertNotIn('数据库已记录事实；不是模型预测', js)
        self.assertNotIn('The neural model covers this entity, but the database record is the primary evidence.', js)
        self.assertNotIn('该实体也被神经模型覆盖，但这里以数据库记录作为主要证据。', js)
        self.assertIn('Model retrieval score', js)
        self.assertNotIn('row.known_association ? "已知" : "潜在"', js)
        self.assertIn('.evidence-section', css)
        self.assertIn('.discovery-section', css)

    def test_open_world_inputs_are_visible_and_preserved_through_confirmation(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        transport = (Path(__file__).resolve().parent / "http_transport.py").read_text(encoding="utf-8")
        self.assertIn("Reaction SMILES, FASTA, or an amino-acid sequence", html)
        self.assertIn("Reaction SMILES 和 FASTA 也可以直接粘贴", html)
        self.assertIn('radio.dataset.reactionSmiles = candidate.reaction_smiles', js)
        self.assertIn('radio.dataset.sequence = candidate.sequence', js)
        self.assertIn('reaction_smiles: reactionSmiles', js)
        self.assertIn('confirmed_seed_inputs: positiveSequenceInputs', js)
        self.assertIn('enzyme_sequence: enzymeSequence', js)
        self.assertIn('reaction_smiles=str(payload.get("reaction_smiles") or "")', transport)
        self.assertIn('enzyme_sequence=str(payload.get("enzyme_sequence") or "")', transport)

    def test_family_level_e2r_is_visible_in_confirmation_execution_and_results(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        transport = (Path(__file__).resolve().parent / "http_transport.py").read_text(encoding="utf-8")
        self.assertIn('protein?.mode === "protein_family"', js)
        self.assertIn('endpoint: "/api/rank-family-reactions"', js)
        self.assertIn('family_id: familyId', js)
        self.assertIn("row.family_support_count", js)
        self.assertIn("家族整合证据", js)
        self.assertIn('parsed.path == "/api/rank-family-reactions"', transport)
        self.assertIn("rank_family_reactions", transport)

    def test_bilingual_ui_defaults_to_english_isolates_sessions_and_keeps_chinese_jargon_free(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        i18n = (frontend / "i18n.js").read_text(encoding="utf-8")
        backend_root = Path(__file__).resolve().parent
        backend = "\n".join(
            (backend_root / name).read_text(encoding="utf-8")
            for name in (
                "serve.py",
                "language_resolver.py",
                "agent_resolution_service.py",
                "retrieval_service.py",
                "route_pathway_service.py",
                "http_transport.py",
            )
        )
        self.assertIn('<html lang="en">', html)
        self.assertIn('id="languageToggle"', html)
        self.assertIn('<script src="/i18n.js" defer></script>', html)
        self.assertIn('data-prompt-en=', html)
        self.assertIn('data-prompt-zh=', html)
        self.assertIn('localStorage.getItem(STORAGE_KEY) || "en"', i18n)
        self.assertIn('location.reload()', i18n)
        self.assertIn('catalyst_finder_session_id_${uiLanguage}', js)
        self.assertIn('tr("Follow-up request:", "用户后续要求：")', js)
        self.assertIn('ui_language: uiLanguage', js)
        self.assertNotRegex(html, r'(?i)(?:data-zh|data-prompt-zh|data-placeholder-zh|data-aria-zh)="[^"]*discovery')
        self.assertNotIn('"Discovery 模型已覆盖"', js)
        self.assertNotIn('已知证据 + discovery', js)
        self.assertNotIn('个 discovery 候选', js)
        self.assertNotIn('数据库事实与 discovery 模型覆盖独立展示', backend)
        self.assertNotIn('模型分数只用于 discovery 候选', backend)
        self.assertIn("Call unrecorded model-ranked associations '新关联候选'", backend)

    def test_bilingual_product_copy_avoids_repetitive_defensive_explanations(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        backend_root = Path(__file__).resolve().parent
        backend = "\n".join(
            (backend_root / name).read_text(encoding="utf-8")
            for name in (
                "serve.py",
                "language_resolver.py",
                "agent_resolution_service.py",
                "retrieval_service.py",
                "route_pathway_service.py",
                "http_transport.py",
            )
        )
        combined = "\n".join([html, js, backend])
        for legacy in [
            "Start from the experimental question, not from model settings.",
            "从实验问题出发，不必先理解模型设置。",
            "These are examples, not separate modes",
            "下面只是示例，不是彼此割裂的模式",
            "Recorded database evidence; not a model prediction",
            "数据库已记录事实；不是模型预测",
            "The neural model covers this entity, but the database record is the primary evidence.",
            "该实体也被神经模型覆盖，但这里以数据库记录作为主要证据。",
            "This association is supported by the database even though the current neural candidate universe does not include this entity.",
            "即使当前神经模型候选空间尚未包含这个实体",
            "Use route-supported iML1515 FBA, not a titer prediction",
            "使用 iML1515 的整路通量约束评估，并不把它解释为产量预测",
            "Interpret compatibility in a chassis context rather than as a one-pot mixture",
            "按宿主细胞环境理解多酶兼容性，而不是按一锅体外体系处理",
            "Database evidence and model exploration are presented separately · model scores are not catalytic efficiency",
            "数据库事实与模型探索结果分开呈现 · 模型评分不等同于实验催化效率",
        ]:
            with self.subTest(legacy=legacy):
                self.assertNotIn(legacy, combined)
        self.assertIn('data-en="Start with your experimental goal."', html)
        self.assertIn('data-zh="从你的实验目标开始。"', html)
        self.assertIn('tr("Retrieval score", "检索分数")', js)
        self.assertIn("direct, natural Simplified Chinese", backend)
        self.assertIn("direct, natural scientific English", backend)

    def test_progressive_capability_guide_exposes_supported_research_scenarios(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        css = (frontend / "styles.css").read_text(encoding="utf-8")
        self.assertIn('id="capabilityGuide"', html)
        self.assertEqual(html.count('class="capability-action"'), 24)
        self.assertIn('class="starter-grid primary-task-grid"', html)
        for label in [
            "给一个反应，寻找催化酶",
            "给一个酶，寻找可能反应",
            "从起始前体设计到目标产物的路线",
            "评估多步路径的酶组合",
            "只找尚未记录的新关联",
            "寻找更远缘的蛋白家族",
            "仅真核候选",
            "仅原核候选",
            "使用 TPS 专用候选库",
            "分析 UniProt 蛋白",
            "从已有活性继续扩展",
            "优先热力学可行性",
            "优先 E. coli 宿主通量",
            "优先项目模型覆盖",
            "只用数据库已知反应",
            "探索预测反应步骤",
            "一锅多酶反应",
            "分步执行反应",
            "细胞内代谢路径",
            "指定 pH、温度和辅因子",
        ]:
            with self.subTest(label=label):
                self.assertIn(label, html)
        self.assertIn('data-prompt-en="For [target reaction], show only the database-recorded enzyme associations."', html)
        self.assertIn('data-prompt-zh="针对【目标反应】，只展示数据库已经记录的酶–反应关联。"', html)
        self.assertIn('data-prompt-en="For [target reaction], return only 10 unrecorded candidate enzyme associations."', html)
        self.assertIn('data-prompt-zh="针对【目标反应】，只给出 10 个尚未记录的新关联候选酶。"', html)
        self.assertIn('search eukaryotic proteins. Include recorded associations and return 10 unrecorded candidates.', html)
        self.assertIn('在真核蛋白中筛选。展示已知关联，并给出 10 个新关联候选。', html)
        self.assertIn('explicitly restrict candidate retrieval to the project\'s TPS / terpene-synthase-specialized library.', html)
        self.assertIn('明确只使用项目的 TPS / 萜类合酶专用候选库筛选。', html)
        self.assertIn('默认仍使用整合通用库', html)
        self.assertIn('using only database-recorded reactions.', html)
        self.assertIn('只使用数据库已经记录的反应。', html)
        self.assertIn('.capability-guide', css)
        self.assertIn('.capability-actions', css)
        self.assertNotIn('data-route-priority', html)
        self.assertNotIn('data-pathway-mode', html)

    def test_single_reaction_starter_is_explicitly_distinct_from_route_design(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        self.assertIn('data-direction-template="reaction_to_enzyme" data-prompt-en="I want to convert [substrate] to [product].', html)
        self.assertIn('data-prompt-zh="我想把【底物】转化为【产物】', html)
        self.assertIn('data-en="Find enzymes for a reaction"', html)
        self.assertIn('data-direction-template="route_design" data-prompt-en="Recommend biosynthetic routes from [starting precursor] to [target product]', html)
        self.assertIn('data-prompt-zh="请推荐从【起始前体】到【目标产物】', html)

    def test_route_design_ui_is_natural_language_first_without_priority_selector(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn("从起始前体设计到目标产物的路线", html)
        self.assertIn('data-direction-template="route_design"', html)
        self.assertIn("优先热力学可行性", html)
        self.assertIn("优先 E. coli 宿主通量", html)
        self.assertIn("只用数据库已知反应", html)
        self.assertIn("探索预测反应步骤", html)
        self.assertIn('"route_design", "pathway_compatibility"', js)
        self.assertNotIn("data-route-priority", html)
        self.assertNotIn("routePriorityOverride", js)
        self.assertIn('resolution.direction === "route_design"', js)
        self.assertIn('endpoint: "/api/route/design"', js)
        self.assertIn("eQuilibrator MDF", js)
        self.assertIn("iML1515", js)
        self.assertIn("MDF 未覆盖", js)
        start = js.index("function renderRouteDesignResult")
        end = js.index("function renderResult", start)
        result_renderer = js[start:end]
        self.assertIn("route-design-template-action", result_renderer)
        self.assertIn("input.value", result_renderer)
        self.assertNotIn("sendPrompt(", result_renderer)

    def test_pathway_ui_is_natural_language_first_without_new_selector(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn("评估多步路径的酶组合", html)
        self.assertIn("一锅多酶反应", html)
        self.assertIn("分步执行反应", html)
        self.assertIn("细胞内代谢路径", html)
        self.assertIn("指定 pH、温度和辅因子", html)
        self.assertIn('data-direction-template="pathway_compatibility"', html)
        self.assertNotIn('data-direction="pathway_compatibility"', html)
        self.assertNotIn('data-pathway-mode', html)
        self.assertIn('resolution.direction === "pathway_compatibility"', js)
        self.assertIn('endpoint: "/api/pathway/analyze"', js)

    def test_edited_starter_template_releases_soft_direction_hint(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn('const starterWasEdited = Boolean(run.card_id && run.prompt_template && text !== run.prompt_template);', js)
        self.assertIn('const effectiveHint = directionHintOneShot && starterWasEdited ? "auto" : directionHint;', js)
        self.assertIn('edited_after_card_click: Boolean(run.card_id && text !== run.prompt_template)', js)

    def test_continuation_carries_scope_as_context_without_locking_direction(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn("previous_association_policy", js)
        self.assertIn("previous_result_mode", js)
        self.assertIn("associationPolicy: continuationMode?.policy", js)
        self.assertIn("const effectiveHint = directionHint", js)
        self.assertNotIn("continuedHint", js)
        self.assertIn("directionHintOneShot", js)

    def test_right_rail_is_default_collapsed_and_expandable(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        css = (frontend / "styles.css").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="railToggle"', html)
        self.assertIn('aria-expanded="false"', html)
        self.assertIn('id="runRail"', html)
        self.assertIn('class="workspace rail-collapsed"', html)
        self.assertIn('.workspace.rail-collapsed{grid-template-columns:minmax(0,1fr)}', css)
        self.assertIn('.workspace.rail-collapsed .run-rail{display:none}', css)
        self.assertIn('function setRailCollapsed(collapsed)', js)
        self.assertIn('setRailCollapsed(true)', js)
        self.assertIn('railToggle?.addEventListener("click"', js)


if __name__ == "__main__":
    unittest.main()
