from __future__ import annotations

import http.client
import json
import re

import requests
import stat
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
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
from scripts.catalyst_finder.agent_harness.contracts import HarnessAction
from scripts.catalyst_finder.agent_harness.tool_registry import HarnessRunContext
from scripts.catalyst_finder.agent_harness.capabilities import public_capabilities
from scripts.catalyst_finder.model_gateway import ModelGateway
from scripts.catalyst_finder.retrieval_service import RetrievalApplicationService
from scripts.catalyst_finder.language_resolver import DeepSeekResolver
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

    def test_runtime_capabilities_include_live_tool_catalog(self) -> None:
        runtime = CatalystFinderRuntime()
        payload = runtime.capabilities()
        tool_names = [item["name"] for item in payload["tools"]]
        self.assertEqual(payload["version"], "catalyst-capabilities-v8")
        self.assertEqual(payload["tool_count"], len(tool_names))
        self.assertIn("resolve_reaction", tool_names)
        self.assertIn("candidate_search", tool_names)
        self.assertIn("lookup_relations", tool_names)
        self.assertIn("list_scope_members", tool_names)
        self.assertIn("resolve_compound", tool_names)
        self.assertTrue(payload["interaction"]["model_led"])
        self.assertTrue(payload["interaction"]["markdown_responses"])

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
        self.assertEqual(payload["agent_controller"], "model_led_scientific_harness")
        self.assertEqual(payload["agent_entrypoint"], "/api/agent/resolve")
        self.assertEqual(payload["agent_capabilities_version"], "catalyst-capabilities-v8")

    def test_contextual_followups_are_generated_from_supplied_result_context(self) -> None:
        resolver = DeepSeekResolver()
        captured = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "id": "followup-test",
                    "choices": [{"message": {"content": json.dumps({
                        "items": [
                            {
                                "prompt": "Inspect the returned structure AF-QTEST-F1 in detail.",
                                "title": "Inspect structure",
                                "reason": "It is present in the verified structure results.",
                                "priority": "high",
                            }
                        ]
                    })}}],
                }

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            return FakeResponse()

        resolver.session.post = fake_post
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            items = resolver.suggest_next_steps(
                result_context={
                    "answer_mode": "research_workspace",
                    "entity": {"kind": "protein", "id": "QTEST1"},
                    "source_panels": [{
                        "section": "structures",
                        "items": [{"id": "AF-QTEST-F1", "source": "AlphaFoldDB"}],
                    }],
                },
                session_facts={"focus": {"protein": "QTEST1"}},
                tool_catalog=[{"name": "inspect_entity", "purpose": "Inspect one verified entity."}],
                ui_language="en",
            )
        self.assertEqual([row["prompt"] for row in items], ["Inspect the returned structure AF-QTEST-F1 in detail."])
        messages = captured["json"]["messages"]
        system_prompt = messages[0]["content"]
        request_payload = json.loads(messages[1]["content"])
        self.assertIn("Use ONLY the supplied current_result", system_prompt)
        self.assertIn("Avoid generic menu text, fixed Top-10 defaults", system_prompt)
        self.assertEqual(request_payload["current_result"]["entity"]["id"], "QTEST1")
        self.assertEqual(request_payload["current_result"]["source_panels"][0]["items"][0]["id"], "AF-QTEST-F1")

    def test_followups_http_endpoint_delegates_without_static_templates(self) -> None:
        class FakeRuntime:
            _route_catalog = {"counts": {}}

            @staticmethod
            def suggest_followups(payload):
                assert payload["result_context"]["entity"]["id"] == "QTEST1"
                return {"status": "ok", "items": [{"prompt": "Inspect QTEST1 annotations."}]}

        original_runtime = Handler.runtime
        Handler.runtime = FakeRuntime()
        server = ProductionHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            body = json.dumps({"result_context": {"entity": {"id": "QTEST1"}}})
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            connection.request("POST", "/api/followups", body=body, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(payload["items"][0]["prompt"], "Inspect QTEST1 annotations.")
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            Handler.runtime = original_runtime

    def test_grounded_synthesis_retries_truncated_json_with_shorter_correction(self) -> None:
        resolver = DeepSeekResolver()

        class Response:
            def __init__(self, content, rid):
                self._content = content
                self._rid = rid
            def raise_for_status(self):
                return None
            def json(self):
                return {"id": self._rid, "choices": [{"message": {"content": self._content}}]}

        class Session:
            def __init__(self):
                self.calls = []
            def post(self, *args, **kwargs):
                self.calls.append(kwargs["json"])
                if len(self.calls) == 1:
                    return Response('{"answer":"truncated', "r1")
                return Response(json.dumps({"answer": "简短且完整的结论。", "evidence_ids": ["MED:1"], "limitations": []}, ensure_ascii=False), "r2")

        fake = Session()
        resolver.session = fake
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key", "DEEPSEEK_MODEL": "fake-model"}, clear=False):
            result = resolver.synthesize_grounded_answer(
                user_text="总结文献", tool_history=[],
                verified_evidence=[{"tool": "inspect_entity", "result": {"immediate_result": {"entities": [{"id": "MED:1"}]}}}],
                current_result={}, ui_language="zh",
            )
        self.assertEqual(result["answer"], "简短且完整的结论。")
        self.assertEqual(result["evidence_ids"], ["MED:1"])
        self.assertEqual(len(fake.calls), 2)
        self.assertIn("previous synthesis response was invalid", fake.calls[1]["messages"][-1]["content"].lower())
        self.assertEqual(fake.calls[1]["max_tokens"], 2600)

    def test_grounded_synthesis_retries_transient_network_failure(self) -> None:
        resolver = DeepSeekResolver()

        class Response:
            def raise_for_status(self): return None
            def json(self):
                return {
                    "id": "network-retry-ok",
                    "choices": [{"message": {"content": json.dumps({
                        "answer": "Recovered evidence-based answer.",
                        "evidence_ids": ["MED:1"],
                        "limitations": [],
                    })}}],
                }

        class Session:
            def __init__(self): self.calls = 0
            def post(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise requests.Timeout("temporary provider timeout")
                return Response()

        fake = Session(); resolver.session = fake
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key", "DEEPSEEK_MODEL": "fake-model"}, clear=False):
            result = resolver.synthesize_grounded_answer(
                user_text="Summarize the paper.", tool_history=[],
                verified_evidence=[{"tool": "inspect_entity", "result": {"immediate_result": {"entities": [{"id": "MED:1"}]}}}],
                current_result={}, ui_language="en",
            )
        self.assertEqual(result["answer"], "Recovered evidence-based answer.")
        self.assertEqual(fake.calls, 2)

    def test_grounded_synthesis_rewrites_unsupported_sensitive_localization(self) -> None:
        resolver = DeepSeekResolver()

        class Response:
            def __init__(self, body): self._body = body
            def raise_for_status(self): return None
            def json(self): return self._body

        class Session:
            def __init__(self): self.calls = []
            def post(self, *args, **kwargs):
                self.calls.append(kwargs["json"])
                if len(self.calls) == 1:
                    answer = "FLCN inhibits cytosolic LDHA."
                else:
                    answer = "FLCN inhibits LDHA."
                content = json.dumps({"answer": answer, "evidence_ids": ["MED:1"], "limitations": []})
                return Response({"id": f"r{len(self.calls)}", "choices": [{"message": {"content": content}}]})

        fake = Session(); resolver.session = fake
        evidence = [{"tool": "inspect_entity", "result": {"immediate_result": {"entities": [{
            "id": "MED:1", "source": "MED", "abstract": "FLCN binds LDHA and acts as an uncompetitive inhibitor."
        }]}}}]
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key", "DEEPSEEK_MODEL": "fake-model"}, clear=False):
            result = resolver.synthesize_grounded_answer(
                user_text="Compare the mechanism.", tool_history=[], verified_evidence=evidence, current_result={}, ui_language="en",
            )
        self.assertEqual(result["answer"], "FLCN inhibits LDHA.")
        self.assertEqual(len(fake.calls), 2)
        self.assertIn("cytosolic localization", fake.calls[1]["messages"][-1]["content"])

    def test_grounded_synthesis_retries_unsupported_protected_identifier(self) -> None:
        resolver = DeepSeekResolver()

        class Response:
            def __init__(self, body): self._body = body
            def raise_for_status(self): return None
            def json(self): return self._body

        class Session:
            def __init__(self): self.calls = []
            def post(self, *args, **kwargs):
                self.calls.append(kwargs["json"])
                if len(self.calls) == 1:
                    content = json.dumps({"answer": "MED:999 报道了额外结论。", "evidence_ids": ["MED:999"], "limitations": []}, ensure_ascii=False)
                else:
                    content = json.dumps({"answer": "MED:1 的已核对内容有限。", "evidence_ids": ["MED:1"], "limitations": []}, ensure_ascii=False)
                return Response({"id": f"r{len(self.calls)}", "choices": [{"message": {"content": content}}]})

        fake = Session(); resolver.session = fake
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key", "DEEPSEEK_MODEL": "fake-model"}, clear=False):
            result = resolver.synthesize_grounded_answer(
                user_text="总结文献", tool_history=[],
                verified_evidence=[{"tool": "inspect_entity", "result": {"immediate_result": {"entities": [{"id": "MED:1"}]}}}],
                current_result={}, ui_language="zh",
            )
        self.assertEqual(result["answer"], "MED:1 的已核对内容有限。")
        self.assertEqual(result["evidence_ids"], ["MED:1"])
        self.assertEqual(len(fake.calls), 2)
        self.assertIn("MED:999", fake.calls[1]["messages"][-1]["content"])

    def test_real_session_selector_honors_current_page_local_ordinal(self) -> None:
        resolver = DeepSeekResolver()
        records = [
            {"kind": "literature", "id": "MED:11", "label": "first visible", "visible": True, "visible_index": 1, "visible_page_index": 1, "related_index": 11},
            {"kind": "literature", "id": "MED:12", "label": "second visible", "visible": True, "visible_index": 2, "visible_page_index": 1, "related_index": 12},
            {"kind": "literature", "id": "MED:2", "label": "global second", "visible": False, "related_index": 2},
        ]
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}, clear=False):
            zh = resolver.select_session_entity_reference(
                user_text="这页第二篇具体讲了什么？", records=records, expected_kind="literature", ui_language="zh",
            )
            en = resolver.select_session_entity_reference(
                user_text="What does the second paper on this page report?", records=records, expected_kind="literature", ui_language="en",
            )
            narrowed = resolver.select_session_entity_reference(
                user_text="第二篇", context_text="这页第二篇具体讲了什么？", records=records, expected_kind="literature", ui_language="zh",
            )
        self.assertEqual(zh["selected_key"], "literature:MED:12")
        self.assertEqual(en["selected_key"], "literature:MED:12")
        self.assertEqual(narrowed["selected_key"], "literature:MED:12")
        self.assertEqual(zh["reference_mode"], "specific")
        self.assertIn("current-page ordinal 2", zh["reason"])

    def test_view_context_endpoint_marks_only_server_verified_entities(self) -> None:
        class Sessions:
            def mark_visible_entities(self, session_id, *, entity_kind, entity_ids, page_index=0):
                self.args = (session_id, entity_kind, list(entity_ids), page_index)
                return {"entity_kind": entity_kind, "visible_count": 2, "page_index": page_index, "visible_ids": entity_ids[:2]}

        sessions = Sessions()
        runtime = SimpleNamespace(agent_sessions=sessions)
        from scripts.catalyst_finder.serve import CatalystFinderRuntime
        result = CatalystFinderRuntime.update_view_context(runtime, {
            "session_id": "s1", "entity_kind": "literature",
            "entity_ids": ["MED:11", "MED:12", "MED:999"], "page_index": 2,
        })
        self.assertEqual(sessions.args, ("s1", "literature", ["MED:11", "MED:12", "MED:999"], 2))
        self.assertEqual(result["visible_count"], 2)

    def test_literature_page_http_endpoint_delegates_cursor_pagination(self) -> None:
        class FakeRuntime:
            _route_catalog = {"counts": {}}

            @staticmethod
            def literature_page(payload):
                assert payload["query"] == "enzyme regulation"
                assert payload["provider"] == "openalex"
                assert payload["cursor"] == "cursor-2"
                return {
                    "id": "literature", "count": 25,
                    "items": [{"id": "11", "title": "page two"}],
                    "pagination": {"mode": "remote", "next_cursor": "cursor-3", "has_more": True},
                }

        original_runtime = Handler.runtime
        Handler.runtime = FakeRuntime()
        server = ProductionHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            body = json.dumps({"query": "enzyme regulation", "provider": "openalex", "cursor": "cursor-2", "page_size": 10, "page_index": 1})
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            connection.request("POST", "/api/research/literature-page", body=body, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(payload["items"][0]["id"], "11")
            self.assertEqual(payload["pagination"]["next_cursor"], "cursor-3")
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            Handler.runtime = original_runtime

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
        runtime.agent_resolution.resolve = lambda _value: {
            "mode": "rhea_id",
            "interpreted_reaction": "A = B",
            "assumptions": [],
            "candidates": [{"rhea_id": "RHEA:32883", "equation": "A = B"}],
            "recommended_id": "RHEA:32883",
        }
        actions = iter([
            HarnessAction(kind="tool", tool="candidate_search", args={"direction": "reaction_to_enzyme", "full_text": text, "reaction_text": "RHEA:32883"}),
        ])
        runtime.deepseek.next_harness_action = lambda **_kwargs: next(actions)
        payload = runtime.agent_resolve(text, ui_language="en")
        self.assertEqual(payload["direction"], "reaction_to_enzyme")
        self.assertEqual(payload["reaction_resolution"]["recommended_id"], "RHEA:32883")
        self.assertEqual(len(payload["positive_enzyme_resolutions"]), 1)
        seed = payload["positive_enzyme_resolutions"][0]["candidates"][0]
        self.assertTrue(seed["id"].startswith("EXT-PROT-"))
        self.assertEqual(seed["sequence"], sequence)
        self.assertEqual(seed["input_mode"], "raw_protein_sequence")
        self.assertEqual(payload["agent_execution"]["steps"][0]["tool"], "candidate_search")

    def test_natural_language_functional_class_uses_generic_semantic_scope(self) -> None:
        runtime = CatalystFinderRuntime()
        runtime.deepseek.parse_protein = lambda _text: {
            "protein_terms": ["membrane prenyltransferase"],
            "organism_terms": [],
            "gene_terms": [],
            "accession_terms": [],
            "interpreted_protein": "membrane prenyltransferase",
        }
        runtime.deepseek.expand_protein_class_terms = lambda **_kwargs: {
            "strict_terms": ["membrane prenyltransferase"],
            "broader_terms": ["prenyltransferase"],
        }
        runtime.family_evidence.summarize_functional_class = lambda spec, ui_language="en": {
            "protein": {"id": "CLASS:semantic", "name": spec["raw_text"], "input_mode": "protein_functional_class"},
            "family": {
                "scope_id": "CLASS:semantic",
                "label": spec["raw_text"],
                "member_count": 7,
                "evidence_member_count": 4,
                "recorded_reaction_count": 3,
            },
            "known_associations": {"count": 3, "items": [], "note": "functional-class evidence"},
            "candidates": [],
            "ranking": {"route_id": "e2r-functional-class-evidence-v1"},
        }
        actions = iter([
            HarnessAction(kind="tool", tool="resolve_protein_scope", args={"text": "membrane prenyltransferase", "scope_hint": "family_or_class"}),
            HarnessAction(kind="tool", tool="lookup_relations", args={"protein_scope_ref": "protein_scope_1"}),
            HarnessAction(kind="return_result"),
        ])
        runtime.deepseek.next_harness_action = lambda **_kwargs: next(actions)
        payload = runtime.agent_resolve("What reactions are recorded for membrane prenyltransferases?", ui_language="en")
        protein = payload["protein_resolution"]
        self.assertEqual(payload["direction"], "enzyme_to_reaction")
        self.assertEqual(protein["mode"], "protein_functional_class")
        self.assertEqual(protein["recommended_id"], "CLASS:semantic")
        self.assertEqual(protein["family"]["member_count"], 7)
        self.assertFalse(payload["agent_execution"]["fallback"])

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
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        result = runtime.agent_tools.execute(
            "resolve_protein_scope",
            {"text": "UniProt P00338 triterpene cyclase family", "scope_hint": "specific_protein"},
            ctx,
        )
        self.assertEqual(result.status, "ok")
        ref = result.payload["protein_scope_ref"]
        scope = ctx.protein_refs[ref]
        self.assertEqual(scope["kind"], "specific_protein")
        self.assertEqual(scope["resolution"]["recommended_id"], "P00338")
        self.assertNotIn("family", scope["resolution"])

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
            "known_association_policy": "separate_known",
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

    def test_result_scope_and_retrieval_modes_have_no_frontend_selector(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        css = (frontend / "styles.css").read_text(encoding="utf-8")
        self.assertNotIn("scope-prompt-hints", html)
        self.assertNotIn("composer-settings", html)
        self.assertNotIn("筛选设置", html)
        self.assertNotIn("data-direction=", html)
        self.assertNotIn("data-route-mode=", html)
        self.assertNotIn("directionHint", js)
        self.assertNotIn("routeMode", js)
        self.assertNotIn("wirePolicyPromptButtons", js)
        self.assertNotIn(".scope-prompt-hints", css)
        self.assertNotIn(".settings-popover", css)
        self.assertNotIn("无需选择模式", html)
        self.assertNotIn("筛选设置", html)

    def test_model_score_uses_engine_evidence_strength_without_query_max_normalization(self) -> None:
        first = RetrievalApplicationService._model_support_fields({
            "score": 0.015, "evidence_passport": {"score": 0.54, "tier": "review_candidate"}
        })
        second = RetrievalApplicationService._model_support_fields({
            "score": 0.99, "evidence_passport": {"score": 0.967, "tier": "priority_candidate"}
        })
        self.assertEqual(first["model_support_index"], 54.0)
        self.assertEqual(second["model_support_index"], 96.7)
        self.assertNotEqual(first["model_support_index"], 100.0)
        self.assertNotEqual(second["model_support_index"], 100.0)

    def test_frontend_shows_model_score_and_keeps_formula_in_technical_fold(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        start = js.index("function renderResult(result, direction)")
        end = js.index("function normalizeRouteFlow", start)
        block = js[start:end]
        self.assertIn('tr("Model score", "模型评分")', block)
        self.assertIn("row.model_support_index", block)
        self.assertIn("Model score formula", block)
        self.assertIn("0.35·rank priority", block)
        self.assertIn("Raw retrieval scores · audit only", block)
        self.assertNotIn('localizedBackendText(result.score_note', block)
        retrieval = (Path(__file__).resolve().parent / "retrieval_service.py").read_text(encoding="utf-8")
        self.assertNotIn('"score_note": _lang_text(', retrieval)
        self.assertNotIn('Number(row.score_fraction || 0) * 100', block)

    def test_results_separate_database_evidence_from_unrecorded_candidate_ranking(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        css = (frontend / "styles.css").read_text(encoding="utf-8")
        self.assertIn('const known = result.known_associations', js)
        self.assertIn('const discoveryRows = mode.knownOnly ? [] : (result.candidates || [])', js)
        self.assertIn('tr("Known enzymes", "已知酶")', js)
        self.assertIn('tr("Model score", "模型评分")', js)
        self.assertIn('tr("Unrecorded candidates", "新关联候选酶")', js)
        self.assertIn('tr("Unified zero-shot model ranking", "统一 Zero-shot 模型排名")', js)
        self.assertIn('row.model_support_index', js)
        self.assertNotIn('Number(row.score_fraction || 0) * 100', js)
        self.assertNotIn('Recorded database evidence; not a model prediction', js)
        self.assertNotIn('数据库已记录事实；不是模型预测', js)
        self.assertNotIn('The neural model covers this entity, but the database record is the primary evidence.', js)
        self.assertNotIn('该实体也被神经模型覆盖，但这里以数据库记录作为主要证据。', js)
        self.assertIn('Model score', js)
        self.assertIn('Raw retrieval scores · audit only', js)
        self.assertNotIn('row.known_association ? "已知" : "潜在"', js)
        self.assertIn('.evidence-section', css)
        self.assertIn('.discovery-section', css)

    def test_open_world_inputs_are_model_led_and_preserved_through_confirmation(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        transport = (Path(__file__).resolve().parent / "http_transport.py").read_text(encoding="utf-8")
        harness = (Path(__file__).resolve().parent / "agent_harness" / "harness.py").read_text(encoding="utf-8")
        capabilities = public_capabilities()
        self.assertIn("Reaction SMILES", json.dumps(capabilities, ensure_ascii=False))
        self.assertIn("amino-acid sequence", capabilities["interaction"]["structured_inputs"])
        self.assertIn("FASTA", capabilities["interaction"]["structured_inputs"])
        self.assertTrue(capabilities["interaction"]["model_led"])
        self.assertNotIn("deterministic_fast_path", harness)
        self.assertIn('radio.dataset.reactionSmiles = candidate.reaction_smiles', js)
        self.assertIn('radio.dataset.sequence = candidate.sequence', js)
        self.assertIn('reaction_smiles: reactionSmiles', js)
        self.assertIn('confirmed_seed_inputs: positiveSequenceInputs', js)
        self.assertIn('enzyme_sequence: enzymeSequence', js)
        self.assertIn('reaction_smiles=str(payload.get("reaction_smiles") or "")', transport)
        self.assertIn('enzyme_sequence=str(payload.get("enzyme_sequence") or "")', transport)
        self.assertNotIn('parsed.path == "/api/resolve"', transport)
        self.assertNotIn('parsed.path == "/api/resolve-protein"', transport)
        agent_block = transport[transport.index('parsed.path == "/api/agent/resolve"'):transport.index('parsed.path == "/api/warmup/protein-encoder"')]
        self.assertNotIn("direction_hint", agent_block)

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

    def test_normal_retrieval_does_not_cap_recorded_associations_at_twenty(self) -> None:
        source = (Path(__file__).resolve().parent / "retrieval_service.py").read_text(encoding="utf-8")
        self.assertNotIn("known_reactions[:20]", source)
        self.assertNotIn("known_association_ids[:20]", source)
        self.assertIn('"truncated": False', source)

    def test_research_workspace_visibly_fuses_live_sources_known_relations_and_model_lens(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        css = (frontend / "styles.css").read_text(encoding="utf-8")
        capabilities = (Path(__file__).resolve().parent / "agent_harness" / "capabilities.py").read_text(encoding="utf-8")
        resolver = (Path(__file__).resolve().parent / "language_resolver.py").read_text(encoding="utf-8")
        self.assertIn('function renderResearchWorkspace(result)', js)
        self.assertIn('result?.answer_mode === "research_workspace"', js)
        self.assertIn('tr("Model lens", "模型视角")', js)
        self.assertIn('selected_sections', js)
        self.assertIn('className = `research-module module-${id}', js)
        self.assertIn('tr("Database annotations", "数据库注释")', js)
        self.assertIn('tr("Literature", "关联文献")', js)
        self.assertNotIn('tr("Current research sources", "当前资料与注释")', js)
        self.assertIn('.research-workspace-composable', css)
        self.assertIn('.research-module>summary', css)
        self.assertIn('"version": "catalyst-capabilities-v8"', capabilities)
        self.assertIn('"title_zh": "科研资料工作区"', capabilities)
        self.assertIn('按本轮问题组合注释、结构、文献、已记录关系、模型分析和下一步优先级', capabilities)
        self.assertNotIn('没有请求的模块不会执行', capabilities)
        self.assertIn('A factual enzyme↔reaction lookup is complete after lookup_relations', resolver)
        self.assertIn('with exactly those requested sections', resolver)

    def test_research_workspace_folded_source_lists_use_shared_nested_pagination(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        start = js.index("function renderResearchWorkspace(result)")
        end = js.index("function normalizeRouteFlow", start)
        block = js[start:end]
        for expected in (
            "paginateInto(facts, panel.facts",
            "paginateInto(tags, panel.participants",
            "paginateInto(rows, panel.catalytic_activities",
            "paginateInto(rows, panel.cofactors",
            "paginateInto(rows, annotationRows",
            "paginateInto(tags, xrefItems",
            "paginateInto(tags, officialProteins",
        ):
            self.assertIn(expected, block)
        for legacy in (
            "panel.facts.slice(0, 8)",
            "panel.participants.slice(0, 12)",
            "panel.catalytic_activities.slice(0, 5)",
            "panel.cofactors.slice(0, 8)",
            "annotationEntries.slice(0, 5)",
            "xrefItems.slice(0, 16)",
        ):
            self.assertNotIn(legacy, block)
        self.assertIn("panel.official_uniprot_items", block)

    def test_scientific_capability_guide_is_comprehensive_but_nested(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        css = (frontend / "styles.css").read_text(encoding="utf-8")
        manifest = public_capabilities()
        groups = {row["id"]: row for row in manifest["groups"]}
        self.assertEqual(set(groups), {"research_workspace", "evidence", "compound_identity", "candidate_retrieval", "route_design", "pathway"})
        self.assertNotIn("conversation", groups)
        self.assertEqual(manifest["version"], "catalyst-capabilities-v8")
        self.assertIn("这个酶", manifest["interaction"]["guide_note_zh"])
        zh_text = "\n".join(
            [str(group.get("title_zh") or "") + " " + str(group.get("description_zh") or "") for group in groups.values()]
            + [str(example.get("title_zh") or "") + " " + str(example.get("description_zh") or "") for group in groups.values() for example in group.get("examples") or []]
        )
        for expected in ("PDB", "AlphaFold", "文献", "MDF", "iML1515", "MINE/Pickaxe", "辅因子", "亚细胞定位", "远缘候选", "真核", "seed", "TPS 专用候选库", "Zero-shot 混排"):
            self.assertIn(expected, zh_text)
        self.assertGreaterEqual(sum(len(group.get("examples") or []) for group in groups.values()), 30)
        self.assertIn('const section = document.createElement("details")', js)
        self.assertIn('section.className = "capability-group"', js)
        self.assertIn('requestContextualFollowUps(card, result, "research_workspace")', js)
        self.assertIn('api("/api/followups"', js)
        self.assertIn('compactFollowUpContext(result, direction)', js)
        self.assertIn('function paginateRemoteInto(', js)
        self.assertIn('/api/research/literature-page', js)
        self.assertIn('provider: panel.provider || panel?.pagination?.provider || "europe_pmc"', js)
        self.assertIn('String(panel?.id || "").startsWith("literature_")', js)
        self.assertIn('/api/session/view-context', js)
        self.assertIn('function publishVisiblePage(', js)
        self.assertIn('viewContext: literatureViewContext', js)
        self.assertIn('panel?.pagination?.mode === "remote"', js)
        source_block = js[js.index('const remoteLiterature = literaturePanel'):js.index('return source;', js.index('const remoteLiterature = literaturePanel'))]
        mount_pos = js.rfind('source.appendChild(items);', 0, js.index('const remoteLiterature = literaturePanel'))
        self.assertGreater(mount_pos, js.index('const literatureViewContext'))
        self.assertLess(mount_pos, js.index('const remoteLiterature = literaturePanel'))
        self.assertNotIn('researchFollowUpPrompts', js)
        self.assertNotIn('genericFollowUps', js)
        self.assertNotIn('Open the first paper above', js)
        self.assertIn('const clean = [...new Set(', js)
        self.assertIn('.result-follow-ups', css)
        self.assertIn('.capability-group>summary', css)

    def test_visual_system_unifies_cards_and_mobile_candidate_rows(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        css = (frontend / "styles.css").read_text(encoding="utf-8")
        index = (frontend / "index.html").read_text(encoding="utf-8")
        self.assertIn("/app.js?v=20260829-agent-tools-v8", index)
        self.assertIn("Unified visual system v1", css)
        for token in ("--ui-card-radius", "--ui-inner-radius", "--ui-card-border", "--ui-card-shadow"):
            self.assertIn(token, css)
        for selector in (".verification-card", ".result-card", ".capability-guide", ".rail-card"):
            self.assertIn(selector, css)
        self.assertIn("@media(max-width:700px)", css)
        self.assertIn("@media(max-width:520px)", css)
        self.assertIn("@media(max-width:430px)", css)
        self.assertIn("@media(max-width:370px)", css)
        self.assertIn(".assistant-message{grid-template-columns:minmax(0,1fr);gap:0}", css)
        self.assertIn(".discovery-table-wrap tbody tr{display:grid", css)
        self.assertIn(".discovery-table-wrap table{display:block;width:100%;min-width:0}", css)
        self.assertNotIn(".assistant-message{grid-template-columns:28px", css)
        self.assertIn("Unbreakable-content guard", css)
        self.assertIn("overflow-wrap:anywhere", css)
        self.assertIn("word-break:break-all", css)
        self.assertIn(".research-reaction-smiles", css)
        self.assertIn(".research-source-item>a", css)

    def test_generic_resolution_logic_has_no_project_fixture_special_cases(self) -> None:
        root = Path(__file__).resolve().parent
        resolver = (root / "language_resolver.py").read_text(encoding="utf-8")
        families = (root / "protein_family_catalog.py").read_text(encoding="utf-8")
        route_design = (root / "route_design.py").read_text(encoding="utf-8")
        for fixture_text in ("miltiradiene", "Salvia miltiorrhiza", "CURATED_FAMILY_ALIASES", "UbiA-type terpene cyclase"):
            self.assertNotIn(fixture_text, resolver + families)
        self.assertNotIn("ALIASES = {", route_design)
        for short_alias in ('"gpp":', '"fpp":', '"ggpp":', '"dmapp":'):
            self.assertNotIn(short_alias, route_design)
        self.assertIn("normalize_compound_terms", resolver)
        self.assertIn("family_or_class", resolver)

    def test_multiturn_state_lifecycle_invalidates_stale_cards_rotates_sessions_and_logs_each_step(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        transport = (Path(__file__).resolve().parent / "http_transport.py").read_text(encoding="utf-8")
        self.assertIn('function rotateSessionId()', js)
        self.assertIn('supersedeActiveVerification("new_user_message")', js)
        self.assertIn('supersedeActiveVerification("conversation_reset")', js)
        self.assertIn('pending.button.textContent = tr("Superseded by later request", "已被后续请求替代")', js)
        self.assertIn('recordClientEvent("confirmation_validation_failed"', js)
        self.assertIn('recordClientEvent("confirmation_execution_failed"', js)
        self.assertIn('recordClientEvent("confirmation_execution_succeeded"', js)
        self.assertIn('candidate_count: card.querySelectorAll(".protein-option input").length', js)
        self.assertIn('event_type="run_step"', transport)
        self.assertIn('"step_type": event_type', transport)
        self.assertNotIn("composerContext", js)
        self.assertNotIn("useContinuation", js)
        self.assertNotIn("previous_target: continuation", js)
        serve_source = (Path(__file__).resolve().parent / "serve.py").read_text(encoding="utf-8")
        self.assertIn("self.agent_sessions.execution_context(session_id, ui_language=ui_language)", serve_source)
        self.assertIn("self.agent_sessions.remember_execution_result", serve_source)

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
        self.assertIn('<script src="/i18n.js?v=20260829-bilingual-v8" defer></script>', html)
        self.assertIn('id="capabilityGuideBody"', html)
        self.assertIn('api("/api/capabilities").then(renderCapabilities)', js)
        self.assertIn("prompt_en", json.dumps(public_capabilities(), ensure_ascii=False))
        self.assertIn("prompt_zh", json.dumps(public_capabilities(), ensure_ascii=False))
        self.assertIn('localStorage.getItem(STORAGE_KEY) || "en"', i18n)
        self.assertIn('location.reload()', i18n)
        self.assertIn('catalyst_finder_session_id_${uiLanguage}', js)
        self.assertNotIn('tr("Follow-up request:", "用户后续要求：")', js)
        self.assertIn("const effectiveText = text;", js)
        self.assertIn('ui_language: uiLanguage', js)
        self.assertNotRegex(html, r'(?i)(?:data-zh|data-prompt-zh|data-placeholder-zh|data-aria-zh)="[^"]*discovery')
        self.assertNotIn('"Discovery 模型已覆盖"', js)
        self.assertNotIn('已知证据 + discovery', js)
        self.assertNotIn('个 discovery 候选', js)
        self.assertNotIn('数据库事实与 discovery 模型覆盖独立展示', backend)
        self.assertNotIn('模型分数只用于 discovery 候选', backend)
        self.assertIn("Call unrecorded model-ranked associations '新关联候选'", backend)
        self.assertIn("已记录/数据库记录/known/recorded", backend)
        self.assertIn("is already an evidence-only restriction", backend)
        self.assertIn("do not infer a protein's usual biological role", backend)
        self.assertIn("do not append an unsolicited menu", backend)
        self.assertIn("explicitly names BOTH a concrete protein and a concrete reaction", backend)
        self.assertIn("then call lookup_relations with that protein_scope_ref", backend)
        self.assertIn("Multiple individual resolve_reaction calls are not a completed pathway analysis", backend)
        self.assertIn("individual reaction identities/equations are never sufficient", backend)
        self.assertIn("Never infer one-pot/pathway compatibility", backend)
        self.assertIn("only changes result policy/view", backend)
        self.assertIn("Never copy an ID/name learned only from session history", backend)
        self.assertIn("INCLUDING a continuation that changes only result/output constraints", backend)

    def test_bilingual_ui_contract_has_no_cross_language_contamination(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        i18n = (frontend / "i18n.js").read_text(encoding="utf-8")
        cjk = re.compile(r"[\u3400-\u9fff]")

        # Static HTML pairs: English slots cannot contain Chinese UI copy.
        for value in re.findall(r'data-en="([^"]*)"', html):
            self.assertIsNone(cjk.search(value), value)
        for value in re.findall(r'data-placeholder-en="([^"]*)"', html):
            self.assertIsNone(cjk.search(value), value)
        for value in re.findall(r'data-aria-en="([^"]*)"', html):
            self.assertIsNone(cjk.search(value), value)

        # Literal tr(en, zh) calls preserve argument direction. Chinese may be
        # language-neutral for technical text such as Top-K, but English cannot
        # accidentally contain Chinese UI copy.
        literal_pairs = re.findall(
            r'tr\(\s*(["`\'])(.*?)\1\s*,\s*(["`\'])(.*?)\3\s*\)', js, re.S
        )
        self.assertGreater(len(literal_pairs), 300)
        for _q1, en, _q2, _zh in literal_pairs:
            self.assertIsNone(cjk.search(en), en)

        # Toggle accessibility text is expressed in the current interface
        # language; the button label itself names the target language.
        self.assertIn('language === "zh" ? "切换到英文界面" : "Switch interface to Chinese"', i18n)
        self.assertIn('language === "zh" ? "切换到英文界面会开启独立会话" : "Switching to Chinese starts a separate session"', i18n)
        self.assertIn('if (containsCjk(zh) && !containsCjk(text)) return zh;', js)
        self.assertIn('new Error(localizedApiError(data, response.status))', js)
        self.assertIn('if (backendMessage && containsCjk(backendMessage)) return backendMessage;', js)
        self.assertIn('if (backendMessage && !containsCjk(backendMessage)) return backendMessage;', js)
        self.assertIn('catalyst_finder_session_id_${uiLanguage}', js)

        manifest = public_capabilities()
        def check_pairs(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key.endswith("_en") and isinstance(item, str):
                        self.assertIsNone(cjk.search(item), f"{key}: {item}")
                        zh_key = key[:-3] + "_zh"
                        if zh_key in value:
                            self.assertTrue(str(value[zh_key]).strip(), zh_key)
                    check_pairs(item)
            elif isinstance(value, list):
                for item in value:
                    check_pairs(item)
        check_pairs(manifest)

    def test_recorded_evidence_route_visible_text_follows_ui_language(self) -> None:
        runtime = CatalystFinderRuntime()
        zh = runtime.evidence_queries.lookup_protein_reactions("P00338", ui_language="zh")
        en = runtime.evidence_queries.lookup_protein_reactions("P00338", ui_language="en")
        zh_node = zh["route_view"]["nodes"][1]
        en_node = en["route_view"]["nodes"][1]
        self.assertEqual(zh_node["subtitle"], "整合证据库")
        self.assertIn("条已记录反应", zh_node["metric"])
        self.assertEqual(en_node["subtitle"], "Integrated evidence catalog")
        self.assertIn("recorded reactions", en_node["metric"])

    def test_public_route_labels_keep_english_and_chinese_fields_separate(self) -> None:
        from scripts.catalyst_finder.route_view import system_route_catalog
        cjk = re.compile(r"[\u3400-\u9fff]")
        catalog = system_route_catalog()
        for group in ("base_routes", "overlays", "downstream_workflows"):
            for row in catalog.get(group) or []:
                english = str(row.get("label_en") or row.get("title_en") or "").strip()
                chinese = str(row.get("label") or row.get("title") or "").strip()
                if english:
                    self.assertIsNone(cjk.search(english), english)
                    self.assertTrue(chinese, row.get("key"))

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
        self.assertIn('<h1>Catalyst Finder</h1>', html)
        self.assertIn('data-en="Ask me a question, or ask what I can do."', html)
        self.assertIn('data-zh="直接提问，也可以先问我能做什么。"', html)
        self.assertIn('data-placeholder-zh="输入你的问题…"', html)
        self.assertNotIn("智能体未强制套用固定任务模式", js)
        self.assertNotIn("本轮智能体调用", js)
        self.assertIn('tr("Model score", "模型评分")', js)
        self.assertIn("direct, natural Simplified Chinese", backend)
        self.assertIn("direct, natural scientific English", backend)

    def test_capability_guide_is_generated_from_live_agent_manifest(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        css = (frontend / "styles.css").read_text(encoding="utf-8")
        manifest = public_capabilities()
        self.assertIn('id="capabilityGuide"', html)
        self.assertIn('id="capabilityGuideBody"', html)
        self.assertNotIn('id="capabilityRibbon"', html)
        self.assertNotIn('id="starterGrid"', html)
        self.assertEqual(html.count('class="capability-action"'), 0)
        self.assertIn('api("/api/capabilities").then(renderCapabilities)', js)
        self.assertIn("function renderCapabilities(payload)", js)
        self.assertTrue(manifest["interaction"]["model_led"])
        self.assertTrue(manifest["interaction"]["natural_language_first"])
        ids = {group["id"] for group in manifest["groups"]}
        self.assertTrue({"evidence", "compound_identity", "candidate_retrieval", "route_design", "pathway"}.issubset(ids))
        self.assertNotIn("conversation", ids)
        joined = json.dumps(manifest, ensure_ascii=False)
        self.assertIn("比较已核对实体", joined)
        self.assertIn("Reaction SMILES", joined)
        self.assertIn("FASTA", joined)
        self.assertIn("热力学", joined)
        self.assertIn("多酶兼容性", joined)
        self.assertIn('.capability-guide', css)
        self.assertIn('.capability-actions', css)

    def test_examples_do_not_encode_hidden_task_direction(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        manifest = public_capabilities()
        self.assertNotIn("data-direction-template", html)
        self.assertNotIn("suggestedDirection", js)
        self.assertNotIn("direction_hint:", js)
        groups = {group["id"]: group for group in manifest["groups"]}
        self.assertTrue(groups["candidate_retrieval"]["examples"])
        self.assertTrue(groups["route_design"]["examples"])
        self.assertNotEqual(
            groups["candidate_retrieval"]["examples"][0]["prompt_zh"],
            groups["route_design"]["examples"][0]["prompt_zh"],
        )

    def test_route_design_is_exposed_as_natural_language_capability_without_selector(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        manifest = public_capabilities()
        route = next(group for group in manifest["groups"] if group["id"] == "route_design")
        self.assertIn("热力学", route["description_zh"])
        self.assertIn("E. coli", route["description_zh"])
        self.assertNotIn("data-route-priority", html)
        self.assertNotIn("routePriorityOverride", js)
        self.assertNotIn("data-direction-template", html)
        self.assertIn('resolution.direction === "route_design"', js)
        self.assertIn('endpoint: "/api/route/design"', js)
        self.assertIn("eQuilibrator MDF", js)
        self.assertIn("iML1515", js)

    def test_pathway_is_exposed_as_natural_language_capability_without_selector(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        manifest = public_capabilities()
        pathway = next(group for group in manifest["groups"] if group["id"] == "pathway")
        self.assertIn("多步路径", pathway["description_zh"])
        self.assertIn("辅因子", pathway["description_zh"])
        self.assertIn("pH", pathway["description_zh"])
        self.assertNotIn('data-pathway-mode', html)
        self.assertNotIn('data-direction-template', html)
        self.assertIn('resolution.direction === "pathway_compatibility"', js)
        self.assertIn('endpoint: "/api/pathway/analyze"', js)

    def test_capability_examples_never_create_direction_hints(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn("function capabilityButton(example, group)", js)
        self.assertIn("button.dataset.capabilityId", js)
        self.assertNotIn("directionHintOneShot", js)
        self.assertNotIn("effectiveHint", js)
        self.assertNotIn("suggestedDirection", js)
        self.assertNotIn("direction_hint:", js)

    def test_continuation_relies_on_session_context_without_rewriting_user_prompt(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        html = (frontend / "index.html").read_text(encoding="utf-8")
        self.assertIn("const effectiveText = text;", js)
        self.assertIn("session_id: run.session_id", js)
        self.assertNotIn("previous_association_policy", js)
        self.assertNotIn("composerContext", js)
        self.assertNotIn("Continue previous task", html)
        self.assertNotIn('tr("Follow-up request:", "用户后续要求：")', js)
        self.assertNotIn("directionHint", js)
        self.assertNotIn("continuedHint", js)

    def test_natural_answer_can_coexist_with_structured_evidence(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn("if (!resolution.immediate_result)", js)
        self.assertIn("if (resolution.immediate_result)", js)
        response_pos = js.index("if (resolution.assistant_response)")
        immediate_pos = js.index("if (resolution.immediate_result)", response_pos)
        self.assertGreater(immediate_pos, response_pos)

    def test_entity_list_results_render_without_association_mislabeling(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        manifest = public_capabilities()
        self.assertIn('function renderEntityListResult(result)', js)
        self.assertIn('result?.answer_mode === "entity_list"', js)
        self.assertIn('const entityListMode = result.answer_mode === "entity_list"', js)
        self.assertIn('paginateInto(grid, entities', js)
        self.assertIn('result.entities?.[0]?.name', js)
        group_ids = {group["id"] for group in manifest["groups"]}
        self.assertIn("compound_identity", group_ids)
        self.assertEqual(manifest["version"], "catalyst-capabilities-v8")

    def test_assistant_markdown_renderer_is_safe_and_used_for_model_text(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        markdown = (frontend / "markdown.js").read_text(encoding="utf-8")
        css = (frontend / "styles.css").read_text(encoding="utf-8")
        self.assertIn('<script src="/markdown.js" defer></script>', html)
        self.assertLess(html.index('/markdown.js'), html.index('/app.js'))
        self.assertIn("CatalystMarkdown.renderInto(copy", js)
        self.assertIn("markdown-body", js)
        self.assertNotIn("innerHTML", markdown)
        self.assertNotIn("eval(", markdown)
        self.assertIn('new Set(["http:", "https:", "mailto:"])', markdown)
        self.assertIn("document.createTextNode", markdown)
        self.assertIn('document.createElement("table")', markdown)
        self.assertIn('document.createElement("pre")', markdown)
        self.assertIn(".markdown-body pre", css)
        self.assertIn(".markdown-table-wrap", css)
        self.assertIn("list.start = Number(orderedMatch[1]) || 1", markdown)
        self.assertNotIn("item.value", markdown)
        self.assertIn("list.start = Number(orderedMatch[1]) || 1", markdown)
        self.assertIn("if (!lines[index].trim())", markdown)
        self.assertIn(".markdown-body p,.markdown-body li{font-size:inherit;line-height:inherit}", css)

    def test_scientific_harness_trace_lives_in_technical_rail_without_exposing_reasoning(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        css = (frontend / "styles.css").read_text(encoding="utf-8")
        self.assertIn('id="technicalAgentTrace"', html)
        self.assertIn("function renderAgentExecution(execution)", js)
        self.assertIn("technicalAgentTrace.replaceChildren()", js)
        self.assertNotIn("本轮智能体调用", js)
        self.assertIn("Tool steps", js)
        self.assertIn(".technical-tool-trace", css)
        trace_start = js.index("function renderAgentExecution(execution)")
        trace_end = js.index("function localizedCapability", trace_start)
        renderer = js[trace_start:trace_end]
        self.assertNotIn("messageShell", renderer)
        self.assertNotIn("step.reason", renderer)
        self.assertNotIn("chain_of_thought", renderer)

    def test_frontend_uses_true_ten_item_pagination_for_long_results(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        css = (frontend / "styles.css").read_text(encoding="utf-8")
        self.assertIn("const RESULT_PAGE_SIZE = 10", js)
        self.assertIn("items.slice(start, start + pageSize)", js)
        self.assertIn("viewport.replaceChildren", js)
        # A partial final page is naturally shorter: 24 results render as 10 / 10 / 4,
        # with no placeholder rows or fixed-height ten-slot window.
        rows = list(range(24))
        pages = [rows[start:start + 10] for start in range(0, len(rows), 10)]
        self.assertEqual([len(page) for page in pages], [10, 10, 4])
        self.assertNotIn("pagination-placeholder", js)
        self.assertIn("paginateInto(grid, entities", js)
        self.assertIn("paginateInto(grid, known.items", js)
        self.assertIn("paginateInto(tbody, discoveryRows", js)
        self.assertIn("paginateInto(list, routes", js)
        self.assertIn(".result-pagination", css)

    def test_first_screen_is_conversation_first_and_science_capabilities_are_folded(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        manifest = public_capabilities()
        self.assertIn('<h1>Catalyst Finder</h1>', html)
        self.assertIn('data-zh="直接提问，也可以先问我能做什么。"', html)
        self.assertIn('data-placeholder-zh="输入你的问题…"', html)
        self.assertIn('id="capabilityGuide"', html)
        self.assertNotIn('id="capabilityRibbon"', html)
        self.assertNotIn('id="starterGrid"', html)
        self.assertNotIn('id="processList"', html)
        self.assertNotIn("自然提问与连续追问", json.dumps(manifest, ensure_ascii=False))
        self.assertNotIn("conversation", {group["id"] for group in manifest["groups"]})

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
