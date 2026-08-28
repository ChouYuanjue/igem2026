from __future__ import annotations

import time
import unittest
from copy import deepcopy
from typing import Any

from scripts.catalyst_finder.agent_harness.contracts import HarnessAction, ToolResult
from scripts.catalyst_finder.agent_harness.harness import CatalystScientificHarness
from scripts.catalyst_finder.agent_harness.session_store import AgentSessionStore
from scripts.catalyst_finder.agent_harness.tool_registry import HarnessRunContext, ScientificToolRegistry


class FakeDeepSeek:
    def __init__(self, actions: list[HarnessAction]) -> None:
        self.actions = list(actions)
        self.calls: list[dict[str, Any]] = []

    def next_harness_action(self, **kwargs: Any) -> HarnessAction:
        self.calls.append(deepcopy(kwargs))
        if not self.actions:
            raise AssertionError("controller called more times than expected")
        return self.actions.pop(0)

    def provenance(self) -> dict[str, Any]:
        return {"provider": "fake", "model": "fake-controller"}


class FakeTools:
    def __init__(self, results: list[ToolResult], terminal_payload: dict[str, Any] | None = None) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.terminal_payload = terminal_payload

    @staticmethod
    def catalog() -> list[dict[str, Any]]:
        return [{"name": "resolve_reaction"}, {"name": "prepare_candidate_retrieval"}]

    def execute(self, tool: str, args: dict[str, Any], ctx: Any) -> ToolResult:
        self.calls.append((tool, args))
        if not self.results:
            raise AssertionError("tool called more times than expected")
        result = self.results.pop(0)
        if result.terminal and result.status == "ok":
            ctx.terminal_resolution = self.terminal_payload or {
                "direction": "reaction_to_enzyme",
                "summary": "terminal",
                "reaction_resolution": None,
                "protein_resolution": None,
                "positive_enzyme_resolutions": [],
            }
        return result


class FakeAgentResolution:
    def __init__(self) -> None:
        self.legacy_calls = 0

    def _direct_open_world_resolution(self, text: str, direction_hint: str, ui_language: str) -> None:
        return None

    def agent_resolve(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.legacy_calls += 1
        return {
            "direction": "enzyme_to_reaction",
            "summary": "legacy fallback",
            "reaction_resolution": None,
            "protein_resolution": None,
            "positive_enzyme_resolutions": [],
        }

    def resolve(self, text: str) -> dict[str, Any]:
        return {"recommended_id": "RHEA:12345", "candidates": []}


class ScientificHarnessLoopTests(unittest.TestCase):
    def build(self, actions: list[HarnessAction], results: list[ToolResult], *, terminal_payload: dict[str, Any] | None = None) -> tuple[CatalystScientificHarness, FakeDeepSeek, FakeTools, FakeAgentResolution]:
        deepseek = FakeDeepSeek(actions)
        tools = FakeTools(results, terminal_payload=terminal_payload)
        legacy = FakeAgentResolution()
        harness = CatalystScientificHarness(
            deepseek=deepseek,
            tools=tools,  # type: ignore[arg-type]
            sessions=AgentSessionStore(ttl_seconds=3600),
            agent_resolution=legacy,
            max_turns=6,
        )
        return harness, deepseek, tools, legacy

    def test_terminal_tool_result_returns_without_extra_controller_turn(self) -> None:
        harness, deepseek, tools, legacy = self.build(
            [HarnessAction(kind="tool", tool="prepare_candidate_retrieval", args={"text": "find candidates"})],
            [ToolResult(tool="prepare_candidate_retrieval", status="ok", summary="prepared", terminal=True)],
        )
        result = harness.run("find candidates", session_id="s1")
        self.assertEqual(result["direction"], "reaction_to_enzyme")
        self.assertFalse(result["agent_execution"]["fallback"])
        self.assertEqual(result["agent_execution"]["turn_count"], 1)
        self.assertEqual(len(deepseek.calls), 1)
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(legacy.legacy_calls, 0)

    def test_recoverable_error_is_fed_back_and_controller_can_change_strategy(self) -> None:
        harness, deepseek, tools, legacy = self.build(
            [
                HarnessAction(kind="tool", tool="resolve_reaction", args={"text": "ambiguous reaction"}),
                HarnessAction(kind="tool", tool="prepare_candidate_retrieval", args={"text": "ambiguous reaction"}),
            ],
            [
                ToolResult(tool="resolve_reaction", status="error", summary="no exact evidence", recoverable=True, error_code="no_match"),
                ToolResult(tool="prepare_candidate_retrieval", status="ok", summary="prepared", terminal=True),
            ],
        )
        result = harness.run("ambiguous reaction")
        self.assertFalse(result["agent_execution"]["fallback"])
        self.assertEqual([call[0] for call in tools.calls], ["resolve_reaction", "prepare_candidate_retrieval"])
        second_history = deepseek.calls[1]["history"]
        self.assertEqual(second_history[-1]["result"]["error_code"], "no_match")
        self.assertEqual(legacy.legacy_calls, 0)

    def test_premature_final_is_rejected_then_tool_can_complete(self) -> None:
        harness, deepseek, tools, legacy = self.build(
            [
                HarnessAction(kind="final", reason="done"),
                HarnessAction(kind="tool", tool="prepare_candidate_retrieval", args={"text": "find candidates"}),
            ],
            [ToolResult(tool="prepare_candidate_retrieval", status="ok", summary="prepared", terminal=True)],
        )
        result = harness.run("find candidates")
        steps = result["agent_execution"]["steps"]
        self.assertEqual(steps[0]["action_kind"], "final")
        self.assertEqual(steps[0]["status"], "rejected")
        self.assertEqual(steps[1]["tool"], "prepare_candidate_retrieval")
        self.assertEqual(legacy.legacy_calls, 0)

    def test_identical_tool_call_is_not_executed_twice(self) -> None:
        same = HarnessAction(kind="tool", tool="resolve_reaction", args={"text": "reaction X"})
        harness, _deepseek, tools, legacy = self.build(
            [same, same.model_copy(deep=True), same.model_copy(deep=True)],
            [ToolResult(tool="resolve_reaction", status="error", summary="try another way", recoverable=True)],
        )
        result = harness.run("reaction X")
        self.assertEqual(len(tools.calls), 1)
        self.assertTrue(result["agent_execution"]["fallback"])
        rejected = [step for step in result["agent_execution"]["steps"] if step["status"] == "rejected"]
        self.assertEqual(len(rejected), 2)
        self.assertEqual(legacy.legacy_calls, 1)

    def test_nonrecoverable_tool_error_falls_back_once(self) -> None:
        harness, _deepseek, tools, legacy = self.build(
            [HarnessAction(kind="tool", tool="resolve_reaction", args={"text": "reaction X"})],
            [ToolResult(tool="resolve_reaction", status="error", summary="backend unavailable", recoverable=False, error_code="backend")],
        )
        result = harness.run("reaction X")
        self.assertEqual(len(tools.calls), 1)
        self.assertTrue(result["agent_execution"]["fallback"])
        self.assertEqual(legacy.legacy_calls, 1)

    def test_verified_session_entity_is_exposed_as_current_run_ref(self) -> None:
        sessions = AgentSessionStore(ttl_seconds=3600)
        sessions.remember_resolution("follow", {
            "direction": "reaction_to_enzyme",
            "reaction_resolution": {"recommended_id": "RHEA:32883", "candidates": []},
        })
        deepseek = FakeDeepSeek([HarnessAction(kind="ask_user", question="scope?")])
        tools = FakeTools([])
        legacy = FakeAgentResolution()
        harness = CatalystScientificHarness(
            deepseek=deepseek,
            tools=tools,  # type: ignore[arg-type]
            sessions=sessions,
            agent_resolution=legacy,
        )
        harness.run("那这个反应呢？", session_id="follow")
        facts = deepseek.calls[0]["session_facts"]
        self.assertEqual(
            facts["current_run_refs"]["reaction_refs"],
            [{"ref": "session_reaction_1", "rhea_id": "RHEA:32883"}],
        )

    def test_plain_six_letter_word_does_not_bypass_harness_as_uniprot(self) -> None:
        harness, deepseek, tools, legacy = self.build(
            [HarnessAction(kind="ask_user", question="Which kinase scope?")],
            [],
        )
        result = harness.run("kinase")
        self.assertEqual(result["summary"], "Which kinase scope?")
        self.assertEqual(len(deepseek.calls), 1)
        self.assertEqual(tools.calls, [])
        self.assertEqual(legacy.legacy_calls, 0)

    def test_ask_user_does_not_touch_legacy_resolver(self) -> None:
        harness, _deepseek, tools, legacy = self.build(
            [HarnessAction(kind="ask_user", question="Which reaction do you mean?")],
            [],
        )
        result = harness.run("that one")
        self.assertEqual(result["direction"], "ambiguous")
        self.assertEqual(result["summary"], "Which reaction do you mean?")
        self.assertEqual(tools.calls, [])
        self.assertEqual(legacy.legacy_calls, 0)


class ScientificToolRecoveryTests(unittest.TestCase):
    def test_functional_class_requires_explicit_broaden_before_parent_evidence(self) -> None:
        class DeepSeek:
            def parse_protein(self, text: str) -> dict[str, Any]:
                return {
                    "interpreted_protein": text,
                    "protein_terms": ["narrow oxidoreductase"],
                    "organism_terms": [],
                    "gene_terms": [],
                    "accession_terms": [],
                }

            def expand_protein_class_terms(self, **kwargs: Any) -> dict[str, list[str]]:
                return {
                    "strict_terms": ["narrow oxidoreductase"],
                    "broader_terms": ["oxidoreductase"],
                }

        class Families:
            def resolve(self, *values: str) -> None:
                return None

        class FamilyEvidence:
            def summarize_functional_class(self, spec: dict[str, Any], *, ui_language: str = "en") -> dict[str, Any]:
                broadened = bool(spec.get("scope_broadened"))
                count = 2 if broadened else 0
                return {
                    "protein": {"id": "CLASS-TEST", "name": "narrow oxidoreductase", "input_mode": "protein_functional_class"},
                    "family": {"evidence_member_count": 1 if broadened else 0},
                    "known_associations": {
                        "count": count,
                        "items": [{"candidate_id": "RHEA:11111"}, {"candidate_id": "RHEA:22222"}] if broadened else [],
                        "note": "parent evidence" if broadened else "no strict evidence",
                    },
                    "candidates": [],
                    "ranking": {"route_id": "e2r-functional-class-evidence-v1"},
                }

        registry = ScientificToolRegistry(
            agent_resolution=FakeAgentResolution(),
            deepseek=DeepSeek(),
            families=Families(),
            family_evidence=FamilyEvidence(),
            evidence_queries=object(),
            route_design_resolve=lambda *a, **k: {},
            pathway_resolve=lambda *a, **k: {},
        )
        ctx = HarnessRunContext(ui_language="en", direction_hint="auto", conversation_context={})
        resolved = registry.execute(
            "resolve_protein_scope",
            {"text": "narrow oxidoreductase", "scope_hint": "family_or_class"},
            ctx,
        )
        strict_ref = resolved.payload["protein_scope_ref"]
        strict = registry.execute("summarize_recorded_relations", {"protein_scope_ref": strict_ref}, ctx)
        self.assertEqual(strict.status, "error")
        self.assertEqual(strict.error_code, "strict_scope_no_evidence")
        self.assertFalse(strict.terminal)

        broaden = registry.execute("broaden_protein_scope", {"protein_scope_ref": strict_ref}, ctx)
        self.assertEqual(broaden.status, "ok")
        self.assertTrue(broaden.payload["approximate_parent_scope"])
        broad_ref = broaden.payload["protein_scope_ref"]
        self.assertNotEqual(broad_ref, strict_ref)

        aggregated = registry.execute("summarize_recorded_relations", {"protein_scope_ref": broad_ref}, ctx)
        self.assertEqual(aggregated.status, "ok")
        self.assertTrue(aggregated.terminal)
        self.assertEqual(aggregated.payload["recorded_reaction_count"], 2)
        self.assertTrue(aggregated.payload["scope_broadened"])
        self.assertEqual(ctx.terminal_resolution["immediate_result"]["known_associations"]["count"], 2)


class ScientificToolCatalogTests(unittest.TestCase):
    def test_catalog_exposes_pydantic_input_schema(self) -> None:
        catalog = {item["name"]: item for item in ScientificToolRegistry.catalog()}
        schema = catalog["resolve_protein_scope"]["input_schema"]
        self.assertIn("text", schema["properties"])
        self.assertIn("scope_hint", schema["properties"])
        self.assertIn("text", schema["required"])
        self.assertIn("family_or_class", str(schema["properties"]["scope_hint"]))


class AgentSessionStoreTests(unittest.TestCase):
    def test_sessions_are_isolated(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("a", {
            "direction": "reaction_to_enzyme",
            "reaction_resolution": {"recommended_id": "RHEA:12345", "candidates": []},
        })
        self.assertEqual(store.snapshot("a")["verified_reaction_ids"], ["RHEA:12345"])
        self.assertEqual(store.snapshot("b"), {})

    def test_freeform_llm_summary_cannot_become_trusted_identifier(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("s", {
            "direction": "reaction_to_enzyme",
            "summary": "I think RHEA:99999 and PFAKE1 are relevant",
            "llm_provenance": {"provider": "fake"},
        })
        snapshot = store.snapshot("s")
        self.assertEqual(snapshot["verified_reaction_ids"], [])
        self.assertEqual(snapshot["verified_protein_ids"], [])
        self.assertEqual(snapshot["verified_family_ids"], [])

    def test_only_supported_protein_modes_are_remembered(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("s", {
            "direction": "enzyme_to_reaction",
            "protein_resolution": {
                "mode": "raw_protein_sequence",
                "recommended_id": "EXT-PROT-UNVERIFIED",
            },
        })
        self.assertEqual(store.snapshot("s")["verified_protein_ids"], [])
        store.remember_resolution("s", {
            "direction": "enzyme_to_reaction",
            "protein_resolution": {
                "mode": "general_merged_sequence_match",
                "recommended_id": "P00338",
            },
        })
        self.assertEqual(store.snapshot("s")["verified_protein_ids"], ["P00338"])

    def test_expired_session_is_pruned(self) -> None:
        store = AgentSessionStore(ttl_seconds=60)
        store.remember_resolution("s", {
            "reaction_resolution": {"recommended_id": "RHEA:12345", "candidates": []},
        })
        # Avoid sleeping: age the internal record under the lock.
        with store._lock:  # noqa: SLF001 - intentional white-box TTL test
            store._states["s"].updated_at = time.time() - 61  # noqa: SLF001
        self.assertEqual(store.snapshot("s"), {})


if __name__ == "__main__":
    unittest.main()
