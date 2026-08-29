from __future__ import annotations

import time
import unittest
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from scripts.catalyst_finder.agent_harness.contracts import HarnessAction, ToolResult
from scripts.catalyst_finder.agent_harness.harness import CatalystScientificHarness
from scripts.catalyst_finder.agent_harness.session_store import AgentSessionStore
from scripts.catalyst_finder.agent_harness.tool_registry import HarnessRunContext, ScientificToolRegistry
from scripts.catalyst_finder.errors import AppError


class FakeDeepSeek:
    def __init__(self, actions: list[HarnessAction]) -> None:
        self.actions = list(actions)
        self.calls: list[dict[str, Any]] = []
        self.synthesis_calls: list[dict[str, Any]] = []
        self.readiness_calls: list[dict[str, Any]] = []

    def next_harness_action(self, **kwargs: Any) -> HarnessAction:
        self.calls.append(deepcopy(kwargs))
        if not self.actions:
            raise AssertionError("controller called more times than expected")
        return self.actions.pop(0)

    def synthesize_grounded_answer(self, **kwargs: Any) -> dict[str, Any]:
        self.synthesis_calls.append(deepcopy(kwargs))
        return {"answer": "Grounded comparison from verified evidence.", "evidence_ids": ["E1", "E2"], "limitations": []}

    def validate_synthesis_readiness(self, **kwargs: Any) -> dict[str, Any]:
        self.readiness_calls.append(deepcopy(kwargs))
        return {"ready": True, "reason": "", "missing_requirements": []}

    def provenance(self) -> dict[str, Any]:
        return {"provider": "fake", "model": "fake-controller"}


class FakeTools:
    def __init__(self, results: list[ToolResult], terminal_payload: dict[str, Any] | None = None) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.terminal_payload = terminal_payload

    @staticmethod
    def catalog() -> list[dict[str, Any]]:
        return [{"name": "resolve_reaction"}, {"name": "candidate_search"}]

    def execute(self, tool: str, args: dict[str, Any], ctx: Any) -> ToolResult:
        self.calls.append((tool, args))
        if not self.results:
            raise AssertionError("tool called more times than expected")
        result = self.results.pop(0)
        if result.status == "ok" and (result.terminal or self.terminal_payload is not None):
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


class HarnessActionProviderShapeTests(unittest.TestCase):
    def test_empty_tool_string_is_normalized_for_non_tool_action(self) -> None:
        action = HarnessAction.model_validate({
            "kind": "respond",
            "tool": "",
            "args": None,
            "reason": None,
            "question": "",
            "message": "I can help with biochemical research tasks.",
        })
        self.assertIsNone(action.tool)
        self.assertEqual(action.args, {})
        self.assertEqual(action.reason, "")


    def test_harness_has_no_automatic_session_ref_seeding_backdoor(self) -> None:
        from pathlib import Path
        source = Path(__file__).with_name("agent_harness").joinpath("harness.py").read_text(encoding="utf-8")
        self.assertNotIn("def _seed_session_refs", source)
        self.assertNotIn("session_protein_scope_group_", source)
        self.assertIn("self.sessions.model_snapshot(session_id)", source)


class ScientificHarnessLoopTests(unittest.TestCase):
    def build(
        self,
        actions: list[HarnessAction],
        results: list[ToolResult],
        *,
        terminal_payload: dict[str, Any] | None = None,
        max_turns: int = 6,
        sessions: AgentSessionStore | None = None,
    ) -> tuple[CatalystScientificHarness, FakeDeepSeek, FakeTools]:
        deepseek = FakeDeepSeek(actions)
        tools = FakeTools(results, terminal_payload=terminal_payload)
        harness = CatalystScientificHarness(
            deepseek=deepseek,
            tools=tools,  # type: ignore[arg-type]
            sessions=sessions or AgentSessionStore(ttl_seconds=3600),
            max_turns=max_turns,
        )
        return harness, deepseek, tools

    def test_every_input_reaches_controller_even_exact_rhea(self) -> None:
        harness, deepseek, tools = self.build(
            [HarnessAction(kind="respond", message="I can decide what to do with that identifier.")],
            [],
        )
        result = harness.run("RHEA:32883")
        self.assertEqual(result["direction"], "conversation")
        self.assertEqual(result["assistant_response"], "I can decide what to do with that identifier.")
        self.assertEqual(len(deepseek.calls), 1)
        self.assertEqual(tools.calls, [])
        self.assertNotEqual(result["agent_execution"]["mode"], "deterministic_fast_path")

    def test_controller_can_answer_product_question_without_tools(self) -> None:
        harness, deepseek, tools = self.build(
            [HarnessAction(kind="respond", message="I can query evidence, rank candidates, design routes, and evaluate pathways.")],
            [],
        )
        result = harness.run("What can you do?")
        self.assertEqual(result["response_type"], "message")
        self.assertIn("rank candidates", result["assistant_response"])
        self.assertEqual(result["agent_execution"]["steps"][0]["action_kind"], "respond")
        self.assertTrue(deepseek.calls[0]["capability_manifest"]["interaction"]["model_led"])
        self.assertGreaterEqual(len(deepseek.calls[0]["capability_manifest"]["groups"]), 5)
        self.assertEqual(tools.calls, [])

    def test_ask_user_is_natural_clarification_without_task_menu(self) -> None:
        harness, _deepseek, tools = self.build(
            [HarnessAction(kind="ask_user", question="Which substrate do you want to start from?")],
            [],
        )
        result = harness.run("Design the route for me")
        self.assertEqual(result["direction"], "conversation")
        self.assertEqual(result["response_type"], "clarification")
        self.assertEqual(result["assistant_response"], "Which substrate do you want to start from?")
        self.assertNotIn("intent_options", result)
        self.assertEqual(tools.calls, [])

    def test_terminal_tool_result_returns_directly_without_extra_controller_turn(self) -> None:
        harness, deepseek, tools = self.build(
            [HarnessAction(kind="tool", tool="candidate_search", args={"direction": "reaction_to_enzyme", "full_text": "find candidates", "reaction_text": "reaction X"})],
            [ToolResult(tool="candidate_search", status="ok", summary="prepared", terminal=True)],
        )
        result = harness.run("find candidates", session_id="s1")
        self.assertEqual(result["direction"], "reaction_to_enzyme")
        self.assertFalse(result["agent_execution"]["fallback"])
        self.assertEqual(result["agent_execution"]["turn_count"], 1)
        self.assertEqual(len(deepseek.calls), 1)
        self.assertEqual(len(tools.calls), 1)

    def test_recoverable_error_is_fed_back_and_controller_changes_strategy(self) -> None:
        harness, deepseek, tools = self.build(
            [
                HarnessAction(kind="tool", tool="resolve_reaction", args={"text": "ambiguous reaction"}),
                HarnessAction(kind="tool", tool="candidate_search", args={"direction": "reaction_to_enzyme", "full_text": "ambiguous reaction", "reaction_text": "ambiguous reaction"}),
            ],
            [
                ToolResult(tool="resolve_reaction", status="error", summary="no exact evidence", recoverable=True, error_code="no_match"),
                ToolResult(tool="candidate_search", status="ok", summary="prepared", terminal=True),
            ],
        )
        result = harness.run("ambiguous reaction")
        self.assertFalse(result["agent_execution"]["fallback"])
        self.assertEqual([call[0] for call in tools.calls], ["resolve_reaction", "candidate_search"])
        self.assertEqual(deepseek.calls[1]["history"][-1]["result"]["error_code"], "no_match")

    def test_verified_evidence_rejects_freeform_followup_and_uses_return_result(self) -> None:
        payload = {
            "direction": "reaction_to_enzyme",
            "summary": "verified evidence",
            "reaction_resolution": {"recommended_id": "RHEA:12345", "candidates": []},
            "protein_resolution": None,
            "positive_enzyme_resolutions": [],
            "immediate_result": {"known_associations": {"count": 1, "items": [{"candidate_id": "PTEST1"}]}},
        }
        harness, _deepseek, _tools = self.build(
            [
                HarnessAction(kind="tool", tool="resolve_reaction", args={"text": "reaction X"}),
                HarnessAction(kind="respond", message="One recorded protein is PTEST1, plus another famous enzyme."),
                HarnessAction(kind="return_result"),
            ],
            [ToolResult(tool="resolve_reaction", status="ok", summary="verified", terminal=False)],
            terminal_payload=payload,
        )
        result = harness.run("Which protein is recorded for reaction X?")
        self.assertNotIn("assistant_response", result)
        self.assertEqual(result["immediate_result"]["known_associations"]["count"], 1)
        steps = result["agent_execution"]["steps"]
        self.assertEqual(steps[1]["action_kind"], "respond")
        self.assertEqual(steps[1]["status"], "rejected")
        self.assertEqual(steps[2]["action_kind"], "return_result")

    def test_relation_lookup_is_composable_without_hidden_workflow_policy(self) -> None:
        terminal_payload = {
            "direction": "reaction_to_enzyme",
            "summary": "recorded evidence",
            "reaction_resolution": {"recommended_id": "RHEA:12345", "candidates": []},
            "protein_resolution": None,
            "positive_enzyme_resolutions": [],
            "immediate_result": {"known_associations": {"count": 2, "items": []}},
        }
        harness, _deepseek, tools = self.build(
            [
                HarnessAction(kind="tool", tool="lookup_relations", args={"reaction_ref": "reaction_1"}),
                HarnessAction(kind="return_result"),
            ],
            [
                ToolResult(tool="lookup_relations", status="ok", summary="recorded evidence", terminal=False, payload={"recorded_count": 2}),
            ],
            terminal_payload=terminal_payload,
        )
        result = harness.run("Which enzymes catalyze this reaction?")
        self.assertEqual(result["immediate_result"]["known_associations"]["count"], 2)
        self.assertEqual([call[0] for call in tools.calls], ["lookup_relations"])

    def test_verified_evidence_can_be_returned_or_followed_by_candidate_workflow(self) -> None:
        evidence_payload = {
            "direction": "reaction_to_enzyme",
            "summary": "verified evidence",
            "reaction_resolution": {"recommended_id": "RHEA:12345", "candidates": []},
            "protein_resolution": None,
            "positive_enzyme_resolutions": [],
            "immediate_result": {"known_associations": {"count": 1, "items": [{"candidate_id": "PTEST1"}]}},
        }
        harness, deepseek, tools = self.build(
            [
                HarnessAction(kind="tool", tool="resolve_reaction", args={"text": "reaction X"}),
                HarnessAction(kind="return_result"),
            ],
            [ToolResult(tool="resolve_reaction", status="ok", summary="verified", terminal=False)],
            terminal_payload=evidence_payload,
        )
        result = harness.run("Which recorded enzyme catalyzes reaction X?")
        self.assertEqual(result["immediate_result"]["known_associations"]["count"], 1)
        self.assertEqual(result["agent_execution"]["steps"][-1]["action_kind"], "return_result")
        self.assertEqual(len(deepseek.calls), 2)
        self.assertEqual(len(tools.calls), 1)

        harness2, deepseek2, tools2 = self.build(
            [
                HarnessAction(kind="tool", tool="resolve_reaction", args={"text": "reaction X"}),
                HarnessAction(kind="tool", tool="candidate_search", args={"direction": "reaction_to_enzyme", "full_text": "show known evidence and candidates", "reaction_text": "reaction X"}),
            ],
            [
                ToolResult(tool="resolve_reaction", status="ok", summary="verified", terminal=False),
                ToolResult(tool="candidate_search", status="ok", summary="prepared", terminal=True),
            ],
            terminal_payload={
                "direction": "reaction_to_enzyme",
                "summary": "candidate workflow",
                "reaction_resolution": {"recommended_id": "RHEA:12345", "candidates": []},
                "protein_resolution": None,
                "positive_enzyme_resolutions": [],
            },
        )
        result2 = harness2.run("Show known evidence and candidates for reaction X")
        self.assertEqual(result2["direction"], "reaction_to_enzyme")
        self.assertEqual([c[0] for c in tools2.calls], ["resolve_reaction", "candidate_search"])
        self.assertEqual(len(deepseek2.calls), 2)

    def test_final_action_is_not_part_of_v2_contract(self) -> None:
        with self.assertRaises(ValueError):
            HarnessAction.model_validate({"kind": "final", "reason": "done"})

    def test_identical_tool_call_is_rejected_without_legacy_fallback(self) -> None:
        same = HarnessAction(kind="tool", tool="resolve_reaction", args={"text": "reaction X"})
        harness, _deepseek, tools = self.build(
            [same, same.model_copy(deep=True), same.model_copy(deep=True)],
            [ToolResult(tool="resolve_reaction", status="error", summary="try another way", recoverable=True)],
        )
        with self.assertRaises(AppError) as ctx:
            harness.run("reaction X")
        self.assertEqual(ctx.exception.code, "agent_repeated_tool_call")
        self.assertEqual(len(tools.calls), 1)

    def test_nonrecoverable_tool_error_still_returns_to_model_for_grounded_decision(self) -> None:
        harness, deepseek, tools = self.build(
            [
                HarnessAction(kind="tool", tool="resolve_reaction", args={"text": "reaction X"}),
                HarnessAction(kind="synthesize"),
            ],
            [ToolResult(tool="resolve_reaction", status="error", summary="backend unavailable", recoverable=False, error_code="backend")],
        )
        result = harness.run("reaction X")
        self.assertEqual(result["response_type"], "grounded_synthesis")
        self.assertEqual(len(deepseek.calls), 2)
        self.assertEqual(len(deepseek.synthesis_calls), 1)
        self.assertEqual(len(tools.calls), 1)
        self.assertFalse(result["agent_execution"]["fallback"])

    def test_verified_session_entity_is_history_not_an_automatic_current_run_ref(self) -> None:
        sessions = AgentSessionStore(ttl_seconds=3600)
        sessions.remember_resolution("follow", {
            "direction": "reaction_to_enzyme",
            "reaction_resolution": {"mode": "rhea_id", "recommended_id": "RHEA:32883", "candidates": [{"rhea_id": "RHEA:32883", "equation": "A = B"}]},
        })
        harness, deepseek, _tools = self.build(
            [HarnessAction(kind="ask_user", question="Which protein family constraint should I apply?")],
            [],
            sessions=sessions,
        )
        harness.run("那这个反应呢？", session_id="follow")
        facts = deepseek.calls[0]["session_facts"]
        self.assertNotIn("current_run_refs", facts)
        history = facts["session_entities"]["history"]
        self.assertEqual(history[0]["kind"], "reaction")
        self.assertEqual(history[0]["id"], "RHEA:32883")
        self.assertNotIn("payload", history[0])

    def test_grounded_synthesis_is_the_only_post_tool_scientific_prose_path(self) -> None:
        terminal_payload = {
            "direction": "conversation",
            "operation": "compare_entities",
            "summary": "comparison evidence ready",
            "reaction_resolution": None,
            "protein_resolution": None,
            "positive_enzyme_resolutions": [],
            "immediate_result": {"answer_mode": "entity_comparison", "entities": [{"id": "E1"}, {"id": "E2"}]},
        }
        harness, deepseek, _tools = self.build(
            [
                HarnessAction(kind="tool", tool="compare_entities", args={"entity_refs": ["ref_1", "ref_2"], "comparison_goal": "compare conclusions"}),
                HarnessAction(kind="synthesize"),
            ],
            [ToolResult(
                tool="compare_entities", status="ok", summary="evidence ready", terminal=False,
                payload={"workflow_incomplete": True, "required_next_action": "synthesize", "evidence_index": [{"id": "E1"}, {"id": "E2"}]},
            )],
            terminal_payload=terminal_payload,
        )
        result = harness.run("Compare the conclusions of the two verified papers.")
        self.assertEqual(result["response_type"], "grounded_synthesis")
        self.assertEqual(result["assistant_response"], "Grounded comparison from verified evidence.")
        self.assertEqual(result["immediate_result"]["analysis"], result["assistant_response"] )
        self.assertEqual(result["grounding"]["source"], "verified_tool_history")
        self.assertEqual(len(deepseek.synthesis_calls), 1)
        self.assertEqual(len(deepseek.readiness_calls), 0)
        self.assertEqual(deepseek.synthesis_calls[0]["current_result"]["immediate_result"]["answer_mode"], "entity_comparison")
        self.assertEqual(deepseek.synthesis_calls[0]["tool_history"][0]["result"]["payload"]["evidence_index"][0]["id"], "E1")
    def test_supplemental_inspection_cannot_replace_comparison_result(self) -> None:
        deepseek = FakeDeepSeek([
            HarnessAction(kind="tool", tool="compare_entities", args={"entity_refs": ["ref_1", "ref_2"], "comparison_goal": "compare"}),
            HarnessAction(kind="tool", tool="inspect_entity", args={"literature_ref": "ref_2"}),
            HarnessAction(kind="synthesize"),
        ])

        class Tools:
            @staticmethod
            def catalog(): return [{"name": "compare_entities"}, {"name": "inspect_entity"}]
            def __init__(self): self.calls = 0
            def execute(self, tool, args, ctx):
                self.calls += 1
                if tool == "compare_entities":
                    ctx.terminal_resolution = {
                        "direction": "conversation", "operation": "compare_entities",
                        "immediate_result": {"answer_mode": "entity_comparison", "entities": [{"id": "E1"}, {"id": "E2"}]},
                    }
                    return ToolResult(tool=tool, status="ok", summary="comparison ready", terminal=False, payload={"required_next_action": "synthesize"})
                ctx.terminal_resolution = {
                    "direction": "conversation", "operation": "inspect_entity",
                    "immediate_result": {"answer_mode": "entity_list", "entities": [{"id": "E2", "abstract": "supplemental evidence"}]},
                }
                return ToolResult(tool=tool, status="ok", summary="supplemental detail", terminal=False, payload={})

        harness = CatalystScientificHarness(deepseek=deepseek, tools=Tools(), sessions=AgentSessionStore(ttl_seconds=3600), max_turns=5)
        result = harness.run("Compare E1 and E2")
        self.assertEqual(result["immediate_result"]["answer_mode"], "entity_comparison")
        self.assertEqual([row["id"] for row in result["immediate_result"]["entities"]], ["E1", "E2"])
        self.assertEqual(result["immediate_result"]["analysis"], result["assistant_response"] )
        self.assertEqual(deepseek.synthesis_calls[0]["current_result"]["immediate_result"]["answer_mode"], "entity_comparison")
        self.assertEqual(len(deepseek.readiness_calls), 0)

    def test_grounded_synthesis_keeps_full_evidence_from_multiple_tools(self) -> None:
        deepseek = FakeDeepSeek([
            HarnessAction(kind="tool", tool="resolve_literature", args={"text": "paper A"}),
            HarnessAction(kind="tool", tool="inspect_entity", args={"literature_ref": "literature_1"}),
            HarnessAction(kind="synthesize"),
        ])

        class MultiEvidenceTools:
            @staticmethod
            def catalog():
                return [{"name": "resolve_literature"}, {"name": "inspect_entity"}]

            def __init__(self):
                self.calls = 0

            def execute(self, tool, args, ctx):
                self.calls += 1
                if self.calls == 1:
                    ctx.literature_refs["literature_1"] = {"id": "111", "source": "MED", "title": "Paper A"}
                    ctx.terminal_resolution = {
                        "direction": "conversation", "operation": "resolve_literature",
                        "immediate_result": {"answer_mode": "entity_list", "entities": [{"id": "MED:111", "name": "Paper A"}]},
                    }
                    return ToolResult(tool="resolve_literature", status="ok", summary="resolved A", terminal=False, payload={"literature_refs": [{"ref": "literature_1", "id": "MED:111"}]})
                ctx.terminal_resolution = {
                    "direction": "conversation", "operation": "inspect_entity",
                    "immediate_result": {"answer_mode": "entity_list", "entities": [{"id": "MED:111", "name": "Paper A", "abstract": "Full verified abstract A"}]},
                }
                return ToolResult(tool="inspect_entity", status="ok", summary="inspected A", terminal=False, payload={"entity_id": "MED:111"})

        harness = CatalystScientificHarness(
            deepseek=deepseek, tools=MultiEvidenceTools(), sessions=AgentSessionStore(ttl_seconds=3600), max_turns=5,
        )
        result = harness.run("Summarize the verified paper")
        self.assertEqual(result["response_type"], "grounded_synthesis")
        ledger = deepseek.synthesis_calls[0]["verified_evidence"]
        self.assertEqual(len(ledger), 2)
        self.assertEqual(ledger[0]["tool"], "resolve_literature")
        self.assertEqual(ledger[1]["tool"], "inspect_entity")
        self.assertEqual(ledger[1]["result"]["immediate_result"]["entities"][0]["abstract"], "Full verified abstract A")


    def test_synthesis_cannot_ignore_explicit_requested_identifier(self) -> None:
        deepseek = FakeDeepSeek([
            HarnessAction(kind="tool", tool="resolve_literature", args={"text": "MED:111"}),
            HarnessAction(kind="synthesize"),
            HarnessAction(kind="tool", tool="resolve_literature", args={"text": "MED:222"}),
            HarnessAction(kind="synthesize"),
        ])

        class Tools:
            @staticmethod
            def catalog():
                return [{"name": "resolve_literature"}]
            def execute(self, tool, args, ctx):
                article_id = "111" if "111" in str(args) else "222"
                ref = f"literature_{article_id}"
                ctx.literature_refs[ref] = {"id": article_id, "pmid": article_id, "source": "MED", "title": f"Paper {article_id}"}
                ctx.terminal_resolution = {
                    "direction": "conversation", "operation": "resolve_literature",
                    "immediate_result": {"answer_mode": "entity_list", "entities": [{"id": f"MED:{article_id}", "name": f"Paper {article_id}"}]},
                }
                return ToolResult(tool="resolve_literature", status="ok", summary=f"resolved {article_id}", terminal=False, payload={"entity_ids": [f"MED:{article_id}"]})

        harness = CatalystScientificHarness(
            deepseek=deepseek, tools=Tools(), sessions=AgentSessionStore(ttl_seconds=3600), max_turns=6,
        )
        result = harness.run("比较 MED:111 和 MED:222 的结论", ui_language="zh")
        steps = result["agent_execution"]["steps"]
        rejected = [step for step in steps if step["action_kind"] == "synthesize" and step["status"] == "rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertIn("MED:222", rejected[0]["summary"])
        self.assertEqual(result["response_type"], "grounded_synthesis")
        ledger = deepseek.synthesis_calls[0]["verified_evidence"]
        self.assertEqual({row["result"]["immediate_result"]["entities"][0]["id"] for row in ledger}, {"MED:111", "MED:222"})

    def test_readiness_critic_can_require_more_evidence_before_synthesis(self) -> None:
        class CriticDeepSeek(FakeDeepSeek):
            def __init__(self):
                super().__init__([
                    HarnessAction(kind="tool", tool="resolve_literature", args={"text": "MED:111"}),
                    HarnessAction(kind="synthesize"),
                    HarnessAction(kind="tool", tool="inspect_entity", args={"literature_ref": "literature_1"}),
                    HarnessAction(kind="synthesize"),
                ])
                self.readiness_calls = 0
            def validate_synthesis_readiness(self, **kwargs):
                self.readiness_calls += 1
                if self.readiness_calls == 1:
                    return {"ready": False, "reason": "citation identity only", "missing_requirements": ["inspect literature content"]}
                return {"ready": True, "reason": "", "missing_requirements": []}

        deepseek = CriticDeepSeek()
        class Tools:
            @staticmethod
            def catalog():
                return [{"name": "resolve_literature"}, {"name": "inspect_entity"}]
            def __init__(self): self.calls = 0
            def execute(self, tool, args, ctx):
                self.calls += 1
                if tool == "resolve_literature":
                    ctx.literature_refs["literature_1"] = {"id": "111", "pmid": "111", "source": "MED", "title": "Paper"}
                    ctx.terminal_resolution = {"direction": "conversation", "operation": "resolve_literature", "immediate_result": {"answer_mode": "entity_list", "entities": [{"id": "MED:111", "name": "Paper"}]}}
                    return ToolResult(tool="resolve_literature", status="ok", summary="resolved", terminal=False, payload={"entity_ids": ["MED:111"]})
                ctx.terminal_resolution = {"direction": "conversation", "operation": "inspect_entity", "immediate_result": {"answer_mode": "entity_list", "entities": [{"id": "MED:111", "name": "Paper", "abstract": "Verified abstract"}]}}
                return ToolResult(tool="inspect_entity", status="ok", summary="inspected", terminal=False, payload={"entity_id": "MED:111"})

        harness = CatalystScientificHarness(deepseek=deepseek, tools=Tools(), sessions=AgentSessionStore(ttl_seconds=3600), max_turns=6)
        result = harness.run("MED:111 的主要结论是什么？", ui_language="zh")
        self.assertEqual(result["response_type"], "grounded_synthesis")
        self.assertEqual(deepseek.readiness_calls, 2)
        self.assertTrue(any(step["action_kind"] == "synthesize" and step["status"] == "rejected" for step in result["agent_execution"]["steps"]))
        self.assertEqual(deepseek.synthesis_calls[0]["verified_evidence"][-1]["result"]["immediate_result"]["entities"][0]["abstract"], "Verified abstract")

    def test_failed_scientific_lookup_cannot_fall_back_to_freeform_model_memory(self) -> None:
        harness, deepseek, tools = self.build(
            [
                HarnessAction(kind="tool", tool="resolve_literature", args={"text": "MED:999999999"}),
                HarnessAction(kind="respond", message="I remember what this paper says."),
                HarnessAction(kind="synthesize"),
            ],
            [ToolResult(
                tool="resolve_literature", status="error", summary="No Europe PMC record matched.",
                terminal=False, recoverable=True, error_code="literature_not_found", payload={"query": "MED:999999999"},
            )],
            max_turns=5,
        )
        result = harness.run("总结 MED:999999999 的主要结论。", ui_language="zh")
        self.assertEqual(result["response_type"], "grounded_synthesis")
        self.assertTrue(any(
            step["action_kind"] == "respond" and step["status"] == "rejected"
            for step in result["agent_execution"]["steps"]
        ))
        self.assertEqual(len(deepseek.synthesis_calls), 1)
        self.assertEqual(deepseek.synthesis_calls[0]["verified_evidence"], [])
        self.assertEqual(tools.calls[0][0], "resolve_literature")

class ScientificToolRecoveryTests(unittest.TestCase):
    def test_functional_class_strict_scope_uses_canonical_terms_not_language_variant_synonyms(self) -> None:
        class DeepSeek:
            def parse_protein(self, _text: str) -> dict[str, Any]:
                return {
                    "interpreted_protein": "cytochrome P450",
                    "protein_terms": ["cytochrome P450"],
                    "organism_terms": [], "gene_terms": [], "accession_terms": [],
                }
            def expand_protein_class_terms(self, *, raw_text: str, **_kwargs: Any) -> dict[str, list[str]]:
                strict = ["CYP", "P450 enzyme"] if "细胞色素" in raw_text else ["cytochrome P450"]
                return {"strict_terms": strict, "broader_terms": ["heme-containing monooxygenase"]}

        registry = ScientificToolRegistry(
            agent_resolution=SimpleNamespace(), deepseek=DeepSeek(),
            families=SimpleNamespace(resolve=lambda *_a: None), family_evidence=SimpleNamespace(),
            evidence_queries=SimpleNamespace(), route_design_resolve=lambda *a, **k: {}, pathway_resolve=lambda *a, **k: {},
        )
        resolved_specs = []
        for text in ("细胞色素 P450 酶", "cytochrome P450 enzymes"):
            ctx = HarnessRunContext(ui_language="zh" if "细胞色素" in text else "en", conversation_context={})
            result = registry.execute("resolve_protein_scope", {"text": text, "scope_hint": "family_or_class"}, ctx)
            self.assertEqual(result.status, "ok")
            scope = ctx.protein_refs[result.payload["protein_scope_ref"]]
            resolved_specs.append(scope["enzyme_spec"])
        self.assertEqual(resolved_specs[0]["strict_terms"], ["cytochrome P450"])
        self.assertEqual(resolved_specs[1]["strict_terms"], ["cytochrome P450"])
        self.assertEqual(resolved_specs[0]["protein_terms"], resolved_specs[1]["protein_terms"])
        self.assertEqual(resolved_specs[0]["broader_terms"], resolved_specs[1]["broader_terms"])
        self.assertEqual(resolved_specs[0]["strict_aliases"], ["CYP", "P450 enzyme"])
        self.assertEqual(resolved_specs[1]["strict_aliases"], [])

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
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        resolved = registry.execute(
            "resolve_protein_scope",
            {"text": "narrow oxidoreductase", "scope_hint": "family_or_class"},
            ctx,
        )
        strict_ref = resolved.payload["protein_scope_ref"]
        strict = registry.execute("lookup_relations", {"protein_scope_ref": strict_ref}, ctx)
        self.assertEqual(strict.status, "error")
        self.assertEqual(strict.error_code, "strict_scope_no_evidence")
        self.assertFalse(strict.terminal)

        broaden = registry.execute("broaden_scope", {"protein_scope_ref": strict_ref}, ctx)
        self.assertEqual(broaden.status, "ok")
        self.assertTrue(broaden.payload["approximate_parent_scope"])
        broad_ref = broaden.payload["protein_scope_ref"]
        self.assertNotEqual(broad_ref, strict_ref)

        aggregated = registry.execute("lookup_relations", {"protein_scope_ref": broad_ref}, ctx)
        self.assertEqual(aggregated.status, "ok")
        self.assertFalse(aggregated.terminal)
        self.assertEqual(aggregated.payload["recorded_reaction_count"], 2)
        self.assertTrue(aggregated.payload["scope_broadened"])
        self.assertEqual(ctx.terminal_resolution["immediate_result"]["known_associations"]["count"], 2)


class StructuredReactionResolutionTests(unittest.TestCase):
    def _registry(self, exact_ids: list[str] | None = None, lookup_calls: list[str] | None = None) -> ScientificToolRegistry:
        exact_ids = list(exact_ids or [])

        class Evidence:
            @staticmethod
            def candidate_reactions_for_smiles(reaction_smiles: str) -> list[str]:
                self.assertEqual(reaction_smiles, "CCO>>CC=O")
                return list(exact_ids)

            @staticmethod
            def reaction_metadata(reaction_id: str) -> dict[str, str]:
                return {"reaction_smiles": "CCO>>CC=O", "equation": "ethanol + NAD(+) = acetaldehyde + NADH + H(+)"}

        class AgentResolution:
            evidence = Evidence()

            @staticmethod
            def resolve(text: str) -> dict[str, Any]:
                raise AssertionError("raw Reaction SMILES must not enter fuzzy natural-language Rhea resolution")

        class DeepSeek:
            @staticmethod
            def provenance() -> dict[str, Any]:
                return {"provider": "fake", "model": "fake-controller"}

        class EvidenceQueries:
            @staticmethod
            def lookup_reaction_proteins(reaction_id: str, **kwargs: Any) -> dict[str, Any]:
                if lookup_calls is not None:
                    lookup_calls.append(reaction_id)
                return {
                    "reaction": {"rhea_id": reaction_id},
                    "known_associations": {"count": 1, "items": [{"candidate_id": "P-FAKE"}], "note": "recorded"},
                    "candidates": [],
                }

        return ScientificToolRegistry(
            agent_resolution=AgentResolution(),
            deepseek=DeepSeek(),
            families=object(),
            family_evidence=object(),
            evidence_queries=EvidenceQueries(),
            route_design_resolve=lambda *a, **k: {},
            pathway_resolve=lambda *a, **k: {},
        )

    def test_raw_reaction_smiles_stays_open_world_without_fuzzy_rhea_recommendation(self) -> None:
        registry = self._registry()
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        result = registry.execute("resolve_reaction", {"text": "CCO>>CC=O"}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["input_mode"], "raw_reaction_smiles")
        self.assertEqual(result.payload["reaction_smiles"], "CCO>>CC=O")
        self.assertEqual(result.payload["exact_rhea_ids"], [])
        self.assertTrue(str(result.payload["recommended_id"]).startswith("EXT-RXN-"))
        ref = result.payload["reaction_ref"]
        self.assertEqual(ctx.reaction_refs[ref]["mode"], "raw_reaction_smiles")
        self.assertEqual(ctx.reaction_refs[ref]["matched_reaction_ids"], [])
        self.assertIn("No fuzzy Rhea assignment", result.summary)

    def test_raw_reaction_ref_can_feed_candidate_retrieval_directly(self) -> None:
        registry = self._registry()
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        resolved = registry.execute("resolve_reaction", {"text": "CCO>>CC=O"}, ctx)
        result = registry.execute(
            "candidate_search",
            {
                "direction": "reaction_to_enzyme",
                "full_text": "Find potential enzymes for CCO>>CC=O",
                "reaction_ref": resolved.payload["reaction_ref"],
            },
            ctx,
        )
        self.assertTrue(result.terminal)
        self.assertEqual(result.payload["direction"], "reaction_to_enzyme")
        self.assertTrue(str(result.payload["reaction_id"]).startswith("EXT-RXN-"))
        self.assertEqual(ctx.terminal_resolution["reaction_resolution"]["mode"], "raw_reaction_smiles")

    def test_recorded_lookup_uses_single_exact_structure_match_only(self) -> None:
        calls: list[str] = []
        registry = self._registry(["RHEA:25290"], calls)
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        resolved = registry.execute("resolve_reaction", {"text": "CCO>>CC=O"}, ctx)
        looked_up = registry.execute(
            "lookup_relations",
            {"reaction_ref": resolved.payload["reaction_ref"]},
            ctx,
        )
        self.assertEqual(looked_up.status, "ok")
        self.assertEqual(calls, ["RHEA:25290"])

    def test_recorded_lookup_does_not_assert_facts_without_exact_structure_match(self) -> None:
        registry = self._registry()
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        resolved = registry.execute("resolve_reaction", {"text": "CCO>>CC=O"}, ctx)
        looked_up = registry.execute(
            "lookup_relations",
            {"reaction_ref": resolved.payload["reaction_ref"]},
            ctx,
        )
        self.assertEqual(looked_up.status, "ok")
        self.assertFalse(looked_up.terminal)
        self.assertEqual(looked_up.payload["recorded_count"], 0)
        self.assertEqual(looked_up.payload["evidence_mapping"], "no_unique_exact_rhea")
        self.assertIsNotNone(ctx.terminal_resolution)
        assert ctx.terminal_resolution is not None
        immediate = ctx.terminal_resolution["immediate_result"]
        self.assertEqual(immediate["known_associations"]["count"], 0)
        self.assertEqual(immediate["reaction"]["input_mode"], "raw_reaction_smiles")
        self.assertIn("not proof", immediate["known_associations"]["note"])


class StructuredProteinRecoveryTests(unittest.TestCase):
    def test_resolve_protein_scope_accepts_fasta_as_specific_protein(self) -> None:
        class AgentResolution:
            @staticmethod
            def _sequence_candidate_payload(item: Any) -> dict[str, Any]:
                return {
                    "id": "EXT-PROT-FAKE",
                    "name": item.header,
                    "sequence": item.sequence,
                    "input_mode": "raw_protein_sequence",
                    "model_ready": False,
                }

        class DeepSeek:
            @staticmethod
            def provenance() -> dict[str, Any]:
                return {"provider": "fake", "model": "fake-controller"}

        registry = ScientificToolRegistry(
            agent_resolution=AgentResolution(),
            deepseek=DeepSeek(),
            families=object(),
            family_evidence=object(),
            evidence_queries=object(),
            route_design_resolve=lambda *a, **k: {},
            pathway_resolve=lambda *a, **k: {},
        )
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        result = registry.execute(
            "resolve_protein_scope",
            {"text": ">query\nMSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVLTAGALA", "scope_hint": "auto"},
            ctx,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["scope_kind"], "specific_protein")
        self.assertEqual(result.payload["input_mode"], "raw_protein_sequence")
        ref = result.payload["protein_scope_ref"]
        self.assertEqual(ctx.protein_refs[ref]["kind"], "specific_protein")

    def test_wrong_r2e_direction_on_sequence_only_request_returns_recovery_hint(self) -> None:
        class AgentResolution:
            @staticmethod
            def _sequence_candidate_payload(item: Any) -> dict[str, Any]:
                return {"id": "EXT-PROT-FAKE", "sequence": item.sequence, "input_mode": "raw_protein_sequence", "model_ready": False}

        class DeepSeek:
            @staticmethod
            def provenance() -> dict[str, Any]:
                return {"provider": "fake", "model": "fake-controller"}

        registry = ScientificToolRegistry(
            agent_resolution=AgentResolution(),
            deepseek=DeepSeek(),
            families=object(),
            family_evidence=object(),
            evidence_queries=object(),
            route_design_resolve=lambda *a, **k: {},
            pathway_resolve=lambda *a, **k: {},
        )
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        result = registry.execute(
            "candidate_search",
            {
                "direction": "reaction_to_enzyme",
                "full_text": "Find possible reactions for this protein.\n>query\nMSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVLTAGALA",
            },
            ctx,
        )
        self.assertEqual(result.status, "error")
        self.assertTrue(result.recoverable)
        self.assertTrue(result.payload["detected_protein_sequence"])
        self.assertEqual(result.payload["suggested_direction"], "enzyme_to_reaction")


class CandidatePreparationToolTests(unittest.TestCase):
    def test_e2r_candidate_preparation_does_not_call_legacy_intent_classifier(self) -> None:
        class AgentResolution:
            def agent_resolve(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                raise AssertionError("legacy agent_resolve must not be called")

            def resolve_protein(self, text: str) -> dict[str, Any]:
                self.last_text = text
                return {
                    "mode": "protein_id",
                    "interpreted_protein": "test protein",
                    "assumptions": [],
                    "normalized": {},
                    "candidates": [{"id": "P00338", "name": "test protein"}],
                    "recommended_id": "P00338",
                }

        class DeepSeek:
            @staticmethod
            def provenance() -> dict[str, Any]:
                return {"provider": "fake", "model": "fake-controller"}

        agent = AgentResolution()
        registry = ScientificToolRegistry(
            agent_resolution=agent,
            deepseek=DeepSeek(),
            families=object(),
            family_evidence=object(),
            evidence_queries=object(),
            route_design_resolve=lambda *a, **k: {},
            pathway_resolve=lambda *a, **k: {},
        )
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        result = registry.execute(
            "candidate_search",
            {
                "direction": "enzyme_to_reaction",
                "full_text": "For UniProt P00338, rank possible reactions.",
                "protein_text": "UniProt P00338",
            },
            ctx,
        )
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.terminal)
        self.assertEqual(result.payload["direction"], "enzyme_to_reaction")
        self.assertEqual(result.payload["protein_id"], "P00338")
        self.assertEqual(agent.last_text, "UniProt P00338")
        self.assertEqual(ctx.terminal_resolution["direction"], "enzyme_to_reaction")

    def test_e2r_candidate_preparation_verifies_explicit_reaction_seed_text_and_ref(self) -> None:
        class AgentResolution:
            def __init__(self): self.reaction_calls = []
            def resolve_protein(self, text: str) -> dict[str, Any]:
                return {
                    "mode": "protein_id", "interpreted_protein": text, "assumptions": [], "normalized": {},
                    "candidates": [{"id": "P00338", "name": "LDHA"}], "recommended_id": "P00338",
                }
            def resolve(self, text: str) -> dict[str, Any]:
                self.reaction_calls.append(text)
                rid = "RHEA:23444" if "23444" in text else "RHEA:25290"
                return {
                    "mode": "rhea_id", "interpreted_reaction": f"verified {rid}", "assumptions": [], "normalized": {},
                    "candidates": [{"rhea_id": rid, "equation": f"equation {rid}"}], "recommended_id": rid,
                }

        class DeepSeek:
            @staticmethod
            def provenance() -> dict[str, Any]: return {"provider": "fake", "model": "fake-controller"}

        agent = AgentResolution()
        registry = ScientificToolRegistry(
            agent_resolution=agent, deepseek=DeepSeek(), families=object(), family_evidence=object(),
            evidence_queries=object(), route_design_resolve=lambda *a, **k: {}, pathway_resolve=lambda *a, **k: {},
        )
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        ctx.reaction_refs["reaction_seed_1"] = {
            "mode": "rhea_id", "interpreted_reaction": "verified RHEA:25290",
            "recommended_id": "RHEA:25290", "normalized": {},
            "candidates": [{"rhea_id": "RHEA:25290", "equation": "equation RHEA:25290"}],
        }
        full_text = "For P00338, use RHEA:23444 as a known activity and also the verified reaction from above."
        result = registry.execute("candidate_search", {
            "direction": "enzyme_to_reaction", "full_text": full_text, "protein_text": "P00338",
            "positive_reaction_texts": ["RHEA:23444"], "positive_reaction_refs": ["reaction_seed_1"],
        }, ctx)
        self.assertEqual(result.status, "ok")
        groups = ctx.terminal_resolution["positive_reaction_resolutions"]
        self.assertEqual([row["recommended_id"] for row in groups], ["RHEA:25290", "RHEA:23444"] )
        self.assertEqual(groups[0]["source_ref"], "reaction_seed_1")
        self.assertEqual(agent.reaction_calls, ["RHEA:23444"] )
        self.assertEqual(result.payload["positive_seed_count"], 2)

        rejected = registry.execute("candidate_search", {
            "direction": "enzyme_to_reaction", "full_text": "For P00338, rank possible reactions.",
            "protein_text": "P00338", "positive_reaction_texts": ["RHEA:23444"],
        }, HarnessRunContext(ui_language="en", conversation_context={}))
        self.assertEqual(rejected.status, "error")
        self.assertEqual(rejected.error_code, "candidate_positive_reaction_not_in_user_text")

    def test_e2r_candidate_preparation_rejects_unknown_positive_reaction_ref(self) -> None:
        class AgentResolution:
            @staticmethod
            def resolve_protein(text: str) -> dict[str, Any]:
                return {"mode": "protein_id", "interpreted_protein": text, "candidates": [{"id": "P00338"}], "recommended_id": "P00338"}
        class DeepSeek:
            @staticmethod
            def provenance() -> dict[str, Any]: return {"provider": "fake", "model": "fake"}
        registry = ScientificToolRegistry(
            agent_resolution=AgentResolution(), deepseek=DeepSeek(), families=object(), family_evidence=object(),
            evidence_queries=object(), route_design_resolve=lambda *a, **k: {}, pathway_resolve=lambda *a, **k: {},
        )
        result = registry.execute("candidate_search", {
            "direction": "enzyme_to_reaction", "full_text": "For P00338 use the reaction above as a positive.",
            "protein_text": "P00338", "positive_reaction_refs": ["reaction_missing"],
        }, HarnessRunContext(ui_language="en", conversation_context={}))
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_code, "unknown_positive_reaction_ref")

    def test_candidate_preparation_reuses_verified_refs_and_rejects_family_as_neural_query(self) -> None:
        class AgentResolution:
            def resolve(self, text: str) -> dict[str, Any]:
                raise AssertionError("verified reaction ref should avoid re-resolution")

            def resolve_protein(self, text: str) -> dict[str, Any]:
                raise AssertionError("verified protein ref should avoid re-resolution")

        class DeepSeek:
            @staticmethod
            def provenance() -> dict[str, Any]:
                return {"provider": "fake", "model": "fake-controller"}

        registry = ScientificToolRegistry(
            agent_resolution=AgentResolution(),
            deepseek=DeepSeek(),
            families=object(),
            family_evidence=object(),
            evidence_queries=object(),
            route_design_resolve=lambda *a, **k: {},
            pathway_resolve=lambda *a, **k: {},
        )
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        ctx.reaction_refs["reaction_1"] = {
            "mode": "rhea_id",
            "recommended_id": "RHEA:32883",
            "candidates": [{"rhea_id": "RHEA:32883"}],
        }
        r2e = registry.execute(
            "candidate_search",
            {"direction": "reaction_to_enzyme", "full_text": "show candidates", "reaction_ref": "reaction_1"},
            ctx,
        )
        self.assertEqual(r2e.status, "ok")
        self.assertEqual(ctx.terminal_resolution["reaction_resolution"]["recommended_id"], "RHEA:32883")

        ctx2 = HarnessRunContext(ui_language="en", conversation_context={})
        ctx2.protein_refs["protein_scope_1"] = {
            "kind": "specific_protein",
            "resolution": {
                "mode": "protein_id",
                "interpreted_protein": "P00338",
                "candidates": [{"id": "P00338"}],
                "recommended_id": "P00338",
            },
        }
        e2r = registry.execute(
            "candidate_search",
            {"direction": "enzyme_to_reaction", "full_text": "show possible reactions", "protein_scope_ref": "protein_scope_1"},
            ctx2,
        )
        self.assertEqual(e2r.status, "ok")
        self.assertEqual(ctx2.terminal_resolution["protein_resolution"]["recommended_id"], "P00338")

        ctx3 = HarnessRunContext(ui_language="en", conversation_context={})
        ctx3.protein_refs["protein_scope_1"] = {"kind": "family", "family_id": "PF01040", "label": "UbiA family"}
        family = registry.execute(
            "candidate_search",
            {"direction": "enzyme_to_reaction", "full_text": "predict family reactions", "protein_scope_ref": "protein_scope_1"},
            ctx3,
        )
        self.assertEqual(family.status, "error")
        self.assertEqual(family.error_code, "candidate_requires_specific_protein")


class NaturalScientificToolTests(unittest.TestCase):
    @staticmethod
    def _registry(*, evidence_queries: Any = None, families: Any = None, compound_resolve: Any = None, family_evidence: Any = None, agent_resolution: Any = None, research_service: Any = None) -> ScientificToolRegistry:
        class DeepSeek:
            @staticmethod
            def provenance() -> dict[str, Any]:
                return {"provider": "fake", "model": "fake"}

        return ScientificToolRegistry(
            agent_resolution=agent_resolution or SimpleNamespace(),
            deepseek=DeepSeek(),
            families=families or SimpleNamespace(),
            family_evidence=family_evidence or SimpleNamespace(),
            evidence_queries=evidence_queries or SimpleNamespace(),
            route_design_resolve=lambda *a, **k: {},
            pathway_resolve=lambda *a, **k: {},
            compound_resolve=compound_resolve,
            research_service=research_service,
        )

    def test_explicit_unknown_pfam_never_becomes_free_text_functional_class(self) -> None:
        families = SimpleNamespace(resolve=lambda *_args: None)
        registry = self._registry(families=families)
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        result = registry.execute(
            "resolve_protein_scope",
            {"text": "Please inspect PF99999 family", "scope_hint": "family_or_class"},
            ctx,
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_code, "protein_family_not_found")
        self.assertEqual(result.payload["family_id"], "PF99999")
        self.assertEqual(ctx.protein_refs, {})

    def test_resolver_cannot_copy_historical_protein_identity_without_reuse(self) -> None:
        class AgentResolution:
            @staticmethod
            def resolve_protein(_text: str) -> dict[str, Any]:
                raise AssertionError("historical identity must be rejected before fresh resolution")

        registry = self._registry(agent_resolution=AgentResolution())
        ctx = HarnessRunContext(
            ui_language="zh",
            conversation_context={},
            user_text="改成只看已记录反应，不要模型。",
            session_facts={
                "session_entities": {
                    "all": [{
                        "kind": "protein", "id": "P00338", "label": "LDHA",
                        "active": True, "focus": True, "role": "confirmed_target",
                    }]
                }
            },
        )
        result = registry.execute(
            "resolve_protein_scope",
            {"text": "P00338", "scope_hint": "specific_protein"},
            ctx,
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_code, "session_identity_requires_reuse")
        self.assertEqual(result.payload["historical_identity"], "P00338")

    def test_resolver_allows_identity_when_latest_user_restates_it(self) -> None:
        class AgentResolution:
            @staticmethod
            def resolve_protein(text: str) -> dict[str, Any]:
                return {
                    "mode": "protein_id", "recommended_id": text,
                    "interpreted_protein": text,
                    "candidates": [{"id": text, "name": text, "input_mode": "protein_id"}],
                }

        registry = self._registry(agent_resolution=AgentResolution())
        ctx = HarnessRunContext(
            ui_language="zh", conversation_context={}, user_text="改查 P00338 已记录反应。",
            session_facts={"session_entities": {"all": [{"kind": "protein", "id": "P00338", "label": "LDHA"}]}},
        )
        result = registry.execute(
            "resolve_protein_scope",
            {"text": "P00338", "scope_hint": "specific_protein"},
            ctx,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["recommended_id"], "P00338")

    def test_specific_protein_recorded_reactions_uses_reverse_evidence_tool(self) -> None:
        class Queries:
            @staticmethod
            def lookup_protein_reactions(protein_id: str, *, ui_language: str):
                self.assertEqual(protein_id, "P_TEST")
                return {
                    "protein": {"id": "P_TEST", "name": "P_TEST"},
                    "known_associations": {"count": 1, "items": [{"candidate_id": "RHEA:12345"}], "note": "recorded"},
                    "candidates": [],
                }

        registry = self._registry(evidence_queries=Queries())
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        ctx.protein_refs["protein_scope_1"] = {
            "kind": "specific_protein",
            "resolution": {"mode": "protein_id", "recommended_id": "P_TEST", "candidates": []},
        }
        result = registry.execute("lookup_relations", {"protein_scope_ref": "protein_scope_1"}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["reaction_ids"], ["RHEA:12345"])
        self.assertEqual(ctx.terminal_resolution["direction"], "enzyme_to_reaction")
        self.assertEqual(ctx.terminal_resolution["immediate_result"]["known_associations"]["count"], 1)

    def test_lookup_relations_aggregates_family_scope(self) -> None:
        class FamilyEvidence:
            @staticmethod
            def summarize(_family_id: str, *, ui_language: str):
                return {
                    "protein": {"id": "PF00001", "name": "Example family"},
                    "family": {"evidence_member_count": 2},
                    "known_associations": {"count": 1, "items": [{"candidate_id": "RHEA:12345"}], "note": "family evidence"},
                    "candidates": [],
                }

        registry = self._registry(family_evidence=FamilyEvidence())
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        ctx.protein_refs["protein_scope_1"] = {"kind": "family", "family_id": "PF00001", "label": "Example family"}
        result = registry.execute("lookup_relations", {"protein_scope_ref": "protein_scope_1"}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["recorded_reaction_count"], 1)
        self.assertEqual(ctx.terminal_resolution["immediate_result"]["known_associations"]["count"], 1)

    def test_lookup_relations_checks_one_concrete_pair(self) -> None:
        class Queries:
            @staticmethod
            def lookup_reaction_proteins(reaction_id: str, *, enzyme_spec: dict, enzyme_scope: str, ui_language: str):
                self.assertEqual(reaction_id, "RHEA:12345")
                self.assertEqual(enzyme_scope, "specific_protein")
                self.assertEqual(enzyme_spec["accession_terms"], ["P_TEST"])
                return {
                    "known_associations": {"count": 1, "items": [{"candidate_id": "P_TEST"}], "note": "pair recorded"},
                    "candidates": [],
                }

        registry = self._registry(evidence_queries=Queries())
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        ctx.reaction_refs["reaction_1"] = {"recommended_id": "RHEA:12345", "candidates": []}
        ctx.protein_refs["protein_scope_1"] = {
            "kind": "specific_protein",
            "resolution": {"recommended_id": "P_TEST", "candidates": []},
        }
        result = registry.execute("lookup_relations", {"reaction_ref": "reaction_1", "protein_scope_ref": "protein_scope_1"}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["protein_ids"], ["P_TEST"])
        self.assertEqual(ctx.terminal_resolution["immediate_result"]["known_associations"]["count"], 1)

    def test_list_family_members_returns_entity_list_without_catalytic_claim(self) -> None:
        family = SimpleNamespace(
            family_id="PF00001",
            label="Example family",
            member_ids=("P1", "P2"),
            source="test_family",
            scope_note="Auditable subset only.",
            scope_note_zh="仅当前可审计子集。",
        )
        families = SimpleNamespace(family=lambda _fid: family)
        evidence = SimpleNamespace(
            protein_metadata=lambda pid: {"canonical_accession": pid},
            is_candidate_protein=lambda _pid: True,
        )
        catalog = SimpleNamespace(protein_by_id={"P1": {"name": "Protein 1"}, "P2": {"name": "Protein 2"}})
        agent_resolution = SimpleNamespace(catalog=catalog, evidence=evidence)
        registry = self._registry(families=families, agent_resolution=agent_resolution)
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        ctx.protein_refs["protein_scope_1"] = {"kind": "family", "family_id": "PF00001", "label": "Example family"}
        result = registry.execute("list_scope_members", {"protein_scope_ref": "protein_scope_1", "limit": 2}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.terminal)
        immediate = ctx.terminal_resolution["immediate_result"]
        self.assertEqual(immediate["answer_mode"], "entity_list")
        self.assertEqual(immediate["entity_kind"], "protein")
        self.assertEqual([row["id"] for row in immediate["entities"]], ["P1", "P2"])
        self.assertIn("subset", immediate["note"].lower())

    def test_compound_resolution_normalizes_noncanonical_names_before_local_id_assignment(self) -> None:
        calls: list[list[str]] = []

        def resolve(terms, *, limit):
            values = list(terms)
            calls.append(values)
            rows = []
            if "p-coumaric acid" in values:
                rows.append({"chebi_id": "CHEBI:12876", "name": "(E)-4-coumarate", "smiles": "O=C([O-])/C=C/c1ccc(O)cc1"})
            if "caffeic acid" in values:
                rows.append({"chebi_id": "CHEBI:57770", "name": "(E)-caffeate", "smiles": "O=C([O-])/C=C/c1ccc(O)c(O)c1"})
            return rows[:limit]

        class DeepSeek:
            @staticmethod
            def provenance() -> dict[str, Any]:
                return {"provider": "fake", "model": "fake"}

            @staticmethod
            def normalize_compound_terms(*, source_terms, target_terms):
                self.assertEqual(source_terms, ["对香豆酸", "咖啡酸"])
                self.assertEqual(target_terms, [])
                return {
                    "source_terms": ["对香豆酸", "p-coumaric acid", "咖啡酸", "caffeic acid"],
                    "target_terms": [],
                }

        registry = self._registry(compound_resolve=resolve)
        registry.deepseek = DeepSeek()
        ctx = HarnessRunContext(ui_language="zh", conversation_context={})
        result = registry.execute("resolve_compound", {"terms": ["对香豆酸", "咖啡酸"], "limit": 8}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["candidate_ids"], ["CHEBI:12876", "CHEBI:57770"])
        self.assertEqual(calls[0], ["对香豆酸", "咖啡酸"])
        self.assertIn("p-coumaric acid", calls[1])
        self.assertIn("caffeic acid", calls[1])


    def test_compound_resolution_uses_local_ids_and_can_be_remembered(self) -> None:
        def resolve(terms, *, limit):
            self.assertIn("p-coumaric acid", terms)
            self.assertEqual(limit, 3)
            return [{"chebi_id": "CHEBI:12876", "name": "(E)-4-coumarate", "smiles": "O=C([O-])/C=C/c1ccc(O)cc1"}]

        registry = self._registry(compound_resolve=resolve)
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        result = registry.execute("resolve_compound", {"terms": ["p-coumaric acid"], "limit": 3}, ctx)
        self.assertEqual(result.status, "ok")
        immediate = ctx.terminal_resolution["immediate_result"]
        self.assertEqual(immediate["entities"][0]["id"], "CHEBI:12876")
        self.assertTrue(result.payload["compound_refs"][0]["ref"].startswith("compound_"))
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("compound-session", ctx.terminal_resolution)
        self.assertEqual(store.snapshot("compound-session")["verified_compound_ids"], ["CHEBI:12876"])


    def test_inspect_entity_reads_only_existing_refs(self) -> None:
        evidence = SimpleNamespace(
            reaction_metadata=lambda rid: {"reaction_smiles": "CCO>>CC=O"} if rid == "RHEA:12345" else None,
            protein_metadata=lambda _pid: None,
            is_candidate_protein=lambda _pid: True,
        )
        catalog = SimpleNamespace(protein_by_id={})
        agent_resolution = SimpleNamespace(evidence=evidence, catalog=catalog)
        registry = self._registry(agent_resolution=agent_resolution)
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        ctx.reaction_refs["reaction_1"] = {
            "mode": "session_verified_rhea",
            "interpreted_reaction": "RHEA:12345",
            "recommended_id": "RHEA:12345",
            "candidates": [{"rhea_id": "RHEA:12345", "equation": "ethanol = acetaldehyde"}],
        }
        reaction = registry.execute("inspect_entity", {"reaction_ref": "reaction_1"}, ctx)
        self.assertEqual(reaction.status, "ok")
        self.assertFalse(reaction.terminal)
        immediate = ctx.terminal_resolution["immediate_result"]
        self.assertEqual(immediate["answer_mode"], "entity_list")
        self.assertEqual(immediate["entity_kind"], "reaction")
        self.assertEqual(immediate["entities"][0]["id"], "RHEA:12345")
        self.assertIn("CCO>>CC=O", immediate["entities"][0]["subtitle"])

        ctx.compound_refs["compound_1"] = {
            "chebi_id": "CHEBI:12876",
            "name": "(E)-4-coumarate",
            "smiles": "O=C([O-])/C=C/c1ccc(O)cc1",
        }
        compound = registry.execute("inspect_entity", {"compound_ref": "compound_1"}, ctx)
        self.assertEqual(compound.status, "ok")
        self.assertEqual(ctx.terminal_resolution["immediate_result"]["entities"][0]["id"], "CHEBI:12876")

        missing = registry.execute("inspect_entity", {"reaction_ref": "missing"}, ctx)
        self.assertEqual(missing.status, "error")
        self.assertEqual(missing.error_code, "unknown_reaction_ref")

    def test_verified_functional_scope_stays_history_until_explicit_reuse(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("scope-session", {
            "direction": "enzyme_to_reaction",
            "protein_resolution": {
                "mode": "protein_functional_class",
                "interpreted_protein": "cytochrome P450",
                "recommended_id": "CLASS-ABC",
                "family": {
                    "scope_id": "CLASS-ABC",
                    "family_id": "CLASS-ABC",
                    "label": "cytochrome P450",
                    "normalized_terms": ["cytochrome P450"],
                    "strict_terms": ["cytochrome P450"],
                    "broader_terms": ["heme monooxygenase"],
                    "scope_broadened": False,
                },
            },
            "immediate_result": {
                "protein": {"id": "CLASS-ABC", "name": "cytochrome P450", "input_mode": "protein_functional_class"},
                "family": {
                    "scope_id": "CLASS-ABC", "family_id": "CLASS-ABC", "label": "cytochrome P450",
                    "normalized_terms": ["cytochrome P450"], "strict_terms": ["cytochrome P450"],
                    "broader_terms": ["heme monooxygenase"],
                },
            },
        })
        snapshot = store.snapshot("scope-session")
        self.assertEqual(snapshot["verified_protein_scopes"][0]["kind"], "functional_class")
        entities = snapshot["session_entities"]["all"]
        scope = next(row for row in entities if row.get("kind") == "protein_scope")
        self.assertEqual(scope["id"], "CLASS-ABC")
        self.assertEqual(scope["payload"]["enzyme_spec"]["strict_terms"], ["cytochrome P450"])
        model_snapshot = store.model_snapshot("scope-session")
        self.assertNotIn("current_run_refs", model_snapshot)
        self.assertIn("reuse_session_entity", model_snapshot["session_entities"]["reuse_rule"])

    def test_database_ids_are_not_current_tool_refs_without_explicit_reuse(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("scope-session", {
            "direction": "enzyme_to_reaction",
            "protein_resolution": {
                "mode": "protein_functional_class",
                "interpreted_protein": "example class",
                "recommended_id": "CLASS-ABC",
                "family": {
                    "scope_id": "CLASS-ABC", "family_id": "CLASS-ABC", "label": "example class",
                    "normalized_terms": ["example class"], "strict_terms": ["example class"], "broader_terms": [],
                },
            },
        })
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        self.assertEqual(ctx.protein_refs, {})
        self.assertEqual(ctx.reaction_refs, {})
        self.assertEqual(ctx.compound_refs, {})
        snapshot = store.model_snapshot("scope-session")
        self.assertNotIn("current_run_refs", snapshot)
        self.assertTrue(any(row.get("id") == "CLASS-ABC" for row in snapshot["session_entities"]["history"]))

    def test_verified_compound_session_ref_can_be_consumed_without_guessing_id(self) -> None:
        calls: list[list[str]] = []
        def resolve(terms, *, limit):
            calls.append(list(terms))
            return [{"chebi_id": "CHEBI:12876", "name": "(E)-4-coumarate", "smiles": "SMILES"}]

        registry = self._registry(compound_resolve=resolve)
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        ctx.compound_refs["session_compound_1"] = {"chebi_id": "CHEBI:12876", "name": "CHEBI:12876", "smiles": ""}
        result = registry.execute("resolve_compound", {"compound_ref": "session_compound_1", "limit": 5}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertEqual(calls, [["CHEBI:12876"]])
        self.assertEqual(ctx.terminal_resolution["immediate_result"]["entities"][0]["id"], "CHEBI:12876")


    def test_research_workspace_remembers_multiple_literature_providers_and_deduplicates_doi_identity(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("multi-lit", {
            "direction": "enzyme_to_reaction", "operation": "research_workspace",
            "protein_resolution": {"mode": "protein_id", "interpreted_protein": "P1", "recommended_id": "P1", "candidates": [{"id": "P1"}]},
            "immediate_result": {
                "answer_mode": "research_workspace",
                "source_panels": [
                    {"id": "literature_curated", "section": "literature", "pagination": {"page_size": 10}, "items": [
                        {"pmid": "111", "doi": "10.1/a", "source": "MED", "provider": "europe_pmc", "title": "Shared paper"},
                    ]},
                    {"id": "literature_openalex", "section": "literature", "pagination": {"page_size": 10}, "items": [
                        {"id": "OPENALEX:W1", "doi": "10.1/a", "source": "OPENALEX", "provider": "openalex", "title": "Shared paper"},
                        {"id": "OPENALEX:W2", "doi": "10.1/b", "source": "OPENALEX", "provider": "openalex", "title": "OpenAlex only"},
                    ]},
                ], "known_associations": {"count": 0, "items": []},
            },
        })
        snap = store.snapshot("multi-lit")
        literature = [row for row in snap["session_entities"]["related"] if row.get("kind") == "literature"]
        ids = {row["id"] for row in literature}
        self.assertIn("MED:111", ids)
        self.assertIn("DOI:10.1/b", ids)
        self.assertEqual(len(ids), 2)

    def test_research_workspace_literature_is_related_session_evidence(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("lit-session", {
            "direction": "enzyme_to_reaction",
            "operation": "research_workspace",
            "protein_resolution": {
                "mode": "protein_id", "interpreted_protein": "P00338",
                "recommended_id": "P00338", "candidates": [{"id": "P00338", "name": "LDHA"}],
            },
            "immediate_result": {
                "answer_mode": "research_workspace",
                "source_panels": [{
                    "id": "literature", "items": [
                        {"id": "111", "source": "MED", "title": "Paper one", "authors": "A et al.", "journal": "J1", "year": "2025", "abstract": "Abstract one"},
                        {"id": "222", "source": "MED", "title": "Paper two", "authors": "B et al.", "journal": "J2", "year": "2026", "abstract": "Abstract two"},
                    ],
                }],
                "known_associations": {"count": 0, "items": []},
            },
        })
        snap = store.snapshot("lit-session")
        literature = [row for row in snap["session_entities"]["related"] if row.get("kind") == "literature"]
        self.assertEqual(len(literature), 2)
        self.assertEqual({row["related_index"] for row in literature}, {1, 2})
        self.assertFalse(any(row.get("active") for row in literature))
        self.assertEqual(next(row for row in literature if row["id"] == "MED:222")["payload"]["abstract"], "Abstract two")

    def test_reuse_and_inspect_second_literature_record(self) -> None:
        class DeepSeek:
            @staticmethod
            def select_session_entity_reference(**kwargs):
                row = next(row for row in kwargs["records"] if row.get("related_index") == 2)
                return {"reference_mode": "specific", "selected_key": f"{row['kind']}:{row['id']}", "reason": "second paper"}
            @staticmethod
            def provenance():
                return {"provider": "fake", "model": "fake"}
        registry = ScientificToolRegistry(
            agent_resolution=SimpleNamespace(), deepseek=DeepSeek(), families=SimpleNamespace(),
            family_evidence=SimpleNamespace(), evidence_queries=SimpleNamespace(),
            route_design_resolve=lambda *a, **k: {}, pathway_resolve=lambda *a, **k: {},
        )
        ctx = HarnessRunContext(
            ui_language="zh", conversation_context={}, user_text="第二篇文献具体讲了什么？",
            session_facts={"session_entities": {"all": [
                {"kind": "literature", "id": "MED:111", "label": "Paper one", "role": "related_evidence", "related_index": 1, "payload": {"id": "111", "source": "MED", "title": "Paper one", "abstract": "A1"}},
                {"kind": "literature", "id": "MED:222", "label": "Paper two", "role": "related_evidence", "related_index": 2, "payload": {"id": "222", "source": "MED", "title": "Paper two", "authors": "B et al.", "journal": "J2", "year": "2026", "abstract": "A2", "url": "https://europepmc.org/article/MED/222"}},
            ]}},
        )
        reused = registry.execute("reuse_session_entity", {"entity_kind": "literature"}, ctx)
        self.assertEqual(reused.status, "ok")
        ref = reused.payload["literature_ref"]
        inspected = registry.execute("inspect_entity", {"literature_ref": ref}, ctx)
        self.assertEqual(inspected.status, "ok")
        row = ctx.terminal_resolution["immediate_result"]["entities"][0]
        self.assertEqual(row["name"], "Paper two")
        self.assertEqual(row["abstract"], "A2")
        self.assertIn("B et al.", row["subtitle"])

    def test_recorded_relation_tools_return_related_entity_refs(self) -> None:
        class Queries:
            @staticmethod
            def lookup_reaction_proteins(_reaction_id: str, **_kwargs):
                return {"known_associations": {"count": 1, "items": [{"candidate_id": "P12345"}], "note": "recorded"}, "candidates": []}

            @staticmethod
            def lookup_protein_reactions(_protein_id: str, **_kwargs):
                return {"known_associations": {"count": 1, "items": [{"candidate_id": "RHEA:12345"}], "note": "recorded"}, "candidates": []}

        evidence = SimpleNamespace(reaction_metadata=lambda rid: {"equation": f"equation {rid}"}, protein_metadata=lambda _pid: {}, is_candidate_protein=lambda _pid: True)
        agent_resolution = SimpleNamespace(evidence=evidence, catalog=SimpleNamespace(protein_by_id={}), proteins=SimpleNamespace(exact_or_search=lambda *_a, **_k: []))
        registry = self._registry(evidence_queries=Queries(), agent_resolution=agent_resolution)
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        ctx.reaction_refs["reaction_1"] = {"mode": "session_verified_rhea", "recommended_id": "RHEA:99999", "candidates": []}
        r2e = registry.execute("lookup_relations", {"reaction_ref": "reaction_1"}, ctx)
        self.assertEqual(r2e.status, "ok")
        pref = r2e.payload["protein_refs"][0]["ref"]
        self.assertEqual(ctx.protein_refs[pref]["resolution"]["recommended_id"], "P12345")

        ctx.protein_refs["protein_scope_1"] = {"kind": "specific_protein", "resolution": {"recommended_id": "P12345"}}
        e2r = registry.execute("lookup_relations", {"protein_scope_ref": "protein_scope_1"}, ctx)
        self.assertEqual(e2r.status, "ok")
        rref = e2r.payload["reaction_refs"][0]["ref"]
        self.assertEqual(ctx.reaction_refs[rref]["recommended_id"], "RHEA:12345")

    def test_session_keeps_related_evidence_separate_from_future_targets(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("result-entities", {
            "direction": "reaction_to_enzyme",
            "immediate_result": {
                "answer_mode": "recorded_association_lookup",
                "known_associations": {"items": [{"candidate_id": "P12345", "name": "related protein"}]},
            },
        })
        first = store.snapshot("result-entities")
        self.assertNotIn("P12345", first["verified_protein_ids"])
        self.assertEqual(first["recent_evidence_ids"], ["P12345"])
        self.assertEqual(first["session_entities"]["related"][0]["id"], "P12345")

        store.remember_resolution("result-entities", {
            "direction": "conversation",
            "operation": "inspect_entity",
            "protein_resolution": {
                "mode": "protein_id",
                "recommended_id": "Q99999",
                "interpreted_protein": "Q99999",
                "candidates": [{"id": "Q99999", "name": "inspected protein", "input_mode": "protein_id"}],
            },
            "immediate_result": {
                "answer_mode": "entity_list",
                "entity_kind": "protein",
                "entities": [{"id": "Q99999", "name": "inspected protein"}],
            },
        })
        snapshot = store.snapshot("result-entities")
        self.assertNotIn("P12345", snapshot["verified_protein_ids"])
        self.assertIn("Q99999", snapshot["verified_protein_ids"])

    def test_reuse_session_entity_promotes_validated_prior_protein_with_executable_candidate(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("reuse", {
            "direction": "enzyme_to_reaction",
            "protein_resolution": {
                "mode": "protein_id",
                "recommended_id": "P00338",
                "interpreted_protein": "L-lactate dehydrogenase",
                "candidates": [{"id": "P00338", "name": "L-lactate dehydrogenase", "input_mode": "protein_id"}],
            },
        })

        class DeepSeek:
            @staticmethod
            def provenance() -> dict[str, Any]:
                return {"provider": "fake", "model": "fake"}

            @staticmethod
            def select_session_entity_reference(**_kwargs: Any) -> dict[str, Any]:
                return {"reference_mode": "focus", "selected_key": "", "reason": "this enzyme refers to current focus"}

        registry = ScientificToolRegistry(
            agent_resolution=SimpleNamespace(resolve_protein=lambda _pid: {}),
            deepseek=DeepSeek(),
            families=SimpleNamespace(),
            family_evidence=SimpleNamespace(),
            evidence_queries=SimpleNamespace(),
            route_design_resolve=lambda *a, **k: {},
            pathway_resolve=lambda *a, **k: {},
        )
        ctx = HarnessRunContext(
            ui_language="en", conversation_context={}, user_text="use this enzyme", session_facts=store.snapshot("reuse")
        )
        result = registry.execute("reuse_session_entity", {"entity_kind": "protein"}, ctx)
        self.assertEqual(result.status, "ok")
        ref = result.payload["protein_scope_ref"]
        resolution = ctx.protein_refs[ref]["resolution"]
        self.assertEqual(resolution["recommended_id"], "P00338")
        self.assertEqual([row["id"] for row in resolution["candidates"]], ["P00338"])

    def test_reuse_session_entity_rejects_stale_target_when_latest_message_switches(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("switch", {
            "direction": "enzyme_to_reaction",
            "protein_resolution": {
                "mode": "protein_id", "recommended_id": "P00338",
                "candidates": [{"id": "P00338", "name": "old protein", "input_mode": "protein_id"}],
            },
        })

        class DeepSeek:
            @staticmethod
            def provenance() -> dict[str, Any]:
                return {"provider": "fake", "model": "fake"}

            @staticmethod
            def select_session_entity_reference(**_kwargs: Any) -> dict[str, Any]:
                return {"reference_mode": "none", "selected_key": "", "reason": "latest message introduces a different named protein"}

        registry = ScientificToolRegistry(
            agent_resolution=SimpleNamespace(resolve_protein=lambda _pid: {}),
            deepseek=DeepSeek(), families=SimpleNamespace(), family_evidence=SimpleNamespace(), evidence_queries=SimpleNamespace(),
            route_design_resolve=lambda *a, **k: {}, pathway_resolve=lambda *a, **k: {},
        )
        ctx = HarnessRunContext(
            ui_language="zh", conversation_context={},
            user_text="不要这个了，换成丹参中的 miltiradiene synthase KSL1",
            session_facts=store.snapshot("switch"),
        )
        result = registry.execute("reuse_session_entity", {"entity_kind": "protein"}, ctx)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_code, "session_entity_not_referenced")
        self.assertEqual(ctx.protein_refs, {})

    def test_raw_sequence_payload_is_server_reusable_but_hidden_from_controller_snapshot(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        sequence = "MSTNPKPQRKTKRNTNRRPQDVKFPGG"
        store.remember_resolution("raw", {
            "direction": "enzyme_to_reaction",
            "protein_resolution": {
                "mode": "raw_protein_sequence",
                "recommended_id": "EXT-PROT-RAW",
                "interpreted_protein": "provided sequence",
                "candidates": [{"id": "EXT-PROT-RAW", "name": "provided sequence", "input_mode": "raw_protein_sequence", "sequence": sequence}],
            },
        })
        full = store.snapshot("raw")
        model = store.model_snapshot("raw")
        payload = full["session_entities"]["history"][0]["payload"]
        self.assertEqual(payload["candidates"][0]["sequence"], sequence)
        self.assertNotIn("payload", model["session_entities"]["history"][0])

    def test_latest_resolved_target_is_focus_even_when_older_target_remains_active(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("focus", {
            "direction": "enzyme_to_reaction",
            "protein_resolution": {"mode": "protein_id", "recommended_id": "P00338", "candidates": [{"id": "P00338", "name": "first", "input_mode": "protein_id"}]},
        })
        store.confirm_protein("focus", protein_id="P00338")
        store.remember_resolution("focus", {
            "direction": "enzyme_to_reaction",
            "protein_resolution": {"mode": "protein_id", "recommended_id": "A0A1W6QDI7", "candidates": [{"id": "A0A1W6QDI7", "name": "second", "input_mode": "protein_id"}]},
        })
        rows = store.model_snapshot("focus")["session_entities"]["history"]
        by_id = {row["id"]: row for row in rows}
        self.assertTrue(by_id["A0A1W6QDI7"]["focus"])
        self.assertFalse(by_id["A0A1W6QDI7"]["active"])
        self.assertTrue(by_id["P00338"]["active"])
        self.assertFalse(by_id["P00338"]["focus"])

    def test_focus_reuse_overrides_selector_choice_of_older_active_target(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("focus-priority", {
            "direction": "enzyme_to_reaction",
            "protein_resolution": {
                "mode": "protein_id", "recommended_id": "P00338",
                "candidates": [{"id": "P00338", "name": "LDHA", "input_mode": "protein_id"}],
            },
        })
        store.confirm_protein("focus-priority", protein_id="P00338")
        store.remember_resolution("focus-priority", {
            "direction": "enzyme_to_reaction",
            "protein_resolution": {
                "mode": "protein_id", "recommended_id": "A0A1W6QDI7",
                "candidates": [{"id": "A0A1W6QDI7", "name": "new focus", "input_mode": "protein_id"}],
            },
        })

        class DeepSeek:
            @staticmethod
            def provenance() -> dict[str, Any]:
                return {"provider": "fake", "model": "fake"}

            @staticmethod
            def select_session_entity_reference(**_kwargs: Any) -> dict[str, Any]:
                # Simulate the exact model mistake observed in the real HTTP flow:
                # it recognizes a historical reference but chooses the older active target.
                return {"reference_mode": "focus", "selected_key": "", "reason": "generic current reference"}

        registry = ScientificToolRegistry(
            agent_resolution=SimpleNamespace(resolve_protein=lambda pid: {
                "mode": "protein_id", "recommended_id": pid,
                "candidates": [{"id": pid, "name": pid, "input_mode": "protein_id"}],
            }),
            deepseek=DeepSeek(), families=SimpleNamespace(), family_evidence=SimpleNamespace(), evidence_queries=SimpleNamespace(),
            route_design_resolve=lambda *a, **k: {}, pathway_resolve=lambda *a, **k: {},
        )
        ctx = HarnessRunContext(
            ui_language="zh", conversation_context={}, user_text="这个酶具体是什么蛋白？",
            session_facts=store.snapshot("focus-priority"),
        )
        result = registry.execute("reuse_session_entity", {"entity_kind": "protein"}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["entity_id"], "A0A1W6QDI7")
        self.assertTrue(result.payload["focus"])
        self.assertFalse(result.payload["active"])

    def test_inspecting_new_focus_does_not_replace_confirmed_active_target(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("inspect-active", {
            "direction": "enzyme_to_reaction",
            "protein_resolution": {"mode": "protein_id", "recommended_id": "P00338", "candidates": [{"id": "P00338", "name": "old active", "input_mode": "protein_id"}]},
        })
        store.confirm_protein("inspect-active", protein_id="P00338")
        store.remember_resolution("inspect-active", {
            "direction": "conversation",
            "operation": "inspect_entity",
            "protein_resolution": {"mode": "protein_id", "recommended_id": "A0A1W6QDI7", "candidates": [{"id": "A0A1W6QDI7", "name": "new focus", "input_mode": "protein_id"}]},
            "immediate_result": {"answer_mode": "entity_list", "entity_kind": "protein", "entities": [{"id": "A0A1W6QDI7", "name": "new focus"}]},
        })
        snap = store.model_snapshot("inspect-active")["session_entities"]
        active = [row["id"] for row in snap["active"] if row["kind"] == "protein"]
        focus = [row["id"] for row in snap["history"] if row["kind"] == "protein" and row.get("focus")]
        self.assertEqual(active, ["P00338"])
        self.assertEqual(focus, ["A0A1W6QDI7"])

    def test_execution_context_comes_from_successful_server_execution(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("execution", {
            "direction": "enzyme_to_reaction",
            "protein_resolution": {"mode": "protein_id", "recommended_id": "P00338", "candidates": [{"id": "P00338", "name": "LDHA", "input_mode": "protein_id"}]},
        })
        store.confirm_protein("execution", protein_id="P00338")
        store.remember_execution_result("execution", {
            "discovery_filter": {"policy": "exclude_recorded_associations", "result_mode": "novel_association_discovery"},
            "ranking": {"route_id": "e2r-test-route"},
        }, direction="enzyme_to_reaction")
        context = store.execution_context("execution", ui_language="zh")
        self.assertEqual(context["previous_direction"], "enzyme_to_reaction")
        self.assertEqual(context["previous_result_mode"], "novel_association_discovery")
        self.assertEqual(context["previous_association_policy"], "exclude_known")
        self.assertEqual(context["previous_route_id"], "e2r-test-route")
        self.assertEqual(context["previous_target"], "P00338")
        self.assertEqual(context["ui_language"], "zh")

    def test_confirmed_protein_becomes_active_even_if_another_target_was_resolved_first(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("confirmed", {
            "direction": "enzyme_to_reaction",
            "protein_resolution": {"mode": "protein_id", "recommended_id": "P00338", "candidates": [{"id": "P00338", "name": "first", "input_mode": "protein_id"}]},
        })
        store.confirm_protein("confirmed", protein_id="A0A1W6QDI7")
        active = store.model_snapshot("confirmed")["session_entities"]["active"]
        self.assertEqual([row["id"] for row in active if row["kind"] == "protein"], ["A0A1W6QDI7"])

    def test_compare_verified_reactions_uses_structured_inspection(self) -> None:
        evidence = SimpleNamespace(
            reaction_metadata=lambda rid: {"equation": f"eq {rid}", "reaction_smiles": f"smiles-{rid}"},
            protein_metadata=lambda _pid: {},
            is_candidate_protein=lambda _pid: True,
        )
        agent_resolution = SimpleNamespace(evidence=evidence, catalog=SimpleNamespace(protein_by_id={}), proteins=SimpleNamespace(exact_or_search=lambda *_a, **_k: []))
        registry = self._registry(agent_resolution=agent_resolution)
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        for i, rid in enumerate(["RHEA:11111", "RHEA:22222"], 1):
            ctx.reaction_refs[f"reaction_{i}"] = {"mode": "session_verified_rhea", "recommended_id": rid, "interpreted_reaction": rid, "candidates": []}
        result = registry.execute("compare_entities", {"entity_refs": ["reaction_1", "reaction_2"]}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.terminal)
        self.assertTrue(result.payload["workflow_incomplete"])
        self.assertEqual(result.payload["required_next_action"], "synthesize")
        immediate = ctx.terminal_resolution["immediate_result"]
        self.assertEqual(immediate["answer_mode"], "entity_comparison")
        self.assertEqual([row["id"] for row in immediate["entities"]], ["RHEA:11111", "RHEA:22222"])
        self.assertEqual(immediate["comparison_rows"][0]["key"], "equation")

    def test_compare_entities_rejects_two_refs_to_same_underlying_entity(self) -> None:
        evidence = SimpleNamespace(
            reaction_metadata=lambda rid: {"equation": f"eq {rid}", "reaction_smiles": f"smiles-{rid}"},
            protein_metadata=lambda _pid: {}, is_candidate_protein=lambda _pid: True,
        )
        agent_resolution = SimpleNamespace(evidence=evidence, catalog=SimpleNamespace(protein_by_id={}), proteins=SimpleNamespace(exact_or_search=lambda *_a, **_k: []))
        registry = self._registry(agent_resolution=agent_resolution)
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        for ref in ("reaction_1", "reaction_2"):
            ctx.reaction_refs[ref] = {"mode": "session_verified_rhea", "recommended_id": "RHEA:11111", "interpreted_reaction": "RHEA:11111", "candidates": []}
        result = registry.execute("compare_entities", {"entity_refs": ["reaction_1", "reaction_2"], "comparison_goal": "compare"}, ctx)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_code, "comparison_duplicate_entities")
        self.assertEqual(result.payload["resolved_ids"], ["RHEA:11111", "RHEA:11111"])

    def test_reuse_session_entity_can_isolate_two_same_kind_reference_spans(self) -> None:
        class DeepSeek:
            @staticmethod
            def select_session_entity_reference(**kwargs):
                text = kwargs["user_text"]
                records = kwargs["records"]
                if text == "这篇文献":
                    return {"reference_mode": "focus", "selected_key": "", "reason": "anaphora"}
                target = next(row for row in records if row["id"] == "MED:222")
                return {"reference_mode": "specific", "selected_key": f"literature:{target['id']}", "reason": "explicit ID"}
            @staticmethod
            def provenance():
                return {"provider": "fake", "model": "fake"}
        registry = ScientificToolRegistry(
            agent_resolution=SimpleNamespace(), deepseek=DeepSeek(), families=SimpleNamespace(), family_evidence=SimpleNamespace(), evidence_queries=SimpleNamespace(),
            route_design_resolve=lambda *a, **k: {}, pathway_resolve=lambda *a, **k: {},
        )
        rows = [
            {"kind": "literature", "id": "MED:111", "label": "Paper one", "role": "resolved_target", "focus": True, "payload": {"id": "111", "source": "MED", "title": "Paper one"}},
            {"kind": "literature", "id": "MED:222", "label": "Paper two", "role": "related_evidence", "focus": False, "payload": {"id": "222", "source": "MED", "title": "Paper two"}},
        ]
        ctx = HarnessRunContext(
            ui_language="zh", conversation_context={},
            user_text="比较这篇文献与 MED:222 的结论",
            session_facts={"session_entities": {"all": rows}},
        )
        first = registry.execute("reuse_session_entity", {"entity_kind": "literature", "reference_text": "这篇文献"}, ctx)
        second = registry.execute("reuse_session_entity", {"entity_kind": "literature", "reference_text": "MED:222", "requested_identity": "MED:222"}, ctx)
        self.assertEqual(first.payload["entity_id"], "MED:111")
        self.assertEqual(second.payload["entity_id"], "MED:222")
        self.assertNotEqual(first.payload["literature_ref"], second.payload["literature_ref"])

    def test_resolve_literature_returns_verified_refs_for_direct_pmids(self) -> None:
        research = SimpleNamespace(resolve_literature=lambda text, limit=6: [{
            "id": "12345", "pmid": "12345", "source": "MED", "title": "Verified paper",
            "authors": "A", "journal": "J", "year": "2026", "url": "https://europepmc.org/article/MED/12345",
        }] if "12345" in text else [])
        registry = self._registry(research_service=research)
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        result = registry.execute("resolve_literature", {"text": "PMID:12345"}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.terminal)
        ref = result.payload["literature_refs"][0]["ref"]
        self.assertEqual(ctx.literature_refs[ref]["pmid"], "12345")

    def test_compare_entities_rejects_mixed_kinds(self) -> None:
        evidence = SimpleNamespace(reaction_metadata=lambda _rid: {}, protein_metadata=lambda _pid: {}, is_candidate_protein=lambda _pid: True)
        agent_resolution = SimpleNamespace(evidence=evidence, catalog=SimpleNamespace(protein_by_id={}), proteins=SimpleNamespace(exact_or_search=lambda *_a, **_k: []))
        registry = self._registry(agent_resolution=agent_resolution, compound_resolve=lambda *_a, **_k: [])
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        ctx.reaction_refs["reaction_1"] = {"mode": "session_verified_rhea", "recommended_id": "RHEA:11111", "interpreted_reaction": "RHEA:11111", "candidates": []}
        ctx.compound_refs["compound_1"] = {"chebi_id": "CHEBI:1", "name": "compound", "smiles": "C"}
        result = registry.execute("compare_entities", {"entity_refs": ["reaction_1", "compound_1"]}, ctx)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_code, "comparison_kind_mismatch")

    def test_inspect_specific_protein_enriches_missing_detail_once(self) -> None:
        calls: list[str] = []
        candidate = SimpleNamespace(name="Remote protein name", organism="Example species", accession="A0A000", source="uniprot")
        proteins = SimpleNamespace(detail_for=lambda accession: (calls.append(accession) or candidate))
        evidence = SimpleNamespace(protein_metadata=lambda _pid: {}, is_candidate_protein=lambda _pid: True)
        agent_resolution = SimpleNamespace(evidence=evidence, catalog=SimpleNamespace(protein_by_id={}), proteins=proteins)
        registry = self._registry(agent_resolution=agent_resolution)
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        ctx.protein_refs["protein_scope_1"] = {"kind": "specific_protein", "resolution": {"recommended_id": "A0A000", "candidates": []}}
        result = registry.execute("inspect_entity", {"protein_scope_ref": "protein_scope_1"}, ctx)
        self.assertEqual(result.status, "ok")
        entity = ctx.terminal_resolution["immediate_result"]["entities"][0]
        self.assertEqual(entity["name"], "Remote protein name")
        self.assertEqual(entity["subtitle"], "Example species")
        self.assertEqual(calls, ["A0A000"])

    def test_inspect_specific_protein_includes_substantive_uniprot_annotations(self) -> None:
        proteins = SimpleNamespace(detail_for=lambda _accession: None)
        evidence = SimpleNamespace(
            protein_metadata=lambda _pid: {"canonical_accession": "P00338"},
            is_candidate_protein=lambda _pid: True,
        )
        agent_resolution = SimpleNamespace(
            evidence=evidence,
            catalog=SimpleNamespace(protein_by_id={}),
            proteins=proteins,
        )
        research = SimpleNamespace(protein_detail=lambda _accession: {
            "record": {"accession": "P00338", "name": "LDHA", "organism": "Homo sapiens", "genes": ["LDHA"]},
            "facts": [{"label": "Annotation score", "value": 5}],
            "catalytic_activities": [{"reaction": "lactate + NAD+ = pyruvate + NADH", "rhea_ids": ["RHEA:23444"]}],
            "cofactors": ["NAD(+)"],
            "annotations": {"FUNCTION": ["Catalyzes lactate/pyruvate interconversion."], "ACTIVITY REGULATION": ["Regulated activity."]},
            "cross_references": {"PDB": ["1I10"]},
        })
        registry = self._registry(agent_resolution=agent_resolution, research_service=research)
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        ctx.protein_refs["protein_scope_1"] = {
            "kind": "specific_protein",
            "resolution": {
                "recommended_id": "P00338",
                "candidates": [{"id": "P00338", "accession": "P00338", "name": "LDHA", "organism": "Homo sapiens"}],
            },
        }
        result = registry.execute("inspect_entity", {"protein_scope_ref": "protein_scope_1"}, ctx)
        self.assertEqual(result.status, "ok")
        entity = ctx.terminal_resolution["immediate_result"]["entities"][0]
        self.assertEqual(entity["content_basis"], "uniprot_record")
        self.assertEqual(entity["gene_names"], ["LDHA"])
        self.assertIn("Catalyzes lactate", entity["function_annotation"])
        self.assertEqual(entity["catalytic_activities"][0]["rhea_ids"], ["RHEA:23444"])
        self.assertEqual(entity["cofactors"], ["NAD(+)"])
        self.assertEqual(entity["cross_references"]["PDB"], ["1I10"])



class ResearchWorkspaceToolTests(unittest.TestCase):
    @staticmethod
    def registry() -> ScientificToolRegistry:
        class Research:
            @staticmethod
            def protein_workspace(accession: str, **_kwargs: Any) -> dict[str, Any]:
                return {
                    "answer_mode": "research_workspace", "workspace_kind": "protein",
                    "title": "Research", "entity": {"id": accession},
                    "source_panels": [{"status": "ok"}],
                    "model_lens": {"status": "ok", "frontier": [{"candidate_id": "RHEA:1"}], "recorded_recovery": {"eligible_recorded": 2, "recovered": 1}},
                    "known_associations": {"count": 2, "items": []},
                }

            @staticmethod
            def reaction_workspace(reaction_id: str, **_kwargs: Any) -> dict[str, Any]:
                return {
                    "answer_mode": "research_workspace", "workspace_kind": "reaction",
                    "title": "Research", "entity": {"id": reaction_id},
                    "source_panels": [{"status": "ok"}, {"status": "ok"}],
                    "model_lens": {"status": "ok", "frontier": [], "recorded_recovery": {"eligible_recorded": 0, "recovered": 0}},
                    "known_associations": {"count": 0, "items": []},
                }

        return ScientificToolRegistry(
            agent_resolution=SimpleNamespace(), deepseek=SimpleNamespace(), families=SimpleNamespace(),
            family_evidence=SimpleNamespace(), evidence_queries=SimpleNamespace(),
            route_design_resolve=lambda *a, **k: {}, pathway_resolve=lambda *a, **k: {},
            research_service=Research(),
        )

    def test_specific_protein_builds_composable_integrated_workspace(self) -> None:
        registry = self.registry()
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        ctx.protein_refs["protein_scope_1"] = {
            "kind": "specific_protein",
            "resolution": {"mode": "protein_id", "recommended_id": "P00338", "candidates": [{"id": "P00338"}]},
        }
        result = registry.execute("research_workspace", {"protein_scope_ref": "protein_scope_1"}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.terminal)
        self.assertEqual(result.payload["model_frontier_count"], 1)
        self.assertEqual(ctx.terminal_resolution["immediate_result"]["answer_mode"], "research_workspace")

    def test_family_scope_does_not_fake_concrete_research_workspace(self) -> None:
        registry = self.registry()
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        ctx.protein_refs["protein_scope_1"] = {"kind": "family", "family_id": "PF00001"}
        result = registry.execute("research_workspace", {"protein_scope_ref": "protein_scope_1"}, ctx)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_code, "research_workspace_requires_specific_protein")

    def test_verified_reaction_builds_composable_integrated_workspace(self) -> None:
        registry = self.registry()
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        ctx.reaction_refs["reaction_1"] = {"recommended_id": "RHEA:12345", "candidates": []}
        result = registry.execute("research_workspace", {"reaction_ref": "reaction_1"}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.terminal)
        self.assertEqual(result.payload["source_count"], 2)


class ScientificToolCatalogTests(unittest.TestCase):
    def test_catalog_exposes_pydantic_input_schema(self) -> None:
        catalog = {item["name"]: item for item in ScientificToolRegistry.catalog()}
        self.assertEqual(set(catalog), {
            "reuse_session_entity", "resolve_reaction", "resolve_protein_scope", "lookup_relations",
            "list_scope_members", "resolve_compound", "resolve_literature", "inspect_entity",
            "compare_entities", "research_workspace", "broaden_scope", "candidate_search",
            "route_design", "pathway_compatibility",
        })
        relation_schema = catalog["lookup_relations"]["input_schema"]
        self.assertEqual(set(relation_schema["properties"]), {"reaction_ref", "protein_scope_ref"})
        self.assertNotIn("research_context", relation_schema["properties"])
        schema = catalog["resolve_protein_scope"]["input_schema"]
        self.assertIn("text", schema["properties"])
        self.assertIn("scope_hint", schema["properties"])
        self.assertIn("text", schema["required"])
        self.assertIn("family_or_class", str(schema["properties"]["scope_hint"]))
        self.assertIn("lookup_relations", catalog)
        self.assertIn("list_scope_members", catalog)
        self.assertIn("resolve_compound", catalog)
        self.assertIn("resolve_literature", catalog)
        self.assertIn("inspect_entity", catalog)
        self.assertIn("research_workspace", catalog)


class AgentSessionStoreTests(unittest.TestCase):
    def test_visible_page_context_uses_page_local_indices_without_creating_entities(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        items = [{"id": str(100 + i), "pmid": str(100 + i), "source": "MED", "title": f"Paper {i}"} for i in range(12)]
        store.remember_resolution("visible-page", {
            "direction": "enzyme_to_reaction", "operation": "research_workspace",
            "protein_resolution": {"mode": "protein_id", "recommended_id": "P1", "candidates": [{"id": "P1"}]},
            "immediate_result": {"answer_mode": "research_workspace", "source_panels": [{"id": "literature", "items": items}]},
        })
        marked = store.mark_visible_entities(
            "visible-page", entity_kind="literature",
            entity_ids=["MED:110", "MED:111", "MED:999999"], page_index=1,
        )
        self.assertEqual(marked["visible_ids"], ["MED:110", "MED:111"])
        snap = store.model_snapshot("visible-page")
        visible = snap["session_entities"]["visible"]
        self.assertEqual([row["id"] for row in visible], ["MED:110", "MED:111"])
        self.assertEqual([row["visible_index"] for row in visible], [1, 2])
        self.assertTrue(all(row["visible_page_index"] == 1 for row in visible))
        self.assertNotIn("MED:999999", [row["id"] for row in snap["session_entities"]["all"]])
        self.assertFalse(any(row.get("focus") for row in visible))

    def test_reuse_can_select_second_item_on_current_visible_page(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        items = [{"id": str(200 + i), "pmid": str(200 + i), "source": "MED", "title": f"Paper {i}"} for i in range(12)]
        store.remember_resolution("visible-reuse", {
            "direction": "enzyme_to_reaction", "operation": "research_workspace",
            "protein_resolution": {"mode": "protein_id", "recommended_id": "P1", "candidates": [{"id": "P1"}]},
            "immediate_result": {"answer_mode": "research_workspace", "source_panels": [{"id": "literature", "items": items}]},
        })
        store.mark_visible_entities("visible-reuse", entity_kind="literature", entity_ids=["MED:210", "MED:211"], page_index=1)

        class DeepSeek:
            @staticmethod
            def select_session_entity_reference(**kwargs):
                row = next(row for row in kwargs["records"] if row.get("visible_index") == 2)
                return {"reference_mode": "specific", "selected_key": f"{row['kind']}:{row['id']}", "reason": "second item on visible page"}
            @staticmethod
            def provenance():
                return {"provider": "fake", "model": "fake"}

        registry = ScientificToolRegistry(
            agent_resolution=SimpleNamespace(), deepseek=DeepSeek(), families=SimpleNamespace(), family_evidence=SimpleNamespace(), evidence_queries=SimpleNamespace(),
            route_design_resolve=lambda *a, **k: {}, pathway_resolve=lambda *a, **k: {},
        )
        ctx = HarnessRunContext(
            ui_language="zh", conversation_context={}, user_text="这页第二篇讲了什么？",
            session_facts=store.snapshot("visible-reuse"),
        )
        result = registry.execute("reuse_session_entity", {"entity_kind": "literature", "reference_text": "这页第二篇"}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["entity_id"], "MED:211")

    def test_explicit_literature_inspection_becomes_conversational_focus(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("lit-focus", {
            "direction": "enzyme_to_reaction", "operation": "research_workspace",
            "protein_resolution": {"mode": "protein_id", "recommended_id": "P1", "candidates": [{"id": "P1"}]},
            "immediate_result": {"answer_mode": "research_workspace", "source_panels": [{"id": "literature", "items": [
                {"id": "111", "source": "MED", "title": "Paper one"},
                {"id": "222", "source": "MED", "title": "Paper two"},
            ]}]},
        })
        store.remember_resolution("lit-focus", {
            "direction": "conversation", "operation": "inspect_entity",
            "immediate_result": {"answer_mode": "entity_list", "entity_kind": "literature", "entities": [
                {"id": "MED:111", "source": "MED", "pmid": "111", "name": "Paper one", "title": "Paper one"}
            ]},
        })
        focused = [row for row in store.model_snapshot("lit-focus")["session_entities"]["focus"] if row["kind"] == "literature"]
        self.assertEqual([row["id"] for row in focused], ["MED:111"])

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

    def test_pending_confirmation_is_bound_to_current_card_target_and_verified_reactions(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("confirm-e2r", {
            "direction": "enzyme_to_reaction",
            "protein_resolution": {
                "mode": "protein_id", "recommended_id": "P00338",
                "candidates": [
                    {"id": "P00338", "name": "LDHA", "input_mode": "protein_id"},
                    {"id": "P07195", "name": "LDHB", "input_mode": "protein_id"},
                ],
            },
            "positive_enzyme_resolutions": [],
            "positive_reaction_resolutions": [{
                "mention": "known reaction", "recommended_id": "RHEA:23444",
                "candidates": [
                    {"rhea_id": "RHEA:23444", "equation": "A = B"},
                    {"rhea_id": "RHEA:23445", "equation": "B = A"},
                ],
            }],
        })
        ok = store.validate_pending_confirmation(
            "confirm-e2r", direction="enzyme_to_reaction", target_id="P07195",
            positive_ids=["RHEA:23445"],
        )
        self.assertTrue(ok["valid"])
        forged = store.validate_pending_confirmation(
            "confirm-e2r", direction="enzyme_to_reaction", target_id="P00338",
            positive_ids=["RHEA:99999"],
        )
        self.assertFalse(forged["valid"])
        self.assertEqual(forged["error_code"], "confirmation_positive_not_verified")
        wrong_target = store.validate_pending_confirmation(
            "confirm-e2r", direction="enzyme_to_reaction", target_id="P00000",
            positive_ids=["RHEA:23444"],
        )
        self.assertEqual(wrong_target["error_code"], "confirmation_target_mismatch")
        other_session = store.validate_pending_confirmation(
            "other", direction="enzyme_to_reaction", target_id="P00338",
            positive_ids=["RHEA:23444"],
        )
        self.assertEqual(other_session["error_code"], "confirmation_context_missing")
        store.consume_pending_confirmation("confirm-e2r", direction="enzyme_to_reaction", target_id="P00338")
        replay = store.validate_pending_confirmation(
            "confirm-e2r", direction="enzyme_to_reaction", target_id="P00338",
            positive_ids=["RHEA:23444"],
        )
        self.assertEqual(replay["error_code"], "confirmation_context_missing")

    def test_pending_sequence_positive_is_bound_to_verified_sequence_digest(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        store.remember_resolution("confirm-r2e", {
            "direction": "reaction_to_enzyme",
            "reaction_resolution": {
                "mode": "rhea_id", "recommended_id": "RHEA:12345",
                "candidates": [{"rhea_id": "RHEA:12345", "equation": "A = B"}],
            },
            "positive_enzyme_resolutions": [{
                "mention": "provided active enzyme", "recommended_id": "EXT-PROT-1",
                "candidates": [
                    {"id": "EXT-PROT-1", "sequence": sequence, "input_mode": "raw_protein_sequence"},
                    {"id": "P12345", "name": "verified protein", "input_mode": "protein_id"},
                ],
            }],
            "positive_reaction_resolutions": [],
        })
        ok = store.validate_pending_confirmation(
            "confirm-r2e", direction="reaction_to_enzyme", target_id="RHEA:12345",
            positive_ids=["P12345"],
            positive_sequence_inputs=[{"id": "EXT-PROT-1", "sequence": "ACD EFGHIKLMNPQRSTVWY"}],
        )
        self.assertTrue(ok["valid"])
        changed = store.validate_pending_confirmation(
            "confirm-r2e", direction="reaction_to_enzyme", target_id="RHEA:12345",
            positive_sequence_inputs=[{"id": "EXT-PROT-1", "sequence": sequence + "A"}],
        )
        self.assertFalse(changed["valid"])
        self.assertEqual(changed["error_code"], "confirmation_sequence_mismatch")

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
