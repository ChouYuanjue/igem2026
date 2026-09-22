from __future__ import annotations

import json
import time
import unittest
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from scripts.starase_navigator.agent_harness.capabilities import controller_self_summary
from scripts.starase_navigator.agent_harness.contracts import HarnessAction, ToolResult
from scripts.starase_navigator.agent_harness.harness import ScientificAgentHarness
from scripts.starase_navigator.agent_harness.session_store import AgentSessionStore
from scripts.starase_navigator.agent_harness.tool_registry import HarnessRunContext, ScientificToolRegistry
from scripts.starase_navigator.errors import AppError


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


    def test_single_agent_action_envelope_is_normalized_but_arbitrary_wrappers_are_not(self) -> None:
        action = HarnessAction.model_validate({
            "agent_action": {
                "kind": "respond",
                "tool": None,
                "args": {},
                "reason": "",
                "question": "",
                "message": "ok",
            }
        })
        self.assertEqual(action.kind, "respond")
        self.assertEqual(action.message, "ok")
        with self.assertRaises(ValueError):
            HarnessAction.model_validate({
                "meta": "unexpected",
                "agent_action": {
                    "kind": "respond",
                    "message": "must not be silently unwrapped",
                },
            })

    def test_action_contract_has_one_primary_response_path(self) -> None:
        for kind in ("respond", "ask_user", "return_result"):
            payload = {"kind": kind}
            if kind == "respond":
                payload["message"] = "done"
            if kind == "ask_user":
                payload["question"] = "which target?"
            action = HarnessAction.model_validate(payload)
            self.assertEqual(action.kind, kind)
        with self.assertRaises(ValueError):
            HarnessAction.model_validate({"kind": "synthesize"})


class ScientificHarnessLoopTests(unittest.TestCase):
    def build(
        self,
        actions: list[HarnessAction],
        results: list[ToolResult],
        *,
        terminal_payload: dict[str, Any] | None = None,
        max_turns: int = 6,
        sessions: AgentSessionStore | None = None,
    ) -> tuple[ScientificAgentHarness, FakeDeepSeek, FakeTools]:
        deepseek = FakeDeepSeek(actions)
        tools = FakeTools(results, terminal_payload=terminal_payload)
        harness = ScientificAgentHarness(
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

    def test_fresh_session_does_not_claim_hidden_context_reuse(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        harness, _deepseek, _tools = self.build(
            [HarnessAction(kind="respond", message="Fresh answer.")],
            [],
            sessions=store,
        )
        result = harness.run("Explain the model.", session_id="fresh-visible-chat")
        self.assertFalse(result["agent_execution"]["session_facts_used"])

        store.remember_dialogue_turn(
            "continued-visible-chat",
            user_text="first",
            assistant_text="first answer",
            response_type="message",
        )
        harness2, _deepseek2, _tools2 = self.build(
            [HarnessAction(kind="respond", message="Follow-up answer.")],
            [],
            sessions=store,
        )
        result2 = harness2.run("continue", session_id="continued-visible-chat")
        self.assertTrue(result2["agent_execution"]["session_facts_used"])

    def test_related_session_evidence_stays_in_workspace_but_not_primary_current_refs(self) -> None:
        class LayeredTools(FakeTools):
            def seed_current_input_handles(self, ctx):
                return []

            def seed_session_handles(self, ctx):
                ctx.protein_refs["session_protein_focus"] = {
                    "kind": "specific_protein",
                    "label": "focused protein",
                    "resolution": {
                        "mode": "protein_id",
                        "recommended_id": "P00338",
                        "candidates": [{"id": "P00338", "input_mode": "protein_id"}],
                    },
                }
                ctx.reaction_refs["session_reaction_related"] = {
                    "mode": "session_verified_rhea",
                    "recommended_id": "RHEA:23444",
                    "candidates": [{"rhea_id": "RHEA:23444"}],
                }
                return [
                    {
                        "ref": "session_protein_focus",
                        "kind": "protein",
                        "id": "P00338",
                        "role": "resolved_target",
                        "focus": True,
                        "active": False,
                        "source": "verified_session_workspace",
                    },
                    {
                        "ref": "session_reaction_related",
                        "kind": "reaction",
                        "id": "RHEA:23444",
                        "role": "related_evidence",
                        "focus": False,
                        "active": False,
                        "source": "verified_session_workspace",
                    },
                ]

        deepseek = FakeDeepSeek([HarnessAction(kind="respond", message="ok")])
        tools = LayeredTools([])
        harness = ScientificAgentHarness(
            deepseek=deepseek,
            tools=tools,  # type: ignore[arg-type]
            sessions=AgentSessionStore(ttl_seconds=3600),
            max_turns=3,
        )
        harness.run("continue with this enzyme")
        call = deepseek.calls[0]
        self.assertIn(
            "session_protein_focus",
            call["current_run_refs"]["protein_scope_ref"],
        )
        self.assertNotIn(
            "session_reaction_related",
            call["current_run_refs"]["reaction_ref"],
        )
        by_ref = {
            row["ref"]: row
            for row in call["workspace_handles"]
            if isinstance(row, dict) and row.get("ref")
        }
        self.assertEqual(
            by_ref["session_reaction_related"]["role"],
            "related_evidence",
        )

    def test_controller_can_answer_product_question_without_tools(self) -> None:
        harness, deepseek, tools = self.build(
            [HarnessAction(kind="respond", message="I can query evidence, rank candidates, design routes, and evaluate pathways.")],
            [],
        )
        result = harness.run("What can you do?")
        self.assertEqual(result["response_type"], "message")
        self.assertIn("rank candidates", result["assistant_response"])
        self.assertEqual(result["agent_execution"]["steps"][0]["action_kind"], "respond")
        self.assertEqual(deepseek.calls[0]["capability_manifest"]["name"], "Starase Navigator")
        self.assertIn("self_inspection", deepseek.calls[0]["capability_manifest"])
        self.assertNotIn("groups", deepseek.calls[0]["capability_manifest"])
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
            [HarnessAction(kind="tool", tool="candidate_search", args={"direction": "reaction_to_enzyme", "full_text": "find candidates", "reaction_text": "reaction X", "known_association_policy": "separate_known"})],
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
                HarnessAction(kind="tool", tool="candidate_search", args={"direction": "reaction_to_enzyme", "full_text": "ambiguous reaction", "reaction_text": "ambiguous reaction", "known_association_policy": "separate_known"}),
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

    def test_failed_tool_observation_remains_in_primary_trace_for_explanation(self) -> None:
        harness, deepseek, tools = self.build(
            [
                HarnessAction(kind="tool", tool="resolve_reaction", args={"text": "missing reaction"}),
                HarnessAction(kind="respond", message="The verified lookup did not find a matching reaction, so I cannot assert a database record from that lookup."),
            ],
            [ToolResult(
                tool="resolve_reaction", status="error", summary="no verified match",
                recoverable=True, error_code="no_match",
            )],
        )
        result = harness.run("What does the database record for this missing reaction?")
        self.assertEqual(result["response_type"], "message")
        self.assertIn("did not find", result["assistant_response"])
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(deepseek.calls[1]["history"][-1]["result"]["error_code"], "no_match")

    def test_identical_tool_call_reuses_observation_without_second_execution(self) -> None:
        same = HarnessAction(kind="tool", tool="resolve_reaction", args={"text": "reaction X"})
        harness, deepseek, tools = self.build(
            [
                same,
                same.model_copy(deep=True),
                HarnessAction(kind="respond", message="The same lookup was already attempted; I will use that observation."),
            ],
            [ToolResult(tool="resolve_reaction", status="error", summary="try another way", recoverable=True)],
            max_turns=5,
        )
        result = harness.run("reaction X")
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(result["agent_execution"]["steps"][1]["status"], "cached")
        self.assertEqual(
            deepseek.calls[2]["history"][-1]["result"]["error_code"],
            "duplicate_tool_call_reused",
        )

    def test_return_result_preserves_controller_summary_as_assistant_response(self) -> None:
        payload = {
            "direction": "route_design",
            "summary": "structured route result",
            "immediate_result": {
                "direction": "route_design",
                "routes": [{"route_id": "RP-test", "steps": []}],
            },
        }
        harness, _deepseek, _tools = self.build(
            [
                HarnessAction(kind="tool", tool="resolve_reaction", args={"text": "route context"}),
                HarnessAction(
                    kind="return_result",
                    message="I kept the unchanged route context and returned the verified structured result.",
                ),
            ],
            [ToolResult(tool="resolve_reaction", status="ok", summary="verified", terminal=False)],
            terminal_payload=payload,
        )
        result = harness.run("Return the verified result with a short explanation.")
        self.assertEqual(
            result["assistant_response"],
            "I kept the unchanged route context and returned the verified structured result.",
        )
        self.assertEqual(result["immediate_result"]["routes"][0]["route_id"], "RP-test")

    def test_turn_limit_synthesizes_verified_trace_instead_of_returning_empty_result(self) -> None:
        payload = {
            "direction": "reaction_to_enzyme",
            "summary": "verified evidence",
            "immediate_result": {
                "known_associations": {"count": 1, "items": [{"candidate_id": "P1"}]},
            },
        }
        repeated = HarnessAction(kind="tool", tool="resolve_reaction", args={"text": "reaction X"})
        harness, deepseek, tools = self.build(
            [repeated, repeated.model_copy(deep=True)],
            [ToolResult(tool="resolve_reaction", status="ok", summary="verified", terminal=False)],
            terminal_payload=payload,
            max_turns=2,
        )
        result = harness.run("Find the verified evidence.")
        self.assertEqual(result["immediate_result"]["known_associations"]["count"], 1)
        self.assertEqual(result["assistant_response"], "Grounded comparison from verified evidence.")
        self.assertEqual(result["agent_execution"]["mode"], "model_led_scientific_harness_final_synthesis")
        self.assertEqual(result["agent_execution"]["steps"][-1]["status"], "synthesized")
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(len(deepseek.synthesis_calls), 1)
        self.assertEqual(len(deepseek.synthesis_calls[0]["verified_evidence"]), 1)
        self.assertEqual(
            [call["execution_budget"]["remaining_turns_after_this"] for call in deepseek.calls],
            [1, 0],
        )

    def test_turn_limit_keeps_structured_fail_soft_when_final_synthesis_fails(self) -> None:
        payload = {
            "direction": "reaction_to_enzyme",
            "summary": "verified evidence",
            "immediate_result": {
                "known_associations": {"count": 1, "items": [{"candidate_id": "P1"}]},
            },
        }
        repeated = HarnessAction(kind="tool", tool="resolve_reaction", args={"text": "reaction X"})
        harness, deepseek, tools = self.build(
            [repeated, repeated.model_copy(deep=True)],
            [ToolResult(tool="resolve_reaction", status="ok", summary="verified", terminal=False)],
            terminal_payload=payload,
            max_turns=2,
        )

        def fail_synthesis(**_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("synthetic synthesis outage")

        deepseek.synthesize_grounded_answer = fail_synthesis  # type: ignore[method-assign]
        result = harness.run("Find the verified evidence.")
        self.assertEqual(result["immediate_result"]["known_associations"]["count"], 1)
        self.assertEqual(result["agent_execution"]["mode"], "model_led_scientific_harness_fail_soft")
        self.assertEqual(result["agent_execution"]["steps"][-1]["status"], "fallback")
        self.assertEqual(len(tools.calls), 1)

    def test_primary_agent_can_answer_after_verified_tool_observation(self) -> None:
        payload = {
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
                HarnessAction(kind="respond", message="The verified observation contains one recorded association: PTEST1."),
            ],
            [ToolResult(tool="resolve_reaction", status="ok", summary="verified", terminal=False)],
            terminal_payload=payload,
        )
        result = harness.run("Which protein is recorded for reaction X?")
        self.assertEqual(result["response_type"], "message")
        self.assertEqual(result["immediate_result"]["known_associations"]["count"], 1)
        self.assertIn("PTEST1", result["assistant_response"])
        self.assertEqual(len(deepseek.calls), 2)
        self.assertEqual(len(tools.calls), 1)

    def test_multi_tool_response_preserves_compact_evidence_from_all_structured_observations(self) -> None:
        class ChangingTools:
            def __init__(self) -> None:
                self.calls: list[str] = []

            @staticmethod
            def catalog() -> list[dict[str, Any]]:
                return [{"name": "compare_entities"}, {"name": "resolve_literature"}]

            @staticmethod
            def seed_current_input_handles(ctx: Any) -> list[dict[str, Any]]:
                return []

            @staticmethod
            def seed_session_handles(ctx: Any) -> list[dict[str, Any]]:
                return []

            def execute(self, tool: str, args: dict[str, Any], ctx: Any) -> ToolResult:
                self.calls.append(tool)
                if tool == "compare_entities":
                    ctx.terminal_resolution = {
                        "direction": "conversation",
                        "operation": "compare_entities",
                        "summary": "comparison",
                        "immediate_result": {
                            "answer_mode": "entity_comparison",
                            "title": "Comparison",
                            "entity_kind": "protein",
                            "entities": [
                                {"id": "P-A", "name": "protein A", "source": "verified"},
                                {"id": "P-B", "name": "protein B", "source": "verified"},
                            ],
                        },
                    }
                    return ToolResult(tool="compare_entities", status="ok", summary="comparison ready", terminal=False)
                ctx.terminal_resolution = {
                    "direction": "conversation",
                    "operation": "resolve_literature",
                    "summary": "literature",
                    "immediate_result": {
                        "answer_mode": "entity_list",
                        "title": "Literature",
                        "entity_kind": "literature",
                        "entities": [
                            {"id": "MED:1", "name": "paper one", "source": "Europe PMC"},
                        ],
                    },
                }
                return ToolResult(tool="resolve_literature", status="ok", summary="literature ready", terminal=False)

        deepseek = FakeDeepSeek([
            HarnessAction(kind="tool", tool="compare_entities", args={"entity_refs": ["a", "b"], "comparison_goal": "compare"}),
            HarnessAction(kind="tool", tool="resolve_literature", args={"text": "supporting evidence", "limit": 4}),
            HarnessAction(kind="respond", message="Grounded synthesis."),
        ])
        tools = ChangingTools()
        harness = ScientificAgentHarness(
            deepseek=deepseek,
            tools=tools,  # type: ignore[arg-type]
            sessions=AgentSessionStore(ttl_seconds=3600),
            max_turns=5,
        )
        result = harness.run("Compare these candidates and check the literature.")
        self.assertEqual(result["assistant_response"], "Grounded synthesis.")
        self.assertEqual(result["immediate_result"]["answer_mode"], "entity_list")
        self.assertEqual(
            [row["answer_mode"] for row in result["agent_evidence"]],
            ["entity_comparison", "entity_list"],
        )
        self.assertEqual(result["agent_evidence"][0]["entities"][0]["id"], "P-A")
        self.assertEqual(result["agent_evidence"][1]["entities"][0]["id"], "MED:1")
        self.assertEqual(tools.calls, ["compare_entities", "resolve_literature"])

    def test_multi_tool_run_persists_intermediate_verified_focus_before_later_result(self) -> None:
        class FocusTools:
            @staticmethod
            def catalog() -> list[dict[str, Any]]:
                return [{"name": "inspect_entity"}, {"name": "lookup_relations"}]

            @staticmethod
            def seed_current_input_handles(ctx: Any) -> list[dict[str, Any]]:
                return []

            @staticmethod
            def seed_session_handles(ctx: Any) -> list[dict[str, Any]]:
                return []

            def execute(self, tool: str, args: dict[str, Any], ctx: Any) -> ToolResult:
                if tool == "inspect_entity":
                    ctx.terminal_resolution = {
                        "direction": "conversation",
                        "operation": "inspect_entity",
                        "summary": "paper inspected",
                        "immediate_result": {
                            "answer_mode": "entity_list",
                            "entity_kind": "literature",
                            "title": "paper detail",
                            "entities": [{
                                "id": "MED:777",
                                "pmid": "777",
                                "name": "A verified paper",
                                "title": "A verified paper",
                                "source": "MED",
                            }],
                        },
                    }
                    return ToolResult(
                        tool="inspect_entity",
                        status="ok",
                        summary="paper inspected",
                        terminal=False,
                    )
                ctx.terminal_resolution = {
                    "direction": "reaction_to_enzyme",
                    "operation": "lookup_relations",
                    "summary": "relation evidence",
                    "reaction_resolution": {
                        "mode": "rhea_id",
                        "recommended_id": "RHEA:12345",
                        "candidates": [{"rhea_id": "RHEA:12345"}],
                    },
                    "immediate_result": {
                        "known_associations": {"count": 1, "items": []},
                    },
                }
                return ToolResult(
                    tool="lookup_relations",
                    status="ok",
                    summary="relation evidence",
                    terminal=False,
                )

        sessions = AgentSessionStore(ttl_seconds=3600)
        harness = ScientificAgentHarness(
            deepseek=FakeDeepSeek([
                HarnessAction(kind="tool", tool="inspect_entity", args={"literature_ref": "literature_1"}),
                HarnessAction(kind="tool", tool="lookup_relations", args={"reaction_ref": "reaction_1"}),
                HarnessAction(kind="respond", message="Combined answer."),
            ]),
            tools=FocusTools(),  # type: ignore[arg-type]
            sessions=sessions,
            max_turns=5,
        )
        result = harness.run("Inspect the paper and then check the reaction evidence.", session_id="multi-focus")
        self.assertEqual(result["assistant_response"], "Combined answer.")
        snapshot = sessions.model_snapshot("multi-focus")
        literature_focus = [
            row for row in snapshot["session_entities"]["focus"]
            if row["kind"] == "literature"
        ]
        reaction_focus = [
            row for row in snapshot["session_entities"]["focus"]
            if row["kind"] == "reaction"
        ]
        self.assertEqual([row["id"] for row in literature_focus], ["MED:777"])
        self.assertEqual([row["id"] for row in reaction_focus], ["RHEA:12345"])

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
                HarnessAction(kind="tool", tool="candidate_search", args={"direction": "reaction_to_enzyme", "full_text": "show known evidence and candidates", "reaction_text": "reaction X", "known_association_policy": "separate_known"}),
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

    def test_unambiguous_candidate_target_executes_production_callback_and_returns_observation(self) -> None:
        class AgentResolution:
            @staticmethod
            def resolve_protein(text: str) -> dict[str, Any]:
                return {
                    "mode": "protein_id",
                    "interpreted_protein": text,
                    "assumptions": [],
                    "normalized": {},
                    "candidates": [{"id": "P-EXACT", "name": "exact protein"}],
                    "recommended_id": "P-EXACT",
                }

        class DeepSeek:
            @staticmethod
            def provenance() -> dict[str, Any]:
                return {"provider": "fake", "model": "fake"}

        calls: list[dict[str, Any]] = []
        def execute_candidate(**kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {
                "protein": {"id": "P-EXACT", "name": "exact protein"},
                "ranking": {"route_id": "test-route", "score_source": "test"},
                "known_associations": {"count": 0, "items": []},
                "candidates": [
                    {"rank": 1, "candidate_id": "RHEA:10001", "name": "candidate A", "correspondence_defect": 0.1},
                    {"rank": 2, "candidate_id": "RHEA:10002", "name": "candidate B", "correspondence_defect": 0.2},
                ],
            }

        registry = ScientificToolRegistry(
            agent_resolution=AgentResolution(), deepseek=DeepSeek(),
            families=object(), family_evidence=object(), evidence_queries=object(),
            route_design_resolve=lambda *a, **k: {}, pathway_resolve=lambda *a, **k: {},
            candidate_execute=execute_candidate,
        )
        ctx = HarnessRunContext(ui_language="en", conversation_context={}, session_id="session-auto")
        result = registry.execute("candidate_search", {
            "direction": "enzyme_to_reaction",
            "full_text": "Rank plausible reactions for P-EXACT.",
            "protein_text": "P-EXACT",
            "known_association_policy": "separate_known",
        }, ctx)
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.terminal)
        self.assertEqual(result.payload["execution"], "production_ranking")
        self.assertEqual(result.payload["candidate_count"], 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["session_id"], "session-auto")
        self.assertEqual(ctx.terminal_resolution["immediate_result"]["candidates"][1]["candidate_id"], "RHEA:10002")

    def test_ambiguous_target_or_declared_positive_seed_preserves_confirmation(self) -> None:
        class AgentResolution:
            @staticmethod
            def resolve_protein(text: str) -> dict[str, Any]:
                if text.strip() == "P-A":
                    return {
                        "mode": "protein_id",
                        "interpreted_protein": text,
                        "assumptions": [],
                        "normalized": {},
                        "candidates": [{"id": "P-A", "name": "candidate A"}],
                        "recommended_id": "P-A",
                    }
                return {
                    "mode": "natural_language",
                    "interpreted_protein": text,
                    "assumptions": [],
                    "normalized": {},
                    "candidates": [
                        {"id": "P-A", "name": "candidate A"},
                        {"id": "P-B", "name": "candidate B"},
                    ],
                    "recommended_id": "P-A",
                }

            @staticmethod
            def resolve(text: str) -> dict[str, Any]:
                return {
                    "mode": "rhea_id",
                    "interpreted_reaction": text,
                    "assumptions": [],
                    "normalized": {},
                    "candidates": [{"rhea_id": "RHEA:10001", "equation": "A = B"}],
                    "recommended_id": "RHEA:10001",
                }

        class DeepSeek:
            @staticmethod
            def provenance() -> dict[str, Any]:
                return {"provider": "fake", "model": "fake"}

        calls: list[dict[str, Any]] = []
        registry = ScientificToolRegistry(
            agent_resolution=AgentResolution(), deepseek=DeepSeek(),
            families=object(), family_evidence=object(), evidence_queries=object(),
            route_design_resolve=lambda *a, **k: {}, pathway_resolve=lambda *a, **k: {},
            candidate_execute=lambda **kwargs: calls.append(kwargs) or {},
        )
        ambiguous = registry.execute("candidate_search", {
            "direction": "enzyme_to_reaction",
            "full_text": "Find reactions for this enzyme description.",
            "protein_text": "this enzyme description",
            "known_association_policy": "separate_known",
        }, HarnessRunContext(ui_language="en", conversation_context={}))
        self.assertTrue(ambiguous.terminal)
        self.assertTrue(ambiguous.payload["requires_confirmation"])
        self.assertEqual(calls, [])

        declared_seed = registry.execute("candidate_search", {
            "direction": "enzyme_to_reaction",
            "full_text": "For P-A, treat RHEA:10001 as a known active reaction and rank alternatives.",
            "protein_text": "P-A",
            "positive_reaction_texts": ["RHEA:10001"],
            "known_association_policy": "separate_known",
        }, HarnessRunContext(ui_language="en", conversation_context={}))
        self.assertTrue(declared_seed.terminal)
        self.assertEqual(declared_seed.payload["positive_seed_count"], 1)
        self.assertEqual(calls, [])

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
            "known_association_policy": "separate_known",
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
            "known_association_policy": "separate_known",
        }, HarnessRunContext(ui_language="en", conversation_context={}))
        self.assertEqual(rejected.status, "error")
        self.assertEqual(rejected.error_code, "candidate_positive_reaction_not_in_user_text")

    def test_candidate_search_carries_only_explicit_assay_conditions(self) -> None:
        class AgentResolution:
            @staticmethod
            def resolve(text: str) -> dict[str, Any]:
                return {
                    "mode":"rhea_id",
                    "interpreted_reaction":text,
                    "candidates":[{"rhea_id":"RHEA:25290"}],
                    "recommended_id":"RHEA:25290",
                }

        class DeepSeek:
            @staticmethod
            def provenance() -> dict[str, Any]:
                return {"provider":"fake","model":"fake"}

        registry=ScientificToolRegistry(
            agent_resolution=AgentResolution(),deepseek=DeepSeek(),
            families=object(),family_evidence=object(),evidence_queries=object(),
            route_design_resolve=lambda *a,**k:{},
            pathway_resolve=lambda *a,**k:{},
        )
        ctx=HarnessRunContext(ui_language="en",conversation_context={})
        result=registry.execute("candidate_search",{
            "direction":"reaction_to_enzyme",
            "full_text":"For RHEA:25290, find candidates at pH 7.2 and 30 C with MgCl2.",
            "reaction_text":"RHEA:25290",
            "target_ph":7.2,
            "target_temperature_c":30.0,
            "target_cofactors":["MgCl2"],
            "known_association_policy":"separate_known",
        },ctx)
        self.assertEqual(result.status,"ok")
        self.assertEqual(ctx.terminal_resolution["target_conditions"],{
            "ph":7.2,"temperature_c":30.0,"cofactors":["MgCl2"],
        })

        rejected=registry.execute("candidate_search",{
            "direction":"reaction_to_enzyme",
            "full_text":"For RHEA:25290, find candidates.",
            "reaction_text":"RHEA:25290",
            "target_ph":7.2,
            "known_association_policy":"separate_known",
        },HarnessRunContext(ui_language="en",conversation_context={}))
        self.assertEqual(rejected.status,"error")
        self.assertEqual(rejected.error_code,"candidate_condition_not_in_user_text")

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
            "known_association_policy": "separate_known",
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
            {"direction": "reaction_to_enzyme", "full_text": "show candidates", "reaction_ref": "reaction_1", "known_association_policy": "separate_known"},
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
            {"direction": "enzyme_to_reaction", "full_text": "show possible reactions", "protein_scope_ref": "protein_scope_1", "known_association_policy": "separate_known"},
            ctx2,
        )
        self.assertEqual(e2r.status, "ok")
        self.assertEqual(ctx2.terminal_resolution["protein_resolution"]["recommended_id"], "P00338")

        ctx3 = HarnessRunContext(ui_language="en", conversation_context={})
        ctx3.protein_refs["protein_scope_1"] = {"kind": "family", "family_id": "PF01040", "label": "UbiA family"}
        family = registry.execute(
            "candidate_search",
            {"direction": "enzyme_to_reaction", "full_text": "predict family reactions", "protein_scope_ref": "protein_scope_1", "known_association_policy": "separate_known"},
            ctx3,
        )
        self.assertEqual(family.status, "error")
        self.assertEqual(family.error_code, "candidate_requires_specific_protein")


class RoutePathwayExecutionToolTests(unittest.TestCase):
    @staticmethod
    def _registry(*, route_resolution: dict[str, Any], pathway_resolution: dict[str, Any], route_execute: Any = None, route_patch_execute: Any = None, pathway_execute: Any = None) -> ScientificToolRegistry:
        class DeepSeek:
            @staticmethod
            def provenance() -> dict[str, Any]:
                return {"provider": "fake", "model": "fake"}
        return ScientificToolRegistry(
            agent_resolution=object(),
            deepseek=DeepSeek(),
            families=object(),
            family_evidence=object(),
            evidence_queries=object(),
            route_design_resolve=lambda *a, **k: deepcopy(route_resolution),
            pathway_resolve=lambda *a, **k: deepcopy(pathway_resolution),
            route_execute=route_execute,
            route_patch_execute=route_patch_execute,
            pathway_execute=pathway_execute,
        )

    def test_unambiguous_route_executes_production_callback_without_confirmation(self) -> None:
        resolution = {
            "direction": "route_design",
            "route_design_resolution": {
                "source_candidates": [{"chebi_id": "CHEBI:1", "name": "source"}],
                "target_candidates": [{"chebi_id": "CHEBI:2", "name": "target"}],
                "recommended_source_id": "CHEBI:1",
                "recommended_target_id": "CHEBI:2",
                "max_steps": 4,
                "route_count": 3,
                "priority": "short",
            },
        }
        calls: list[dict[str, Any]] = []
        registry = self._registry(
            route_resolution=resolution,
            pathway_resolution={},
            route_execute=lambda **kwargs: calls.append(kwargs) or {
                "direction": "route_design",
                "routes": [{"route_id": "route-1", "steps": []}],
                "route_count": 1,
            },
        )
        ctx = HarnessRunContext(ui_language="en", conversation_context={}, session_id="route-session")
        result = registry.execute("route_design", {"text": "source to target"}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.terminal)
        self.assertEqual(result.payload["execution"], "production_route_design")
        self.assertEqual(result.payload["route_count"], 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["session_id"], "route-session")
        self.assertEqual(ctx.terminal_resolution["immediate_result"]["routes"][0]["route_id"], "route-1")

    def test_ambiguous_route_preserves_confirmation(self) -> None:
        resolution = {
            "direction": "route_design",
            "route_design_resolution": {
                "source_candidates": [
                    {"chebi_id": "CHEBI:1"},
                    {"chebi_id": "CHEBI:3"},
                ],
                "target_candidates": [{"chebi_id": "CHEBI:2"}],
                "recommended_source_id": "CHEBI:1",
                "recommended_target_id": "CHEBI:2",
            },
        }
        calls: list[dict[str, Any]] = []
        registry = self._registry(
            route_resolution=resolution,
            pathway_resolution={},
            route_execute=lambda **kwargs: calls.append(kwargs) or {},
        )
        result = registry.execute("route_design", {"text": "ambiguous source to target"}, HarnessRunContext(ui_language="en", conversation_context={}))
        self.assertTrue(result.terminal)
        self.assertTrue(result.payload["requires_confirmation"])
        self.assertEqual(calls, [])

    def test_single_low_confidence_compound_candidate_does_not_auto_execute_route(self) -> None:
        resolution = {
            "direction": "route_design",
            "route_design_resolution": {
                "source_candidates": [{
                    "chebi_id": "CHEBI:1",
                    "identity_confident": True,
                }],
                "target_candidates": [{
                    "chebi_id": "CHEBI:2",
                    "identity_confident": False,
                    "match_type": "lexical_candidate",
                }],
                "recommended_source_id": "CHEBI:1",
                "recommended_target_id": "CHEBI:2",
            },
        }
        calls: list[dict[str, Any]] = []
        registry = self._registry(
            route_resolution=resolution,
            pathway_resolution={},
            route_execute=lambda **kwargs: calls.append(kwargs) or {},
        )
        result = registry.execute(
            "route_design",
            {"text": "source to target"},
            HarnessRunContext(ui_language="en", conversation_context={}),
        )
        self.assertTrue(result.terminal)
        self.assertTrue(result.payload["requires_confirmation"])
        self.assertEqual(calls, [])

    def test_route_design_binds_verified_compound_refs_and_uses_original_user_request(self) -> None:
        captured: list[dict[str, Any]] = []

        class DeepSeek:
            @staticmethod
            def provenance() -> dict[str, Any]:
                return {"provider": "fake", "model": "fake"}

        def resolve(text: str, **kwargs: Any) -> dict[str, Any]:
            captured.append({"text": text, **kwargs})
            source = dict(kwargs["source_candidate"])
            target = dict(kwargs["target_candidate"])
            return {
                "direction": "route_design",
                "route_design_resolution": {
                    "source_candidates": [source],
                    "target_candidates": [target],
                    "recommended_source_id": source["chebi_id"],
                    "recommended_target_id": target["chebi_id"],
                    "max_steps": 4,
                    "route_count": 2,
                    "priority": "short",
                },
            }

        executions: list[dict[str, Any]] = []
        registry = ScientificToolRegistry(
            agent_resolution=object(),
            deepseek=DeepSeek(),
            families=object(),
            family_evidence=object(),
            evidence_queries=object(),
            route_design_resolve=resolve,
            pathway_resolve=lambda *a, **k: {},
            route_execute=lambda **kwargs: executions.append(kwargs) or {
                "direction": "route_design",
                "routes": [{"route_id": "RR-bound", "steps": []}],
                "route_count": 1,
            },
        )
        ctx = HarnessRunContext(
            ui_language="en",
            conversation_context={},
            user_text="Design two short routes from the verified source to the verified target.",
            session_id="bound-route",
        )
        ctx.compound_refs["compound_source"] = {
            "chebi_id": "CHEBI:101",
            "name": "verified source",
            "identity_confident": True,
        }
        ctx.compound_refs["compound_target"] = {
            "chebi_id": "CHEBI:202",
            "name": "verified target",
            "identity_confident": True,
        }
        result = registry.execute(
            "route_design",
            {
                "text": "controller rewrite mentioning unrelated compounds",
                "source_compound_ref": "compound_source",
                "target_compound_ref": "compound_target",
            },
            ctx,
        )
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.terminal)
        self.assertEqual(captured[0]["text"], ctx.user_text)
        self.assertEqual(captured[0]["source_candidate"]["chebi_id"], "CHEBI:101")
        self.assertEqual(captured[0]["target_candidate"]["chebi_id"], "CHEBI:202")
        self.assertEqual(result.payload["source_identity_bound"], True)
        self.assertEqual(result.payload["target_identity_bound"], True)
        self.assertEqual(len(executions), 1)

    def test_route_patch_requires_one_contiguous_segment_and_delegates_workspace_payload(self) -> None:
        route = {
            "route_id": "RR-parent",
            "steps": [
                {"step_index": 1, "rhea_id": "RHEA:1", "source": "CHEBI:1", "target": "CHEBI:2"},
                {"step_index": 2, "rhea_id": "RHEA:2", "source": "CHEBI:2", "target": "CHEBI:3"},
                {"step_index": 3, "rhea_id": "RHEA:3", "source": "CHEBI:3", "target": "CHEBI:4"},
            ],
        }
        calls: list[dict[str, Any]] = []
        registry = self._registry(
            route_resolution={},
            pathway_resolution={},
            route_patch_execute=lambda **kwargs: calls.append(kwargs) or {
                "direction": "route_design",
                "answer_mode": "route_patch",
                "routes": [{"route_id": "RP-new", "steps": []}],
                "patch_context": {"excluded_rhea_ids": ["RHEA:2", "RHEA:3"]},
            },
        )
        ctx = HarnessRunContext(ui_language="en", conversation_context={}, session_id="patch-session")
        ctx.route_refs["route_1"] = route
        ctx.route_step_refs["step_1"] = {"route_id": "RR-parent", "step_index": 1, "step": route["steps"][0]}
        ctx.route_step_refs["step_2"] = {"route_id": "RR-parent", "step_index": 2, "step": route["steps"][1]}
        ctx.route_step_refs["step_3"] = {"route_id": "RR-parent", "step_index": 3, "step": route["steps"][2]}

        result = registry.execute(
            "patch_route_segment",
            {
                "route_ref": "route_1",
                "route_step_refs": ["step_2", "step_3"],
                "replacement_count": 2,
                "max_replacement_steps": 3,
            },
            ctx,
        )
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.terminal)
        self.assertEqual(result.payload["parent_route_id"], "RR-parent")
        self.assertEqual(result.payload["start_step_index"], 2)
        self.assertEqual(result.payload["end_step_index"], 3)
        self.assertEqual(len(calls), 1)
        self.assertEqual([row["step_index"] for row in calls[0]["segment_steps"]], [2, 3])
        self.assertEqual(calls[0]["replacement_count"], 2)
        self.assertEqual(calls[0]["max_replacement_steps"], 3)
        self.assertEqual(ctx.terminal_resolution["operation"], "patch_route_segment")

        noncontiguous = registry.execute(
            "patch_route_segment",
            {
                "route_ref": "route_1",
                "route_step_refs": ["step_1", "step_3"],
            },
            ctx,
        )
        self.assertEqual(noncontiguous.status, "error")
        self.assertEqual(noncontiguous.error_code, "route_patch_segment_not_contiguous")

        ctx.route_step_refs["foreign"] = {
            "route_id": "RR-other",
            "step_index": 2,
            "step": {"step_index": 2},
        }
        mixed = registry.execute(
            "patch_route_segment",
            {
                "route_ref": "route_1",
                "route_step_refs": ["foreign"],
            },
            ctx,
        )
        self.assertEqual(mixed.status, "error")
        self.assertEqual(mixed.error_code, "route_patch_mixed_routes")

    def test_unambiguous_pathway_executes_and_ambiguous_specified_enzyme_does_not(self) -> None:
        unique = {
            "direction": "pathway_compatibility",
            "pathway_resolution": {
                "execution_mode": "auto",
                "steps": [
                    {
                        "reaction_resolution": {
                            "recommended_id": "RHEA:10001",
                            "candidates": [{"rhea_id": "RHEA:10001", "orientation": "forward"}],
                        },
                        "enzyme_resolution": {"specified": False, "candidates": [], "recommended_id": None},
                    },
                    {
                        "reaction_resolution": {
                            "recommended_id": "RHEA:10002",
                            "candidates": [{"rhea_id": "RHEA:10002", "orientation": "forward"}],
                        },
                        "enzyme_resolution": {
                            "specified": True,
                            "recommended_id": "P-ONE",
                            "candidates": [{"id": "P-ONE"}],
                        },
                    },
                ],
            },
        }
        calls: list[dict[str, Any]] = []
        registry = self._registry(
            route_resolution={},
            pathway_resolution=unique,
            pathway_execute=lambda **kwargs: calls.append(kwargs) or {
                "direction": "pathway_compatibility",
                "steps": [{"step_index": 1}, {"step_index": 2}],
                "verdict": "compatible",
            },
        )
        ctx = HarnessRunContext(ui_language="en", conversation_context={}, session_id="path-session")
        result = registry.execute("pathway_compatibility", {"text": "analyze two steps"}, ctx)
        self.assertFalse(result.terminal)
        self.assertEqual(result.payload["execution"], "production_pathway_analysis")
        self.assertEqual(len(calls), 1)

        ambiguous = deepcopy(unique)
        ambiguous["pathway_resolution"]["steps"][1]["enzyme_resolution"] = {
            "specified": True,
            "recommended_id": "P-ONE",
            "candidates": [{"id": "P-ONE"}, {"id": "P-TWO"}],
        }
        blocked_calls: list[dict[str, Any]] = []
        registry2 = self._registry(
            route_resolution={},
            pathway_resolution=ambiguous,
            pathway_execute=lambda **kwargs: blocked_calls.append(kwargs) or {},
        )
        result2 = registry2.execute("pathway_compatibility", {"text": "analyze two steps"}, HarnessRunContext(ui_language="en", conversation_context={}))
        self.assertTrue(result2.terminal)
        self.assertTrue(result2.payload["requires_confirmation"])
        self.assertEqual(blocked_calls, [])


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

    def test_verified_session_entity_is_mounted_as_stable_workspace_handle(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("workspace", {
            "direction": "enzyme_to_reaction",
            "protein_resolution": {
                "mode": "protein_id",
                "recommended_id": "P00338",
                "interpreted_protein": "LDHA",
                "candidates": [{"id": "P00338", "name": "LDHA", "input_mode": "protein_id"}],
            },
        })
        registry = self._registry()
        snapshot = store.snapshot("workspace")
        ctx1 = HarnessRunContext(ui_language="en", conversation_context={}, session_facts=snapshot)
        ctx2 = HarnessRunContext(ui_language="en", conversation_context={}, session_facts=snapshot)
        handles1 = registry.seed_session_handles(ctx1)
        handles2 = registry.seed_session_handles(ctx2)
        protein1 = next(row for row in handles1 if row["kind"] == "protein")
        protein2 = next(row for row in handles2 if row["kind"] == "protein")
        self.assertEqual(protein1["ref"], protein2["ref"])
        self.assertTrue(protein1["ref"].startswith("session_protein_scope_"))
        self.assertIn(protein1["ref"], ctx1.protein_refs)
        self.assertEqual(
            ctx1.protein_refs[protein1["ref"]]["resolution"]["recommended_id"],
            "P00338",
        )

    def test_visible_session_entities_are_direct_workspace_handles(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("visible", {
            "direction": "conversation",
            "operation": "research_workspace",
            "immediate_result": {
                "answer_mode": "research_workspace",
                "source_panels": [{
                    "id": "literature",
                    "items": [
                        {"id": "111", "pmid": "111", "source": "MED", "title": "Paper one"},
                        {"id": "222", "pmid": "222", "source": "MED", "title": "Paper two"},
                    ],
                }],
            },
        })
        store.mark_visible_entities(
            "visible", entity_kind="literature",
            entity_ids=["MED:111", "MED:222"], page_index=0,
        )
        registry = self._registry()
        ctx = HarnessRunContext(
            ui_language="en", conversation_context={},
            session_facts=store.snapshot("visible"),
        )
        handles = registry.seed_session_handles(ctx)
        literature = sorted(
            [row for row in handles if row["kind"] == "literature" and row["visible"]],
            key=lambda row: row["visible_index"],
        )
        self.assertEqual([row["id"] for row in literature], ["MED:111", "MED:222"])
        self.assertEqual([row["visible_index"] for row in literature], [1, 2])
        self.assertTrue(all(row["ref"] in ctx.literature_refs for row in literature))

    def test_invalid_tool_arguments_are_json_serializable_observations(self) -> None:
        registry = self._registry()
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        result = registry.execute("lookup_relations", {}, ctx)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_code, "invalid_tool_arguments")
        encoded = __import__("json").dumps(result.model_view())
        self.assertIn("validation", encoded)

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
        self.assertEqual(result.payload["evidence_scope"], "database_recorded_only")
        self.assertFalse(result.payload["exhaustive_of_biochemical_capability"])
        self.assertFalse(result.payload["includes_model_predictions"])
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
        self.assertEqual(result.payload["evidence_scope"], "database_recorded_only")
        self.assertFalse(result.payload["exhaustive_of_biochemical_capability"])
        self.assertFalse(result.payload["includes_model_predictions"])
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
        self.assertEqual(result.payload["evidence_scope"], "database_recorded_only")
        self.assertFalse(result.payload["exhaustive_of_biochemical_capability"])
        self.assertFalse(result.payload["includes_model_predictions"])
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


    def test_low_confidence_compound_candidate_is_not_promoted_to_verified_session_target(self) -> None:
        def resolve(terms, *, limit):
            self.assertEqual(list(terms), ["base compound"])
            return [{
                "chebi_id": "CHEBI:999",
                "name": "modified-base compound",
                "smiles": "CCC",
                "matched_term": "base compound",
                "match_type": "lexical_candidate",
                "match_source": "local_rhea_name_substring",
                "identity_confident": False,
            }]

        registry = self._registry(compound_resolve=resolve)
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        result = registry.execute(
            "resolve_compound",
            {"terms": ["base compound"], "limit": 3},
            ctx,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["candidate_ids"], ["CHEBI:999"])
        self.assertEqual(result.payload["identity_confident_refs"], [])
        self.assertFalse(result.payload["compound_refs"][0]["identity_confident"])
        self.assertEqual(ctx.terminal_resolution["compound_resolution"]["recommended_id"], "")
        self.assertFalse(ctx.terminal_resolution["compound_resolution"]["identity_confident"])

        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("compound-low-confidence", ctx.terminal_resolution)
        snapshot = store.snapshot("compound-low-confidence")
        self.assertEqual(snapshot["verified_compound_ids"], [])
        listed = [
            row for row in store.snapshot("compound-low-confidence")["session_entities"]["all"]
            if row["kind"] == "compound"
        ]
        self.assertEqual([row["id"] for row in listed], ["CHEBI:999"])
        self.assertFalse(listed[0]["payload"]["identity_confident"])
        self.assertEqual(listed[0]["role"], "related_evidence")

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

        compound_ctx = HarnessRunContext(ui_language="en", conversation_context={})
        compound_ctx.compound_refs["compound_1"] = {
            "chebi_id": "CHEBI:12876",
            "name": "(E)-4-coumarate",
            "smiles": "O=C([O-])/C=C/c1ccc(O)cc1",
        }
        compound = registry.execute("inspect_entity", {"compound_ref": "compound_1"}, compound_ctx)
        self.assertEqual(compound.status, "ok")
        self.assertEqual(compound_ctx.terminal_resolution["immediate_result"]["entities"][0]["id"], "CHEBI:12876")

        missing = registry.execute("inspect_entity", {"reaction_ref": "missing"}, ctx)
        self.assertEqual(missing.status, "error")
        self.assertEqual(missing.error_code, "unknown_reaction_ref")

    def test_supporting_inspection_does_not_clobber_primary_resolution(self) -> None:
        evidence = SimpleNamespace(
            reaction_metadata=lambda rid: {"reaction_smiles": "CCO>>CC=O"} if rid == "RHEA:12345" else None,
            protein_metadata=lambda _pid: None,
            is_candidate_protein=lambda _pid: True,
        )
        registry = self._registry(
            agent_resolution=SimpleNamespace(
                evidence=evidence,
                catalog=SimpleNamespace(protein_by_id={}),
                proteins=SimpleNamespace(detail_for=lambda _accession: None),
            )
        )
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        ctx.terminal_resolution = {
            "direction": "enzyme_to_reaction",
            "summary": "primary relation result",
            "reaction_resolution": None,
            "protein_resolution": {
                "mode": "protein_id",
                "recommended_id": "P00338",
                "candidates": [{"id": "P00338", "input_mode": "protein_id"}],
            },
            "positive_enzyme_resolutions": [],
            "immediate_result": {
                "answer_mode": "known_associations_only",
                "known_associations": {"count": 1, "items": [{"candidate_id": "RHEA:12345"}]},
            },
        }
        ctx.reaction_refs["reaction_1"] = {
            "mode": "session_verified_rhea",
            "interpreted_reaction": "RHEA:12345",
            "recommended_id": "RHEA:12345",
            "candidates": [{"rhea_id": "RHEA:12345", "equation": "ethanol = acetaldehyde"}],
        }
        result = registry.execute("inspect_entity", {"reaction_ref": "reaction_1"}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertEqual(ctx.terminal_resolution["direction"], "enzyme_to_reaction")
        self.assertEqual(
            ctx.terminal_resolution["protein_resolution"]["recommended_id"],
            "P00338",
        )
        self.assertIsNone(ctx.terminal_resolution["reaction_resolution"])
        self.assertEqual(
            ctx.terminal_resolution["supporting_inspections"][0]["entity_id"],
            "RHEA:12345",
        )

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
        self.assertTrue(result.payload["evidence_ready"])
        self.assertNotIn("required_next_action", result.payload)
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

    def test_self_inspection_loads_only_requested_public_sections(self) -> None:
        registry = self.registry()
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        result = registry.execute(
            "inspect_self",
            {"topics": ["model_principles", "ranking_interpretation"], "detail": "brief"},
            ctx,
        )
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.terminal)
        self.assertEqual(
            [row["topic"] for row in result.payload["sections"]],
            ["model_principles", "ranking_interpretation"],
        )
        serialized = json.dumps(result.payload, ensure_ascii=False)
        for internal in ("seed_policy", "current_run_refs", "known_association_policy", "action schema"):
            self.assertNotIn(internal, serialized)

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
            "resolve_reaction", "resolve_protein_scope", "lookup_relations",
            "list_scope_members", "resolve_compound", "resolve_literature", "inspect_entity",
            "compare_entities", "research_workspace", "broaden_scope", "candidate_search",
            "route_design", "patch_route_segment", "pathway_compatibility", "inspect_self",
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
        self.assertIn("inspect_self", catalog)
        patch_schema = catalog["patch_route_segment"]["input_schema"]
        self.assertIn("route_ref", patch_schema["properties"])
        self.assertIn("route_step_refs", patch_schema["properties"])
        self.assertIn("replacement_count", patch_schema["properties"])
        self.assertNotIn("args", catalog["candidate_search"])
        self.assertIn("input_schema", catalog["candidate_search"])

    def test_l0_self_model_stays_compact_and_has_no_workflow_manual(self) -> None:
        summary = controller_self_summary()
        serialized = json.dumps(summary, ensure_ascii=False)
        self.assertLess(len(serialized), 2000)
        self.assertNotIn("groups", summary)
        self.assertNotIn("seed_policy", serialized)
        self.assertNotIn("current_run_refs", serialized)


class AgentSessionStoreTests(unittest.TestCase):
    def test_conversation_and_execution_histories_are_append_only_for_controller_context(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_dialogue_turn(
            "history", user_text="first question", assistant_text="first answer",
            response_type="message",
        )
        store.remember_execution_result(
            "history",
            {
                "candidates": [{"rank": 1, "candidate_id": "RXN-1", "name": "reaction one"}],
                "ranking": {"route_id": "route-1"},
                "discovery_filter": {"result_mode": "evidence_plus_unrecorded"},
            },
            direction="enzyme_to_reaction",
        )
        store.remember_dialogue_turn(
            "history", user_text="follow up", assistant_text="second answer",
            response_type="message",
        )
        store.remember_execution_result(
            "history",
            {
                "candidates": [{"rank": 1, "candidate_id": "RXN-2", "name": "reaction two"}],
                "ranking": {"route_id": "route-2"},
                "discovery_filter": {"result_mode": "novel_association_discovery"},
            },
            direction="enzyme_to_reaction",
        )
        self.assertEqual(
            store.model_history("history"),
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "follow up"},
                {"role": "assistant", "content": "second answer"},
            ],
        )
        snapshot = store.model_snapshot("history")
        self.assertNotIn("recent_dialogue", snapshot)
        self.assertEqual(len(snapshot["execution_history"]), 2)
        self.assertEqual(
            [row["candidates"][0]["candidate_id"] for row in snapshot["execution_history"]],
            ["RXN-1", "RXN-2"],
        )
        self.assertEqual(snapshot["last_result_context"]["route_id"], "route-2")

    def test_executed_model_candidates_are_reusable_hypotheses_not_verified_focus(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("candidate-workspace", {
            "direction": "enzyme_to_reaction",
            "protein_resolution": {
                "mode": "protein_id",
                "interpreted_protein": "query protein",
                "recommended_id": "P-TARGET",
                "candidates": [{"id": "P-TARGET", "name": "query protein"}],
            },
        })
        store.remember_execution_result(
            "candidate-workspace",
            {
                "protein": {"id": "P-TARGET", "name": "query protein"},
                "ranking": {"route_id": "route-test"},
                "known_associations": {"count": 0, "items": []},
                "candidates": [
                    {"rank": 1, "candidate_id": "RHEA:10001", "name": "candidate reaction A", "correspondence_defect": 0.1},
                    {"rank": 2, "candidate_id": "RHEA:10002", "name": "candidate reaction B", "correspondence_defect": 0.2},
                ],
            },
            direction="enzyme_to_reaction",
        )
        snapshot = store.snapshot("candidate-workspace")
        model_snapshot = store.model_snapshot("candidate-workspace")
        self.assertEqual(snapshot["verified_reaction_ids"], [])
        candidates = model_snapshot["session_entities"]["candidates"]
        self.assertEqual({row["id"] for row in candidates}, {"RHEA:10001", "RHEA:10002"})
        self.assertTrue(all(row["role"] == "model_candidate" for row in candidates))
        self.assertTrue(all(row.get("hypothesis") for row in candidates))
        self.assertFalse(any(row.get("focus") or row.get("active") for row in candidates))
        protein_focus = [
            row for row in model_snapshot["session_entities"]["focus"]
            if row["kind"] == "protein"
        ]
        self.assertEqual([row["id"] for row in protein_focus], ["P-TARGET"])

        registry = ScientificToolRegistry(
            agent_resolution=object(), deepseek=object(), families=object(),
            family_evidence=object(), evidence_queries=object(),
            route_design_resolve=lambda *a, **k: {}, pathway_resolve=lambda *a, **k: {},
        )
        ctx = HarnessRunContext(
            ui_language="en", conversation_context={},
            session_facts=snapshot,
        )
        handles = registry.seed_session_handles(ctx)
        candidate_handles = [row for row in handles if row.get("role") == "model_candidate"]
        self.assertEqual(len(candidate_handles), 2)
        self.assertTrue(all(row["source"] == "executed_model_candidate" for row in candidate_handles))
        self.assertTrue(all(row["hypothesis"] for row in candidate_handles))
        self.assertEqual(len(ctx.reaction_refs), 2)

    def test_executed_route_steps_become_reusable_handles_and_inspection_exposes_scientific_refs(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("route-workspace", {
            "direction": "route_design",
            "operation": "route_design",
            "immediate_result": {
                "direction": "route_design",
                "routes": [{
                    "route_id": "RR-test-route",
                    "base_rank": 1,
                    "score": 91.2,
                    "route_type": "known_rhea",
                    "compound_ids": ["CHEBI:1", "CHEBI:2", "CHEBI:3"],
                    "compound_names": ["A", "B", "C"],
                    "steps": [
                        {
                            "step_index": 1,
                            "rhea_id": "RHEA:10001",
                            "orientation": "forward",
                            "source": "CHEBI:1",
                            "target": "CHEBI:2",
                            "source_name": "A",
                            "target_name": "B",
                            "swissprot_count": 3,
                        },
                        {
                            "step_index": 2,
                            "rhea_id": "RHEA:10002",
                            "directed_rhea_id": "RHEA:10004",
                            "orientation": "reverse",
                            "source": "CHEBI:2",
                            "target": "CHEBI:3",
                            "source_name": "B",
                            "target_name": "C",
                            "swissprot_count": 5,
                        },
                    ],
                }],
            },
        })
        snapshot = store.model_snapshot("route-workspace")
        routes = [row for row in snapshot["session_entities"]["all"] if row["kind"] == "route"]
        steps = [row for row in snapshot["session_entities"]["all"] if row["kind"] == "route_step"]
        self.assertEqual([row["id"] for row in routes], ["RR-test-route"])
        self.assertEqual([row["step_index"] for row in sorted(steps, key=lambda row: row["step_index"])], [1, 2])
        self.assertEqual([row["id"] for row in snapshot["session_entities"]["visible"]], ["RR-test-route"])

        registry = ScientificToolRegistry(
            agent_resolution=object(),
            deepseek=object(),
            families=object(),
            family_evidence=object(),
            evidence_queries=object(),
            route_design_resolve=lambda *a, **k: {},
            pathway_resolve=lambda *a, **k: {},
        )
        ctx = HarnessRunContext(
            ui_language="en",
            conversation_context={},
            session_facts=store.snapshot("route-workspace"),
        )
        handles = registry.seed_session_handles(ctx)
        route_handle = next(row for row in handles if row["kind"] == "route")
        step_handle = next(row for row in handles if row["kind"] == "route_step" and row["step_index"] == 2)
        self.assertEqual(route_handle["rank"], 1)
        self.assertEqual(step_handle["reaction_id"], "RHEA:10002")
        self.assertEqual(step_handle["parent_route_id"], "RR-test-route")

        inspected = registry.execute(
            "inspect_entity",
            {"route_step_ref": step_handle["ref"]},
            ctx,
        )
        self.assertEqual(inspected.status, "ok")
        self.assertEqual(inspected.payload["entity_kind"], "route_step")
        self.assertEqual(inspected.payload["evidence"]["rhea_id"], "RHEA:10002")
        self.assertEqual(inspected.payload["evidence"]["directed_rhea_id"], "RHEA:10004")
        reaction_ref = inspected.payload["reaction_ref"]
        self.assertEqual(ctx.reaction_refs[reaction_ref]["recommended_id"], "RHEA:10002")
        compound_links = inspected.payload["compound_refs"]
        self.assertEqual(
            [(row["role"], row["chebi_id"]) for row in compound_links],
            [("source", "CHEBI:2"), ("target", "CHEBI:3")],
        )
        self.assertEqual(ctx.terminal_resolution["immediate_result"]["entities"][0]["step_index"], 2)

        store.remember_resolution("route-workspace", ctx.terminal_resolution)
        focused_steps = [
            row for row in store.model_snapshot("route-workspace")["session_entities"]["focus"]
            if row["kind"] == "route_step"
        ]
        self.assertEqual([row["id"] for row in focused_steps], ["RR-test-route::step:2"])

    def test_derived_route_lineage_is_visible_in_workspace_handle_and_inspection(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("route-lineage", {
            "direction": "route_design",
            "operation": "patch_route_segment",
            "immediate_result": {
                "direction": "route_design",
                "answer_mode": "route_patch",
                "routes": [{
                    "route_id": "RP-child",
                    "parent_route_id": "RP-parent",
                    "root_route_id": "RR-root",
                    "generation": 2,
                    "rank": 1,
                    "route_type": "patched_known_rhea",
                    "score": 77.0,
                    "compound_ids": ["CHEBI:1", "CHEBI:2"],
                    "compound_names": ["A", "B"],
                    "patch": {
                        "start_step_index": 2,
                        "end_step_index": 3,
                        "original_rhea_ids": ["RHEA:1", "RHEA:2"],
                    },
                    "steps": [{
                        "step_index": 1,
                        "rhea_id": "RHEA:9",
                        "source": "CHEBI:1",
                        "target": "CHEBI:2",
                        "source_name": "A",
                        "target_name": "B",
                    }],
                }],
            },
        })
        snapshot = store.model_snapshot("route-lineage")
        route_row = next(row for row in snapshot["session_entities"]["all"] if row["kind"] == "route")
        self.assertEqual(route_row["parent_route_id"], "RP-parent")
        self.assertEqual(route_row["root_route_id"], "RR-root")
        self.assertEqual(route_row["generation"], 2)
        self.assertEqual(route_row["patch_start_step_index"], 2)
        self.assertEqual(route_row["patch_end_step_index"], 3)

        registry = ScientificToolRegistry(
            agent_resolution=object(), deepseek=object(), families=object(),
            family_evidence=object(), evidence_queries=object(),
            route_design_resolve=lambda *a, **k: {}, pathway_resolve=lambda *a, **k: {},
        )
        ctx = HarnessRunContext(
            ui_language="en", conversation_context={},
            session_facts=store.snapshot("route-lineage"),
        )
        handle = next(row for row in registry.seed_session_handles(ctx) if row["kind"] == "route")
        self.assertEqual(handle["parent_route_id"], "RP-parent")
        self.assertEqual(handle["root_route_id"], "RR-root")
        self.assertEqual(handle["generation"], 2)
        self.assertEqual(handle["patch_start_step_index"], 2)
        self.assertEqual(handle["patch_end_step_index"], 3)

        inspected = registry.execute("inspect_entity", {"route_ref": handle["ref"]}, ctx)
        evidence = inspected.payload["evidence"]
        self.assertEqual(evidence["parent_route_id"], "RP-parent")
        self.assertEqual(evidence["root_route_id"], "RR-root")
        self.assertEqual(evidence["generation"], 2)
        self.assertEqual(evidence["patch"]["start_step_index"], 2)
        self.assertEqual(evidence["patch"]["end_step_index"], 3)

    def test_direct_entity_list_becomes_ordered_reusable_workspace_without_focus_promotion(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("direct-list", {
            "direction": "conversation",
            "operation": "resolve_literature",
            "immediate_result": {
                "answer_mode": "entity_list",
                "entity_kind": "literature",
                "entities": [
                    {"id": "MED:301", "name": "Paper A", "source": "Europe PMC"},
                    {"id": "MED:302", "name": "Paper B", "source": "Europe PMC"},
                    {"id": "MED:303", "name": "Paper C", "source": "Europe PMC"},
                ],
            },
        })
        snap = store.model_snapshot("direct-list")
        visible = snap["session_entities"]["visible"]
        self.assertEqual([row["id"] for row in visible], ["MED:301", "MED:302", "MED:303"])
        self.assertEqual([row["visible_index"] for row in visible], [1, 2, 3])
        self.assertTrue(all(row["role"] == "related_evidence" for row in visible))
        self.assertFalse(any(row.get("focus") or row.get("active") for row in visible))

        registry = ScientificToolRegistry(
            agent_resolution=object(), deepseek=object(), families=object(),
            family_evidence=object(), evidence_queries=object(),
            route_design_resolve=lambda *a, **k: {}, pathway_resolve=lambda *a, **k: {},
        )
        ctx = HarnessRunContext(
            ui_language="en",
            conversation_context={},
            session_facts=store.snapshot("direct-list"),
        )
        handles = registry.seed_session_handles(ctx)
        papers = [row for row in handles if row.get("kind") == "literature"]
        self.assertEqual([row["id"] for row in sorted(papers, key=lambda x: x.get("visible_index") or 999)], ["MED:301", "MED:302", "MED:303"])
        self.assertEqual([row["visible_index"] for row in sorted(papers, key=lambda x: x.get("visible_index") or 999)], [1, 2, 3])

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
