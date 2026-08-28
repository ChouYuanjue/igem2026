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
            [HarnessAction(kind="tool", tool="prepare_candidate_retrieval", args={"direction": "reaction_to_enzyme", "full_text": "find candidates", "reaction_text": "reaction X"})],
            [ToolResult(tool="prepare_candidate_retrieval", status="ok", summary="prepared", terminal=True)],
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
                HarnessAction(kind="tool", tool="prepare_candidate_retrieval", args={"direction": "reaction_to_enzyme", "full_text": "ambiguous reaction", "reaction_text": "ambiguous reaction"}),
            ],
            [
                ToolResult(tool="resolve_reaction", status="error", summary="no exact evidence", recoverable=True, error_code="no_match"),
                ToolResult(tool="prepare_candidate_retrieval", status="ok", summary="prepared", terminal=True),
            ],
        )
        result = harness.run("ambiguous reaction")
        self.assertFalse(result["agent_execution"]["fallback"])
        self.assertEqual([call[0] for call in tools.calls], ["resolve_reaction", "prepare_candidate_retrieval"])
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
                HarnessAction(kind="tool", tool="prepare_candidate_retrieval", args={"direction": "reaction_to_enzyme", "full_text": "show known evidence and candidates", "reaction_text": "reaction X"}),
            ],
            [
                ToolResult(tool="resolve_reaction", status="ok", summary="verified", terminal=False),
                ToolResult(tool="prepare_candidate_retrieval", status="ok", summary="prepared", terminal=True),
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
        self.assertEqual([c[0] for c in tools2.calls], ["resolve_reaction", "prepare_candidate_retrieval"])
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

    def test_nonrecoverable_tool_error_still_returns_to_model_for_decision(self) -> None:
        harness, deepseek, tools = self.build(
            [
                HarnessAction(kind="tool", tool="resolve_reaction", args={"text": "reaction X"}),
                HarnessAction(kind="respond", message="The evidence service is unavailable, so I cannot verify that claim right now."),
            ],
            [ToolResult(tool="resolve_reaction", status="error", summary="backend unavailable", recoverable=False, error_code="backend")],
        )
        result = harness.run("reaction X")
        self.assertEqual(result["response_type"], "message")
        self.assertEqual(len(deepseek.calls), 2)
        self.assertEqual(len(tools.calls), 1)
        self.assertFalse(result["agent_execution"]["fallback"])

    def test_verified_session_entity_is_exposed_as_current_run_ref(self) -> None:
        sessions = AgentSessionStore(ttl_seconds=3600)
        sessions.remember_resolution("follow", {
            "direction": "reaction_to_enzyme",
            "reaction_resolution": {"recommended_id": "RHEA:32883", "candidates": []},
        })
        harness, deepseek, _tools = self.build(
            [HarnessAction(kind="ask_user", question="Which protein family constraint should I apply?")],
            [],
            sessions=sessions,
        )
        harness.run("那这个反应呢？", session_id="follow")
        facts = deepseek.calls[0]["session_facts"]
        self.assertEqual(
            facts["current_run_refs"]["reaction_refs"],
            [{"ref": "session_reaction_1", "rhea_id": "RHEA:32883"}],
        )

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
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
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
            "prepare_candidate_retrieval",
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
            "lookup_recorded_associations",
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
            "lookup_recorded_associations",
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
            "prepare_candidate_retrieval",
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
            "prepare_candidate_retrieval",
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
            "prepare_candidate_retrieval",
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
            "prepare_candidate_retrieval",
            {"direction": "enzyme_to_reaction", "full_text": "show possible reactions", "protein_scope_ref": "protein_scope_1"},
            ctx2,
        )
        self.assertEqual(e2r.status, "ok")
        self.assertEqual(ctx2.terminal_resolution["protein_resolution"]["recommended_id"], "P00338")

        ctx3 = HarnessRunContext(ui_language="en", conversation_context={})
        ctx3.protein_refs["protein_scope_1"] = {"kind": "family", "family_id": "PF01040", "label": "UbiA family"}
        family = registry.execute(
            "prepare_candidate_retrieval",
            {"direction": "enzyme_to_reaction", "full_text": "predict family reactions", "protein_scope_ref": "protein_scope_1"},
            ctx3,
        )
        self.assertEqual(family.status, "error")
        self.assertEqual(family.error_code, "candidate_requires_specific_protein")


class NaturalScientificToolTests(unittest.TestCase):
    @staticmethod
    def _registry(*, evidence_queries: Any = None, families: Any = None, compound_resolve: Any = None, family_evidence: Any = None, agent_resolution: Any = None) -> ScientificToolRegistry:
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
        )

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
        result = registry.execute("lookup_recorded_protein_reactions", {"protein_scope_ref": "protein_scope_1"}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["reaction_ids"], ["RHEA:12345"])
        self.assertEqual(ctx.terminal_resolution["direction"], "enzyme_to_reaction")
        self.assertEqual(ctx.terminal_resolution["immediate_result"]["known_associations"]["count"], 1)

    def test_reverse_evidence_tool_rejects_family_scope(self) -> None:
        registry = self._registry()
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        ctx.protein_refs["protein_scope_1"] = {"kind": "family", "family_id": "PF00001"}
        result = registry.execute("lookup_recorded_protein_reactions", {"protein_scope_ref": "protein_scope_1"}, ctx)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_code, "scope_not_specific_protein")

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
        result = registry.execute("list_protein_scope_members", {"protein_scope_ref": "protein_scope_1", "limit": 2}, ctx)
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.terminal)
        immediate = ctx.terminal_resolution["immediate_result"]
        self.assertEqual(immediate["answer_mode"], "entity_list")
        self.assertEqual(immediate["entity_kind"], "protein")
        self.assertEqual([row["id"] for row in immediate["entities"]], ["P1", "P2"])
        self.assertIn("subset", immediate["note"].lower())

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


    def test_inspect_verified_entity_reads_only_existing_refs(self) -> None:
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
        reaction = registry.execute("inspect_verified_entity", {"reaction_ref": "reaction_1"}, ctx)
        self.assertEqual(reaction.status, "ok")
        self.assertTrue(reaction.terminal)
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
        compound = registry.execute("inspect_verified_entity", {"compound_ref": "compound_1"}, ctx)
        self.assertEqual(compound.status, "ok")
        self.assertEqual(ctx.terminal_resolution["immediate_result"]["entities"][0]["id"], "CHEBI:12876")

        missing = registry.execute("inspect_verified_entity", {"reaction_ref": "missing"}, ctx)
        self.assertEqual(missing.status, "error")
        self.assertEqual(missing.error_code, "unknown_reaction_ref")

    def test_verified_functional_scope_rehydrates_as_session_tool_ref(self) -> None:
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
                    "scope_id": "CLASS-ABC",
                    "family_id": "CLASS-ABC",
                    "label": "cytochrome P450",
                    "normalized_terms": ["cytochrome P450"],
                    "strict_terms": ["cytochrome P450"],
                    "broader_terms": ["heme monooxygenase"],
                },
            },
        })
        snapshot = store.snapshot("scope-session")
        self.assertEqual(snapshot["verified_protein_scopes"][0]["kind"], "functional_class")
        harness = CatalystScientificHarness(
            deepseek=FakeDeepSeek([HarnessAction(kind="ask_user", question="ok?")]),
            tools=FakeTools([]),
            sessions=store,
        )
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        enriched = harness._seed_session_refs(snapshot, ctx)  # noqa: SLF001 - session-ref contract
        refs = enriched["current_run_refs"]["protein_scope_refs"]
        group_ref = next(row["ref"] for row in refs if row.get("scope_kind") == "functional_class")
        self.assertEqual(ctx.protein_refs[group_ref]["kind"], "functional_class")
        self.assertEqual(ctx.protein_refs[group_ref]["enzyme_spec"]["strict_terms"], ["cytochrome P450"])

    def test_session_refs_explicitly_distinguish_tool_refs_from_database_ids(self) -> None:
        store = AgentSessionStore(ttl_seconds=3600)
        store.remember_resolution("scope-session", {
            "direction": "enzyme_to_reaction",
            "protein_resolution": {
                "mode": "protein_functional_class",
                "interpreted_protein": "example class",
                "recommended_id": "CLASS-ABC",
                "family": {
                    "scope_id": "CLASS-ABC",
                    "family_id": "CLASS-ABC",
                    "label": "example class",
                    "normalized_terms": ["example class"],
                    "strict_terms": ["example class"],
                    "broader_terms": [],
                },
            },
        })
        harness = CatalystScientificHarness(
            deepseek=FakeDeepSeek([HarnessAction(kind="ask_user", question="ok?")]),
            tools=FakeTools([]),
            sessions=store,
        )
        ctx = HarnessRunContext(ui_language="en", conversation_context={})
        enriched = harness._seed_session_refs(store.snapshot("scope-session"), ctx)  # noqa: SLF001
        current = enriched["current_run_refs"]
        self.assertIn("ONLY", current["usage_rule"])
        self.assertIn("CLASS-", current["usage_rule"])
        ref = current["protein_scope_refs"][0]["ref"]
        self.assertIn(ref, ctx.protein_refs)
        self.assertNotIn("CLASS-ABC", ctx.protein_refs)

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



class ScientificToolCatalogTests(unittest.TestCase):
    def test_catalog_exposes_pydantic_input_schema(self) -> None:
        catalog = {item["name"]: item for item in ScientificToolRegistry.catalog()}
        schema = catalog["resolve_protein_scope"]["input_schema"]
        self.assertIn("text", schema["properties"])
        self.assertIn("scope_hint", schema["properties"])
        self.assertIn("text", schema["required"])
        self.assertIn("family_or_class", str(schema["properties"]["scope_hint"]))
        self.assertIn("lookup_recorded_protein_reactions", catalog)
        self.assertIn("list_protein_scope_members", catalog)
        self.assertIn("resolve_compound", catalog)
        self.assertIn("inspect_verified_entity", catalog)


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
