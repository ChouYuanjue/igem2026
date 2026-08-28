from __future__ import annotations

import json
import re
from typing import Any

from scripts.catalyst_finder.agent_harness.contracts import HarnessTraceStep, ToolResult
from scripts.catalyst_finder.agent_harness.session_store import AgentSessionStore
from scripts.catalyst_finder.agent_harness.tool_registry import HarnessRunContext, ScientificToolRegistry
from scripts.catalyst_finder.errors import AppError
from scripts.catalyst_finder.formatting import probable_uniprot
from scripts.catalyst_finder.resolution_helpers import explicit_uniprot_accession


class CatalystScientificHarness:
    """Small bounded agent loop over Catalyst Finder's scientific capabilities."""

    def __init__(
        self,
        *,
        deepseek: Any,
        tools: ScientificToolRegistry,
        sessions: AgentSessionStore,
        agent_resolution: Any,
        max_turns: int = 6,
    ) -> None:
        self.deepseek = deepseek
        self.tools = tools
        self.sessions = sessions
        self.agent_resolution = agent_resolution
        self.max_turns = max(2, min(int(max_turns), 8))

    @staticmethod
    def _signature(tool: str, args: dict[str, Any]) -> str:
        return f"{tool}:{json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"

    def _decorate(
        self,
        resolution: dict[str, Any],
        *,
        steps: list[HarnessTraceStep],
        fallback: bool,
        session_facts_used: bool,
        mode: str = "scientific_harness",
    ) -> dict[str, Any]:
        output = dict(resolution)
        output["agent_execution"] = {
            "mode": mode,
            "version": "catalyst-scientific-harness-v1",
            "turn_count": len(steps),
            "fallback": fallback,
            "session_facts_used": session_facts_used,
            "steps": [step.model_dump() for step in steps],
        }
        provenance = dict(output.get("llm_provenance") or self.deepseek.provenance())
        provenance["used_for"] = provenance.get("used_for") or mode
        output["llm_provenance"] = provenance
        return output

    def _legacy(
        self,
        text: str,
        *,
        direction_hint: str,
        conversation_context: dict[str, Any],
        ui_language: str,
        steps: list[HarnessTraceStep],
        session_facts_used: bool,
    ) -> dict[str, Any]:
        resolution = self.agent_resolution.agent_resolve(
            text,
            direction_hint=direction_hint,
            conversation_context=conversation_context,
            ui_language=ui_language,
            resolve_reaction=self.agent_resolution.resolve,
        )
        steps.append(HarnessTraceStep(
            turn=len(steps) + 1,
            action_kind="fallback",
            tool="legacy_agent_resolution",
            status="ok",
            summary="Used the legacy resolver as a compatibility fallback.",
        ))
        return self._decorate(
            resolution,
            steps=steps,
            fallback=True,
            session_facts_used=session_facts_used,
            mode="scientific_harness_legacy_fallback",
        )

    def _seed_session_refs(
        self,
        session_facts: dict[str, Any],
        run_ctx: HarnessRunContext,
    ) -> dict[str, Any]:
        """Expose verified prior entities as refs valid in the current tool run."""
        enriched = dict(session_facts)
        reaction_refs: list[dict[str, str]] = []
        for index, reaction_id in enumerate(session_facts.get("verified_reaction_ids") or []):
            rid = str(reaction_id or "").strip()
            if not rid or index >= 4:
                continue
            ref = f"session_reaction_{index + 1}"
            run_ctx.reaction_refs[ref] = {
                "mode": "session_verified_rhea",
                "interpreted_reaction": rid,
                "assumptions": [],
                "candidates": [],
                "recommended_id": rid,
            }
            reaction_refs.append({"ref": ref, "rhea_id": rid})

        protein_refs: list[dict[str, str]] = []
        for index, protein_id in enumerate(session_facts.get("verified_protein_ids") or []):
            pid = str(protein_id or "").strip()
            if not pid or index >= 4:
                continue
            ref = f"session_protein_scope_{index + 1}"
            run_ctx.protein_refs[ref] = {
                "kind": "specific_protein",
                "label": pid,
                "resolution": {
                    "mode": "session_verified_protein",
                    "interpreted_protein": pid,
                    "assumptions": [],
                    "normalized": {},
                    "candidates": [],
                    "recommended_id": pid,
                },
            }
            protein_refs.append({"ref": ref, "protein_id": pid})

        if reaction_refs or protein_refs:
            enriched["current_run_refs"] = {
                "reaction_refs": reaction_refs,
                "protein_scope_refs": protein_refs,
            }
        return enriched

    @staticmethod
    def _clarification(question: str, ui_language: str) -> dict[str, Any]:
        zh = str(ui_language or "").lower().startswith("zh")
        return {
            "direction": "ambiguous",
            "summary": question,
            "confidence": 0.0,
            "alternative_direction": "",
            "ambiguity": True,
            "intent_options": [
                {"direction": "reaction_to_enzyme", "label": "从反应找酶" if zh else "Find enzymes for a reaction"},
                {"direction": "enzyme_to_reaction", "label": "从酶找反应" if zh else "Find reactions for an enzyme"},
                {"direction": "route_design", "label": "设计合成路线" if zh else "Design a biosynthetic route"},
                {"direction": "pathway_compatibility", "label": "评估多步路径" if zh else "Evaluate a multi-step pathway"},
            ],
        }

    def run(
        self,
        text: str,
        *,
        direction_hint: str = "auto",
        conversation_context: dict[str, Any] | None = None,
        ui_language: str = "en",
        session_id: str = "",
    ) -> dict[str, Any]:
        text = str(text or "").strip()
        if not text:
            raise AppError("empty_input", "告诉我你想查询的反应、酶、路线或路径。", 422)
        context = dict(conversation_context or {})
        steps: list[HarnessTraceStep] = []
        session_facts = self.sessions.snapshot(session_id)
        session_facts_used = bool(session_facts)

        # Deterministic structured payloads and exact IDs already have a safer/faster
        # parser than an LLM loop. They still enter the same downstream runtime.
        direct = self.agent_resolution._direct_open_world_resolution(text, direction_hint, ui_language)
        exact_database_id = bool(
            re.fullmatch(r"\s*(?:RHEA\s*:\s*)?\d{5}\s*", text, re.IGNORECASE)
            or probable_uniprot(text)
            or explicit_uniprot_accession(text)
        )
        if direct is not None or exact_database_id:
            resolution = direct or self.agent_resolution.agent_resolve(
                text,
                direction_hint=direction_hint,
                conversation_context=context,
                ui_language=ui_language,
                resolve_reaction=self.agent_resolution.resolve,
            )
            steps.append(HarnessTraceStep(
                turn=1,
                action_kind="deterministic_fast_path",
                status="ok",
                summary="Used deterministic structured-input/database-ID resolution.",
            ))
            output = self._decorate(
                resolution,
                steps=steps,
                fallback=False,
                session_facts_used=session_facts_used,
                mode="deterministic_fast_path",
            )
            self.sessions.remember_resolution(session_id, output)
            return output

        history: list[dict[str, Any]] = []
        seen_calls: set[str] = set()
        repeated_rejections = 0
        run_ctx = HarnessRunContext(
            ui_language=ui_language,
            direction_hint=direction_hint,
            conversation_context=context,
        )
        controller_session_facts = self._seed_session_refs(session_facts, run_ctx)

        try:
            for turn in range(1, self.max_turns + 1):
                action = self.deepseek.next_harness_action(
                    user_text=text,
                    direction_hint=direction_hint,
                    session_facts=controller_session_facts,
                    tool_catalog=self.tools.catalog(),
                    history=history,
                    ui_language=ui_language,
                )
                if action.kind == "ask_user":
                    steps.append(HarnessTraceStep(
                        turn=turn,
                        action_kind="ask_user",
                        status="needs_input",
                        summary=action.question,
                    ))
                    output = self._decorate(
                        self._clarification(action.question, ui_language),
                        steps=steps,
                        fallback=False,
                        session_facts_used=session_facts_used,
                    )
                    return output
                if action.kind == "final":
                    if run_ctx.terminal_resolution is not None:
                        steps.append(HarnessTraceStep(
                            turn=turn,
                            action_kind="final",
                            status="ok",
                            summary=action.reason or "Accepted terminal scientific tool result.",
                        ))
                        output = self._decorate(
                            run_ctx.terminal_resolution,
                            steps=steps,
                            fallback=False,
                            session_facts_used=session_facts_used,
                        )
                        self.sessions.remember_resolution(session_id, output)
                        return output
                    synthetic = ToolResult(
                        tool="prepare_candidate_retrieval",
                        status="error",
                        summary="final is not allowed before a terminal scientific tool result exists.",
                        recoverable=True,
                        error_code="premature_final",
                    )
                    history.append({"turn": turn, "action": action.model_dump(), "result": synthetic.model_view()})
                    steps.append(HarnessTraceStep(turn=turn, action_kind="final", status="rejected", summary=synthetic.summary))
                    continue

                assert action.tool is not None
                signature = self._signature(str(action.tool), action.args)
                if signature in seen_calls:
                    repeated_rejections += 1
                    repeated = ToolResult(
                        tool=action.tool,
                        status="error",
                        summary="This identical tool call has already been executed in this run. Choose a different capability or arguments.",
                        recoverable=True,
                        error_code="repeated_tool_call",
                    )
                    history.append({"turn": turn, "action": action.model_dump(), "result": repeated.model_view()})
                    steps.append(HarnessTraceStep(turn=turn, action_kind="tool", tool=str(action.tool), status="rejected", summary=repeated.summary))
                    if repeated_rejections >= 2:
                        break
                    continue
                seen_calls.add(signature)
                result = self.tools.execute(action.tool, action.args, run_ctx)
                history.append({"turn": turn, "action": action.model_dump(), "result": result.model_view()})
                steps.append(HarnessTraceStep(
                    turn=turn,
                    action_kind="tool",
                    tool=str(action.tool),
                    status=result.status,
                    summary=result.summary[:700],
                ))
                if result.terminal and result.status == "ok" and run_ctx.terminal_resolution is not None:
                    output = self._decorate(
                        run_ctx.terminal_resolution,
                        steps=steps,
                        fallback=False,
                        session_facts_used=session_facts_used,
                    )
                    self.sessions.remember_resolution(session_id, output)
                    return output
                if result.status == "error" and not result.recoverable:
                    break
        except AppError as exc:
            steps.append(HarnessTraceStep(
                turn=len(steps) + 1,
                action_kind="controller_error",
                status="error",
                summary=f"{getattr(exc, 'code', 'controller_error')}: {exc}",
            ))

        output = self._legacy(
            text,
            direction_hint=direction_hint,
            conversation_context=context,
            ui_language=ui_language,
            steps=steps,
            session_facts_used=session_facts_used,
        )
        self.sessions.remember_resolution(session_id, output)
        return output
