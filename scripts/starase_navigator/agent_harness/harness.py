from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from scripts.starase_navigator.agent_harness.capabilities import controller_self_summary
from scripts.starase_navigator.agent_harness.contracts import HarnessTraceStep, ToolResult
from scripts.starase_navigator.agent_harness.session_store import AgentSessionStore
from scripts.starase_navigator.agent_harness.tool_registry import HarnessRunContext, ScientificToolRegistry
from scripts.starase_navigator.errors import AppError


class ScientificAgentHarness:
    """Model-led, bounded scientific-agent loop over Starase Navigator capabilities.

    Every non-empty user message reaches the controller model first. Python validates
    tool contracts and scientific evidence, but does not pre-classify the task.
    """

    def __init__(
        self,
        *,
        deepseek: Any,
        tools: ScientificToolRegistry,
        sessions: AgentSessionStore,
        max_turns: int = 12,
    ) -> None:
        self.deepseek = deepseek
        self.tools = tools
        self.sessions = sessions
        self.max_turns = max(2, min(int(max_turns), 16))

    @staticmethod
    def _signature(tool: str, args: dict[str, Any]) -> str:
        return f"{tool}:{json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"


    @staticmethod
    def _compact_agent_evidence(evidence_history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []
        for entry in list(evidence_history or [])[-8:]:
            if not isinstance(entry, dict):
                continue
            result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
            immediate = result.get("immediate_result") if isinstance(result.get("immediate_result"), dict) else {}
            if not immediate:
                continue
            entities = [row for row in immediate.get("entities") or [] if isinstance(row, dict)]
            candidates = [row for row in immediate.get("candidates") or [] if isinstance(row, dict)]
            known = immediate.get("known_associations") if isinstance(immediate.get("known_associations"), dict) else {}
            artifact = {
                "turn": entry.get("turn"),
                "tool": str(entry.get("tool") or ""),
                "direction": str(result.get("direction") or ""),
                "operation": str(result.get("operation") or ""),
                "answer_mode": str(immediate.get("answer_mode") or ""),
                "title": str(immediate.get("title") or result.get("summary") or "")[:300],
                "entity_kind": str(immediate.get("entity_kind") or ""),
                "entity_count": len(entities),
                "candidate_count": len(candidates),
                "recorded_association_count": int(known.get("count") or 0),
                "entities": [
                    {
                        "id": str(row.get("id") or row.get("candidate_id") or "")[:160],
                        "name": str(row.get("name") or row.get("title") or "")[:300],
                        "source": str(row.get("source") or "")[:160],
                    }
                    for row in entities[:6]
                ],
                "candidates": [
                    {
                        "rank": row.get("rank"),
                        "id": str(row.get("candidate_id") or row.get("id") or "")[:160],
                        "name": str(row.get("name") or row.get("substrate_name") or row.get("product_name") or "")[:300],
                        "known_association": bool(row.get("known_association")),
                    }
                    for row in candidates[:6]
                ],
                "note": str(immediate.get("note") or "")[:500],
            }
            artifacts.append(artifact)
        return artifacts

    def _decorate(
        self,
        resolution: dict[str, Any],
        *,
        steps: list[HarnessTraceStep],
        session_facts_used: bool,
        evidence_history: list[dict[str, Any]] | None = None,
        mode: str = "model_led_scientific_harness",
    ) -> dict[str, Any]:
        output = dict(resolution)
        compact_evidence = self._compact_agent_evidence(evidence_history)
        if compact_evidence:
            output["agent_evidence"] = compact_evidence
        output["agent_execution"] = {
            "mode": mode,
            "version": "starase-navigator-agent-v6",
            "turn_count": len(steps),
            "fallback": False,
            "session_facts_used": session_facts_used,
            "steps": [step.model_dump() for step in steps],
        }
        provenance = dict(output.get("llm_provenance") or self.deepseek.provenance())
        provenance["used_for"] = provenance.get("used_for") or mode
        output["llm_provenance"] = provenance
        return output

    @staticmethod
    def _conversation_payload(message: str, *, clarification: bool) -> dict[str, Any]:
        return {
            "direction": "conversation",
            "response_type": "clarification" if clarification else "message",
            "assistant_response": message,
            "summary": message,
            "needs_user_input": clarification,
            "reaction_resolution": None,
            "protein_resolution": None,
            "positive_enzyme_resolutions": [],
        }

    def run(
        self,
        text: str,
        *,
        conversation_context: dict[str, Any] | None = None,
        ui_language: str = "en",
        session_id: str = "",
    ) -> dict[str, Any]:
        text = str(text or "").strip()
        if not text:
            raise AppError("empty_input", "请直接告诉我你想做什么，或粘贴反应、蛋白和路径信息。", 422)

        context = dict(conversation_context or {})
        steps: list[HarnessTraceStep] = []
        history: list[dict[str, Any]] = []
        # Full verified result snapshots are kept separately from the compact controller
        # history. This is the run-scoped verified observation ledger supplied to the primary controller so
        # later tool calls cannot accidentally erase evidence produced earlier in the run.
        evidence_history: list[dict[str, Any]] = []
        seen_calls: dict[str, dict[str, Any]] = {}
        session_facts = self.sessions.snapshot(session_id)
        controller_session_facts = self.sessions.model_snapshot(session_id)
        conversation_history = self.sessions.model_history(session_id)
        session_entities = (controller_session_facts.get("session_entities") or {}).get("all") or []
        session_facts_used = bool(
            conversation_history
            or session_entities
            or controller_session_facts.get("last_target")
            or controller_session_facts.get("last_direction")
            or controller_session_facts.get("last_result_context")
        )
        run_ctx = HarnessRunContext(
            ui_language=ui_language,
            conversation_context=context,
            user_text=text,
            session_id=session_id,
            session_facts=session_facts,
        )
        seed_current=getattr(self.tools,"seed_current_input_handles",None)
        current_handles=(
            list(seed_current(run_ctx) or []) if callable(seed_current) else []
        )
        seed_handles=getattr(self.tools,"seed_session_handles",None)
        session_handles=(
            list(seed_handles(run_ctx) or []) if callable(seed_handles) else []
        )
        workspace_handles=current_handles + session_handles
        secondary_session_refs = {
            str(row.get("ref") or "")
            for row in session_handles
            if isinstance(row, dict)
            and str(row.get("role") or "") in {"related_evidence", "model_candidate"}
            and str(row.get("ref") or "")
        }
        capability_manifest = controller_self_summary()

        def current_refs(values: dict[str, Any]) -> list[str]:
            # Historical evidence/model hypotheses stay available in workspace_handles
            # but are not promoted to the primary current-ref pool. Any refs created
            # during this run are not in secondary_session_refs and therefore remain current.
            return [ref for ref in values.keys() if ref not in secondary_session_refs]

        for turn in range(1, self.max_turns + 1):
            action = self.deepseek.next_harness_action(
                user_text=text,
                session_facts=controller_session_facts,
                tool_catalog=self.tools.catalog(),
                capability_manifest=capability_manifest,
                conversation_history=conversation_history,
                workspace_handles=workspace_handles,
                history=history,
                verified_evidence=evidence_history,
                current_run_refs={
                    "reaction_ref": current_refs(run_ctx.reaction_refs),
                    "protein_scope_ref": current_refs(run_ctx.protein_refs),
                    "compound_ref": current_refs(run_ctx.compound_refs),
                    "literature_ref": current_refs(run_ctx.literature_refs),
                    "route_ref": current_refs(run_ctx.route_refs),
                    "route_step_ref": current_refs(run_ctx.route_step_refs),
                },
                execution_budget={
                    "current_turn": turn,
                    "max_turns": self.max_turns,
                    "remaining_turns_after_this": self.max_turns - turn,
                    "successful_tool_observations": len(evidence_history),
                },
                ui_language=ui_language,
            )

            if action.kind == "respond":
                steps.append(HarnessTraceStep(
                    turn=turn,
                    action_kind="respond",
                    status="ok",
                    summary="Returned the agent's natural-language response from the current working trace.",
                ))
                if run_ctx.terminal_resolution is not None:
                    resolution=deepcopy(run_ctx.terminal_resolution)
                    resolution["assistant_response"]=action.message.strip()
                    resolution["response_type"]="message"
                    resolution["summary"]=action.message.strip()[:800]
                else:
                    resolution=self._conversation_payload(action.message.strip(), clarification=False)
                output = self._decorate(
                    resolution,
                    steps=steps,
                    session_facts_used=session_facts_used,
                    evidence_history=evidence_history,
                )
                if run_ctx.terminal_resolution is not None:
                    self.sessions.remember_resolution(session_id, output)
                return output

            if action.kind == "ask_user":
                steps.append(HarnessTraceStep(
                    turn=turn,
                    action_kind="ask_user",
                    status="needs_input",
                    summary="Asked one concrete clarification question.",
                ))
                return self._decorate(
                    self._conversation_payload(action.question.strip(), clarification=True),
                    steps=steps,
                    session_facts_used=session_facts_used,
                    evidence_history=evidence_history,
                )


            if action.kind == "return_result":
                if run_ctx.terminal_resolution is None:
                    steps.append(HarnessTraceStep(
                        turn=turn,
                        action_kind="return_result",
                        status="rejected",
                        summary="No verified structured result is available to return yet.",
                    ))
                    history.append({
                        "turn": turn,
                        "action": action.model_dump(),
                        "result": {
                            "status": "error",
                            "summary": "No verified structured result is available. Call a scientific tool first.",
                            "recoverable": True,
                            "error_code": "no_result_to_return",
                        },
                    })
                    continue
                steps.append(HarnessTraceStep(
                    turn=turn,
                    action_kind="return_result",
                    status="ok",
                    summary="Returned the current verified structured result.",
                ))
                output = self._decorate(
                    run_ctx.terminal_resolution,
                    steps=steps,
                    session_facts_used=session_facts_used,
                    evidence_history=evidence_history,
                )
                if action.message.strip():
                    output["assistant_response"] = action.message.strip()
                    output["summary"] = action.message.strip()[:800]
                self.sessions.remember_resolution(session_id, output)
                return output

            assert action.tool is not None
            signature = self._signature(str(action.tool), action.args)
            if signature in seen_calls:
                previous=deepcopy(seen_calls[signature])
                repeated=ToolResult(
                    tool=action.tool,
                    status="ok" if str(previous.get("status") or "")=="ok" else "error",
                    summary="This identical tool call is already present in the working trace; reuse its previous observation instead of spending another call.",
                    payload={"previous_observation":previous},
                    recoverable=True,
                    error_code="duplicate_tool_call_reused",
                )
                history.append({"turn": turn, "action": action.model_dump(), "result": repeated.model_view()})
                steps.append(HarnessTraceStep(
                    turn=turn,
                    action_kind="tool",
                    tool=str(action.tool),
                    status="cached",
                    summary=repeated.summary,
                ))
                continue

            previous_resolution = run_ctx.terminal_resolution
            result = self.tools.execute(action.tool, action.args, run_ctx)
            seen_calls[signature]=result.model_view()
            if result.status == "ok":
                evidence_entry: dict[str, Any] = {
                    "turn": turn,
                    "tool": str(action.tool),
                    "payload": deepcopy(result.payload),
                }
                if run_ctx.terminal_resolution is not None and run_ctx.terminal_resolution is not previous_resolution:
                    evidence_entry["result"] = deepcopy(run_ctx.terminal_resolution)
                    # Verified intermediate observations are durable workspace state,
                    # not disposable chain-of-thought. Persist them immediately so a
                    # later tool cannot erase an inspected/focused object merely by
                    # replacing terminal_resolution in the same run.
                    self.sessions.remember_resolution(
                        session_id,
                        deepcopy(run_ctx.terminal_resolution),
                    )
                evidence_history.append(evidence_entry)
                # Bound pathological tool chains without dropping the newest evidence.
                del evidence_history[:-8]
            history_entry={"turn": turn, "action": action.model_dump(), "result": result.model_view()}
            if run_ctx.terminal_resolution is not None and run_ctx.terminal_resolution is not previous_resolution:
                history_entry["verified_result"]=deepcopy(run_ctx.terminal_resolution)
            history.append(history_entry)
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
                    session_facts_used=session_facts_used,
                    evidence_history=evidence_history,
                )
                self.sessions.remember_resolution(session_id, output)
                return output

        if run_ctx.terminal_resolution is not None:
            synthesis_answer = ""
            synthesis = getattr(self.deepseek, "synthesize_grounded_answer", None)
            if callable(synthesis):
                try:
                    synthesized = synthesis(
                        user_text=text,
                        conversation_history=conversation_history,
                        terminal_resolution=deepcopy(run_ctx.terminal_resolution),
                        verified_evidence=deepcopy(evidence_history),
                        execution_history=deepcopy(history),
                        ui_language=ui_language,
                    )
                    if isinstance(synthesized, dict):
                        synthesis_answer = str(synthesized.get("answer") or "").strip()
                    else:
                        synthesis_answer = str(synthesized or "").strip()
                except Exception:
                    synthesis_answer = ""
            if synthesis_answer:
                steps.append(HarnessTraceStep(
                    turn=self.max_turns + 1,
                    action_kind="respond",
                    status="synthesized",
                    summary="Synthesized the final answer from the verified execution trace after the controller tool budget was exhausted.",
                ))
                resolution = deepcopy(run_ctx.terminal_resolution)
                resolution["assistant_response"] = synthesis_answer
                resolution["response_type"] = "message"
                resolution["summary"] = synthesis_answer[:800]
                output = self._decorate(
                    resolution,
                    steps=steps,
                    session_facts_used=session_facts_used,
                    evidence_history=evidence_history,
                    mode="model_led_scientific_harness_final_synthesis",
                )
                self.sessions.remember_resolution(session_id, output)
                return output
            steps.append(HarnessTraceStep(
                turn=self.max_turns + 1,
                action_kind="return_result",
                status="fallback",
                summary="Returned the latest verified structured result after the controller reached its turn limit; final synthesis was unavailable.",
            ))
            output = self._decorate(
                run_ctx.terminal_resolution,
                steps=steps,
                session_facts_used=session_facts_used,
                evidence_history=evidence_history,
                mode="model_led_scientific_harness_fail_soft",
            )
            self.sessions.remember_resolution(session_id, output)
            return output
        raise AppError(
            "agent_turn_limit",
            "智能体在本轮内没有形成可执行结果。请补充目标或约束后再试。",
            502,
        )
