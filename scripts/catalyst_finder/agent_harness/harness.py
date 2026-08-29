from __future__ import annotations

import json
from typing import Any

from scripts.catalyst_finder.agent_harness.capabilities import public_capabilities
from scripts.catalyst_finder.agent_harness.contracts import HarnessTraceStep, ToolResult
from scripts.catalyst_finder.agent_harness.session_store import AgentSessionStore
from scripts.catalyst_finder.agent_harness.tool_registry import HarnessRunContext, ScientificToolRegistry
from scripts.catalyst_finder.errors import AppError


class CatalystScientificHarness:
    """Model-led, bounded scientific-agent loop over Catalyst Finder capabilities.

    Every non-empty user message reaches the controller model first. Python validates
    tool contracts and scientific evidence, but does not pre-classify the task.
    """

    def __init__(
        self,
        *,
        deepseek: Any,
        tools: ScientificToolRegistry,
        sessions: AgentSessionStore,
        max_turns: int = 6,
    ) -> None:
        self.deepseek = deepseek
        self.tools = tools
        self.sessions = sessions
        self.max_turns = max(2, min(int(max_turns), 8))

    @staticmethod
    def _signature(tool: str, args: dict[str, Any]) -> str:
        return f"{tool}:{json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"

    def _decorate(
        self,
        resolution: dict[str, Any],
        *,
        steps: list[HarnessTraceStep],
        session_facts_used: bool,
        mode: str = "model_led_scientific_harness",
    ) -> dict[str, Any]:
        output = dict(resolution)
        output["agent_execution"] = {
            "mode": mode,
            "version": "catalyst-model-led-agent-v5",
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
        seen_calls: set[str] = set()
        repeated_rejections = 0
        session_facts = self.sessions.snapshot(session_id)
        controller_session_facts = self.sessions.model_snapshot(session_id)
        session_facts_used = bool(controller_session_facts)
        run_ctx = HarnessRunContext(
            ui_language=ui_language,
            conversation_context=context,
            user_text=text,
            session_facts=session_facts,
        )
        capability_manifest = public_capabilities()

        for turn in range(1, self.max_turns + 1):
            action = self.deepseek.next_harness_action(
                user_text=text,
                session_facts=controller_session_facts,
                tool_catalog=self.tools.catalog(),
                capability_manifest=capability_manifest,
                history=history,
                current_run_refs={
                    "reaction_ref": list(run_ctx.reaction_refs.keys()),
                    "protein_scope_ref": list(run_ctx.protein_refs.keys()),
                    "compound_ref": list(run_ctx.compound_refs.keys()),
                    "literature_ref": list(run_ctx.literature_refs.keys()),
                },
                ui_language=ui_language,
            )

            if action.kind == "respond":
                session_rows = ((controller_session_facts.get("session_entities") or {}).get("all") or []) if isinstance(controller_session_facts, dict) else []
                if session_rows:
                    session_reference = self.deepseek.select_session_entity_reference(
                        user_text=text, records=[row for row in session_rows if isinstance(row, dict)],
                        expected_kind="", requested_identity="", ui_language=ui_language,
                    )
                    reference_mode = str(session_reference.get("reference_mode") or "none")
                    if reference_mode in {"focus", "active"} or (reference_mode == "specific" and str(session_reference.get("selected_key") or "").strip()):
                        summary = "The latest message refers to a verified prior session entity. Use reuse_session_entity and scientific tools instead of answering from model memory."
                        history.append({
                            "turn": turn, "action": action.model_dump(),
                            "result": {"status": "error", "summary": summary, "recoverable": True, "error_code": "session_reference_requires_tool"},
                        })
                        steps.append(HarnessTraceStep(turn=turn, action_kind="respond", status="rejected", summary=summary))
                        continue
                has_successful_scientific_result = any(
                    isinstance(entry, dict)
                    and str((entry.get("result") or {}).get("status") or "") == "ok"
                    for entry in history
                )
                if has_successful_scientific_result:
                    summary = (
                        "A successful scientific tool result already exists. Free-form scientific claims cannot be added now; "
                        "continue with scientific tools, ask one minimal clarification, or use return_result for the verified structured result."
                    )
                    history.append({
                        "turn": turn,
                        "action": action.model_dump(),
                        "result": {
                            "status": "error",
                            "summary": summary,
                            "recoverable": True,
                            "error_code": "post_tool_freeform_response_disallowed",
                        },
                    })
                    steps.append(HarnessTraceStep(
                        turn=turn,
                        action_kind="respond",
                        status="rejected",
                        summary=summary,
                    ))
                    continue
                steps.append(HarnessTraceStep(
                    turn=turn,
                    action_kind="respond",
                    status="ok",
                    summary="Returned a natural-language agent response before any scientific tool result existed.",
                ))
                return self._decorate(
                    self._conversation_payload(action.message.strip(), clarification=False),
                    steps=steps,
                    session_facts_used=session_facts_used,
                )

            if action.kind == "ask_user":
                session_entities = ((session_facts.get("session_entities") or {}).get("all") or []) if isinstance(session_facts, dict) else []
                reusable_kinds = []
                for kind in ("protein", "reaction", "protein_scope", "compound", "literature"):
                    rows = [row for row in session_entities if isinstance(row, dict) and str(row.get("kind") or "") == kind]
                    if not rows:
                        continue
                    try:
                        ref_check = self.deepseek.select_session_entity_reference(
                            user_text=text, records=rows, expected_kind=kind, requested_identity="", ui_language=ui_language,
                        )
                    except Exception:
                        ref_check = {"reference_mode": "none"}
                    if str(ref_check.get("reference_mode") or "none") in {"focus", "active", "specific"}:
                        reusable_kinds.append(kind)
                if len(reusable_kinds) == 1:
                    summary = (
                        f"A verified {reusable_kinds[0]} session reference already resolves this follow-up. "
                        "Do not ask the user to reconfirm it; call reuse_session_entity instead."
                    )
                    steps.append(HarnessTraceStep(turn=turn, action_kind="ask_user", status="rejected", summary=summary))
                    history.append({
                        "turn": turn, "action": action.model_dump(),
                        "result": {"status": "error", "summary": summary, "recoverable": True, "error_code": "unnecessary_session_reconfirmation"},
                    })
                    continue
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
                )

            if action.kind == "return_result":
                latest_success = next((
                    entry for entry in reversed(history)
                    if isinstance(entry, dict) and str((entry.get("result") or {}).get("status") or "") == "ok"
                ), None)
                latest_payload = ((latest_success or {}).get("result") or {}).get("payload") if latest_success else {}
                if isinstance(latest_payload, dict) and latest_payload.get("workflow_incomplete"):
                    required = str(latest_payload.get("required_next_tool") or "build_research_workspace")
                    summary = (
                        f"The current factual relation lookup is an intermediate step in the integrated research workflow. "
                        f"Continue with {required} on the same verified target; return_result is allowed here only when the user explicitly requested evidence-only output."
                    )
                    steps.append(HarnessTraceStep(
                        turn=turn, action_kind="return_result", status="rejected", summary=summary,
                    ))
                    history.append({
                        "turn": turn, "action": action.model_dump(),
                        "result": {"status": "error", "summary": summary, "recoverable": True, "error_code": "integrated_research_incomplete"},
                    })
                    continue
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
                )
                self.sessions.remember_resolution(session_id, output)
                return output

            assert action.tool is not None
            signature = self._signature(str(action.tool), action.args)
            if signature in seen_calls:
                repeated_rejections += 1
                repeated = ToolResult(
                    tool=action.tool,
                    status="error",
                    summary="This identical tool call has already been executed. Change the plan or arguments.",
                    recoverable=True,
                    error_code="repeated_tool_call",
                )
                history.append({"turn": turn, "action": action.model_dump(), "result": repeated.model_view()})
                steps.append(HarnessTraceStep(
                    turn=turn,
                    action_kind="tool",
                    tool=str(action.tool),
                    status="rejected",
                    summary=repeated.summary,
                ))
                if repeated_rejections >= 2:
                    raise AppError(
                        "agent_repeated_tool_call",
                        "智能体连续重复了无效操作，请重新描述目标后再试。",
                        502,
                    )
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
                    session_facts_used=session_facts_used,
                )
                self.sessions.remember_resolution(session_id, output)
                return output

        raise AppError(
            "agent_turn_limit",
            "智能体在本轮内没有形成可执行结果。请补充目标或约束后再试。",
            502,
        )
