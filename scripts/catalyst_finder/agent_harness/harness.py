from __future__ import annotations

import json
import re
from copy import deepcopy
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

    @staticmethod
    def _explicit_identifiers(text: str) -> set[str]:
        value = str(text or "")
        found: set[str] = set()
        patterns = [
            r"(?i)\bRHEA:\d+", r"(?i)\bCHEBI:\d+", r"(?i)\bMED:\d+",
            r"(?i)\bPMID\s*[:#]?\s*\d+", r"(?i)\bPMC\d+", r"(?i)\bPF\d{5}",
            r"(?i)\b10\.\d{4,9}/[^\s<>()\[\]{}，。；;]+",
            r"(?i)\b(?:UniProt(?:KB)?\s*[:#]?\s*)?([OPQ][0-9][A-Z0-9]{3}[0-9]|A0A[A-Z0-9]{7})\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, value):
                token = match.group(0).strip().rstrip(".,;:，。；")
                if re.match(r"(?i)^PMID", token):
                    digits = re.search(r"\d+", token)
                    if digits:
                        found.add(f"MED:{digits.group(0)}")
                elif re.match(r"(?i)^UniProt", token):
                    acc = re.search(r"([OPQ][0-9][A-Z0-9]{3}[0-9]|A0A[A-Z0-9]{7})", token, re.I)
                    if acc:
                        found.add(acc.group(1).upper())
                else:
                    found.add(token.upper() if token.upper().startswith(("RHEA:", "CHEBI:", "MED:", "PMC", "PF")) else token)
        return found

    @staticmethod
    def _evidence_identifiers(evidence_history: list[dict[str, Any]]) -> set[str]:
        found: set[str] = set()
        def visit(value: Any) -> None:
            if isinstance(value, dict):
                source = str(value.get("source") or "").strip().upper()
                raw_id = str(value.get("id") or "").strip()
                if raw_id:
                    found.add(raw_id)
                    found.add(raw_id.upper())
                    if source == "MED" and raw_id.isdigit():
                        found.add(f"MED:{raw_id}")
                pmid = str(value.get("pmid") or "").strip()
                if pmid.isdigit():
                    found.add(f"MED:{pmid}")
                for key in ("pmcid", "doi", "rhea_id", "chebi_id", "candidate_id", "recommended_id", "accession", "canonical_accession", "family_id"):
                    token = str(value.get(key) or "").strip()
                    if token:
                        found.add(token)
                        found.add(token.upper())
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(evidence_history)
        return found

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
        # Full verified result snapshots are kept separately from the compact controller
        # history. This is the run-scoped evidence ledger used by grounded synthesis so
        # later tool calls cannot accidentally erase evidence produced earlier in the run.
        evidence_history: list[dict[str, Any]] = []
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
                has_scientific_tool_attempt = any(
                    isinstance(entry, dict)
                    and str((entry.get("action") or {}).get("kind") or "") == "tool"
                    for entry in history
                )
                if has_scientific_tool_attempt:
                    summary = (
                        "A scientific tool has already been attempted in this run. Ordinary free-form response is no longer allowed; "
                        "continue with scientific tools, ask one minimal clarification, use grounded synthesis to explain verified evidence/tool limitations, or return a verified structured result."
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

            if action.kind == "synthesize":
                tool_attempts = [
                    entry for entry in history
                    if isinstance(entry, dict) and str((entry.get("action") or {}).get("kind") or "") == "tool"
                ]
                if not tool_attempts:
                    summary = "Grounded synthesis requires at least one scientific tool attempt in this run."
                    steps.append(HarnessTraceStep(turn=turn, action_kind="synthesize", status="rejected", summary=summary))
                    history.append({
                        "turn": turn, "action": action.model_dump(),
                        "result": {"status": "error", "summary": summary, "recoverable": True, "error_code": "synthesis_without_evidence"},
                    })
                    continue
                explicit_ids = self._explicit_identifiers(text)
                if explicit_ids:
                    present_ids = self._evidence_identifiers(evidence_history)
                    attempted_ids = self._evidence_identifiers(tool_attempts)
                    for attempted in tool_attempts:
                        args_payload = (attempted.get("action") or {}).get("args") if isinstance(attempted, dict) else {}
                        attempted_ids.update(self._explicit_identifiers(json.dumps(args_payload or {}, ensure_ascii=False)))
                    missing_ids = sorted(
                        identifier for identifier in explicit_ids
                        if identifier not in present_ids and identifier.upper() not in present_ids
                        and identifier not in attempted_ids and identifier.upper() not in attempted_ids
                    )
                    if missing_ids:
                        summary = (
                            "Synthesis is premature because explicitly requested identifier(s) have not yet produced verified evidence: "
                            + ", ".join(missing_ids[:8])
                            + ". Resolve or inspect the missing requested entities before synthesizing."
                        )
                        steps.append(HarnessTraceStep(turn=turn, action_kind="synthesize", status="rejected", summary=summary))
                        history.append({
                            "turn": turn, "action": action.model_dump(),
                            "result": {"status": "error", "summary": summary, "recoverable": True, "error_code": "synthesis_missing_explicit_evidence", "payload": {"missing_identifiers": missing_ids[:8]}},
                        })
                        continue
                readiness = self.deepseek.validate_synthesis_readiness(
                    user_text=text, tool_history=history, verified_evidence=evidence_history, ui_language=ui_language,
                )
                if not bool(readiness.get("ready", True)):
                    missing = [str(value).strip() for value in readiness.get("missing_requirements") or [] if str(value).strip()]
                    reason = str(readiness.get("reason") or "").strip()
                    summary = "Synthesis is premature; more requested evidence is still obtainable."
                    if reason:
                        summary += f" {reason}"
                    if missing:
                        summary += " Missing: " + "; ".join(missing[:6])
                    steps.append(HarnessTraceStep(turn=turn, action_kind="synthesize", status="rejected", summary=summary[:700]))
                    history.append({
                        "turn": turn, "action": action.model_dump(),
                        "result": {"status": "error", "summary": summary[:1200], "recoverable": True, "error_code": "synthesis_not_ready", "payload": {"missing_requirements": missing[:6]}},
                    })
                    continue
                synthesized = self.deepseek.synthesize_grounded_answer(
                    user_text=text,
                    tool_history=history,
                    verified_evidence=evidence_history,
                    current_result=run_ctx.terminal_resolution or {},
                    ui_language=ui_language,
                )
                answer = str(synthesized.get("answer") or "").strip()
                if run_ctx.terminal_resolution is not None:
                    resolution = dict(run_ctx.terminal_resolution)
                else:
                    resolution = self._conversation_payload(answer, clarification=False)
                resolution["assistant_response"] = answer
                resolution["response_type"] = "grounded_synthesis"
                resolution["summary"] = answer[:800]
                resolution["grounding"] = {
                    "evidence_ids": list(synthesized.get("evidence_ids") or []),
                    "limitations": list(synthesized.get("limitations") or []),
                    "source": "verified_tool_history",
                }
                immediate = resolution.get("immediate_result") if isinstance(resolution.get("immediate_result"), dict) else None
                if immediate is not None and str(immediate.get("answer_mode") or "") == "entity_comparison":
                    immediate = dict(immediate)
                    immediate["analysis"] = answer
                    immediate["evidence_ids"] = list(synthesized.get("evidence_ids") or [])
                    immediate["limitations"] = list(synthesized.get("limitations") or [])
                    resolution["immediate_result"] = immediate
                steps.append(HarnessTraceStep(
                    turn=turn, action_kind="synthesize", status="ok",
                    summary="Composed a scientific answer strictly from verified tool evidence in this run.",
                ))
                output = self._decorate(
                    resolution, steps=steps, session_facts_used=session_facts_used,
                    mode="model_led_scientific_harness+grounded_synthesis",
                )
                self.sessions.remember_resolution(session_id, output)
                return output

            if action.kind == "return_result":
                latest_success = next((
                    entry for entry in reversed(history)
                    if isinstance(entry, dict) and str((entry.get("result") or {}).get("status") or "") == "ok"
                ), None)
                latest_payload = ((latest_success or {}).get("result") or {}).get("payload") if latest_success else {}
                if isinstance(latest_payload, dict) and latest_payload.get("workflow_incomplete"):
                    required = str(latest_payload.get("required_next_action") or latest_payload.get("required_next_tool") or "the required next step")
                    summary = (
                        f"The current verified result is explicitly marked incomplete. Continue with {required}; "
                        "do not return an intermediate result as the completed answer."
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
            previous_resolution = run_ctx.terminal_resolution
            result = self.tools.execute(action.tool, action.args, run_ctx)
            if result.status == "ok":
                evidence_entry: dict[str, Any] = {
                    "turn": turn,
                    "tool": str(action.tool),
                    "payload": deepcopy(result.payload),
                }
                if run_ctx.terminal_resolution is not None and run_ctx.terminal_resolution is not previous_resolution:
                    evidence_entry["result"] = deepcopy(run_ctx.terminal_resolution)
                evidence_history.append(evidence_entry)
                # Bound pathological tool chains without dropping the newest evidence.
                del evidence_history[:-8]
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
