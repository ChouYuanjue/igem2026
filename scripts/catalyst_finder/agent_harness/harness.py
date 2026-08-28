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
            "version": "catalyst-model-led-agent-v4",
            "turn_count": len(steps),
            "fallback": False,
            "session_facts_used": session_facts_used,
            "steps": [step.model_dump() for step in steps],
        }
        provenance = dict(output.get("llm_provenance") or self.deepseek.provenance())
        provenance["used_for"] = provenance.get("used_for") or mode
        output["llm_provenance"] = provenance
        return output

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

        for index, scope_snapshot in enumerate(session_facts.get("verified_protein_scopes") or []):
            if index >= 4 or not isinstance(scope_snapshot, dict):
                continue
            kind = str(scope_snapshot.get("kind") or "").strip()
            scope_id = str(scope_snapshot.get("id") or scope_snapshot.get("family_id") or scope_snapshot.get("scope_id") or "").strip()
            if kind not in {"family", "functional_class"} or not scope_id:
                continue
            ref = f"session_protein_scope_group_{index + 1}"
            if kind == "family":
                run_ctx.protein_refs[ref] = {
                    "kind": "family",
                    "family_id": str(scope_snapshot.get("family_id") or scope_id),
                    "label": str(scope_snapshot.get("label") or scope_id),
                    "enzyme_spec": {
                        "raw_text": str(scope_snapshot.get("label") or scope_id),
                        "protein_terms": [str(scope_snapshot.get("label") or scope_id)],
                        "organism_terms": [],
                        "gene_terms": [],
                        "accession_terms": [],
                    },
                }
            else:
                run_ctx.protein_refs[ref] = {
                    "kind": "functional_class",
                    "label": str(scope_snapshot.get("label") or scope_id),
                    "scope_id": str(scope_snapshot.get("scope_id") or scope_id),
                    "enzyme_spec": dict(scope_snapshot.get("enzyme_spec") or {}),
                }
            protein_refs.append({"ref": ref, "scope_kind": kind, "scope_id": scope_id, "label": str(scope_snapshot.get("label") or scope_id)})

        compound_refs: list[dict[str, str]] = []
        for index, compound_id in enumerate(session_facts.get("verified_compound_ids") or []):
            cid = str(compound_id or "").strip().upper()
            if not cid.startswith("CHEBI:") or index >= 4:
                continue
            ref = f"session_compound_{index + 1}"
            run_ctx.compound_refs[ref] = {"chebi_id": cid, "name": cid, "smiles": ""}
            compound_refs.append({"ref": ref, "chebi_id": cid})

        if reaction_refs or protein_refs or compound_refs:
            enriched["current_run_refs"] = {
                "usage_rule": (
                    "For any tool argument whose name ends with _ref, copy ONLY a value from a current_run_refs item.ref "
                    "or from a tool result's returned ref. Database identifiers such as RHEA:..., UniProt IDs, CHEBI:..., "
                    "PFxxxxx, CLASS-..., scope_id, protein_id, reaction_id or chebi_id are identities, not tool refs."
                ),
                "reaction_refs": reaction_refs,
                "protein_scope_refs": protein_refs,
                "compound_refs": compound_refs,
            }
        return enriched

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
        session_facts_used = bool(session_facts)
        run_ctx = HarnessRunContext(
            ui_language=ui_language,
            conversation_context=context,
        )
        controller_session_facts = self._seed_session_refs(session_facts, run_ctx)
        capability_manifest = public_capabilities()

        for turn in range(1, self.max_turns + 1):
            action = self.deepseek.next_harness_action(
                user_text=text,
                session_facts=controller_session_facts,
                tool_catalog=self.tools.catalog(),
                capability_manifest=capability_manifest,
                history=history,
                ui_language=ui_language,
            )

            if action.kind == "respond":
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
