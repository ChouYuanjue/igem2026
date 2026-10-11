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
    """Model-led, bounded scientific-agent loop over COMPASS capabilities.

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
                        "model_score": row.get("score"),
                        "active_experts": [
                            {"name": str(part.get("expert") or ""),
                             "contribution": part.get("contribution"),
                             "rank_change": part.get("rank_change")}
                            for part in (row.get("expert_contributions") or [])
                            if isinstance(part, dict) and part.get("active") is True
                        ][:8],
                    }
                    for row in candidates[:6]
                ],
                "note": str(immediate.get("note") or "")[:500],
            }
            artifacts.append(artifact)
        return artifacts

    @staticmethod
    def _structured_observations(
        evidence_history: list[dict[str, Any]] | None,
        final_resolution: dict[str, Any],
        *,
        display_result_turns: list[int] | None = None,
        user_text: str = "",
    ) -> list[dict[str, Any]]:
        """Render the controller's selected VERIFIED tool results, not its tool trace.

        Candidate searches are only eligible if their executed pool satisfies
        explicit user constraints. A later scoped database lookup can contribute
        known evidence to the SAME candidate result without replacing its ranking.
        If an older controller omits display_result_turns, use the last eligible
        candidate ranking, never an earlier unfiltered search.
        """
        snapshots: list[dict[str, Any]] = []
        seen: set[str] = set()
        for entry in list(evidence_history or []):
            if not isinstance(entry, dict):
                continue
            resolution = entry.get("result")
            if not isinstance(resolution, dict):
                continue
            immediate = resolution.get("immediate_result")
            if not isinstance(immediate, dict) or not immediate:
                continue
            if not any(immediate.get(k) for k in (
                "candidates", "entities", "routes", "steps", "comparison_rows",
                "source_panels", "model_lens", "opportunities",
            )) and not ((immediate.get("known_associations") or {}).get("count")
                       or (immediate.get("known_associations") or {}).get("items")):
                continue
            fingerprint = json.dumps(
                [resolution.get("direction"), immediate],
                ensure_ascii=False, sort_keys=True, default=str,
            )
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            snapshots.append({
                "turn": entry.get("turn"),
                "tool": str(entry.get("tool") or ""),
                "direction": str(resolution.get("direction") or ""),
                "result": deepcopy(immediate),
                "_retrieval_plan": dict(resolution.get("retrieval_plan") or {}),
            })
        if not snapshots:
            immediate = final_resolution.get("immediate_result")
            if isinstance(immediate, dict) and immediate:
                snapshots.append({
                    "turn": None,
                    "tool": str(final_resolution.get("operation") or ""),
                    "direction": str(final_resolution.get("direction") or ""),
                    "result": deepcopy(immediate),
                    "_retrieval_plan": dict(final_resolution.get("retrieval_plan") or {}),
                })

        if not snapshots:
            return []

        def predicted(row: dict[str, Any]) -> bool:
            immediate = row["result"]
            return bool(immediate.get("candidates") or (
                (immediate.get("model_lens") or {}).get("frontier")
            ))

        def target(row: dict[str, Any]) -> str:
            result = row["result"]
            return str(
                (result.get("reaction") or {}).get("rhea_id")
                or (result.get("protein") or {}).get("id")
                or (result.get("entity") or {}).get("id")
                or ""
            )

        ranked = [row for row in snapshots if predicted(row)]
        turns = list(dict.fromkeys(display_result_turns or []))
        chosen = [
            next(row for row in snapshots if row["turn"] == turn)
            for turn in turns
            if any(row["turn"] == turn for row in snapshots)
        ]
        if not chosen:
            # Compatibility path used when the controller/provider supplies no
            # explicit selection; the normal flow passes audited turn IDs.
            chosen = [snapshots[-1]]

        # Keep model-selected order, but place a same-target evidence-only
        # lookup next to its prediction for composition.
        visible: list[dict[str, Any]] = []
        for row in chosen:
            immediate = deepcopy(row["result"])
            is_known_lookup = row["tool"] == "lookup_relations" and not predicted(row)
            parent = next((
                preceding for preceding in visible
                if predicted(preceding) and preceding["direction"] == row["direction"]
                and target(preceding) and target(preceding) == target(row)
            ), None)
            if is_known_lookup and parent is not None:
                narrow = immediate.get("known_associations")
                if isinstance(narrow, dict):
                    prior = parent["result"].get("known_associations")
                    if isinstance(prior, dict):
                        parent["result"]["broader_known_associations"] = prior
                    parent["result"]["known_associations"] = deepcopy(narrow)
                    parent["result"]["recorded_observation_turn"] = row["turn"]
                continue
            visible.append({
                "turn": row["turn"],
                "tool": row["tool"],
                "direction": row["direction"],
                "result": immediate,
            })
        return visible[:8]

    @staticmethod
    def _presentation_review_manifest(
        evidence_history: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Small immutable facts about what each tool actually returned."""
        rows: list[dict[str, Any]] = []
        for entry in evidence_history or []:
            if not isinstance(entry, dict) or not isinstance(entry.get("turn"), int):
                continue
            resolution = entry.get("result")
            if not isinstance(resolution, dict):
                continue
            result = resolution.get("immediate_result")
            if not isinstance(result, dict) or not result:
                continue
            candidates = result.get("candidates")
            if not isinstance(candidates, list):
                candidates = []
            known = result.get("known_associations")
            if not isinstance(known, dict):
                known = {}
            ranking = result.get("ranking") or {}
            routing = result.get("routing") or {}
            query = result.get("reaction") or result.get("protein") or result.get("entity") or {}
            rows.append({
                "turn": entry["turn"],
                "tool": entry.get("tool"),
                "direction": resolution.get("direction"),
                "operation": resolution.get("operation"),
                "answer_mode": result.get("answer_mode"),
                "target": {
                    "id": query.get("rhea_id") or query.get("id") or query.get("uniprot_id"),
                    "description": str(query.get("equation") or query.get("name") or "")[:220],
                },
                "requested_retrieval_plan": dict(resolution.get("retrieval_plan") or {}),
                "executed_pool": {
                    "routing": {
                        key: routing.get(key)
                        for key in (
                            "enzyme_taxonomy_scope", "retrieval_scope",
                            "known_association_policy", "ranking_objective",
                            "candidate_universe", "analysis_depth",
                        ) if routing.get(key) is not None
                    },
                    "ranking": {
                        key: ranking.get(key)
                        for key in (
                            "candidate_universe", "candidate_universe_size",
                            "retrieval_scope", "ranking_objective",
                            "top_k", "result_scope", "score_source",
                        ) if ranking.get(key) is not None
                    },
                },
                "candidate_count": len(candidates),
                "sample_candidates": [
                    {
                        "id": candidate.get("candidate_id") or candidate.get("id"),
                        "species": candidate.get("species"),
                        "rank": candidate.get("rank"),
                        "known_association": candidate.get("known_association"),
                    }
                    for candidate in candidates[:4] if isinstance(candidate, dict)
                ],
                "known_association_count": known.get("count"),
                "known_sample": [
                    row.get("candidate_id")
                    for row in (known.get("items") or [])[:4]
                    if isinstance(row, dict)
                ],
            })
        return rows

    def _decorate(
        self,
        resolution: dict[str, Any],
        *,
        steps: list[HarnessTraceStep],
        session_facts_used: bool,
        evidence_history: list[dict[str, Any]] | None = None,
        mode: str = "model_led_scientific_harness",
        display_result_turns: list[int] | None = None,
        user_text: str = "",
    ) -> dict[str, Any]:
        output = dict(resolution)
        chosen_turns = list(display_result_turns or [])
        manifest = self._presentation_review_manifest(evidence_history)
        reviewer = getattr(self.deepseek, "review_result_presentation", None)
        if user_text and len(manifest) > 1 and callable(reviewer):
            try:
                reviewed = reviewer(
                    user_text=user_text, proposed_turns=chosen_turns,
                    observations=manifest,
                )
                if isinstance(reviewed, list):
                    chosen_turns = reviewed
            except Exception:
                # Provider failure cannot manufacture evidence or break delivery.
                pass
        structured = self._structured_observations(
            evidence_history, output,
            display_result_turns=chosen_turns,
        )
        if structured:
            output["structured_results"] = structured
            output["immediate_result"] = deepcopy(structured[0]["result"])
            output["direction"] = structured[0]["direction"]
            output["display_result_turns"] = [row["turn"] for row in structured if row["turn"] is not None]
            # Make session follow-ups reference the selected result, not an
            # incidental final lookup from the tool chain.
            primary = next((
                entry.get("result") for entry in (evidence_history or [])
                if entry.get("turn") == structured[0]["turn"]
                and isinstance(entry.get("result"), dict)
            ), None)
            if primary:
                for key in (
                    "operation", "reaction_resolution", "protein_resolution",
                    "retrieval_plan", "target_conditions", "reaction_constraints",
                    "positive_enzyme_resolutions", "positive_reaction_resolutions",
                ):
                    if key in primary:
                        output[key] = deepcopy(primary[key])
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

    def _synthesize_verified_resolution(
        self,
        *,
        user_text: str,
        conversation_history: list[dict[str, str]],
        terminal_resolution: dict[str, Any],
        evidence_history: list[dict[str, Any]],
        execution_history: list[dict[str, Any]],
        ui_language: str,
    ) -> str:
        synthesis = getattr(self.deepseek, "synthesize_grounded_answer", None)
        if not callable(synthesis):
            return ""
        try:
            synthesized = synthesis(
                user_text=user_text,
                conversation_history=conversation_history,
                terminal_resolution=deepcopy(terminal_resolution),
                verified_evidence=deepcopy(evidence_history),
                execution_history=deepcopy(execution_history),
                ui_language=ui_language,
            )
            if isinstance(synthesized, dict):
                return str(synthesized.get("answer") or "").strip()
            return str(synthesized or "").strip()
        except Exception:
            return ""

    def _concise_scientific_narration(
        self,
        original: str,
        *,
        user_text: str,
        conversation_history: list[dict[str, str]],
        terminal_resolution: dict[str, Any],
        evidence_history: list[dict[str, Any]],
        execution_history: list[dict[str, Any]],
        ui_language: str,
    ) -> str:
        """Ask the same scientific model to condense an unusually long final reply.

        Normal responses pass through untouched. This protects the chat from
        unrequested prose reproductions of candidate lists, without altering
        the verified structured result or the model's tool autonomy.
        """
        original = str(original or "").strip()
        limit = 480 if str(ui_language).lower().startswith("zh") else 850
        if len(original) <= limit or not evidence_history:
            return original
        compact = self._synthesize_verified_resolution(
            user_text=user_text,
            conversation_history=conversation_history,
            terminal_resolution=terminal_resolution,
            evidence_history=evidence_history,
            execution_history=execution_history,
            ui_language=ui_language,
        )
        return compact if compact and len(compact) < len(original) else original

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
                    narration = self._concise_scientific_narration(
                        action.message, user_text=text, conversation_history=conversation_history,
                        terminal_resolution=resolution, evidence_history=evidence_history,
                        execution_history=history, ui_language=ui_language,
                    )
                    resolution["assistant_response"]=narration
                    resolution["response_type"]="message"
                    resolution["summary"]=narration[:800]
                else:
                    resolution=self._conversation_payload(action.message.strip(), clarification=False)
                output = self._decorate(
                    resolution,
                    steps=steps,
                    session_facts_used=session_facts_used,
                    evidence_history=evidence_history,
                    display_result_turns=action.display_result_turns,
                    user_text=text,
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
                if run_ctx.terminal_resolution is not None:
                    resolution = deepcopy(run_ctx.terminal_resolution)
                    resolution["assistant_response"] = action.question.strip()
                    resolution["response_type"] = "verification"
                    resolution["needs_user_input"] = True
                    resolution["summary"] = action.question.strip()[:800]
                    output = self._decorate(
                        resolution,
                        steps=steps,
                        session_facts_used=session_facts_used,
                        evidence_history=evidence_history,
                    )
                    self.sessions.remember_resolution(session_id, output)
                    return output
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
                    display_result_turns=action.display_result_turns,
                    user_text=text,
                )
                if action.message.strip():
                    narration = self._concise_scientific_narration(
                        action.message, user_text=text, conversation_history=conversation_history,
                        terminal_resolution=output, evidence_history=evidence_history,
                        execution_history=history, ui_language=ui_language,
                    )
                    output["assistant_response"] = narration
                    output["summary"] = narration[:800]
                self.sessions.remember_resolution(session_id, output)
                return output

            assert action.tool is not None
            if action.tool == "candidate_search":
                plan_reviewer = getattr(self.deepseek, "review_candidate_search_plan", None)
                if callable(plan_reviewer):
                    try:
                        previous_plans = [
                            {
                                "turn": row.get("turn"),
                                "retrieval_plan": (
                                    (row.get("result") or {}).get("retrieval_plan") or {}
                                ),
                            }
                            for row in evidence_history
                            if row.get("tool") == "candidate_search"
                        ]
                        audited = plan_reviewer(
                            user_text=text, proposed_args=deepcopy(action.args),
                            conversation_history=conversation_history,
                            previous_searches=previous_plans,
                        )
                        if isinstance(audited, dict):
                            from scripts.starase_navigator.agent_harness.contracts import TOOL_ARG_MODELS
                            validated = TOOL_ARG_MODELS["candidate_search"].model_validate(audited)
                            action.args = validated.model_dump(exclude_unset=True)
                    except Exception:
                        # The controller's typed tool contract still applies if
                        # the independent scientific review is unavailable.
                        pass
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
            session_cached = self.sessions.tool_observation(session_id, signature)
            cache_source = ""
            if session_cached is not None:
                result = ToolResult.model_validate(session_cached["result"])
                cached_terminal = session_cached.get("terminal_resolution")
                if isinstance(cached_terminal, dict):
                    run_ctx.terminal_resolution = deepcopy(cached_terminal)
                cache_source = "session_verified"
            else:
                result = self.tools.execute(action.tool, action.args, run_ctx)
                if result.status == "ok":
                    self.sessions.remember_tool_observation(
                        session_id,
                        signature,
                        tool=str(action.tool),
                        result=result.model_view(),
                        terminal_resolution=run_ctx.terminal_resolution,
                    )
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
            if cache_source:
                history_entry["cache_source"] = cache_source
            if run_ctx.terminal_resolution is not None and run_ctx.terminal_resolution is not previous_resolution:
                history_entry["verified_result"]=deepcopy(run_ctx.terminal_resolution)
            history.append(history_entry)
            steps.append(HarnessTraceStep(
                turn=turn,
                action_kind="tool",
                tool=str(action.tool),
                status=("cached" if cache_source else result.status),
                summary=(
                    "Reused an exact verified tool observation from this conversation."
                    if cache_source
                    else result.summary[:700]
                ),
            ))
            if result.terminal and result.status == "ok" and run_ctx.terminal_resolution is not None:
                resolution = deepcopy(run_ctx.terminal_resolution)
                narration = str(resolution.get("assistant_response") or "").strip()
                if not narration:
                    narration = self._synthesize_verified_resolution(
                        user_text=text,
                        conversation_history=conversation_history,
                        terminal_resolution=resolution,
                        evidence_history=evidence_history,
                        execution_history=history,
                        ui_language=ui_language,
                    )
                if narration:
                    resolution["assistant_response"] = narration
                    resolution["summary"] = narration[:800]
                    steps.append(HarnessTraceStep(
                        turn=turn + 1,
                        action_kind="respond",
                        status="synthesized",
                        summary=(
                            "Explained the verified terminal state without starting another tool workflow."
                        ),
                    ))
                else:
                    fallback_message = str(result.summary or "").strip()
                    if fallback_message:
                        resolution["assistant_response"] = fallback_message
                        resolution["summary"] = fallback_message[:800]
                output = self._decorate(
                    resolution,
                    steps=steps,
                    session_facts_used=session_facts_used,
                    evidence_history=evidence_history,
                    mode=(
                        "model_led_scientific_harness_verification"
                        if bool(resolution.get("needs_user_input"))
                        else "model_led_scientific_harness"
                    ),
                )
                self.sessions.remember_resolution(session_id, output)
                return output

        if run_ctx.terminal_resolution is not None:
            synthesis_answer = self._synthesize_verified_resolution(
                user_text=text,
                conversation_history=conversation_history,
                terminal_resolution=run_ctx.terminal_resolution,
                evidence_history=evidence_history,
                execution_history=history,
                ui_language=ui_language,
            )
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
                    user_text=text,
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
                user_text=text,
            )
            self.sessions.remember_resolution(session_id, output)
            return output
        raise AppError(
            "agent_turn_limit",
            "智能体在本轮内没有形成可执行结果。请补充目标或约束后再试。",
            502,
        )
