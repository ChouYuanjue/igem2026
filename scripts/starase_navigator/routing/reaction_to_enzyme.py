from __future__ import annotations

import os
import inspect
import re
from typing import Any, Callable, Literal
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from projects.active.bridge.core.candidate_universes import DEFAULT_CANDIDATE_UNIVERSE
from projects.active.bridge.core.routing import resolve_route
from projects.active.bridge.core.taxonomy_scope import validate_seed_scope

SUPPORTED_TOP_K = {3, 5, 10, 20}
SUPPORTED_TAXONOMY = {"all", "eukaryote", "prokaryote"}
SUPPORTED_SEED_MODES = {"none", "explicit", "catalog_known"}
SUPPORTED_HOMOLOGY_POLICIES = {"allow", "cross_cluster"}
SUPPORTED_KNOWN_ASSOCIATION_POLICIES = {"separate_known", "rank_with_known", "known_only", "exclude_known"}
SUPPORTED_RETRIEVAL_SCOPES = {"broad", "application_domain"}
SUPPORTED_ANALYSIS_DEPTHS = {"standard", "deep"}
DEFAULT_PLAN = {
    "top_k": 10,
    "enzyme_taxonomy_scope": "all",
    "known_enzyme_ids": [],
    "seed_mode": "none",
    "seed_source": "none",
    "homology_policy": "allow",
    "known_association_policy": "separate_known",
    "retrieval_scope": "broad",
    "analysis_depth": "standard",
    "observation_mode": "standard",
    "candidate_universe": DEFAULT_CANDIDATE_UNIVERSE,
    "candidate_universe_source": "semantic_scope_default",
}

class RouteState(TypedDict, total=False):
    user_text: str
    reaction_equation: str
    route_mode: str
    is_current: bool
    orientation: str
    explicit_known_ids: list[str]
    confirmed_known_ids: list[str]
    catalog_known_ids: list[str]
    conversation_context: dict[str, Any]
    base_plan: dict[str, Any]
    ai_proposal: dict[str, Any]
    proposal_error: str
    proposal_error_code: str
    proposal_error_detail: str
    plan: dict[str, Any]


class RoutePlanner:
    """Constrained reaction-to-enzyme route planner.

    The LLM can only propose intent-level knobs. LangGraph applies deterministic
    constraints before the production router sees anything. In particular:

    * a user-named seed must be an ID that really exists in the deployed protein
      universe and appears explicitly in the user's text;
    * catalog records stay evidence only; only confirmed novel relations may
      update query-local episodic memory; training relations belong to long-term memory;
    * user-supplied positive enzymes are accepted only after identity/confirmation
      validation and are merged, deduplicated, with database positives;
    * "near homolog" is not taxonomy and is not an arbitrary embedding cutoff.
      The planner maps remote-family intent to a separate cross-cluster novelty
      overlay, whose runtime implementation uses the repository's MMseqs2 50%
      identity / 80% coverage family boundary;
    * if AI routing fails, the graph deterministically falls back to Top-10
      with long-term memory and only genuinely confirmed episodic support.
    """

    def __init__(
        self,
        *,
        proposal_fn: Callable[..., dict[str, Any]],
        protein_ids: list[str] | set[str],
    ) -> None:
        self.proposal_fn = proposal_fn
        self._protein_id_lookup = {str(value).casefold(): str(value) for value in protein_ids}
        builder = StateGraph(RouteState)
        builder.add_node("defaults", self._defaults)
        builder.add_node("ai_proposal", self._ai_proposal)
        builder.add_node("guardrails", self._guardrails)
        builder.add_edge(START, "defaults")
        builder.add_conditional_edges(
            "defaults",
            self._after_defaults,
            {"ai_proposal": "ai_proposal", "guardrails": "guardrails"},
        )
        builder.add_edge("ai_proposal", "guardrails")
        builder.add_edge("guardrails", END)
        self.graph = builder.compile()

    def explicit_protein_ids(self, text: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9_.:-]{3,64}", str(text or ""))
        result: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            canonical = self._protein_id_lookup.get(token.casefold())
            if canonical and canonical not in seen:
                seen.add(canonical)
                result.append(canonical)
        return result

    def plan(
        self,
        *,
        user_text: str,
        reaction_equation: str,
        route_mode: str,
        is_current: bool,
        orientation: str,
        known_association_ids: list[str] | None = None,
        confirmed_known_ids: list[str] | None = None,
        conversation_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        mode = "default" if route_mode == "default" else "intelligent"
        catalog_known = []
        seen: set[str] = set()
        for value in known_association_ids or []:
            canonical = self._protein_id_lookup.get(str(value).casefold())
            if canonical and canonical not in seen:
                seen.add(canonical)
                catalog_known.append(canonical)
        confirmed = list(dict.fromkeys(str(value).strip() for value in (confirmed_known_ids or []) if str(value).strip()))
        state = self.graph.invoke({
            "user_text": str(user_text or ""),
            "reaction_equation": str(reaction_equation or ""),
            "route_mode": mode,
            "is_current": bool(is_current),
            "orientation": "reverse" if orientation == "reverse" else "forward",
            "explicit_known_ids": self.explicit_protein_ids(user_text),
            "confirmed_known_ids": confirmed,
            "catalog_known_ids": catalog_known,
            "conversation_context": dict(conversation_context or {}),
        })
        return dict(state["plan"])

    def plan_from_proposal(
        self,
        *,
        proposal: dict[str, Any],
        user_text: str,
        reaction_equation: str,
        is_current: bool,
        orientation: str,
        known_association_ids: list[str] | None = None,
        confirmed_known_ids: list[str] | None = None,
        conversation_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate a primary-agent retrieval plan without a second LLM proposal."""
        catalog_known=[]
        seen:set[str]=set()
        for value in known_association_ids or []:
            canonical=self._protein_id_lookup.get(str(value).casefold())
            if canonical and canonical not in seen:
                seen.add(canonical)
                catalog_known.append(canonical)
        confirmed=list(dict.fromkeys(
            str(value).strip() for value in (confirmed_known_ids or []) if str(value).strip()
        ))
        state: RouteState={
            "user_text":str(user_text or ""),
            "reaction_equation":str(reaction_equation or ""),
            "route_mode":"intelligent",
            "is_current":bool(is_current),
            "orientation":"reverse" if orientation=="reverse" else "forward",
            "explicit_known_ids":self.explicit_protein_ids(user_text),
            "confirmed_known_ids":confirmed,
            "catalog_known_ids":catalog_known,
            "conversation_context":dict(conversation_context or {}),
        }
        state.update(self._defaults(state))
        state["ai_proposal"]={
            **dict(proposal or {}),
            "_semantic_source":"primary_agent",
        }
        guarded=self._guardrails(state)
        return dict(guarded["plan"])

    @staticmethod
    def _defaults(state: RouteState) -> dict[str, Any]:
        confirmed = list(state.get("confirmed_known_ids") or [])
        has_confirmed = bool(confirmed)
        application_default = bool(state.get("is_current")) and state.get("orientation") != "reverse"
        return {
            "base_plan": {
                **DEFAULT_PLAN,
                "retrieval_scope": "application_domain" if application_default else "broad",
                # Application-domain intent selects richer evidence and analysis,
                # not a smaller ranking universe. BRIDGE always owns the full broad pool.
                "candidate_universe": DEFAULT_CANDIDATE_UNIVERSE,
                "candidate_universe_source": (
                    "current_entity_application_default"
                    if application_default else "semantic_scope_default"
                ),
                "known_enzyme_ids": confirmed,
                "seed_mode": "explicit" if has_confirmed else "none",
                "seed_source": "user_confirmed" if has_confirmed else "none",
                "selected_by": "default",
                "reason": (
                    "BRIDGE 训练图长期记忆按模型门控参与排序；仅用户确认的新增催化关系可进入情景记忆。"
                    "数据库已记录关联与预测结果分开作为证据，均不自动更新情景记忆。"
                ),
                "warnings": [],
            }
        }

    @staticmethod
    def _after_defaults(state: RouteState) -> Literal["ai_proposal", "guardrails"]:
        return "ai_proposal" if state.get("route_mode") == "intelligent" else "guardrails"

    def _ai_proposal(self, state: RouteState) -> dict[str, Any]:
        try:
            authorized = list(dict.fromkeys(
                list(state.get("explicit_known_ids", [])) + list(state.get("confirmed_known_ids", []))
            ))
            args = [
                state.get("user_text", ""),
                state.get("reaction_equation", ""),
                authorized,
                len(state.get("catalog_known_ids", [])),
                list(state.get("catalog_known_ids", [])),
                dict(state.get("conversation_context") or {}),
            ]
            try:
                parameter_count = len(inspect.signature(self.proposal_fn).parameters)
            except (TypeError, ValueError):
                parameter_count = len(args)
            proposal = self.proposal_fn(*args[:parameter_count])
            if not isinstance(proposal, dict):
                raise TypeError("route proposal must be an object")
            return {"ai_proposal": proposal}
        except Exception as exc:
            code = str(getattr(exc, "code", "") or type(exc).__name__)[:120]
            detail = str(getattr(exc, "detail", "") or "").strip()[:800]
            return {
                "proposal_error": f"{type(exc).__name__}: {exc}",
                "proposal_error_code": code,
                "proposal_error_detail": detail,
            }

    @staticmethod
    def _int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _guardrails(self, state: RouteState) -> dict[str, Any]:
        plan = dict(state.get("base_plan") or DEFAULT_PLAN)
        plan["warnings"] = list(plan.get("warnings") or [])
        proposal = state.get("ai_proposal") if state.get("route_mode") == "intelligent" else None
        explicit_ids = list(state.get("explicit_known_ids", []))
        confirmed_ids = list(state.get("confirmed_known_ids", []))
        authorized_ids = list(dict.fromkeys(explicit_ids + confirmed_ids))
        catalog_known_ids = list(state.get("catalog_known_ids", []))
        user_text = str(state.get("user_text") or "")

        if state.get("proposal_error"):
            plan["warnings"].append("智能路由暂时不可用，已自动使用默认路线。")
            plan["fallback_reason"] = "ai_route_failed"
            plan["fallback_error_code"] = str(state.get("proposal_error_code") or "route_proposal_failed")
            if os.environ.get("STARASE_NAVIGATOR_DEBUG", "").strip() == "1" and state.get("proposal_error_detail"):
                plan["fallback_detail"] = str(state.get("proposal_error_detail"))[:800]
        elif isinstance(proposal, dict):
            semantic_source=str(proposal.get("_semantic_source") or "")
            semantic_proposal = semantic_source in {"deepseek","primary_agent"}
            top_k = self._int(proposal.get("top_k"), 10)
            if top_k not in SUPPORTED_TOP_K:
                top_k = 3 if top_k <= 3 else 5 if top_k <= 5 else 10 if top_k <= 10 else 20
                plan["warnings"].append("候选数量已映射到当前交互支持的 Top 3 / 5 / 10 / 20。")

            scope = str(proposal.get("enzyme_taxonomy_scope") or "all").strip().lower()
            if scope not in SUPPORTED_TAXONOMY:
                scope = "all"
                plan["warnings"].append("无法安全解释物种范围，已使用全部候选酶。")

            requested_seed_mode = str(proposal.get("seed_mode") or "").strip().lower()
            if requested_seed_mode not in SUPPORTED_SEED_MODES:
                requested_seed_mode = ""

            # Compatibility: seed_mode is an engine transport alias. Never
            # promote catalog records or an arbitrary model-mentioned ID to a
            # confirmed runtime episodic relation.
            proposed_ids = proposal.get("known_enzyme_ids") or []
            if not isinstance(proposed_ids, list):
                proposed_ids = []
            if requested_seed_mode == "catalog_known":
                plan["warnings"].append(
                    "存量数据库关联仅作为证据；情景记忆必须由用户确认的新增关系激活。"
                )
                requested_seed_mode = "none"
            if not requested_seed_mode:
                requested_seed_mode = "explicit" if confirmed_ids else "none"

            known_ids: list[str] = []
            seed_mode = "none"
            seed_source = "none"
            if requested_seed_mode == "explicit":
                verified = {value.casefold(): value for value in confirmed_ids}
                proposed = proposed_ids or confirmed_ids
                for value in proposed:
                    canonical = verified.get(str(value).casefold())
                    if canonical and canonical not in known_ids:
                        known_ids.append(canonical)
                    elif str(value).strip():
                        plan["warnings"].append(
                            "未确认的酶关系无法进入情景记忆；请先核验该关系。"
                        )
                if known_ids:
                    seed_mode = "explicit"
                    seed_source = "user_confirmed"
                else:
                    plan["warnings"].append(
                        "没有已确认的新增催化关系，本次仅使用训练图长期记忆。"
                    )
            elif proposed_ids:
                plan["warnings"].append(
                    "没有启用情景记忆；未核验的候选酶 ID 不参与当前查询更新。"
                )

            if known_ids and scope != "all":
                # User/model-chosen candidate scope is authoritative. Database
                # positives are optional supporting seeds: an incompatible
                # positive must not silently widen the actual search space.
                compatible_seeds: list[str] = []
                incompatible_seeds: list[str] = []
                for seed_id in known_ids:
                    try:
                        validate_seed_scope([seed_id], scope)
                    except (ValueError, OSError):
                        incompatible_seeds.append(seed_id)
                    else:
                        compatible_seeds.append(seed_id)
                if incompatible_seeds:
                    plan["excluded_seed_count"] = len(incompatible_seeds)
                    plan["warnings"].append(
                        f"已排除 {len(incompatible_seeds)} 个与指定物种范围不兼容或未分类的参考正例；候选池范围保持不变。"
                    )
                known_ids = compatible_seeds
                if not known_ids:
                    seed_mode = "none"
                    seed_source = "no_compatible_seed_in_requested_scope"

            homology_policy = str(proposal.get("homology_policy") or "allow").strip().lower()
            if proposal.get("exclude_near_source"):
                homology_policy = "cross_cluster"
            if homology_policy not in SUPPORTED_HOMOLOGY_POLICIES:
                homology_policy = "allow"
            if (
                homology_policy == "cross_cluster"
                and not semantic_proposal
            ):
                homology_policy = "allow"
                plan["warnings"].append("远缘筛选属于语义策略；未经过 DeepSeek 语义解析的提议不能启用该约束，已保留默认同源策略。")

            homology_anchor_ids: list[str] = []
            homology_anchor_source = "none"
            homology_filter_requested = homology_policy == "cross_cluster"
            homology_filter_applied = False
            if homology_filter_requested:
                if known_ids:
                    homology_anchor_ids = list(known_ids)
                    homology_anchor_source = seed_source
                    homology_filter_applied = True
                elif catalog_known_ids:
                    # This branch is reached only when the user explicitly disabled
                    # seed scoring but still requested a cross-cluster filter. Catalog
                    # positives remain valid anchors for defining the excluded families.
                    homology_anchor_ids = list(catalog_known_ids)
                    homology_anchor_source = "catalog_known_associations_filter_only"
                    homology_filter_applied = True
                    plan["warnings"].append("远缘筛选将以数据库已知阳性酶作为 50% identity cluster 排除锚点；本次用户明确关闭了 seed 打分。")
                else:
                    homology_policy = "allow"
                    plan["warnings"].append("检测到远缘发现意图，但该反应没有可用阳性锚点，无法定义“相对谁远缘”，因此未应用跨簇筛选。")

            association_policy = str(proposal.get("known_association_policy") or "separate_known").strip().lower()
            if association_policy not in SUPPORTED_KNOWN_ASSOCIATION_POLICIES:
                association_policy = "separate_known"
            if association_policy in {"rank_with_known", "known_only"}:
                # Retrospective and known-only scoring must not use the same
                # evaluated relations as runtime episodic updates.
                known_ids = []
                seed_mode = "none"
                seed_source = (
                    "mixed_ranking_forces_zero_shot"
                    if association_policy == "rank_with_known"
                    else "known_only_scoring_forces_zero_shot"
                )
            base_retrieval_scope = str(
                plan.get("retrieval_scope") or "broad"
            ).strip().lower()
            retrieval_scope = str(
                proposal.get("retrieval_scope") or base_retrieval_scope
            ).strip().lower()
            analysis_depth = str(proposal.get("analysis_depth") or "standard").strip().lower()
            if not semantic_proposal:
                # Intent belongs to the semantic planner. Deterministic code only
                # validates output and preserves the already verified entity scope.
                retrieval_scope = base_retrieval_scope
                analysis_depth = "standard"
            if retrieval_scope not in SUPPORTED_RETRIEVAL_SCOPES:
                retrieval_scope = base_retrieval_scope
                plan["warnings"].append("智能语义范围无效，已保留已验证实体的默认检索范围。")
            if analysis_depth not in SUPPORTED_ANALYSIS_DEPTHS:
                analysis_depth = "standard"
                plan["warnings"].append("智能分析深度无效，已使用常规观测预算。")
            # Scope controls evidence/analysis semantics only. Candidate
            # cardinality remains the deployment-wide BRIDGE universe so search-space
            # difficulty is not silently changed by application-domain routing.
            candidate_universe = DEFAULT_CANDIDATE_UNIVERSE
            candidate_universe_source = (
                f"{semantic_source}_semantic_scope" if semantic_proposal else "semantic_scope_default"
            )
            # DeepSeek semantic planner decides association scope. Keyword matching
            # cannot reliably resolve conversational follow-ups such as "只看潜在的"
            # or references to the previous result. The graph only validates that
            # the selected value is in the supported enum.

            plan.update({
                "top_k": top_k,
                "enzyme_taxonomy_scope": scope,
                "known_enzyme_ids": known_ids,
                "seed_mode": seed_mode,
                "seed_source": seed_source,
                "homology_policy": homology_policy,
                "homology_filter_requested": homology_filter_requested,
                "homology_filter_applied": homology_filter_applied,
                "homology_anchor_ids": homology_anchor_ids,
                "homology_anchor_source": homology_anchor_source,
                "known_association_policy": association_policy,
                "retrieval_scope": retrieval_scope,
                "analysis_depth": analysis_depth,
                "observation_mode": analysis_depth,
                "candidate_universe": candidate_universe,
                "candidate_universe_source": candidate_universe_source,
                "selected_by": "ai",
                "reason": str(proposal.get("reason") or "根据输入中的实验目标选择受支持的检索策略。").strip()[:300],
            })

        # Association scope is semantic policy. The guardrail validates model output
        # but never guesses user intent from keyword lists. Without a semantic proposal,
        # preserve the documented separate-known default.
        proposal_source=str(proposal.get("_semantic_source") or "") if isinstance(proposal,dict) else ""
        has_semantic_scope = (
            isinstance(proposal,dict)
            and proposal_source in {"deepseek","primary_agent"}
            and "known_association_policy" in proposal
        )
        if has_semantic_scope:
            plan["known_association_policy_source"] = f"{proposal_source}_semantic"
        else:
            if isinstance(proposal, dict) and str(proposal.get("known_association_policy") or "separate_known") != "separate_known":
                plan["warnings"].append("结果范围属于语义策略；未经过 DeepSeek 语义解析的提议不能改变默认的已知证据与新候选分层展示。")
            plan["known_association_policy"] = "separate_known"
            plan["known_association_policy_source"] = "default_fallback"

        if plan.get("known_association_policy") in {"rank_with_known", "known_only"}:
            policy = str(plan.get("known_association_policy"))
            plan["known_enzyme_ids"] = []
            plan["seed_mode"] = "none"
            plan["seed_source"] = (
                "mixed_ranking_forces_zero_shot"
                if policy == "rank_with_known"
                else "known_only_scoring_forces_zero_shot"
            )



        objective = {3: "top3", 5: "top10", 10: "top10", 20: "top20"}[int(plan.get("top_k", 10))]
        is_current = bool(state.get("is_current")) and state.get("orientation") != "reverse"
        route = resolve_route(
            direction="reaction_to_enzyme",
            objective=objective,
            is_current=is_current,
            has_seed=bool(plan.get("known_enzyme_ids")),
            enzyme_taxonomy_scope=str(plan.get("enzyme_taxonomy_scope") or "all"),
        )
        plan.update({
            "route_mode": state.get("route_mode", "intelligent"),
            "ranking_objective": objective,
            "planned_route_id": route.route_id,
            "scope": "current" if is_current else "external",
            "shot_mode": "few_shot" if plan.get("known_enzyme_ids") else "zero_shot",
            "memory_mode": "long_term_plus_episodic" if plan.get("known_enzyme_ids") else "long_term_only",
            "default_route": {
                "top_k": 10,
                "ranking_objective": "top10",
                "enzyme_taxonomy_scope": "all",
                "memory_mode": "long_term_plus_confirmed_episodic_if_provided",
                "homology_policy": "allow",
                "known_association_policy": "separate_known",
            },
            "constraints": {
                "top_k": [3, 5, 10, 20],
                "enzyme_taxonomy_scope": ["all", "eukaryote", "prokaryote"],
                "episodic_memory": ["only_verified_user_updates", "never_automatic_catalog", "training_relations_long_term_only"],
                "homology_policy": ["allow", "cross_cluster"],
                "known_association_policy": ["separate_known_default", "rank_with_known_explicit_zero_shot", "known_only_when_explicitly_requested", "exclude_known_when_explicitly_requested"],
                "retrieval_scope": sorted(SUPPORTED_RETRIEVAL_SCOPES),
                "analysis_depth": sorted(SUPPORTED_ANALYSIS_DEPTHS),
                "cross_cluster_definition": "MMseqs2 min identity 0.50, coverage 0.80",
            },
        })
        return {"plan": plan}
