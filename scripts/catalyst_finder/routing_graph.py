from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Literal
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from projects.active.terpene_screening.core.candidate_universes import (
    DEFAULT_CANDIDATE_UNIVERSE,
    SUPPORTED_CANDIDATE_UNIVERSES,
)
from projects.active.terpene_screening.core.routing import resolve_route
from projects.active.terpene_screening.core.taxonomy_scope import validate_seed_scope

SUPPORTED_TOP_K = {3, 5, 10, 20}
SUPPORTED_TAXONOMY = {"all", "eukaryote", "prokaryote"}
SUPPORTED_SEED_MODES = {"none", "explicit", "catalog_known"}
SUPPORTED_HOMOLOGY_POLICIES = {"allow", "cross_cluster"}
SUPPORTED_KNOWN_ASSOCIATION_POLICIES = {"allow_known", "known_only", "exclude_known"}
DEFAULT_PLAN = {
    "top_k": 10,
    "enzyme_taxonomy_scope": "all",
    "known_enzyme_ids": [],
    "seed_mode": "none",
    "seed_source": "none",
    "homology_policy": "allow",
    "known_association_policy": "allow_known",
    "candidate_universe": DEFAULT_CANDIDATE_UNIVERSE,
    "candidate_universe_source": "default",
}

KNOWN_POSITIVE_INTENT = re.compile(
    r"(已知.{0,5}(阳性|催化|酶)|阳性酶|已知酶|known\s+(positive|catalyst)|few[- ]?shot|seed)",
    re.IGNORECASE,
)
REMOTE_INTENT = re.compile(
    r"(远缘|远源|跨簇|跨.{0,4}家族|不同.{0,4}家族|排除.{0,8}(近缘|近源|同源)|避免.{0,8}(近缘|近源|同源)|"
    r"remote\s+homolog|cross[- ]?cluster|exclude.{0,12}(homolog|near))",
    re.IGNORECASE,
)

EXCLUDE_KNOWN_ASSOCIATION_INTENT = re.compile(
    r"((排除|去掉|过滤|屏蔽|不返回|不要返回).{0,10}(已知|已有|已记录|已经记录).{0,10}(酶|催化酶|关联)|"
    r"只.{0,5}(找|看|要).{0,8}(未收录|未记录|新关联)|"
    r"exclude.{0,12}(known|recorded).{0,12}(enzyme|association)|novel[- ]association[- ]only)",
    re.IGNORECASE,
)
KNOWN_ONLY_ASSOCIATION_INTENT = re.compile(
    r"((只|仅).{0,5}(看|要|返回|保留|显示|排序).{0,8}(已知|已有|已记录|已经记录).{0,8}(酶|催化酶|关联)|"
    r"(只|仅).{0,8}(已知|已有|已记录|已经记录).{0,8}(酶|催化酶|关联)|"
    r"known[- ]only|recorded[- ]only|only.{0,8}(known|recorded).{0,8}(enzyme|association))",
    re.IGNORECASE,
)
ALLOW_KNOWN_ASSOCIATION_INTENT = re.compile(
    r"((保留|包含|允许).{0,10}(已知|已有|已记录|已经记录).{0,10}(酶|关联)|"
    r"(不要|不).{0,4}(排除|过滤|屏蔽).{0,8}(已知|已有|已记录|已经记录)|"
    r"include.{0,12}(known|recorded)|keep.{0,12}(known|recorded))",
    re.IGNORECASE,
)


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
    plan: dict[str, Any]


class RoutePlanner:
    """Constrained reaction-to-enzyme route planner.

    The LLM can only propose intent-level knobs. LangGraph applies deterministic
    constraints before the production router sees anything. In particular:

    * a user-named seed must be an ID that really exists in the deployed protein
      universe and appears explicitly in the user's text;
    * database-known positive enzymes can be used as seeds only when the user
      explicitly asks to use known positives;
    * "near homolog" is not taxonomy and is not an arbitrary embedding cutoff.
      The planner maps remote-family intent to a separate cross-cluster novelty
      overlay, whose runtime implementation uses the repository's MMseqs2 50%
      identity / 80% coverage family boundary;
    * if AI routing fails, the graph deterministically falls back to Top-10,
      unrestricted, zero-shot, homolog-allowed retrieval.
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

    @staticmethod
    def _defaults(state: RouteState) -> dict[str, Any]:
        return {
            "base_plan": {
                **DEFAULT_PLAN,
                "known_enzyme_ids": [],
                "selected_by": "default",
                "reason": "默认路线：Top 10、全部候选酶、Zero-shot、允许同源候选，并保留当前知识库中的已记录催化酶。",
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
            return {"proposal_error": f"{type(exc).__name__}: {exc}"}

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
        elif isinstance(proposal, dict):
            semantic_proposal = proposal.get("_semantic_source") == "deepseek"
            top_k = self._int(proposal.get("top_k"), 10)
            if top_k not in SUPPORTED_TOP_K:
                top_k = 3 if top_k <= 3 else 5 if top_k <= 5 else 10 if top_k <= 10 else 20
                plan["warnings"].append("候选数量已映射到当前交互支持的 Top 3 / 5 / 10 / 20。")

            scope = str(proposal.get("enzyme_taxonomy_scope") or "all").strip().lower()
            if scope not in SUPPORTED_TAXONOMY:
                scope = "all"
                plan["warnings"].append("无法安全解释物种范围，已使用全部候选酶。")

            seed_mode = str(proposal.get("seed_mode") or "none").strip().lower()
            if seed_mode not in SUPPORTED_SEED_MODES:
                seed_mode = "none"

            # Backward-compatible interpretation of earlier proposal schema.
            proposed_ids = proposal.get("known_enzyme_ids") or []
            if not isinstance(proposed_ids, list):
                proposed_ids = []
            if proposed_ids and seed_mode == "none":
                seed_mode = "explicit"

            known_ids: list[str] = []
            seed_source = "none"
            if seed_mode == "explicit":
                authorized_lookup = {value.casefold(): value for value in authorized_ids}
                requested = proposed_ids or authorized_ids
                rejected_seed = False
                for value in requested:
                    canonical = authorized_lookup.get(str(value).casefold())
                    if canonical and canonical not in known_ids:
                        known_ids.append(canonical)
                    elif str(value).strip():
                        rejected_seed = True
                if rejected_seed:
                    plan["warnings"].append("语言模型提出的未确认酶 ID 已被 LangGraph 约束层拒绝。")
                if known_ids:
                    seed_source = "user_confirmed" if any(value in confirmed_ids for value in known_ids) else "user_explicit"
                else:
                    seed_mode = "none"
            elif seed_mode == "catalog_known":
                if catalog_known_ids and (semantic_proposal or KNOWN_POSITIVE_INTENT.search(user_text)):
                    known_ids = list(catalog_known_ids)
                    seed_source = "catalog_known_associations"
                else:
                    seed_mode = "none"
                    if not catalog_known_ids:
                        plan["warnings"].append("本地已知关联中没有可用阳性酶，因此保持 Zero-shot。")
                    else:
                        plan["warnings"].append("只有在用户明确要求使用已知阳性时，系统才会把数据库关联作为 Few-shot seed。")

            if known_ids and scope != "all":
                try:
                    validate_seed_scope(known_ids, scope)
                except Exception:
                    scope = "all"
                    plan["warnings"].append("所选阳性 seed 与物种限制不兼容；为保留 seed 证据，候选范围已回退到全部蛋白。")

            homology_policy = str(proposal.get("homology_policy") or "allow").strip().lower()
            if proposal.get("exclude_near_source"):
                homology_policy = "cross_cluster"
            if homology_policy not in SUPPORTED_HOMOLOGY_POLICIES:
                homology_policy = "allow"
            if (
                homology_policy == "cross_cluster"
                and not semantic_proposal
                and not REMOTE_INTENT.search(user_text)
            ):
                homology_policy = "allow"
                plan["warnings"].append("远缘筛选属于强制新颖性约束；用户没有明确表达远缘/排除同源意图，因此未自动启用。")

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
                    # This does NOT turn the ranking into few-shot. Catalog positives are
                    # used only to define which 50%-identity families should be hidden.
                    homology_anchor_ids = list(catalog_known_ids)
                    homology_anchor_source = "catalog_known_associations_filter_only"
                    homology_filter_applied = True
                    plan["warnings"].append("远缘筛选将以数据库已知阳性酶作为 50% identity cluster 排除锚点；这些酶不会作为 Zero-shot 模型的 seed 打分证据。")
                else:
                    homology_policy = "allow"
                    plan["warnings"].append("检测到远缘发现意图，但该反应没有可用阳性锚点，无法定义“相对谁远缘”，因此未应用跨簇筛选。")

            association_policy = str(proposal.get("known_association_policy") or "allow_known").strip().lower()
            if association_policy not in SUPPORTED_KNOWN_ASSOCIATION_POLICIES:
                association_policy = "allow_known"
            candidate_universe = str(
                proposal.get("candidate_universe") or DEFAULT_CANDIDATE_UNIVERSE
            ).strip().lower()
            candidate_universe_source = (
                "deepseek_semantic"
                if semantic_proposal and "candidate_universe" in proposal
                else "default"
            )
            if candidate_universe not in SUPPORTED_CANDIDATE_UNIVERSES:
                candidate_universe = DEFAULT_CANDIDATE_UNIVERSE
                candidate_universe_source = "guardrail_default"
                plan["warnings"].append("无法安全解释候选库范围，已使用默认通用候选库。")
            elif candidate_universe != DEFAULT_CANDIDATE_UNIVERSE and not semantic_proposal:
                candidate_universe = DEFAULT_CANDIDATE_UNIVERSE
                candidate_universe_source = "guardrail_default"
                plan["warnings"].append("专用候选库只能由经过语义解析的明确用户请求启用，已保留默认通用候选库。")
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
                "candidate_universe": candidate_universe,
                "candidate_universe_source": candidate_universe_source,
                "selected_by": "ai",
                "reason": str(proposal.get("reason") or "根据输入中的实验目标选择受支持的检索策略。").strip()[:300],
            })

        # Scope selection is a semantic decision made by DeepSeek when an AI
        # proposal exists. For default/offline routes there is no semantic planner,
        # so preserve the existing explicit user fallback behavior.
        has_semantic_scope = isinstance(proposal, dict) and proposal.get("_semantic_source") == "deepseek" and "known_association_policy" in proposal
        if has_semantic_scope:
            plan["known_association_policy_source"] = "deepseek_semantic"
        else:
            if KNOWN_ONLY_ASSOCIATION_INTENT.search(user_text):
                plan["known_association_policy"] = "known_only"
                plan["known_association_policy_source"] = "natural_language"
            elif ALLOW_KNOWN_ASSOCIATION_INTENT.search(user_text):
                plan["known_association_policy"] = "allow_known"
                plan["known_association_policy_source"] = "natural_language"
            elif EXCLUDE_KNOWN_ASSOCIATION_INTENT.search(user_text):
                plan["known_association_policy"] = "exclude_known"
                plan["known_association_policy_source"] = "natural_language"
            else:
                plan["known_association_policy"] = "allow_known"
                plan["known_association_policy_source"] = "default_fallback"
                if isinstance(proposal, dict) and "known_association_policy" in proposal:
                    plan["warnings"].append("未验证的路由提议未获得语义授权，因此保留已知关联混排。")

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
            "default_route": {
                "top_k": 10,
                "ranking_objective": "top10",
                "enzyme_taxonomy_scope": "all",
                "shot_mode": "zero_shot",
                "homology_policy": "allow",
                "known_association_policy": "allow_known",
            },
            "constraints": {
                "top_k": [3, 5, 10, 20],
                "enzyme_taxonomy_scope": ["all", "eukaryote", "prokaryote"],
                "seed_mode": ["none", "explicit_user_ids", "catalog_known_when_explicitly_requested"],
                "homology_policy": ["allow", "cross_cluster"],
                "known_association_policy": ["allow_known", "known_only_when_explicitly_requested", "exclude_known_when_explicitly_requested"],
                "candidate_universe": sorted(SUPPORTED_CANDIDATE_UNIVERSES),
                "cross_cluster_definition": "MMseqs2 min identity 0.50, coverage 0.80",
                "manual_model_override": "not_ai_selectable",
                "temporary_candidate_universe": "requires_explicit_file_input",
            },
        })
        return {"plan": plan}
