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

SUPPORTED_TOP_K = {3, 5, 10, 20}
SUPPORTED_KNOWN_ASSOCIATION_POLICIES = {"separate_known", "rank_with_known", "known_only", "exclude_known"}
DEFAULT_PLAN = {
    "top_k": 10,
    "use_known_activity_seeds": False,
    "known_reaction_ids": [],
    "mask_reaction_ids": [],
    "known_association_policy": "separate_known",
    "candidate_universe": DEFAULT_CANDIDATE_UNIVERSE,
    "candidate_universe_source": "default",
}

SEED_INTENT = re.compile(
    r"(基于.{0,8}(已知|已有).{0,8}(反应|活性|功能)|扩展.{0,8}(已知|已有).{0,8}(反应|活性)|"
    r"known.{0,8}(activity|reaction).{0,8}(seed|expand)|few[- ]?shot|seed)",
    re.IGNORECASE,
)
MASK_INTENT = re.compile(
    r"((排除|去掉|过滤|屏蔽|不返回|不要返回).{0,10}(已知|已有|已记录|已经记录).{0,10}(反应|活性|功能|关联)|"
    r"只.{0,5}(找|看|要).{0,8}(未收录|未记录|新反应|新功能|未知活性|新关联|潜在.{0,2}(反应|功能)|可能.{0,2}(反应|功能))|"
    r"novel.{0,8}(reaction|activity|association)|exclude.{0,12}(known|recorded))",
    re.IGNORECASE,
)
KNOWN_ONLY_INTENT = re.compile(
    r"((只|仅).{0,5}(看|要|返回|保留|显示|排序).{0,8}(已知|已有|已记录|已经记录).{0,8}(反应|活性|功能|关联)|"
    r"(只|仅).{0,8}(已知|已有|已记录|已经记录).{0,8}(反应|活性|功能|关联)|"
    r"known[- ]only|recorded[- ]only|only.{0,8}(known|recorded).{0,8}(reaction|activity|association))",
    re.IGNORECASE,
)
ALLOW_KNOWN_INTENT = re.compile(
    r"((保留|包含|允许).{0,10}(已知|已有|已记录|已经记录).{0,10}(反应|活性|关联)|"
    r"(不要|不).{0,4}(排除|过滤|屏蔽).{0,8}(已知|已有|已记录|已经记录)|"
    r"include.{0,12}(known|recorded)|keep.{0,12}(known|recorded))",
    re.IGNORECASE,
)
MIXED_RANKING_INTENT = re.compile(
    r"(混排|一起排序|统一排序|同一.{0,6}(排名|列表)|已知.{0,8}(未知|候选).{0,8}(一起|混合|混排|排序)|"
    r"(known|recorded).{0,12}(unknown|novel|candidate).{0,12}(same ranking|rank together|mixed)|"
    r"rank.{0,12}(known|recorded).{0,12}(with|alongside).{0,12}(unknown|novel|candidate))",
    re.IGNORECASE,
)


class E2RState(TypedDict, total=False):
    user_text: str
    route_mode: str
    is_current: bool
    catalog_known_reactions: list[str]
    conversation_context: dict[str, Any]
    base_plan: dict[str, Any]
    ai_proposal: dict[str, Any]
    proposal_error: str
    plan: dict[str, Any]


class E2RRoutePlanner:
    """Constrained enzyme-to-reaction route planner.

    The LLM proposes only experimental intent. The graph decides whether catalog
    reactions are allowed to become few-shot seeds or mask-only exclusions and
    then delegates the actual model-family choice to the repository router.
    """

    def __init__(self, *, proposal_fn: Callable[..., dict[str, Any]]) -> None:
        self.proposal_fn = proposal_fn
        graph = StateGraph(E2RState)
        graph.add_node("defaults", self._defaults)
        graph.add_node("ai_proposal", self._ai_proposal)
        graph.add_node("guardrails", self._guardrails)
        graph.add_edge(START, "defaults")
        graph.add_conditional_edges(
            "defaults",
            self._after_defaults,
            {"ai_proposal": "ai_proposal", "guardrails": "guardrails"},
        )
        graph.add_edge("ai_proposal", "guardrails")
        graph.add_edge("guardrails", END)
        self.graph = graph.compile()

    def plan(
        self,
        *,
        user_text: str,
        route_mode: str,
        is_current: bool,
        catalog_known_reactions: list[str] | None = None,
        conversation_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        known = list(dict.fromkeys(str(x) for x in (catalog_known_reactions or []) if str(x).strip()))
        state = self.graph.invoke({
            "user_text": str(user_text or ""),
            "route_mode": "default" if route_mode == "default" else "intelligent",
            "is_current": bool(is_current),
            "catalog_known_reactions": known,
            "conversation_context": dict(conversation_context or {}),
        })
        return dict(state["plan"])

    @staticmethod
    def _defaults(state: E2RState) -> dict[str, Any]:
        return {"base_plan": {
            **DEFAULT_PLAN,
            "selected_by": "default",
            "reason": "默认路线：Top 10，不使用已知反应 seed，并把已记录关系与模型候选分层呈现。",
            "warnings": [],
        }}

    @staticmethod
    def _after_defaults(state: E2RState) -> Literal["ai_proposal", "guardrails"]:
        return "ai_proposal" if state.get("route_mode") == "intelligent" else "guardrails"

    def _ai_proposal(self, state: E2RState) -> dict[str, Any]:
        try:
            args = [
                str(state.get("user_text") or ""),
                len(state.get("catalog_known_reactions") or []),
                list(state.get("catalog_known_reactions") or []),
                dict(state.get("conversation_context") or {}),
            ]
            try:
                parameter_count = len(inspect.signature(self.proposal_fn).parameters)
            except (TypeError, ValueError):
                parameter_count = len(args)
            proposal = self.proposal_fn(*args[:parameter_count])
            if not isinstance(proposal, dict):
                raise TypeError("E2R route proposal must be an object")
            return {"ai_proposal": proposal}
        except Exception as exc:
            return {"proposal_error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _normalize_top_k(value: Any) -> int:
        try:
            top_k = int(value)
        except (TypeError, ValueError):
            return 10
        if top_k in SUPPORTED_TOP_K:
            return top_k
        return 3 if top_k <= 3 else 5 if top_k <= 5 else 10 if top_k <= 10 else 20

    def _guardrails(self, state: E2RState) -> dict[str, Any]:
        plan = dict(state.get("base_plan") or DEFAULT_PLAN)
        plan["warnings"] = list(plan.get("warnings") or [])
        proposal = state.get("ai_proposal") if state.get("route_mode") == "intelligent" else None
        user_text = str(state.get("user_text") or "")
        known = list(state.get("catalog_known_reactions") or [])

        if state.get("proposal_error"):
            plan["warnings"].append("智能路由暂时不可用，已回到 E2R 默认路线。")
            plan["fallback_reason"] = "ai_route_failed"
        elif isinstance(proposal, dict):
            semantic_proposal = proposal.get("_semantic_source") == "deepseek"
            top_k = self._normalize_top_k(proposal.get("top_k"))
            use_known_activity_seeds = proposal.get("use_known_activity_seeds") is True
            if use_known_activity_seeds and not semantic_proposal and not SEED_INTENT.search(user_text):
                use_known_activity_seeds = False
                plan["warnings"].append("用户没有明确要求从已有活性扩展，因此没有自动启用 E2R Few-shot。")
            if use_known_activity_seeds and not known:
                use_known_activity_seeds = False
                plan["warnings"].append("这个酶在本地目录中没有可用已知反应，因此保持 Zero-shot。")

            association_policy = str(proposal.get("known_association_policy") or "separate_known").strip().lower()
            if association_policy not in SUPPORTED_KNOWN_ASSOCIATION_POLICIES:
                association_policy = "separate_known"
            if association_policy == "rank_with_known":
                use_known_activity_seeds = False
            # DeepSeek semantic planner owns the interpretation of result scope.
            # Seeding and result filtering are orthogonal: a user may seed from known
            # activities while still requesting only unrecorded outputs.
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

            plan.update({
                "top_k": top_k,
                "use_known_activity_seeds": use_known_activity_seeds,
                "known_reaction_ids": list(known) if use_known_activity_seeds else [],
                "known_association_policy": association_policy,
                "candidate_universe": candidate_universe,
                "candidate_universe_source": candidate_universe_source,
                "selected_by": "ai",
                "reason": str(proposal.get("reason") or "根据用户对反应范围和探索深度的描述选择 E2R 路线。").strip()[:300],
            })

        # Result scope is selected by the semantic planner. Regexes are not used
        # as the authority for changing candidate scope because follow-up requests
        # often refer to previous results implicitly.
        association_policy = str(plan.get("known_association_policy") or "separate_known").strip().lower()
        if association_policy not in SUPPORTED_KNOWN_ASSOCIATION_POLICIES:
            association_policy = "separate_known"
        if isinstance(proposal, dict) and proposal.get("_semantic_source") == "deepseek" and "known_association_policy" in proposal:
            plan["known_association_policy_source"] = "deepseek_semantic"
        else:
            if KNOWN_ONLY_INTENT.search(user_text):
                association_policy = "known_only"
                plan["known_association_policy_source"] = "natural_language"
            elif MIXED_RANKING_INTENT.search(user_text):
                association_policy = "rank_with_known"
                plan["known_association_policy_source"] = "natural_language_explicit_mixed_ranking"
            elif MASK_INTENT.search(user_text):
                association_policy = "exclude_known"
                plan["known_association_policy_source"] = "natural_language"
            elif ALLOW_KNOWN_INTENT.search(user_text):
                association_policy = "separate_known"
                plan["known_association_policy_source"] = "natural_language_restore_default"
        if association_policy == "rank_with_known":
            plan["use_known_activity_seeds"] = False
            plan["known_reaction_ids"] = []
            plan["mask_reaction_ids"] = []
        elif association_policy == "exclude_known":
            plan["mask_reaction_ids"] = list(known)
        else:
            plan["mask_reaction_ids"] = []
        plan["discovery_default_applied"] = False
        plan["known_association_policy"] = association_policy

        objective = {3: "top3", 5: "top10", 10: "top10", 20: "top20"}[int(plan.get("top_k", 10))]
        route = resolve_route(
            direction="enzyme_to_reaction",
            objective=objective,
            is_current=bool(state.get("is_current")),
            has_seed=bool(plan.get("known_reaction_ids")),
            masked_discovery=bool(plan.get("mask_reaction_ids")),
        )
        plan.update({
            "route_mode": state.get("route_mode", "intelligent"),
            "ranking_objective": objective,
            "planned_route_id": route.route_id,
            "scope": "current" if state.get("is_current") else "external",
            "shot_mode": "few_shot" if plan.get("known_reaction_ids") else "zero_shot",
            "default_route": {
                "top_k": 10,
                "ranking_objective": "top10",
                "use_known_activity_seeds": False,
                "known_association_policy": "separate_known",
                "shot_mode": "zero_shot",
            },
            "constraints": {
                "top_k": [3, 5, 10, 20],
                "use_known_activity_seeds": [False, True],
                "known_association_policy": ["separate_known_default", "rank_with_known_explicit_zero_shot", "known_only_when_explicitly_requested", "exclude_known_when_explicitly_requested"],
                "candidate_universe": sorted(SUPPORTED_CANDIDATE_UNIVERSES),
            },
        })
        return {"plan": plan}
