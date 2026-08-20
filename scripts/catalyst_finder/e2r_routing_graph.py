from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Literal
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from projects.active.terpene_screening.core.routing import resolve_route

SUPPORTED_TOP_K = {3, 5, 10, 20}
SUPPORTED_KNOWN_ACTIVITY_POLICIES = {"none", "seed_known", "mask_known"}
SUPPORTED_KNOWN_ASSOCIATION_POLICIES = {"allow_known", "known_only", "exclude_known"}
DEFAULT_PLAN = {
    "top_k": 10,
    "known_activity_policy": "none",
    "known_reaction_ids": [],
    "mask_reaction_ids": [],
    "known_association_policy": "allow_known",
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
            "reason": "默认路线：Top 10，不把已知反应作为正向 seed，并保留当前知识库中已经记录的反应。",
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
            top_k = self._normalize_top_k(proposal.get("top_k"))
            policy = str(proposal.get("known_activity_policy") or "none").strip().lower()
            if policy not in SUPPORTED_KNOWN_ACTIVITY_POLICIES:
                policy = "none"
            if policy == "seed_known" and not SEED_INTENT.search(user_text):
                policy = "none"
                plan["warnings"].append("用户没有明确要求用已有活性引导扩展，因此没有自动启用 E2R Few-shot。")
            if policy != "none" and not known:
                policy = "none"
                plan["warnings"].append("这个酶在本地目录中没有可用已知反应，因此保持 Zero-shot。")

            association_policy = str(proposal.get("known_association_policy") or "allow_known").strip().lower()
            if association_policy not in SUPPORTED_KNOWN_ASSOCIATION_POLICIES:
                association_policy = "exclude_known" if policy == "mask_known" else "allow_known"
            # DeepSeek semantic planner owns the interpretation of user scope.
            # Do not downgrade exclude_known/known_only using keyword rules;
            # phrases such as "只看潜在的反应" and conversational follow-ups
            # require semantic understanding rather than lexical matching.
            if policy == "mask_known":
                association_policy = "exclude_known"

            plan.update({
                "top_k": top_k,
                "known_activity_policy": policy,
                "known_reaction_ids": list(known) if policy == "seed_known" else [],
                "known_association_policy": association_policy,
                "selected_by": "ai",
                "reason": str(proposal.get("reason") or "根据用户对反应范围和探索深度的描述选择 E2R 路线。").strip()[:300],
            })

        # Result scope is selected by the semantic planner. Regexes are not used
        # as the authority for changing candidate scope because follow-up requests
        # often refer to previous results implicitly.
        association_policy = str(plan.get("known_association_policy") or "allow_known").strip().lower()
        if association_policy not in SUPPORTED_KNOWN_ASSOCIATION_POLICIES:
            association_policy = "allow_known"
        if isinstance(proposal, dict) and proposal.get("_semantic_source") == "deepseek" and "known_association_policy" in proposal:
            plan["known_association_policy_source"] = "deepseek_semantic"
        else:
            if KNOWN_ONLY_INTENT.search(user_text):
                association_policy = "known_only"
                plan["known_association_policy_source"] = "natural_language"
            elif ALLOW_KNOWN_INTENT.search(user_text):
                association_policy = "allow_known"
                plan["known_association_policy_source"] = "natural_language"
            elif MASK_INTENT.search(user_text):
                association_policy = "exclude_known"
                plan["known_association_policy_source"] = "natural_language"
        if association_policy == "exclude_known":
            plan["mask_reaction_ids"] = list(known)
            if plan.get("known_activity_policy") == "none":
                plan["known_activity_policy"] = "mask_known"
        else:
            plan["mask_reaction_ids"] = []
            if plan.get("known_activity_policy") == "mask_known":
                plan["known_activity_policy"] = "none"
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
                "known_activity_policy": "none",
                "known_association_policy": "allow_known",
                "shot_mode": "zero_shot",
            },
            "constraints": {
                "top_k": [3, 5, 10, 20],
                "known_activity_policy": ["none", "seed_known", "mask_known"],
                "known_association_policy": ["allow_known", "known_only_when_explicitly_requested", "exclude_known_when_explicitly_requested"],
                "manual_model_override": "not_ai_selectable",
                "temporary_reaction_universe": "requires_explicit_file_input",
            },
        })
        return {"plan": plan}
