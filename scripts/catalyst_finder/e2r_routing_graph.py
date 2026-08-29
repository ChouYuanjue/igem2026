from __future__ import annotations

import os
import inspect
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
    "seed_mode": "none",
    "seed_source": "none",
    "mask_reaction_ids": [],
    "known_association_policy": "separate_known",
    "candidate_universe": DEFAULT_CANDIDATE_UNIVERSE,
    "candidate_universe_source": "default",
}

class E2RState(TypedDict, total=False):
    user_text: str
    route_mode: str
    is_current: bool
    catalog_known_reactions: list[str]
    confirmed_known_reactions: list[str]
    conversation_context: dict[str, Any]
    base_plan: dict[str, Any]
    ai_proposal: dict[str, Any]
    proposal_error: str
    proposal_error_code: str
    proposal_error_detail: str
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
        confirmed_known_reactions: list[str] | None = None,
        conversation_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        known = list(dict.fromkeys(str(x).strip() for x in (catalog_known_reactions or []) if str(x).strip()))
        confirmed = list(dict.fromkeys(str(x).strip() for x in (confirmed_known_reactions or []) if str(x).strip()))
        state = self.graph.invoke({
            "user_text": str(user_text or ""),
            "route_mode": "default" if route_mode == "default" else "intelligent",
            "is_current": bool(is_current),
            "catalog_known_reactions": known,
            "confirmed_known_reactions": confirmed,
            "conversation_context": dict(conversation_context or {}),
        })
        return dict(state["plan"])

    @staticmethod
    def _defaults(state: E2RState) -> dict[str, Any]:
        known = list(state.get("catalog_known_reactions") or [])
        has_known = bool(known)
        return {"base_plan": {
            **DEFAULT_PLAN,
            "use_known_activity_seeds": has_known,
            "known_reaction_ids": known,
            "seed_mode": "catalog_known" if has_known else "none",
            "seed_source": "catalog_known_associations" if has_known else "none",
            "selected_by": "default",
            "reason": (
                "默认路线：Top 10；数据库存在已核对反应时，作为 Few-shot 反应锚点，同时把已记录关系与未记录模型候选分层呈现。"
                if has_known else
                "默认路线：Top 10；当前没有可用已知反应，因此使用 Zero-shot，并把数据库证据与模型候选分层呈现。"
            ),
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
                list(state.get("confirmed_known_reactions") or []),
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
            code = str(getattr(exc, "code", "") or type(exc).__name__)[:120]
            detail = str(getattr(exc, "detail", "") or "").strip()[:800]
            return {
                "proposal_error": f"{type(exc).__name__}: {exc}",
                "proposal_error_code": code,
                "proposal_error_detail": detail,
            }

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
        confirmed = list(state.get("confirmed_known_reactions") or [])
        all_positive = list(dict.fromkeys(known + confirmed))

        if state.get("proposal_error"):
            plan["warnings"].append("智能路由暂时不可用，已回到 E2R 默认路线。")
            plan["fallback_reason"] = "ai_route_failed"
            plan["fallback_error_code"] = str(state.get("proposal_error_code") or "route_proposal_failed")
            if os.environ.get("CATALYST_FINDER_DEBUG", "").strip() == "1" and state.get("proposal_error_detail"):
                plan["fallback_detail"] = str(state.get("proposal_error_detail"))[:800]
        elif isinstance(proposal, dict):
            semantic_proposal = proposal.get("_semantic_source") == "deepseek"
            top_k = self._normalize_top_k(proposal.get("top_k"))
            requested_seed_mode = str(proposal.get("seed_mode") or "").strip().lower()
            if requested_seed_mode not in {"none", "catalog_known", "explicit"}:
                requested_seed_mode = ""
            # Semantic routing decides whether the user explicitly opted out of the
            # normal positive-context behavior. Non-semantic proposals cannot change
            # this intent-level choice; they inherit the deterministic default.
            if semantic_proposal and requested_seed_mode == "none":
                use_known_activity_seeds = False
                seed_mode = "none"
                seed_source = "semantic_zero_shot"
            elif all_positive:
                use_known_activity_seeds = True
                seed_mode = "explicit" if confirmed else "catalog_known"
                seed_source = (
                    "catalog_known_plus_user_confirmed" if known and confirmed else
                    "user_confirmed" if confirmed else
                    "catalog_known_associations"
                )
            else:
                use_known_activity_seeds = False
                seed_mode = "none"
                seed_source = "none"

            association_policy = str(proposal.get("known_association_policy") or "separate_known").strip().lower()
            if association_policy not in SUPPORTED_KNOWN_ASSOCIATION_POLICIES:
                association_policy = "separate_known"
            if association_policy == "rank_with_known":
                use_known_activity_seeds = False
                seed_mode = "none"
                seed_source = "mixed_ranking_forces_zero_shot"
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
                "known_reaction_ids": list(all_positive) if use_known_activity_seeds else [],
                "seed_mode": seed_mode,
                "seed_source": seed_source,
                "known_association_policy": association_policy,
                "candidate_universe": candidate_universe,
                "candidate_universe_source": candidate_universe_source,
                "selected_by": "ai",
                "reason": str(proposal.get("reason") or "根据用户对反应范围和探索深度的描述选择 E2R 路线。").strip()[:300],
            })

        # Result scope is an intent-level semantic decision. Deterministic code
        # validates the enum but never infers it from keyword lists. If the semantic
        # planner is unavailable, the public default remains separate_known.
        association_policy = str(plan.get("known_association_policy") or "separate_known").strip().lower()
        if association_policy not in SUPPORTED_KNOWN_ASSOCIATION_POLICIES:
            association_policy = "separate_known"
        if isinstance(proposal, dict) and proposal.get("_semantic_source") == "deepseek" and "known_association_policy" in proposal:
            plan["known_association_policy_source"] = "deepseek_semantic"
        else:
            if isinstance(proposal, dict) and str(proposal.get("known_association_policy") or "separate_known") != "separate_known":
                plan["warnings"].append("结果范围属于语义策略；未经过 DeepSeek 语义解析的提议不能改变默认的已知证据与新候选分层展示。")
            association_policy = "separate_known"
            plan["known_association_policy_source"] = "default_fallback"

        if association_policy in {"rank_with_known", "known_only"}:
            plan["use_known_activity_seeds"] = False
            plan["known_reaction_ids"] = []
            plan["seed_mode"] = "none"
            plan["seed_source"] = (
                "mixed_ranking_forces_zero_shot" if association_policy == "rank_with_known"
                else "known_only_scoring_forces_zero_shot"
            )
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
                "use_known_activity_seeds": "if_database_positive",
                "known_association_policy": "separate_known",
                "shot_mode": "few_shot_if_database_positive_else_zero_shot",
            },
            "constraints": {
                "top_k": [3, 5, 10, 20],
                "seed_mode": ["catalog_known_by_default", "none_when_semantically_requested_zero_shot"],
                "known_association_policy": ["separate_known_default", "rank_with_known_explicit_zero_shot", "known_only_when_explicitly_requested", "exclude_known_when_explicitly_requested"],
                "candidate_universe": sorted(SUPPORTED_CANDIDATE_UNIVERSES),
            },
        })
        return {"plan": plan}
