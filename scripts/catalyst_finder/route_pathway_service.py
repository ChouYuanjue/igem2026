from __future__ import annotations

from http import HTTPStatus
from typing import Any, Callable

from scripts.catalyst_finder.errors import AppError
from scripts.catalyst_finder.protein_resolution import compact_query_terms
from scripts.catalyst_finder.rhea_client import (
    RHEA_ENTRY_BASE,
    RHEA_ID_RE,
    canonical_rhea_id,
)
from scripts.catalyst_finder.route_design import RouteDesignError
from scripts.catalyst_finder.resolution_helpers import unique


def _clean_string_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = str(item or "").strip()
        if text and len(text) <= 240:
            result.append(text)
    return unique(result)[:limit]


class RoutePathwayService:
    """Application service for route design and multi-step pathway analysis."""

    def __init__(
        self,
        *,
        catalog: Any,
        deepseek: Any,
        proteins: Any,
        route_designer: Any,
        route_feasibility: Any,
        pathway: Any,
        resolve_reaction: Callable[[str], dict[str, Any]],
        resolve_reaction_from_terms: Callable[..., dict[str, Any]],
        rank_model: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.catalog = catalog
        self.deepseek = deepseek
        self.proteins = proteins
        self.route_designer = route_designer
        self.route_feasibility = route_feasibility
        self.pathway = pathway
        self.resolve = resolve_reaction
        self._resolve_reaction_from_terms = resolve_reaction_from_terms
        self._rank_model = rank_model

    @staticmethod
    def _candidate_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
        rows = result.get("candidates") if isinstance(result, dict) else None
        return [dict(row) for row in rows or [] if isinstance(row, dict)]

    def _annotate_model_enzyme_frontier(
        self, routes: list[dict[str, Any]], *, max_unique_steps: int = 12
    ) -> dict[str, Any]:
        """Attach bounded production-retrieval evidence to known Rhea route steps.

        This is deliberately diagnostic: it never changes route score/order. The
        expensive model layer is invoked only for enzyme-availability-priority route
        design, and only for a bounded set of unique known Rhea steps.
        """
        if self._rank_model is None:
            return {"status": "unavailable", "reason": "model_gateway_not_configured", "scored_steps": 0}
        step_refs: dict[str, list[dict[str, Any]]] = {}
        for route in routes:
            for step in route.get("steps") or []:
                if not isinstance(step, dict) or str(step.get("evidence_type") or "known_rhea") != "known_rhea":
                    continue
                rid = str(step.get("rhea_id") or "").strip()
                if rid:
                    step_refs.setdefault(rid, []).append(step)
        selected = list(step_refs)[: max(0, int(max_unique_steps))]
        failures: list[dict[str, str]] = []
        reverse_checked = 0
        reverse_recovered = 0
        for rid in selected:
            try:
                r2e = self._rank_model("rank-enzymes", {
                    "reaction_id": rid, "candidate_universe": "general_merged",
                    "top_k": 3, "ranking_objective": "top3", "conformal_mode": "disabled",
                })
                candidates = self._candidate_rows(r2e)[:3]
                query = dict(r2e.get("query") or {})
                annotation: dict[str, Any] = {
                    "status": "ok" if candidates else "empty",
                    "route_id": query.get("route_id"),
                    "score_source": query.get("score_source"),
                    "top_candidates": [
                        {
                            "rank": int(row.get("rank") or i + 1),
                            "protein_id": str(row.get("candidate_id") or ""),
                            "selection_source": row.get("selection_source"),
                        }
                        for i, row in enumerate(candidates)
                    ],
                    "reverse_recovery_at_20": None,
                }
                if candidates:
                    top1 = str(candidates[0].get("candidate_id") or "").strip()
                    if top1:
                        reverse_checked += 1
                        e2r = self._rank_model("rank-reactions", {
                            "enzyme_id": top1, "candidate_universe": "general_merged",
                            "top_k": 20, "ranking_objective": "top20", "conformal_mode": "disabled",
                        })
                        e2r_rows = self._candidate_rows(e2r)
                        recovered_rank = next(
                            (int(row.get("rank") or i + 1) for i, row in enumerate(e2r_rows)
                             if str(row.get("candidate_id") or "") == rid),
                            None,
                        )
                        recovered = recovered_rank is not None and recovered_rank <= 20
                        reverse_recovered += int(recovered)
                        annotation["reverse_recovery_at_20"] = bool(recovered)
                        annotation["reverse_recovery_rank"] = recovered_rank
                        annotation["reverse_route_id"] = (e2r.get("query") or {}).get("route_id")
                for step in step_refs[rid]:
                    step["model_enzyme_frontier"] = dict(annotation)
            except Exception as exc:
                failures.append({"rhea_id": rid, "error": f"{type(exc).__name__}: {exc}"})
                for step in step_refs[rid]:
                    step["model_enzyme_frontier"] = {"status": "failed", "error": type(exc).__name__}
        return {
            "status": "completed" if selected else "not_applicable",
            "policy": "diagnostic_only_no_route_rescoring",
            "requested_unique_steps": len(step_refs),
            "scored_steps": len(selected),
            "truncated": len(step_refs) > len(selected),
            "max_unique_steps": int(max_unique_steps),
            "reverse_checked_steps": reverse_checked,
            "reverse_recovered_at_20_steps": reverse_recovered,
            "failures": failures,
        }

    def route_design_resolve(self, text: str, ui_language: str = "en") -> dict[str, Any]:
        parsed = self.deepseek.interpret_route_design_request(text, ui_language=ui_language)
        source_terms = list(parsed["source_terms"])
        target_terms = list(parsed["target_terms"])
        try:
            sources = self.route_designer.resolve_compound(source_terms, limit=6) if source_terms else []
            targets = self.route_designer.resolve_compound(target_terms, limit=6)
        except RouteDesignError as exc:
            raise AppError("route_design_resolution_failed", str(exc), HTTPStatus.UNPROCESSABLE_ENTITY) from exc
        if (source_terms and not sources) or not targets:
            normalized = self.deepseek.normalize_compound_terms(
                source_terms=source_terms,
                target_terms=target_terms,
            )
            source_terms = list(normalized.get("source_terms") or source_terms)
            target_terms = list(normalized.get("target_terms") or target_terms)
            try:
                sources = self.route_designer.resolve_compound(source_terms, limit=6) if source_terms else []
                targets = self.route_designer.resolve_compound(target_terms, limit=6)
            except RouteDesignError as exc:
                raise AppError("route_design_resolution_failed", str(exc), HTTPStatus.UNPROCESSABLE_ENTITY) from exc
        if source_terms and not sources:
            raise AppError("route_design_source_unresolved", "没有在 Rhea 参与物中核对到起始前体，请换用标准英文名称或 ChEBI ID。", HTTPStatus.UNPROCESSABLE_ENTITY)
        if not targets:
            raise AppError("route_design_target_unresolved", "没有在 Rhea 参与物中核对到目标产物，请换用标准英文名称或 ChEBI ID。", HTTPStatus.UNPROCESSABLE_ENTITY)
        host_norm = parsed["host"].casefold()
        host_pool_supported = bool(parsed["host"] and ("coli" in host_norm or "escherichia" in host_norm or "大肠杆菌" in parsed["host"]))
        if not sources and not host_pool_supported:
            raise AppError("route_design_source_missing", "路线推荐需要一个起始前体；如果你是从宿主代谢网络出发，也可以直接说明宿主。目前可直接使用 E. coli / 大肠杆菌的 iML1515 代谢物池。", HTTPStatus.UNPROCESSABLE_ENTITY)
        return {
            "direction": "route_design",
            "summary": parsed["summary"],
            "route_design_resolution": {
                "source_terms": source_terms,
                "target_terms": target_terms,
                "source_candidates": sources,
                "target_candidates": targets,
                "recommended_source_id": sources[0]["chebi_id"] if sources else None,
                "recommended_target_id": targets[0]["chebi_id"] if targets else None,
                "host": parsed["host"],
                "host_pool_supported": host_pool_supported,
                "max_steps": parsed["max_steps"],
                "route_count": parsed["route_count"],
                "priority": parsed["priority"],
                "exploration_policy": parsed["exploration_policy"],
                "analysis_layers": list(parsed.get("analysis_layers") or []),
            },
            "reaction_resolution": None,
            "positive_enzyme_resolutions": [],
            "protein_resolution": None,
            "pathway_resolution": None,
            "llm_provenance": {**self.deepseek.provenance(), "used_for": "route_design_interpretation"},
        }


    def design_routes(self, payload: dict[str, Any]) -> dict[str, Any]:
        source_id = str(payload.get("source_chebi_id") or "").strip()
        target_id = str(payload.get("target_chebi_id") or "").strip()
        host = str(payload.get("host") or "").strip()
        priority = str(payload.get("priority") or "balanced").strip()
        if priority not in {"balanced", "short", "enzyme_available", "project_covered", "thermodynamic", "host_flux"}:
            priority = "balanced"
        exploration_policy = str(payload.get("exploration_policy") or "known_first").strip()
        if exploration_policy not in {"known_first", "known_only", "explore"}:
            exploration_policy = "known_first"
        analysis_layers = [
            str(value).strip() for value in (payload.get("analysis_layers") or [])
            if str(value).strip() in {"thermodynamics", "host_flux"}
        ]
        analysis_layers = list(dict.fromkeys(analysis_layers))
        run_thermodynamics = "thermodynamics" in analysis_layers
        run_host_flux = "host_flux" in analysis_layers
        requested_count = max(1, min(int(payload.get("route_count") or 10), 20))
        host_norm = host.casefold()
        host_is_ecoli = bool(host and ("escherichia coli" in host_norm or "e. coli" in host_norm or "e coli" in host_norm or "大肠杆菌" in host or host_norm == "ecoli"))
        if host_is_ecoli and run_host_flux:
            candidate_limit = min(30, max(20, requested_count * 2))
        elif priority == "thermodynamic" and run_thermodynamics:
            candidate_limit = min(30, max(10, requested_count * 2))
        else:
            candidate_limit = min(24, max(10, requested_count))
        try:
            result = self.route_designer.design(
                source_terms=[source_id] if source_id else [],
                target_terms=[target_id] if target_id else _clean_string_list(payload.get("target_terms"), 8),
                host=host,
                max_steps=int(payload.get("max_steps") or 6),
                limit=requested_count,
                candidate_limit=candidate_limit,
                priority=priority,
                local_reaction_ids=self.catalog.reaction_by_id.keys(),
            )
        except RouteDesignError as exc:
            raise AppError("route_design_failed", str(exc), HTTPStatus.UNPROCESSABLE_ENTITY) from exc
        except Exception as exc:
            raise AppError("route_design_failed", "候选路线生成没有完成。", HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}") from exc

        feasibility = self.route_feasibility.evaluate(
            list(result.get("routes") or []),
            host=host,
            priority=priority,
            requested_count=requested_count,
            run_thermodynamics=run_thermodynamics,
            run_host_flux=run_host_flux,
        )
        result["routes"] = list(feasibility.get("routes") or [])
        result["route_count"] = len(result["routes"])
        result["feasibility"] = feasibility.get("summary") or {}
        result["thermodynamics_run"] = feasibility.get("thermo_run") or {}
        result["host_feasibility_run"] = feasibility.get("host_run") or {}
        if priority == "enzyme_available":
            result["model_enzyme_frontier_run"] = self._annotate_model_enzyme_frontier(
                list(result.get("routes") or []), max_unique_steps=12
            )
        else:
            result["model_enzyme_frontier_run"] = {
                "status": "not_requested",
                "policy": "run_only_for_enzyme_available_priority",
                "scored_steps": 0,
            }

        exploration: dict[str, Any] = {"status": "not_requested", "routes": []}
        should_explore = bool(
            source_id
            and (
                exploration_policy == "explore"
                or (exploration_policy == "known_first" and not result.get("routes"))
            )
        )
        if should_explore:
            try:
                exploration = self.route_designer.explore_predicted_bridges(
                    source_chebi_id=source_id,
                    target_chebi_id=target_id,
                    max_steps=int(payload.get("max_steps") or 6),
                    limit=min(5, int(payload.get("route_count") or 10)),
                    priority=priority,
                    local_reaction_ids=self.catalog.reaction_by_id.keys(),
                )
            except RouteDesignError as exc:
                exploration = {"status": "unavailable", "message": str(exc), "routes": []}
        elif exploration_policy == "explore" and not source_id:
            exploration = {
                "status": "needs_explicit_source",
                "message": "规则预测扩展当前只对已经确认的单一起始前体运行；宿主代谢物池仍使用全 Rhea 已知路线搜索。",
                "routes": [],
            }
        result["exploratory_routes"] = list(exploration.get("routes") or [])
        result["exploration_run"] = {k: v for k, v in exploration.items() if k != "routes"}

        # Route cards already carry Rhea IDs, participant names and directions from
        # the cached official graph. Avoid N additional network calls here; the full
        # Rhea equation is fetched later only when a route is selected for pathway
        # compatibility analysis.
        for route in list(result.get("routes", [])) + list(result.get("exploratory_routes", [])):
            for step in route.get("steps", []):
                rid = str(step.get("rhea_id") or "")
                if rid:
                    step["url"] = f"{RHEA_ENTRY_BASE}{rid.split(':')[-1]}"

        result.update({
            "direction": "route_design",
            "exploration_policy": exploration_policy,
            "analysis_layers": analysis_layers,
            "exploration_backend": {
                "known_rhea": "active",
                "predicted_rules": exploration.get("status") or "not_requested",
                "predicted_engine": "MINE/Pickaxe + MetaCyc generalized rules",
                "available": self.route_designer.pickaxe_available(),
                "worker": exploration.get("worker"),
                "mapped_bridge_count": exploration.get("mapped_bridge_count"),
                "known_duplicate_count": exploration.get("known_duplicate_count"),
                "predicted_note": (
                    "预测探索已完成；预测路线单独排序，并为预测步骤保留独立证据标签。"
                    if exploration.get("status") == "completed"
                    else exploration.get("message")
                    or "预测反应探索按需运行，并与 Rhea 已知路线分别展示。"
                ),
            },
            "route_view": {
                "route_id": "route-design-rhea-known-v1",
                "title": "候选生物合成路线生成与排序",
                "summary": "按本轮请求生成 Rhea 候选路线，并只执行用户要求的热力学或宿主通量分析。DeepSeek 负责解析起点、目标、宿主、排序偏好和分析层。",
                "direction": "route_design",
                "active_overlays": ["route-design-pickaxe-isolated"] if exploration.get("status") == "completed" else [],
                "nodes": [
                    {"id": "route-design-parse", "title": "理解路线目标", "subtitle": "natural language → source / target / host", "kind": "input", "metric": f"{priority} · Top {int(payload.get('route_count') or 10)}", "detail": "DeepSeek 规范化用户描述的起点、目标、宿主和排序偏好。"},
                    {"id": "route-design-rhea-graph", "title": "加载全量 Rhea 已知反应图", "subtitle": "official Rhea directed reaction SMILES", "kind": "universe", "metric": f"{result.get('graph_stats', {}).get('route_nodes', 0):,} nodes · {result.get('graph_stats', {}).get('route_edges', 0):,} edges", "detail": "使用 Rhea 官方定向 reaction SMILES、ChEBI 结构、方向和 Swiss-Prot 映射构造已知生化路线空间。"},
                    {"id": "route-design-main-transform", "title": "提取主转化连接", "subtitle": "currency exclusion + structure continuity", "kind": "filter", "metric": "Rhea ID retained", "detail": "过滤水、质子、ATP/ADP、NAD(P)H、CoA、磷酸/焦磷酸等高频辅因子捷径，并按结构连续性提取可能的主底物→主产物连接；完整 Rhea 方程仍保留用于复核。"},
                    {"id": "route-design-kpaths", "title": "枚举候选简单路线", "subtitle": "NetworkX shortest_simple_paths", "kind": "model", "metric": f"{result.get('feasibility', {}).get('preliminary_route_count', 0)} preliminary · ≤ {int(payload.get('max_steps') or 6)} steps", "detail": "先生成比最终返回数更大的候选池，再交给科学可行性层复核，避免旧图分过早截断真正可行路线。"},
                    *([{"id": "route-design-stoichiometry", "title": "恢复完整 Rhea 化学计量", "subtitle": "directed reaction SMILES → exact ChEBI participants", "kind": "trust", "metric": "full hyper-reaction", "detail": "仅在热力学或宿主通量分析需要时恢复完整底物、产物与辅因子。"}] if (run_thermodynamics or run_host_flux) else []),
                    *([{"id": "route-design-thermo", "title": "计算整路热力学驱动力", "subtitle": "eQuilibrator · MDF", "kind": "trust", "metric": f"{result.get('feasibility', {}).get('thermo_complete_count', 0)} routes with MDF", "detail": "本轮明确请求热力学分析，因此运行 eQuilibrator MDF。"}] if run_thermodynamics else []),
                    *([{"id": "route-design-fba", "title": "检查宿主可承载通量", "subtitle": "COBRApy · iML1515 route-supported FBA", "kind": "filter", "metric": f"filtered {result.get('feasibility', {}).get('host_infeasible_filtered_count', 0)} zero-flux routes", "detail": "本轮明确请求宿主通量分析，因此运行 iML1515 route-supported FBA。"}] if run_host_flux else []),
                    *([{"id": "route-design-bime-frontier", "title": "逐步检索候选酶", "subtitle": "deployed retrieval → Top-3 + reverse recovery audit", "kind": "model", "metric": f"{result.get('model_enzyme_frontier_run', {}).get('scored_steps', 0)} unique Rhea steps", "detail": "仅在 enzyme_available 优先级下，对最终候选路线中的去重 Rhea 步骤调用部署中的双向检索：每步给出 Top-3 酶，并对 Top-1 做 E2R Top-20 反向恢复诊断。该诊断不改写路线主分。"}] if priority == "enzyme_available" else []),
                    {"id": "route-design-rank", "title": "排序候选路线", "subtitle": "requested evidence layers only", "kind": "rank", "metric": f"{len(result.get('routes', []))} routes", "detail": "只使用本轮实际执行的基础路线分和可选分析层进行相对排序。"},
                    {"id": "route-design-next", "title": "衔接整条路径酶评估", "subtitle": "selected route → pathway compatibility", "kind": "output", "metric": "natural-language follow-up", "detail": "用户选定候选路线后，可直接把该路线填入输入框，继续复用现有逐步 R2E、UniProt 条件证据和多酶全局兼容性评估。"},
                ],
            },
            "score_note": (
                "路线分数用于候选间相对排序；本轮只使用基础路线分"
                + ("、MDF" if run_thermodynamics else "")
                + ("和 E. coli route-supported FBA" if run_host_flux else "")
                + "。未请求的分析层没有执行。"
            ),
        })
        return result


    def pathway_resolve(self, text: str, ui_language: str = "en") -> dict[str, Any]:
        parsed = self.deepseek.interpret_pathway_request(text, ui_language=ui_language)
        groups: list[dict[str, Any]] = []
        for index, step in enumerate(parsed["steps"]):
            reaction_spec = step.get("reaction") or {}
            raw = str(reaction_spec.get("raw_text") or step.get("raw_text") or "").strip()
            rhea_match = RHEA_ID_RE.search(raw)
            if rhea_match:
                reaction_resolution = self.resolve(f"RHEA:{rhea_match.group(1)}")
            else:
                substrates = list(reaction_spec.get("substrate_terms") or [])
                products = list(reaction_spec.get("product_terms") or [])
                if substrates or products:
                    reaction_resolution = self._resolve_reaction_from_terms(
                        substrate_terms=substrates,
                        product_terms=products,
                        interpreted_reaction=raw,
                    )
                elif raw:
                    reaction_resolution = self.resolve(raw)
                else:
                    raise AppError("pathway_reaction_missing", f"第 {index + 1} 步没有识别出可核对的反应。", HTTPStatus.UNPROCESSABLE_ENTITY)

            enzyme_spec = step.get("enzyme") if isinstance(step.get("enzyme"), dict) else {}
            terms = compact_query_terms(enzyme_spec)
            enzyme_rows = []
            enzyme_raw = str(enzyme_spec.get("raw_text") or "").strip()
            if enzyme_raw or any(terms.values()):
                exact = self.proteins.exact_or_search(enzyme_raw, limit=6) if enzyme_raw else []
                enzyme_rows = exact or self.proteins.search(**{**terms, "limit": 6})
            groups.append({
                "step_index": index + 1,
                "mention": str(step.get("raw_text") or raw or f"第 {index + 1} 步").strip(),
                "reaction_resolution": reaction_resolution,
                "enzyme_resolution": {
                    "specified": bool(enzyme_raw or any(terms.values())),
                    "interpreted_protein": enzyme_raw,
                    "normalized": terms,
                    "candidates": [row.as_dict() for row in enzyme_rows],
                    "recommended_id": enzyme_rows[0].identifier if enzyme_rows else None,
                },
            })
        return {
            "direction": "pathway_compatibility",
            "summary": parsed["summary"],
            "pathway_resolution": {
                "execution_mode": parsed["execution_mode"],
                "host": parsed["host"],
                "target_conditions": parsed.get("target_conditions") or {},
                "evidence_dimensions": list(parsed.get("evidence_dimensions") or []),
                "steps": groups,
            },
            "reaction_resolution": None,
            "positive_enzyme_resolutions": [],
            "protein_resolution": None,
            "llm_provenance": {**self.deepseek.provenance(), "used_for": "pathway_interpretation"},
        }


    def analyze_pathway(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
        steps: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_steps[:8]):
            if not isinstance(raw, dict):
                continue
            rid = canonical_rhea_id(str(raw.get("rhea_id") or ""))
            if not rid:
                raise AppError("pathway_step_invalid", f"第 {index + 1} 步缺少有效的 Rhea ID。", HTTPStatus.UNPROCESSABLE_ENTITY)
            steps.append({
                "rhea_id": rid,
                "orientation": "reverse" if str(raw.get("orientation") or "forward") == "reverse" else "forward",
                "equation": str(raw.get("equation") or "").strip(),
                "enzyme_id": str(raw.get("enzyme_id") or "").strip(),
            })
        if len(steps) < 2:
            raise AppError("pathway_steps_missing", "整条路径评估至少需要两步已经确认的反应。", HTTPStatus.UNPROCESSABLE_ENTITY)
        try:
            result = self.pathway.analyze(
                steps=steps,
                user_text=str(payload.get("user_text") or ""),
                execution_mode=str(payload.get("execution_mode") or "auto"),
                host=str(payload.get("host") or ""),
                target_conditions=payload.get("target_conditions") if isinstance(payload.get("target_conditions"), dict) else {},
                evidence_dimensions=[
                    str(value).strip() for value in (payload.get("evidence_dimensions") or [])
                    if str(value).strip() in {"ph", "temperature", "cofactors", "localization", "cross_step_activity"}
                ],
            )
        except ValueError as exc:
            raise AppError("pathway_analysis_invalid", str(exc), HTTPStatus.UNPROCESSABLE_ENTITY) from exc
        except Exception as exc:
            raise AppError("pathway_analysis_failed", "整条路径兼容性评估没有完成。", HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}") from exc
        return result
