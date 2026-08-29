from __future__ import annotations

import csv
import hashlib
import re
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from projects.active.terpene_screening.core.candidate_universes import (
    DEFAULT_CANDIDATE_UNIVERSE,
    TPS_SPECIALIZED_UNIVERSE,
)
from projects.active.terpene_screening.core.input_audit import audit_protein_sequence
from scripts.catalyst_finder.errors import AppError
from scripts.catalyst_finder.formatting import (
    lang_text as _lang_text,
    probable_uniprot as _probable_uniprot,
    ui_language as _ui_language,
)
from scripts.catalyst_finder.open_world_inputs import (
    detect_direct_open_world_inputs,
    stable_protein_query_id,
)
from scripts.catalyst_finder.rhea_client import canonical_rhea_id
from scripts.catalyst_finder.route_view import build_e2r_route_view, build_r2e_route_view

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = ROOT / "results/catalyst_finder_runtime"


class RetrievalApplicationService:
    """Application service for bidirectional enzyme/reaction retrieval.

    Database evidence, semantic planning, model execution, and result reconciliation
    meet here. HTTP transport and process-level model lifecycle remain separate.
    """

    def __init__(
        self,
        *,
        catalog: Any,
        evidence: Any,
        proteins: Any,
        rhea: Any,
        route_planner: Any,
        e2r_planner: Any,
        homology: Any,
        route_designer: Any,
        model_gateway: Any,
    ) -> None:
        self.catalog = catalog
        self.evidence = evidence
        self.proteins = proteins
        self.rhea = rhea
        self.route_planner = route_planner
        self.e2r_planner = e2r_planner
        self.homology = homology
        self.route_designer = route_designer
        self.model_gateway = model_gateway

    def _protein_in_candidate_universe(self, protein_id: str, universe: str) -> bool:
        if universe == TPS_SPECIALIZED_UNIVERSE:
            return str(protein_id) in self.catalog.protein_by_id
        return self.evidence.is_candidate_protein(protein_id)

    def _reaction_in_candidate_universe(self, reaction_id: str, universe: str) -> bool:
        if universe == TPS_SPECIALIZED_UNIVERSE:
            return str(reaction_id) in self.catalog.reaction_by_id
        return self.evidence.is_candidate_reaction(reaction_id)

    def _prepare_seed_inputs(
        self,
        identifiers: list[str],
        sequence_inputs: list[dict[str, Any]] | None = None,
    ) -> tuple[list[str], Path | None, list[dict[str, Any]]]:
        canonical_ids: list[str] = []
        external_rows: list[tuple[str, str]] = []
        verified: list[dict[str, Any]] = []
        for raw in identifiers[:5]:
            value = str(raw or "").strip()
            if not value:
                continue
            candidate_id = self.evidence.canonical_protein_id(value)
            if self.evidence.is_candidate_protein(candidate_id):
                if candidate_id not in canonical_ids:
                    canonical_ids.append(candidate_id)
                    project_meta = self.catalog.protein_by_id.get(candidate_id, {})
                    merged_meta = self.evidence.protein_metadata(candidate_id) or {}
                    verified.append(
                        {
                            "id": candidate_id,
                            "requested_id": value,
                            "source": str(merged_meta.get("source_layer") or "general_merged_candidate"),
                            "name": project_meta.get("name"),
                            "organism": project_meta.get("species"),
                        }
                    )
                continue
            try:
                exact = self.proteins.uniprot.exact(value)
            except requests.RequestException as exc:
                raise AppError("positive_enzyme_unverified", f"无法在 UniProt 核对阳性酶 {value}。", HTTPStatus.UNPROCESSABLE_ENTITY, str(exc)) from exc
            accession = str(exact.get("accession") or value).strip()
            sequence = str(exact.get("sequence") or "").strip()
            if not sequence:
                raise AppError("positive_enzyme_sequence_missing", f"UniProt 条目 {accession} 没有可用蛋白序列。", HTTPStatus.UNPROCESSABLE_ENTITY)
            if accession not in canonical_ids:
                canonical_ids.append(accession)
                external_rows.append((accession, sequence))
                verified.append({"id": accession, "source": "uniprot_external", "name": exact.get("name"), "organism": exact.get("organism")})

        for raw in (sequence_inputs or [])[:5]:
            if not isinstance(raw, dict):
                continue
            sequence_value = str(raw.get("sequence") or "").strip()
            if not sequence_value:
                continue
            try:
                sequence, audit = audit_protein_sequence(sequence_value, policy="strict")
            except ValueError as exc:
                raise AppError(
                    "positive_enzyme_sequence_invalid",
                    "提供的阳性酶序列没有通过蛋白序列检查。",
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    str(exc),
                ) from exc
            existing = self.evidence.candidate_protein_for_sequence(sequence)
            if existing:
                if existing not in canonical_ids:
                    canonical_ids.append(existing)
                    merged_meta = self.evidence.protein_metadata(existing) or {}
                    project_meta = self.catalog.protein_by_id.get(existing, {})
                    verified.append(
                        {
                            "id": existing,
                            "requested_id": str(raw.get("id") or raw.get("query_id") or ""),
                            "source": "user_sequence_matched_general_merged",
                            "name": project_meta.get("name") or str(raw.get("header") or "") or None,
                            "organism": project_meta.get("species"),
                            "sequence_length": audit.sequence_length,
                            "candidate_source": merged_meta.get("source_layer"),
                        }
                    )
                continue
            external_id = str(raw.get("id") or raw.get("query_id") or "").strip()
            if not external_id.startswith("EXT-PROT-"):
                external_id = stable_protein_query_id(sequence)
            if external_id not in canonical_ids:
                canonical_ids.append(external_id)
                external_rows.append((external_id, sequence))
                verified.append(
                    {
                        "id": external_id,
                        "source": "user_provided_sequence",
                        "name": str(raw.get("header") or "").strip() or None,
                        "organism": None,
                        "sequence_length": audit.sequence_length,
                    }
                )

        if not external_rows:
            return canonical_ids, None, verified
        digest = hashlib.sha256("|".join(sorted(value for value, _ in external_rows)).encode("utf-8")).hexdigest()[:16]
        directory = RUNTIME_ROOT / "temp_inputs"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"positive_seeds_{digest}.csv"
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["enzyme_id", "sequence"])
                writer.writerows(external_rows)
            tmp.replace(path)
        return canonical_ids, path, verified


    def rank_reactions(
        self,
        protein_id: str = "",
        *,
        enzyme_sequence: str = "",
        query_id: str = "",
        user_text: str = "",
        route_mode: str = "intelligent",
        conversation_context: dict[str, Any] | None = None,
        ui_language: str = "en",
    ) -> dict[str, Any]:
        ui_language = _ui_language(ui_language)
        requested = str(protein_id or "").strip()
        provided_sequence = str(enzyme_sequence or "").strip()
        sequence_matched_candidate = ""
        if provided_sequence:
            try:
                provided_sequence, sequence_audit = audit_protein_sequence(
                    provided_sequence, policy="strict"
                )
            except ValueError as exc:
                raise AppError(
                    "protein_sequence_invalid",
                    "提供的蛋白序列没有通过输入检查。",
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    str(exc),
                ) from exc
            sequence_matched_candidate = (
                self.evidence.candidate_protein_for_sequence(provided_sequence) or ""
            )
            if sequence_matched_candidate:
                requested = sequence_matched_candidate
            elif not requested:
                requested = str(query_id or "").strip() or stable_protein_query_id(
                    provided_sequence
                )
        if not requested and not provided_sequence:
            raise AppError(
                "protein_required",
                "请提供蛋白记录、UniProt 登录号或蛋白序列。",
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        local_id = self.proteins.canonical_local_id(requested) if requested else ""
        candidate_id = self.evidence.canonical_protein_id(requested) if requested else ""
        is_current = bool(local_id and self.catalog.protein_by_id.get(local_id, {}).get("seen"))
        is_model_ready = self.evidence.is_candidate_protein(candidate_id)
        display_meta: dict[str, Any]
        model_payload: dict[str, Any]
        resolved_query_id = candidate_id if is_model_ready else (str(query_id or "").strip() or requested)
        external_sequence = ""
        if is_model_ready:
            meta = self.catalog.protein_by_id.get(candidate_id, {})
            merged_meta = self.evidence.protein_metadata(candidate_id) or {}
            accession = (
                str(meta.get("uniprot_id") or "").strip()
                or _probable_uniprot(requested)
                or _probable_uniprot(candidate_id)
            )
            display_meta = {
                "id": candidate_id,
                "requested_id": requested,
                "accession": accession or None,
                "name": meta.get("name"),
                "organism": meta.get("species"),
                "url": f"https://www.uniprot.org/uniprotkb/{quote(str(accession or candidate_id), safe='')}",
                "input_mode": "general_merged_candidate_id",
                "candidate_source": merged_meta.get("source_layer"),
            }
            if provided_sequence:
                display_meta["input_mode"] = "general_merged_sequence_match"
                display_meta["provided_sequence_id"] = str(query_id or "").strip() or stable_protein_query_id(provided_sequence)
                display_meta["sequence_length"] = len(provided_sequence)
        else:
            if provided_sequence:
                external_sequence = provided_sequence
                resolved_query_id = str(query_id or "").strip() or stable_protein_query_id(
                    external_sequence
                )
                display_meta = {
                    "id": resolved_query_id,
                    "accession": None,
                    "name": None,
                    "organism": None,
                    "url": None,
                    "input_mode": "raw_protein_sequence",
                    "sequence_length": len(external_sequence),
                }
            else:
                try:
                    exact = self.proteins.uniprot.exact(requested)
                except requests.RequestException as exc:
                    raise AppError("protein_unverified", "无法从 UniProt 取得这个蛋白。", HTTPStatus.UNPROCESSABLE_ENTITY, str(exc)) from exc
                external_sequence = str(exact.get("sequence") or "").strip()
                if not external_sequence:
                    raise AppError("protein_sequence_missing", "这个 UniProt 条目没有可用于模型的蛋白序列。", HTTPStatus.UNPROCESSABLE_ENTITY)
                resolved_query_id = str(exact.get("accession") or requested)
                display_meta = {
                    "id": resolved_query_id,
                    "accession": resolved_query_id,
                    "name": exact.get("name"),
                    "organism": exact.get("organism"),
                    "url": f"https://www.uniprot.org/uniprotkb/{quote(resolved_query_id, safe='')}",
                    "input_mode": "uniprot_sequence",
                    "sequence_length": len(external_sequence),
                }

        local_known_reactions = [
            str(row.get("reaction_id") or "")
            for row in self.catalog.pairs_by_protein.get(local_id or "", [])
            if str(row.get("reaction_id") or "")
        ]
        evidence_protein_id = candidate_id if is_model_ready else requested
        integrated_evidence_rows = (
            self.evidence.known_reactions(evidence_protein_id)
            if evidence_protein_id and self.evidence.is_candidate_protein(evidence_protein_id)
            else []
        )
        integrated_known_reactions = list(
            dict.fromkeys(row.reaction_id for row in integrated_evidence_rows if row.reaction_id)
        )
        official_accession = str(display_meta.get("accession") or "").strip()
        if official_accession and _probable_uniprot(official_accession):
            try:
                official_known_reactions = self.route_designer.known_rhea_ids(official_accession)
            except Exception as exc:
                official_known_reactions = []
                official_known_reactions_error = f"{type(exc).__name__}: {exc}"
            else:
                official_known_reactions_error = ""
        else:
            official_known_reactions = []
            official_known_reactions_error = ""
        known_reactions = list(
            dict.fromkeys(
                integrated_known_reactions + local_known_reactions + official_known_reactions
            )
        )
        route_plan = self.e2r_planner.plan(
            user_text=str(user_text or ""),
            route_mode=route_mode,
            is_current=is_current,
            catalog_known_reactions=known_reactions,
            conversation_context={**dict(conversation_context or {}), "ui_language": ui_language},
        )
        selected_top_k = int(route_plan["top_k"])
        ranking_objective = str(route_plan.get("ranking_objective") or "top10")
        association_policy = str(route_plan.get("known_association_policy") or "allow_known")
        candidate_universe = str(
            route_plan.get("candidate_universe") or DEFAULT_CANDIDATE_UNIVERSE
        )
        retain_recorded_associations_only = association_policy == "known_only"
        candidate_known_reactions = {
            rid
            for rid in known_reactions
            if self._reaction_in_candidate_universe(rid, candidate_universe)
        }
        engine_top_k = selected_top_k
        query_is_in_selected_universe = bool(
            is_model_ready
            and self._protein_in_candidate_universe(candidate_id, candidate_universe)
        )
        if is_model_ready and not query_is_in_selected_universe:
            # Query coverage and candidate-universe coverage are independent. A
            # protein can have a precomputed embedding in the merged general library
            # while the user explicitly asks to rank only within the smaller TPS
            # candidate universe. In that case it remains a valid external query.
            if provided_sequence:
                external_sequence = provided_sequence
            else:
                accession = str(display_meta.get("accession") or "").strip()
                if not accession:
                    raise AppError(
                        "protein_sequence_required_for_selected_universe",
                        "这个蛋白不在所选专用候选库的预计算蛋白集合中，请提供蛋白序列后继续。",
                        HTTPStatus.UNPROCESSABLE_ENTITY,
                    )
                try:
                    exact = self.proteins.uniprot.exact(accession)
                except requests.RequestException as exc:
                    raise AppError(
                        "protein_sequence_unavailable_for_selected_universe",
                        "无法取得这个蛋白的序列，请直接提供 FASTA 或氨基酸序列后继续。",
                        HTTPStatus.UNPROCESSABLE_ENTITY,
                        str(exc),
                    ) from exc
                external_sequence = str(exact.get("sequence") or "").strip()
                if not external_sequence:
                    raise AppError(
                        "protein_sequence_required_for_selected_universe",
                        "这个蛋白不在所选专用候选库的预计算蛋白集合中，请提供蛋白序列后继续。",
                        HTTPStatus.UNPROCESSABLE_ENTITY,
                    )

        if query_is_in_selected_universe:
            model_payload = {
                "enzyme_id": candidate_id,
                "top_k": engine_top_k,
                "candidate_universe": candidate_universe,
                "ranking_objective": ranking_objective,
                "reliability_policy": "annotate",
            }
        else:
            model_payload = {
                "query_id": resolved_query_id,
                "enzyme_sequence": external_sequence,
                "protein_input_policy": "warn",
                "top_k": engine_top_k,
                "candidate_universe": candidate_universe,
                "ranking_objective": ranking_objective,
                "reliability_policy": "annotate",
            }
        if route_plan.get("known_reaction_ids"):
            model_payload["known_reaction_ids"] = list(route_plan["known_reaction_ids"])
        engine_masked_reaction_ids = candidate_known_reactions | set(
            route_plan.get("mask_reaction_ids") or []
        )
        if engine_masked_reaction_ids:
            model_payload["mask_reaction_ids"] = sorted(engine_masked_reaction_ids)
        try:
            result = self.model_gateway.rank("rank-reactions", model_payload)
        except Exception as exc:
            raise AppError("e2r_model_failed", "反应排序没有完成。", HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}") from exc
        query = dict(result.get("query") or {})
        route_plan["actual_route_id"] = query.get("route_id")
        route_plan["route_match"] = query.get("route_id") == route_plan.get("planned_route_id")
        route_plan["known_reaction_count"] = len(known_reactions)
        policy_masked_reaction_ids = set(route_plan.get("mask_reaction_ids") or [])
        seeded_reaction_ids = set(route_plan.get("known_reaction_ids") or [])
        known_reaction_ids = set(known_reactions)
        model_ranked_rows = list(result.get("candidates") or [])
        model_ranked_by_id = {str(row.get("candidate_id") or ""): row for row in model_ranked_rows}
        before_known_filter = len(model_ranked_rows)
        if retain_recorded_associations_only:
            rows = []
            filter_policy = "retain_recorded_associations_only"
            result_mode = "known_associations_only"
        elif association_policy == "exclude_known":
            rows = [
                row for row in model_ranked_rows
                if str(row.get("candidate_id") or "") not in known_reaction_ids
            ]
            filter_policy = "exclude_recorded_associations"
            result_mode = "novel_association_discovery"
        else:
            # The engine masks candidate-universe known reactions before Top-K. Keep
            # this second check as a provenance/alias safety net, not as over-fetch logic.
            rows = [
                row for row in model_ranked_rows
                if str(row.get("candidate_id") or "") not in known_reaction_ids
            ]
            filter_policy = "allow_recorded_associations"
            result_mode = "full_ranking"
        discovery_filter = {
            "policy": filter_policy,
            "result_mode": result_mode,
            "applied": association_policy != "allow_known",
            "recorded_association_count": len(known_reactions),
            "project_catalog_recorded_association_count": len(local_known_reactions),
            "rhea_swissprot_association_count": len(official_known_reactions),
            "integrated_database_association_count": len(integrated_known_reactions),
            "candidate_universe_recorded_association_count": len(candidate_known_reactions),
            "excluded_count": len(candidate_known_reactions) if association_policy == "exclude_known" else 0,
            "discovery_removed_known_count": sum(1 for row in model_ranked_rows if str(row.get("candidate_id") or "") in known_reaction_ids),
            "retained_count": len(known_reactions) if retain_recorded_associations_only else None,
            "known_ids": list(known_reactions),
            "masked_ids": sorted(engine_masked_reaction_ids),
            "policy_masked_ids": sorted(policy_masked_reaction_ids),
            "candidate_mask_applied": bool(engine_masked_reaction_ids),
            "seed_examples_removed": sorted(seeded_reaction_ids),
            "source": "integrated_database_plus_live_rhea_swissprot",
            "scope_note": _lang_text(ui_language,
                "Recorded associations are unioned from the integrated database evidence and the live official Rhea/Swiss-Prot mapping.",
                "“已记录”来自统一数据库证据与实时 Rhea/Swiss-Prot 官方映射的并集。"),
        }
        if official_known_reactions_error:
            discovery_filter["rhea_swissprot_error"] = official_known_reactions_error
        route_plan["discovery_filter"] = discovery_filter
        if retain_recorded_associations_only:
            query["empirical_reliability_status"] = "not_applicable_known_associations_only"
            query["empirical_reliability_tier"] = "uncalibrated"
        rows = rows[:selected_top_k]
        max_abs = max((abs(float(row.get("score") or 0.0)) for row in rows), default=1.0) or 1.0
        candidates = []
        for final_rank, row in enumerate(rows, start=1):
            rid = str(row.get("candidate_id") or "").strip()
            meta = self.catalog.reaction_by_id.get(rid, {})
            merged_meta = self.evidence.reaction_metadata(rid) or {}
            rhea_url = f"https://www.rhea-db.org/rhea/{rid.split(':',1)[1]}" if re.fullmatch(r"RHEA:\d{5}", rid) else None
            score = float(row.get("score") or 0.0)
            candidates.append({
                "rank": final_rank,
                "candidate_id": rid,
                "score": score,
                "score_fraction": abs(score) / max_abs,
                "name": meta.get("name") if meta.get("name") != rid else None,
                "substrate_name": meta.get("substrate_name"),
                "product_name": meta.get("product_name"),
                "reaction_source": merged_meta.get("source_layer") or meta.get("source"),
                "rhea_url": rhea_url,
                "selection_source": row.get("selection_source") or "primary",
                "known_association": rid in known_reaction_ids,
            })
        ranked_candidate_by_id = model_ranked_by_id
        evidence_sources_by_reaction: dict[str, list[str]] = {}
        for evidence_row in integrated_evidence_rows:
            sources = evidence_sources_by_reaction.setdefault(evidence_row.reaction_id, [])
            if evidence_row.source not in sources:
                sources.append(evidence_row.source)
        known_association_items = []
        for reaction_id in known_reactions:
            meta = self.catalog.reaction_by_id.get(reaction_id, {})
            ranked = ranked_candidate_by_id.get(reaction_id)
            rhea_url = f"https://www.rhea-db.org/rhea/{reaction_id.split(':', 1)[1]}" if re.fullmatch(r"RHEA:\d{5}", reaction_id) else None
            evidence_sources = list(evidence_sources_by_reaction.get(reaction_id, ()))
            if reaction_id in official_known_reactions and "rhea_swissprot_live" not in evidence_sources:
                evidence_sources.append("rhea_swissprot_live")
            if not evidence_sources and reaction_id in local_known_reactions:
                evidence_sources.append("project_catalog")
            known_association_items.append({
                "candidate_id": reaction_id,
                "rhea_url": rhea_url,
                "name": meta.get("name") if meta.get("name") != reaction_id else None,
                "substrate_name": meta.get("substrate_name"),
                "product_name": meta.get("product_name"),
                "source": ";".join(evidence_sources) or "integrated_database",
                "sources": evidence_sources,
                "in_model_catalog": self.evidence.is_candidate_reaction(reaction_id),
                "in_candidate_universe": self._reaction_in_candidate_universe(
                    reaction_id, candidate_universe
                ),
                "model_score": float(ranked.get("score")) if ranked and ranked.get("score") is not None else None,
                "model_rank": int(ranked.get("rank")) if ranked and ranked.get("rank") is not None else None,
            })
        known_associations = {
            "count": len(known_reactions),
            "rhea_swissprot_count": len(official_known_reactions),
            "project_catalog_count": len(local_known_reactions),
            "integrated_database_count": len(integrated_known_reactions),
            "items": known_association_items,
            "truncated": False,
            "source_record_url": display_meta.get("url"),
            "note": _lang_text(ui_language,
                "Recorded reactions come from the integrated database evidence and Rhea/Swiss-Prot.",
                "已记录反应来自统一数据库证据与 Rhea/Swiss-Prot。"),
        }
        route_view = build_e2r_route_view(protein=display_meta, query=query, routing=route_plan, candidates=candidates)
        return {
            "protein": display_meta,
            "routing": route_plan,
            "ranking": {
                "top_k": selected_top_k,
                "ranking_objective": query.get("ranking_objective") or ranking_objective,
                "route_id": query.get("route_id"),
                "scope": query.get("scope"),
                "shot_mode": query.get("shot_mode"),
                "score_source": query.get("score_source"),
                "candidate_universe": query.get("candidate_universe") or candidate_universe,
                "candidate_universe_size": query.get("candidate_universe_size"),
                "reliability_status": query.get("empirical_reliability_status"),
            },
            "route_view": route_view,
            "discovery_filter": discovery_filter,
            "known_associations": known_associations,
            "candidates": candidates,
            "score_note": _lang_text(ui_language,
                "Retrieval scores compare model priority among the unrecorded reaction candidates.",
                "检索分数用于比较新关联候选反应的模型优先级。"),
        }


    def rank(
        self,
        rhea_id: str = "",
        *,
        reaction_smiles: str = "",
        query_id: str = "",
        orientation: str = "forward",
        user_text: str = "",
        route_mode: str = "intelligent",
        top_k: int | None = None,
        confirmed_seed_ids: list[str] | None = None,
        confirmed_seed_inputs: list[dict[str, Any]] | None = None,
        conversation_context: dict[str, Any] | None = None,
        ui_language: str = "en",
    ) -> dict[str, Any]:
        ui_language = _ui_language(ui_language)
        provided_reaction_smiles = str(reaction_smiles or "").strip()
        matched_reaction_ids: list[str] = []
        rhea_entry: RheaCandidate | None = None
        if provided_reaction_smiles:
            direct_reaction = detect_direct_open_world_inputs(provided_reaction_smiles).reaction
            if direct_reaction is None:
                raise AppError(
                    "reaction_smiles_invalid",
                    "提供的 Reaction SMILES 需要包含完整的 reactants>>products 结构。",
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                )
            provided_reaction_smiles = direct_reaction.reaction_smiles
            rid = str(query_id or "").strip() or direct_reaction.query_id
            orientation = "forward"
            reaction_equation = provided_reaction_smiles
            reaction_url: str | None = None
            matched_reaction_ids = self.evidence.candidate_reactions_for_smiles(
                provided_reaction_smiles
            )
            is_current = False
        else:
            rid = canonical_rhea_id(rhea_id)
            orientation = "reverse" if orientation == "reverse" else "forward"
            rhea_entry = self.rhea.exact(rid)
            reaction_equation = rhea_entry.equation
            reaction_url = rhea_entry.url
            matched_reaction_ids = [rid]
            is_current = rid in self.catalog.reaction_by_id and orientation != "reverse"

        evidence_reaction_ids = list(
            dict.fromkeys(
                value for value in matched_reaction_ids
                if self.evidence.canonical_rhea(value)
            )
        )

        # Known associations are contextual evidence, not automatically few-shot
        # seeds. LangGraph may use them only when the user explicitly requests
        # known-positive guidance, or as filter-only anchors when the user
        # explicitly asks for remote/cross-cluster discovery.
        local_known_association_ids = list(
            dict.fromkeys(
                str(row.get("protein_id") or "")
                for reaction_id in evidence_reaction_ids
                for row in self.catalog.pairs_by_reaction.get(reaction_id, [])
                if str(row.get("protein_id") or "") in self.catalog.protein_by_id
            )
        )
        integrated_evidence_rows = [
            row
            for reaction_id in evidence_reaction_ids
            for row in self.evidence.known_proteins(reaction_id)
        ]
        integrated_reported_ids = list(
            dict.fromkeys(row.protein_id for row in integrated_evidence_rows if row.protein_id)
        )
        integrated_canonical_ids = list(
            dict.fromkeys(
                (row.canonical_protein_id or row.protein_id)
                for row in integrated_evidence_rows
                if (row.canonical_protein_id or row.protein_id)
            )
        )
        rhea_swissprot_ids: list[str] = []
        rhea_swissprot_errors: list[str] = []
        for evidence_reaction_id in evidence_reaction_ids:
            try:
                rows = self.route_designer.known_uniprot_ids(evidence_reaction_id)
            except Exception as exc:
                rhea_swissprot_errors.append(
                    f"{evidence_reaction_id}: {type(exc).__name__}: {exc}"
                )
                continue
            for value in rows:
                if value not in rhea_swissprot_ids:
                    rhea_swissprot_ids.append(value)
        rhea_swissprot_error = "; ".join(rhea_swissprot_errors)
        live_canonical_ids = [self.evidence.canonical_protein_id(value) for value in rhea_swissprot_ids]
        known_association_ids = list(
            dict.fromkeys(integrated_reported_ids + local_known_association_ids + rhea_swissprot_ids)
        )
        planner_known_association_ids = list(
            dict.fromkeys(integrated_canonical_ids + live_canonical_ids + local_known_association_ids)
        )
        verified_seed_ids: list[str] = []
        external_seed_file: Path | None = None
        verified_seed_meta: list[dict[str, Any]] = []
        if route_mode != "default" and (confirmed_seed_ids or confirmed_seed_inputs):
            verified_seed_ids, external_seed_file, verified_seed_meta = self._prepare_seed_inputs(
                list(confirmed_seed_ids or []),
                list(confirmed_seed_inputs or []),
            )
        route_plan = self.route_planner.plan(
            user_text=str(user_text or ""),
            reaction_equation=reaction_equation,
            route_mode=route_mode,
            is_current=is_current,
            orientation=orientation,
            known_association_ids=planner_known_association_ids,
            confirmed_known_ids=verified_seed_ids,
            conversation_context={**dict(conversation_context or {}), "ui_language": ui_language},
        )
        selected_top_k = int(route_plan["top_k"])
        taxonomy_scope = str(route_plan["enzyme_taxonomy_scope"])
        known_enzyme_ids = list(route_plan.get("known_enzyme_ids") or [])
        ranking_objective = str(route_plan.get("ranking_objective") or "top10")
        candidate_universe = str(
            route_plan.get("candidate_universe") or DEFAULT_CANDIDATE_UNIVERSE
        )

        homology_filter: dict[str, Any] = {
            "requested": bool(route_plan.get("homology_filter_requested")),
            "applied": False,
            "anchor_count": 0,
            "anchor_source": route_plan.get("homology_anchor_source", "none"),
            "excluded_count": 0,
        }
        excluded_homolog_ids: set[str] = set()
        if route_plan.get("homology_filter_applied"):
            anchors = list(route_plan.get("homology_anchor_ids") or [])
            try:
                excluded_homolog_ids, cluster_meta = self.homology.exclusion_set(anchors)
                homology_filter.update(cluster_meta)
                homology_filter.update({
                    "applied": bool(excluded_homolog_ids),
                    "anchor_count": len(anchors),
                    "anchor_source": route_plan.get("homology_anchor_source", "none"),
                    "cluster_member_count": len(excluded_homolog_ids),
                })
                if not excluded_homolog_ids:
                    route_plan.setdefault("warnings", []).append("50% identity cluster 中没有找到可排除候选，因此保持普通排序。")
            except Exception as exc:
                route_plan.setdefault("warnings", []).append("远缘筛选索引不可用，已保留生产基础路线。")
                homology_filter["error"] = f"{type(exc).__name__}: {exc}"
                excluded_homolog_ids = set()

        # Database evidence and neural coverage are separate. Exact-sequence merging
        # may choose a canonical accession different from the accession reported by a
        # source database, so both identities are masked from the discovery ranking.
        recorded_association_ids = set(known_association_ids) | set(planner_known_association_ids)
        association_policy = str(route_plan.get("known_association_policy") or "allow_known")
        exclude_recorded_associations = association_policy == "exclude_known"
        retain_recorded_associations_only = association_policy == "known_only"
        expanded_for_novelty = bool(excluded_homolog_ids)
        candidate_recorded_ids = {
            value
            for value in planner_known_association_ids
            if self._protein_in_candidate_universe(value, candidate_universe)
        }
        masked_candidate_ids = set(candidate_recorded_ids)
        masked_candidate_ids.update(
            self.evidence.canonical_protein_id(value)
            for value in excluded_homolog_ids
            if self._protein_in_candidate_universe(
                self.evidence.canonical_protein_id(value), candidate_universe
            )
        )
        # The model can mask arbitrary IDs before Top-K selection, so no fixed-size
        # over-fetch or historical 2,085-candidate assumption is necessary.
        engine_top_k = selected_top_k

        input_mode = "registered_id"
        model_rhea_id = rid
        if provided_reaction_smiles:
            input_mode = "raw_reaction_smiles"
            model_rhea_id = matched_reaction_ids[0] if matched_reaction_ids else rid
            model_payload = {
                "query_id": rid,
                "reaction_smiles": provided_reaction_smiles,
                "reaction_feature_policy": "warn",
                "top_k": engine_top_k,
                "candidate_universe": candidate_universe,
                "ranking_objective": ranking_objective,
                "reliability_policy": "annotate",
                "enzyme_taxonomy_scope": taxonomy_scope,
            }
        elif is_current:
            model_payload: dict[str, Any] = {
                "reaction_id": rid,
                "top_k": engine_top_k,
                "candidate_universe": candidate_universe,
                "ranking_objective": ranking_objective,
                "reliability_policy": "annotate",
                "enzyme_taxonomy_scope": taxonomy_scope,
            }
        else:
            smiles = self.rhea.reaction_smiles(rid, orientation=orientation)
            input_mode = "rhea_smiles_reverse" if orientation == "reverse" else "rhea_smiles_external"
            model_rhea_id = smiles["source_rhea_id"]
            model_payload = {
                "query_id": rid,
                "reaction_smiles": smiles["reaction_smiles"],
                "reaction_feature_policy": "warn",
                "top_k": engine_top_k,
                "candidate_universe": candidate_universe,
                "ranking_objective": ranking_objective,
                "reliability_policy": "annotate",
                "enzyme_taxonomy_scope": taxonomy_scope,
            }
        if known_enzyme_ids:
            model_payload["known_enzyme_ids"] = known_enzyme_ids
            if external_seed_file is not None and any(
                not self.evidence.is_candidate_protein(value) for value in known_enzyme_ids
            ):
                model_payload["external_enzymes_csv"] = external_seed_file
        if masked_candidate_ids:
            model_payload["mask_enzyme_ids"] = sorted(masked_candidate_ids)
        if expanded_for_novelty:
            # CAGE is a separate current-Top20 result-assembly overlay. Mixing it
            # into a full-universe ordering would make the semantics ambiguous;
            # remote-family discovery therefore uses the locked base score route
            # and its explicit cluster filter only.
            model_payload["cage_rescue_slots"] = 0
            model_payload["conformal_mode"] = "disabled"

        try:
            result = self.model_gateway.rank("rank-enzymes", model_payload)
        except Exception as exc:
            raise AppError("model_failed", "候选酶排序没有完成。", HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}") from exc

        model_ranked_rows = list(result.get("candidates", []))
        model_ranked_by_id = {str(row.get("candidate_id") or ""): row for row in model_ranked_rows}
        before_known_filter = len(model_ranked_rows)
        if retain_recorded_associations_only:
            # Known evidence is presented in `known_associations`; it is not a discovery list.
            raw_rows = []
        else:
            # Default and discovery-only modes both reserve the model list for unrecorded
            # associations. Known rows still retain their auxiliary model score above.
            raw_rows = [
                row for row in model_ranked_rows
                if str(row.get("candidate_id") or "") not in recorded_association_ids
            ]
        if retain_recorded_associations_only:
            result_mode = "known_associations_only"
            filter_policy = "retain_recorded_associations_only"
        elif exclude_recorded_associations:
            result_mode = "novel_association_discovery"
            filter_policy = "exclude_recorded_associations"
        else:
            result_mode = "full_ranking"
            filter_policy = "allow_recorded_associations"
        discovery_filter = {
            "policy": filter_policy,
            "result_mode": result_mode,
            "applied": association_policy != "allow_known",
            "recorded_association_count": len(known_association_ids),
            "candidate_universe_recorded_association_count": len(candidate_recorded_ids),
            # Backward-compatible field name for older frontends; its semantics now
            # refer to the active neural candidate universe rather than the old TPS catalog.
            "model_catalog_recorded_association_count": len(candidate_recorded_ids),
            "rhea_swissprot_association_count": len(rhea_swissprot_ids),
            "integrated_database_association_count": len(integrated_reported_ids),
            "excluded_count": before_known_filter - len(raw_rows) if exclude_recorded_associations else 0,
            "discovery_removed_known_count": sum(1 for row in model_ranked_rows if str(row.get("candidate_id") or "") in recorded_association_ids),
            "retained_count": len(known_association_ids) if retain_recorded_associations_only else None,
            "known_ids": list(known_association_ids),
            "masked_ids": sorted(masked_candidate_ids),
            "candidate_mask_applied": bool(masked_candidate_ids),
            "source": "integrated_database_plus_live_rhea_swissprot",
            "scope_note": _lang_text(ui_language,
                "Recorded associations are unioned from the integrated database evidence and the live official Rhea/Swiss-Prot mapping.",
                "“已记录”来自统一数据库证据与实时 Rhea/Swiss-Prot 官方映射的并集。"),
        }
        if rhea_swissprot_error:
            discovery_filter["rhea_swissprot_error"] = rhea_swissprot_error
        if expanded_for_novelty:
            before = len(raw_rows)
            raw_rows = [row for row in raw_rows if str(row.get("candidate_id") or "") not in excluded_homolog_ids]
            excluded_in_eligible = before - len(raw_rows)
            homology_filter["excluded_count"] = excluded_in_eligible
            homology_filter["eligible_after_filter"] = len(raw_rows)
            route_plan.setdefault("warnings", []).append(
                "跨 50% identity cluster 后的候选集合改变了校准总体；本次不沿用原 unrestricted reliability / conformal 保证。"
            )
        raw_rows = raw_rows[:selected_top_k]

        candidates: list[dict[str, Any]] = []
        max_abs_score = max((abs(float(row.get("score") or 0.0)) for row in raw_rows), default=1.0) or 1.0
        for final_rank, row in enumerate(raw_rows, start=1):
            cid = str(row.get("candidate_id") or "").strip()
            meta = self.catalog.protein_by_id.get(cid, {})
            merged_meta = self.evidence.protein_metadata(cid) or {}
            uniprot_id = str(meta.get("uniprot_id") or "").strip() or _probable_uniprot(cid)
            if uniprot_id:
                uniprot_url = f"https://www.uniprot.org/uniprotkb/{quote(uniprot_id, safe='')}"
            else:
                uniprot_url = f"https://www.uniprot.org/uniprotkb?query={quote(cid, safe='')}"
            score = float(row.get("score") or 0.0)
            candidates.append({
                "rank": final_rank,
                "base_rank": int(row.get("rank") or final_rank),
                "candidate_id": cid,
                "score": score,
                "score_fraction": abs(score) / max_abs_score,
                "uniprot_id": uniprot_id or None,
                "uniprot_url": uniprot_url,
                "name": meta.get("name") if meta.get("name") != cid else None,
                "species": meta.get("species"),
                "candidate_source": str(merged_meta.get("source_layer") or ("registered" if meta.get("registered") else "reference")),
                "selection_source": row.get("selection_source") or "primary",
                "known_association": cid in recorded_association_ids,
            })

        ranked_candidate_by_id = model_ranked_by_id
        evidence_sources_by_id: dict[str, list[str]] = {}
        canonical_by_reported: dict[str, str] = {}
        for evidence_row in integrated_evidence_rows:
            reported = evidence_row.protein_id
            canonical = evidence_row.canonical_protein_id or reported
            canonical_by_reported[reported] = canonical
            sources = evidence_sources_by_id.setdefault(reported, [])
            if evidence_row.source not in sources:
                sources.append(evidence_row.source)
        known_association_items = []
        for association_id in known_association_ids:
            meta = self.catalog.protein_by_id.get(association_id, {})
            accession = str(meta.get("uniprot_id") or "").strip() or _probable_uniprot(association_id) or association_id
            canonical_id = canonical_by_reported.get(association_id) or self.evidence.canonical_protein_id(association_id)
            ranked = ranked_candidate_by_id.get(canonical_id) or ranked_candidate_by_id.get(association_id)
            evidence_sources = list(evidence_sources_by_id.get(association_id, ()))
            if association_id in rhea_swissprot_ids and "rhea_swissprot_live" not in evidence_sources:
                evidence_sources.append("rhea_swissprot_live")
            if not evidence_sources and association_id in local_known_association_ids:
                evidence_sources.append("project_catalog")
            known_association_items.append({
                "candidate_id": association_id,
                "canonical_candidate_id": canonical_id,
                "uniprot_id": accession,
                "uniprot_url": f"https://www.uniprot.org/uniprotkb/{quote(accession, safe='')}",
                "name": meta.get("name") if meta else None,
                "species": meta.get("species") if meta else None,
                "source": ";".join(evidence_sources) or "integrated_database",
                "sources": evidence_sources,
                "in_model_catalog": self.evidence.is_candidate_protein(canonical_id),
                "in_candidate_universe": self._protein_in_candidate_universe(
                    canonical_id, candidate_universe
                ),
                "model_score": float(ranked.get("score")) if ranked and ranked.get("score") is not None else None,
                "model_rank": int(ranked.get("rank")) if ranked and ranked.get("rank") is not None else None,
            })
        known_associations = {
            "count": len(known_association_ids),
            "rhea_swissprot_count": len(rhea_swissprot_ids),
            "project_catalog_count": len(local_known_association_ids),
            "integrated_database_count": len(integrated_reported_ids),
            "items": known_association_items,
            "truncated": False,
            "source_record_url": reaction_url,
            "note": _lang_text(ui_language,
                "Recorded enzymes come from the integrated database evidence and Rhea/Swiss-Prot.",
                "已记录酶来自统一数据库证据与 Rhea/Swiss-Prot。"),
        }

        query = dict(result.get("query", {}))
        if discovery_filter.get("applied") and not expanded_for_novelty:
            query["empirical_reliability_status"] = (
                "not_applicable_known_associations_only"
                if retain_recorded_associations_only
                else "not_applicable_known_associations_masked"
            )
            query["empirical_reliability_tier"] = "uncalibrated"
        if expanded_for_novelty:
            query["empirical_reliability_status"] = "not_applicable_cross_cluster_filter"
            query["empirical_reliability_tier"] = "uncalibrated"
            query["conformal_retrieval_set"] = {
                **(query.get("conformal_retrieval_set") or {}),
                "status": "not_applicable_cross_cluster_filter",
                "recommendation": "manual_review_remote_family_shortlist",
            }
        actual_route_id = query.get("route_id")
        route_plan["actual_route_id"] = actual_route_id
        route_plan["route_match"] = actual_route_id == route_plan.get("planned_route_id")
        route_plan["known_association_count"] = len(known_association_ids)
        route_plan["confirmed_positive_enzymes"] = verified_seed_meta
        route_plan["temporary_seed_extension"] = bool(external_seed_file and any(value not in self.catalog.protein_by_id for value in known_enzyme_ids))
        route_plan["homology_filter"] = homology_filter
        route_plan["discovery_filter"] = discovery_filter

        reaction_payload = {
            "rhea_id": rid,
            "model_rhea_id": model_rhea_id,
            "equation": reaction_equation,
            "url": reaction_url,
            "input_mode": input_mode,
            "reaction_smiles": provided_reaction_smiles or None,
            "matched_reaction_ids": matched_reaction_ids if provided_reaction_smiles else [],
        }
        route_view = build_r2e_route_view(
            reaction=reaction_payload,
            query=query,
            routing=route_plan,
            candidates=candidates,
        )
        return {
            "reaction": reaction_payload,
            "routing": route_plan,
            "ranking": {
                "top_k": selected_top_k,
                "ranking_objective": query.get("ranking_objective") or ranking_objective,
                "route_id": actual_route_id,
                "scope": query.get("scope"),
                "shot_mode": query.get("shot_mode"),
                "score_source": query.get("score_source"),
                "candidate_universe": query.get("candidate_universe") or candidate_universe,
                "candidate_universe_size": query.get("candidate_universe_size"),
                "candidate_universe_pre_taxonomy_size": query.get("candidate_universe_pre_taxonomy_size"),
                "candidate_universe_post_taxonomy_size": query.get("candidate_universe_post_taxonomy_size"),
                "enzyme_taxonomy_scope": query.get("enzyme_taxonomy_scope"),
                "reliability_status": query.get("empirical_reliability_status"),
            },
            "route_view": route_view,
            "discovery_filter": discovery_filter,
            "known_associations": known_associations,
            "candidates": candidates,
            "score_note": _lang_text(ui_language,
                "Retrieval scores compare model priority among the unrecorded enzyme candidates.",
                "检索分数用于比较新关联候选酶的模型优先级。"),
        }
