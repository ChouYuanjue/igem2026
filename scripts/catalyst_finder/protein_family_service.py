from __future__ import annotations

import hashlib
from collections import defaultdict
from http import HTTPStatus
from typing import Any

from scripts.catalyst_finder.errors import AppError


class ProteinFamilyEvidenceService:
    """Aggregate recorded enzyme→reaction evidence over family/class scopes."""

    def __init__(self, *, families: Any, evidence: Any, rhea: Any, proteins: Any, official_rhea: Any | None = None) -> None:
        self.families = families
        self.evidence = evidence
        self.rhea = rhea
        self.proteins = proteins
        self.official_rhea = official_rhea

    def _summarize_scope(
        self,
        *,
        scope: dict[str, Any],
        member_ids: list[str],
        ui_language: str,
    ) -> dict[str, Any]:
        reaction_members: dict[str, set[str]] = defaultdict(set)
        reaction_sources: dict[str, set[str]] = defaultdict(set)
        evidence_members: set[str] = set()
        for protein_id in member_ids:
            for row in self.evidence.known_reactions(protein_id):
                if not row.reaction_id:
                    continue
                reaction_members[row.reaction_id].add(protein_id)
                reaction_sources[row.reaction_id].add(row.source)
                evidence_members.add(protein_id)
            if self.official_rhea is not None:
                try:
                    official_ids = self.official_rhea.known_rhea_ids(protein_id)
                except Exception:
                    official_ids = []
                for reaction_id in official_ids:
                    reaction_members[reaction_id].add(protein_id)
                    reaction_sources[reaction_id].add("rhea_swissprot")
                    evidence_members.add(protein_id)

        ranked_reactions = sorted(
            reaction_members.items(),
            key=lambda item: (-len(item[1]), str(item[0])),
        )
        total_reaction_count = len(ranked_reactions)
        display_reactions = ranked_reactions[:30]
        items: list[dict[str, Any]] = []
        for index, (reaction_id, members) in enumerate(display_reactions):
            rhea_url = f"https://www.rhea-db.org/rhea/{reaction_id.split(':')[-1]}"
            local_meta = self.evidence.reaction_metadata(reaction_id) or {}
            equation = str(local_meta.get("equation") or local_meta.get("reaction_smiles") or "").strip()
            # The merged universe already carries local structure metadata for most
            # reactions. Only fall back to live Rhea enrichment when the local record
            # has no useful display text, keeping broad class queries off the network
            # hot path while preserving canonical Rhea links.
            if not equation and (total_reaction_count <= 12 or index < 12):
                try:
                    record = self.rhea.exact(reaction_id)
                except Exception:
                    record = None
                if record is not None:
                    equation = str(record.equation or "")
                    rhea_url = str(record.url or rhea_url)
            items.append(
                {
                    "candidate_id": reaction_id,
                    "name": equation,
                    "rhea_url": rhea_url,
                    "source": "integrated_family_evidence",
                    "family_support_count": len(members),
                    "family_member_count": len(member_ids),
                    "family_support_fraction": len(members) / max(1, len(member_ids)),
                    "supporting_member_ids": sorted(members)[:12],
                    "evidence_sources": sorted(reaction_sources[reaction_id]),
                    "model_score": None,
                }
            )

        zh = str(ui_language or "").lower().startswith("zh")
        family_like = scope.get("scope_type") == "auditable_family"
        route_id = "e2r-family-evidence-v1" if family_like else "e2r-functional-class-evidence-v1"
        if family_like:
            note = (
                "家族级结果只汇总成员的数据库已记录反应；不会把整个家族虚构成一条平均蛋白序列进行神经预测。需要预测潜在反应时，请进一步选择具体成员或提供具体序列。"
                if zh
                else "Family-level results aggregate recorded reactions across member proteins. The neural E2R model is sequence-specific, so no fictitious average-family sequence is predicted; choose a concrete member or sequence for model exploration."
            )
        else:
            note = (
                "该结果汇总当前功能类成员范围内的数据库已记录反应。功能类不是一条具体蛋白序列，因此不会虚构平均序列进行神经预测；需要预测潜在反应时，请进一步选择具体成员或提供序列。"
                if zh
                else "This result aggregates database-recorded reactions across the current functional-class member scope. A functional class is not one concrete sequence, so no fictitious average sequence is sent to the neural model; choose a member or sequence for model exploration."
            )
        scope_payload = {
            **scope,
            "member_count": len(member_ids),
            "member_ids_sample": member_ids[:12],
            "evidence_member_count": len(evidence_members),
            "recorded_reaction_count": total_reaction_count,
        }
        return {
            "direction": "enzyme_to_reaction",
            "protein": {
                "id": str(scope.get("family_id") or scope.get("scope_id") or ""),
                "name": str(scope.get("label") or "Protein family/class"),
                "input_mode": "protein_family" if family_like else "protein_functional_class",
                "member_count": len(member_ids),
            },
            "family": scope_payload,
            "known_associations": {
                "count": total_reaction_count,
                "rhea_swissprot_count": 0,
                "project_catalog_count": 0,
                "integrated_database_count": total_reaction_count,
                "items": items,
                "truncated": total_reaction_count > len(items),
                "source_record_url": None,
                "note": note,
            },
            "candidates": [],
            "ranking": {
                "top_k": 0,
                "ranking_objective": "family_recorded_evidence",
                "route_id": route_id,
                "scope": "family_or_class",
                "shot_mode": "not_applicable",
                "score_source": "database_evidence_aggregation",
                "candidate_universe": "resolved_member_scope",
                "candidate_universe_size": len(member_ids),
                "reliability_status": "not_applicable_database_evidence",
            },
            "discovery_filter": {
                "policy": "family_evidence_only",
                "result_mode": "known_associations_only",
                "applied": True,
                "recorded_association_count": total_reaction_count,
                "integrated_database_association_count": total_reaction_count,
                "candidate_universe_recorded_association_count": total_reaction_count,
                "excluded_count": 0,
                "known_ids": [str(reaction_id) for reaction_id, _members in ranked_reactions],
                "source": "integrated_database_family_aggregation",
                "scope_note": note,
            },
            "score_note": note,
            "route_view": {
                "direction": "enzyme_to_reaction",
                "route_id": route_id,
                "base_route_id": route_id,
                "active_overlays": [],
                "title": "蛋白家族/功能类 · 已记录反应汇总" if zh else "Protein family/class · recorded reaction evidence",
                "decision": {"scope": "family_or_class", "objective": "recorded_evidence"},
                "nodes": [
                    {
                        "id": "family-resolve",
                        "title": "确认成员范围" if zh else "Resolve member scope",
                        "subtitle": str(scope.get("label") or ""),
                        "kind": "input",
                        "detail": str(scope.get("scope_note") or scope.get("caution") or ""),
                        "metric": f"{len(member_ids)} members",
                    },
                    {
                        "id": "family-evidence",
                        "title": "汇总成员数据库证据" if zh else "Aggregate member evidence",
                        "subtitle": "Integrated enzyme↔reaction associations",
                        "kind": "evidence",
                        "detail": note,
                        "metric": f"{len(evidence_members)} members with evidence",
                    },
                    {
                        "id": "family-rhea",
                        "title": "合并 Rhea 反应" if zh else "Merge Rhea reactions",
                        "subtitle": "Deduplicate by Rhea ID",
                        "kind": "rank",
                        "detail": "按记录该反应的成员数排序。" if zh else "Rank reactions by the number of members with recorded support.",
                        "metric": f"{total_reaction_count} recorded reactions",
                    },
                    {
                        "id": "family-output",
                        "title": "成员范围反应证据" if zh else "Member-scope reaction evidence",
                        "subtitle": "Database evidence only",
                        "kind": "output",
                        "detail": note,
                        "metric": f"{total_reaction_count} reactions",
                    },
                ],
                "edges": [
                    {"from": "family-resolve", "to": "family-evidence"},
                    {"from": "family-evidence", "to": "family-rhea"},
                    {"from": "family-rhea", "to": "family-output"},
                ],
                "summary": note,
            },
        }

    def summarize(self, family_id: str, *, ui_language: str = "en") -> dict[str, Any]:
        family = self.families.family(family_id)
        if family is None:
            raise AppError(
                "protein_family_not_found",
                "没有找到可核对的蛋白家族范围。",
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        zh = str(ui_language or "").lower().startswith("zh")
        scope = {
            **family.as_dict(),
            "scope_type": "auditable_family",
            "caution": family.caution_zh if zh else family.caution,
            "scope_note": family.scope_note_zh if zh else family.scope_note,
        }
        return self._summarize_scope(scope=scope, member_ids=list(family.member_ids), ui_language=ui_language)

    def summarize_functional_class(
        self,
        enzyme_spec: dict[str, Any],
        *,
        ui_language: str = "en",
    ) -> dict[str, Any]:
        terms = [str(x).strip() for x in enzyme_spec.get("protein_terms") or [] if str(x).strip()]
        raw = str(enzyme_spec.get("raw_text") or "").strip()
        if not terms and raw:
            terms = [raw]
        if not terms:
            raise AppError(
                "protein_class_unresolved",
                "没有识别出可用于构建功能类成员集合的蛋白描述。",
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        rows = self.proteins.search_class_members(
            protein_terms=terms,
            organism_terms=list(enzyme_spec.get("organism_terms") or []),
            gene_terms=list(enzyme_spec.get("gene_terms") or []),
            limit=40,
        )
        member_ids: list[str] = []
        for row in rows:
            identifier = str(row.identifier or "").strip()
            if identifier and identifier not in member_ids:
                member_ids.append(identifier)
        label = raw or terms[0]
        digest = hashlib.sha256("\n".join(x.casefold() for x in terms).encode("utf-8")).hexdigest()[:12].upper()
        zh = str(ui_language or "").lower().startswith("zh")
        scope_note = (
            "当前范围由本地目录与 UniProt 检索得到；反应汇总仅统计这些可核对成员。"
            if zh
            else "This scope comes from the local catalog and UniProt search; reaction evidence is aggregated over these verified members."
        )
        caution = ""
        scope = {
            "scope_id": f"CLASS-{digest}",
            "family_id": f"CLASS-{digest}",
            "label": label,
            "source": "local_catalog+uniprot_functional_class_search",
            "query_scope": "search_derived_functional_class_subset",
            "scope_type": "functional_class",
            "normalized_terms": terms,
            "strict_terms": list(enzyme_spec.get("strict_terms") or []),
            "broader_terms": list(enzyme_spec.get("broader_terms") or []),
            "organism_terms": list(enzyme_spec.get("organism_terms") or []),
            "gene_terms": list(enzyme_spec.get("gene_terms") or []),
            "scope_broadened": bool(enzyme_spec.get("scope_broadened")),
            "scope_note": scope_note,
            "caution": caution,
        }
        return self._summarize_scope(scope=scope, member_ids=member_ids, ui_language=ui_language)
