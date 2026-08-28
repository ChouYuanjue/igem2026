from __future__ import annotations

from collections import defaultdict
from typing import Any

from scripts.catalyst_finder.errors import AppError
from http import HTTPStatus


class ProteinFamilyEvidenceService:
    """Aggregate recorded enzyme→reaction evidence over a resolved protein family."""

    def __init__(self, *, families: Any, evidence: Any, rhea: Any) -> None:
        self.families = families
        self.evidence = evidence
        self.rhea = rhea

    def summarize(self, family_id: str, *, ui_language: str = "en") -> dict[str, Any]:
        family = self.families.family(family_id)
        if family is None:
            raise AppError(
                "protein_family_not_found",
                "没有找到可核对的蛋白家族范围。",
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )

        reaction_members: dict[str, set[str]] = defaultdict(set)
        reaction_sources: dict[str, set[str]] = defaultdict(set)
        evidence_members: set[str] = set()
        for protein_id in family.member_ids:
            for row in self.evidence.known_reactions(protein_id):
                if not row.reaction_id:
                    continue
                reaction_members[row.reaction_id].add(protein_id)
                reaction_sources[row.reaction_id].add(row.source)
                evidence_members.add(protein_id)

        items: list[dict[str, Any]] = []
        for reaction_id, members in reaction_members.items():
            equation = ""
            rhea_url = f"https://www.rhea-db.org/rhea/{reaction_id.split(':')[-1]}"
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
                    "family_member_count": len(family.member_ids),
                    "family_support_fraction": len(members) / max(1, len(family.member_ids)),
                    "supporting_member_ids": sorted(members)[:12],
                    "evidence_sources": sorted(reaction_sources[reaction_id]),
                    "model_score": None,
                }
            )
        items.sort(key=lambda row: (-int(row["family_support_count"]), str(row["candidate_id"])))

        zh = str(ui_language or "").lower().startswith("zh")
        note = (
            "家族级结果只汇总成员的数据库已记录反应；不会把整个家族虚构成一条平均蛋白序列进行神经预测。需要预测潜在反应时，请进一步选择具体成员或提供具体序列。"
            if zh
            else "Family-level results aggregate recorded reactions across member proteins. The neural E2R model is sequence-specific, so no fictitious average-family sequence is predicted; choose a concrete member or sequence for model exploration."
        )
        caution = family.caution_zh if zh else family.caution
        scope_note = family.scope_note_zh if zh else family.scope_note

        family_payload = {
            **family.as_dict(),
            "evidence_member_count": len(evidence_members),
            "recorded_reaction_count": len(items),
            "caution": caution,
            "scope_note": scope_note,
        }
        return {
            "direction": "enzyme_to_reaction",
            "protein": {
                "id": family.family_id,
                "name": family.label,
                "input_mode": "protein_family",
                "member_count": len(family.member_ids),
            },
            "family": family_payload,
            "known_associations": {
                "count": len(items),
                "rhea_swissprot_count": 0,
                "project_catalog_count": 0,
                "integrated_database_count": len(items),
                "items": items,
                "truncated": False,
                "source_record_url": None,
                "note": note,
            },
            "candidates": [],
            "ranking": {
                "top_k": 0,
                "ranking_objective": "family_recorded_evidence",
                "route_id": "e2r-family-evidence-v1",
                "scope": "family",
                "shot_mode": "not_applicable",
                "score_source": "database_evidence_aggregation",
                "candidate_universe": "family_members",
                "candidate_universe_size": len(family.member_ids),
                "reliability_status": "not_applicable_database_evidence",
            },
            "discovery_filter": {
                "policy": "family_evidence_only",
                "result_mode": "known_associations_only",
                "applied": True,
                "recorded_association_count": len(items),
                "integrated_database_association_count": len(items),
                "candidate_universe_recorded_association_count": len(items),
                "excluded_count": 0,
                "known_ids": [str(row["candidate_id"]) for row in items],
                "source": "integrated_database_family_aggregation",
                "scope_note": note,
            },
            "score_note": note,
            "route_view": {
                "direction": "enzyme_to_reaction",
                "route_id": "e2r-family-evidence-v1",
                "base_route_id": "e2r-family-evidence-v1",
                "active_overlays": [],
                "title": "蛋白家族 · 已记录反应汇总" if zh else "Protein family · recorded reaction evidence",
                "decision": {"scope": "family", "objective": "recorded_evidence"},
                "nodes": [
                    {
                        "id": "family-resolve",
                        "title": "确认家族范围" if zh else "Resolve family scope",
                        "subtitle": family.label,
                        "kind": "input",
                        "detail": caution,
                        "metric": f"{len(family.member_ids)} members",
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
                        "detail": "按支持该反应的家族成员数排序。" if zh else "Rank reactions by the number of family members with recorded support.",
                        "metric": f"{len(items)} recorded reactions",
                    },
                    {
                        "id": "family-output",
                        "title": "家族反应证据" if zh else "Family reaction evidence",
                        "subtitle": "Database evidence only",
                        "kind": "output",
                        "detail": note,
                        "metric": f"{len(items)} reactions",
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
