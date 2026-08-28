from __future__ import annotations

from typing import Any

from scripts.catalyst_finder.errors import AppError
from scripts.catalyst_finder.formatting import probable_uniprot as _probable_uniprot
from scripts.catalyst_finder.rhea_client import canonical_rhea_id


class AssociationEvidenceQueryService:
    """Answer finite, factual enzyme↔reaction relation queries.

    This service never invents associations. It starts from the integrated evidence
    graph, then optionally applies a verified family membership intersection or a
    semantic filter over the already-recorded protein rows.
    """

    def __init__(
        self,
        *,
        evidence: Any,
        families: Any,
        proteins: Any,
        rhea: Any,
        deepseek: Any,
        catalog: Any,
    ) -> None:
        self.evidence = evidence
        self.families = families
        self.proteins = proteins
        self.rhea = rhea
        self.deepseek = deepseek
        self.catalog = catalog
        self._protein_display_cache: dict[str, dict[str, Any]] = {}

    def _protein_record(self, protein_id: str) -> dict[str, Any]:
        protein_id = str(protein_id or "").strip()
        cached = self._protein_display_cache.get(protein_id)
        if cached is not None:
            return dict(cached)
        local = self.catalog.protein_by_id.get(protein_id, {})
        record = {
            "id": protein_id,
            "name": str(local.get("name") or protein_id),
            "organism": str(local.get("species") or "") or None,
            "gene_names": [],
            "uniprot_url": f"https://www.uniprot.org/uniprotkb/{protein_id}" if _probable_uniprot(protein_id) else None,
        }
        # General merged metadata intentionally stays compact and often does not carry
        # names. For a finite evidence set, enrich accession-like rows from UniProt so
        # semantic class filtering has auditable names rather than opaque IDs.
        if _probable_uniprot(protein_id) and (record["name"] == protein_id or not record["organism"]):
            try:
                exact = self.proteins.uniprot.exact(protein_id)
            except Exception:
                exact = None
            if exact:
                record.update(
                    {
                        "name": str(exact.get("name") or protein_id),
                        "organism": exact.get("organism"),
                        "gene_names": list(exact.get("gene_names") or []),
                    }
                )
        self._protein_display_cache[protein_id] = dict(record)
        return record

    def lookup_reaction_proteins(
        self,
        reaction_id: str,
        *,
        enzyme_spec: dict[str, Any] | None = None,
        enzyme_scope: str = "unspecified",
        ui_language: str = "en",
    ) -> dict[str, Any]:
        reaction_id = canonical_rhea_id(reaction_id)
        evidence_rows = self.evidence.known_proteins(reaction_id)
        ids: list[str] = []
        sources: dict[str, set[str]] = {}
        for row in evidence_rows:
            protein_id = str(row.canonical_protein_id or row.protein_id or "").strip()
            if not protein_id:
                continue
            if protein_id not in ids:
                ids.append(protein_id)
            sources.setdefault(protein_id, set()).add(str(row.source or "integrated_database"))

        records = {protein_id: self._protein_record(protein_id) for protein_id in ids}
        spec = dict(enzyme_spec or {})
        raw_constraint = str(spec.get("raw_text") or "").strip()
        normalized_terms = [str(x).strip() for x in spec.get("protein_terms") or [] if str(x).strip()]
        constraint_text = raw_constraint or "; ".join(normalized_terms)
        constraint_payload: dict[str, Any] | None = None
        selected_ids = list(ids)

        family = self.families.resolve(
            raw_constraint,
            *normalized_terms,
            *(spec.get("accession_terms") or []),
        ) if constraint_text else None
        # An auditable backend family match is stronger than a probabilistic scope
        # label from the language model. This also makes relation queries robust to
        # small LLM classification variance without hardcoding any one family name.
        if family is not None:
            allowed = set(family.member_ids)
            selected_ids = [protein_id for protein_id in ids if protein_id in allowed]
            constraint_payload = {
                "type": "auditable_family",
                "label": family.label,
                "family": family.as_dict(),
                "selection_source": "exact_membership_intersection",
            }
        elif enzyme_scope == "family_or_class" and constraint_text:
            selection = self.deepseek.select_evidence_records(
                constraint_text=constraint_text,
                records=list(records.values()),
                ui_language=ui_language,
            )
            allowed = set(selection.get("selected_ids") or [])
            selected_ids = [protein_id for protein_id in ids if protein_id in allowed]
            constraint_payload = {
                "type": "semantic_functional_class",
                "label": raw_constraint or (normalized_terms[0] if normalized_terms else constraint_text),
                "selection_source": "semantic_filter_over_recorded_associations",
                "reason": str(selection.get("reason") or ""),
            }
        elif enzyme_scope == "specific_protein":
            explicit = [
                str(x).strip().upper()
                for x in spec.get("accession_terms") or []
                if str(x).strip()
            ]
            if explicit:
                explicit_set = set(explicit)
                selected_ids = [protein_id for protein_id in ids if protein_id.upper() in explicit_set]

        try:
            reaction = self.rhea.exact(reaction_id)
        except Exception:
            reaction = None
        equation = str(getattr(reaction, "equation", "") or "")
        reaction_url = str(getattr(reaction, "url", "") or f"https://www.rhea-db.org/rhea/{reaction_id.split(':')[-1]}")

        items = []
        for protein_id in selected_ids:
            record = records[protein_id]
            items.append(
                {
                    "candidate_id": protein_id,
                    "name": record.get("name"),
                    "species": record.get("organism"),
                    "gene_names": list(record.get("gene_names") or []),
                    "uniprot_url": record.get("uniprot_url"),
                    "source": "integrated_recorded_association",
                    "evidence_sources": sorted(sources.get(protein_id, ())),
                    "model_score": None,
                }
            )

        zh = str(ui_language or "").lower().startswith("zh")
        constraint_label = str((constraint_payload or {}).get("label") or "").strip()
        if constraint_label:
            note = (
                f"在数据库已记录的 {reaction_id} 催化蛋白中，再按“{constraint_label}”约束筛选。"
                if zh
                else f"Filtered the proteins already recorded for {reaction_id} by the constraint “{constraint_label}”."
            )
        else:
            note = (
                f"列出数据库中已记录与 {reaction_id} 关联的蛋白。"
                if zh
                else f"Database-recorded proteins associated with {reaction_id}."
            )
        return {
            "direction": "reaction_to_enzyme",
            "answer_mode": "recorded_association_lookup",
            "reaction": {
                "rhea_id": reaction_id,
                "equation": equation,
                "url": reaction_url,
            },
            "constraint": constraint_payload,
            "known_associations": {
                "count": len(items),
                "rhea_swissprot_count": 0,
                "project_catalog_count": 0,
                "integrated_database_count": len(items),
                "items": items,
                "truncated": False,
                "source_record_url": reaction_url,
                "note": note,
            },
            "candidates": [],
            "ranking": {
                "top_k": 0,
                "ranking_objective": "recorded_association_lookup",
                "route_id": "evidence-association-lookup-v1",
                "scope": "recorded_evidence",
                "shot_mode": "not_applicable",
                "score_source": "database_evidence",
                "candidate_universe": "recorded_associations",
                "candidate_universe_size": len(ids),
                "reliability_status": "not_applicable_database_evidence",
            },
            "discovery_filter": {
                "policy": "retain_recorded_associations_only",
                "result_mode": "known_associations_only",
                "applied": True,
                "recorded_association_count": len(items),
                "integrated_database_association_count": len(items),
                "candidate_universe_recorded_association_count": len(ids),
                "excluded_count": max(0, len(ids) - len(items)),
                "known_ids": [row["candidate_id"] for row in items],
                "source": "integrated_database_evidence",
                "scope_note": note,
            },
            "score_note": note,
            "route_view": {
                "direction": "reaction_to_enzyme",
                "route_id": "evidence-association-lookup-v1",
                "base_route_id": "evidence-association-lookup-v1",
                "active_overlays": [],
                "title": "已记录关联查询" if zh else "Recorded association lookup",
                "decision": {"scope": "recorded_evidence", "objective": "association_lookup"},
                "nodes": [
                    {
                        "id": "evidence-reaction",
                        "title": "确认 Rhea 反应" if zh else "Resolve Rhea reaction",
                        "subtitle": reaction_id,
                        "kind": "input",
                        "detail": equation,
                        "metric": reaction_id,
                    },
                    {
                        "id": "evidence-associations",
                        "title": "读取已记录蛋白关联" if zh else "Read recorded protein associations",
                        "subtitle": "Integrated evidence catalog",
                        "kind": "evidence",
                        "detail": note,
                        "metric": f"{len(ids)} recorded proteins",
                    },
                    {
                        "id": "evidence-constraint",
                        "title": "应用实体约束" if zh else "Apply entity constraint",
                        "subtitle": constraint_label or ("无附加约束" if zh else "No additional constraint"),
                        "kind": "decision",
                        "detail": note,
                        "metric": f"{len(items)} matched proteins",
                    },
                    {
                        "id": "evidence-output",
                        "title": "返回数据库记录" if zh else "Return database records",
                        "subtitle": "Evidence only",
                        "kind": "output",
                        "detail": note,
                        "metric": f"{len(items)} proteins",
                    },
                ],
                "edges": [
                    {"from": "evidence-reaction", "to": "evidence-associations"},
                    {"from": "evidence-associations", "to": "evidence-constraint"},
                    {"from": "evidence-constraint", "to": "evidence-output"},
                ],
                "summary": note,
            },
        }

    def lookup_protein_reactions(
        self,
        protein_id: str,
        *,
        ui_language: str = "en",
    ) -> dict[str, Any]:
        """Return database-recorded reactions for one concrete protein.

        Association identity comes from the integrated local evidence graph. Reaction
        display metadata is read from the merged reaction table, so this factual
        reverse lookup does not require live UniProt/Rhea HTTP calls.
        """
        requested_id = str(protein_id or "").strip()
        canonical_id = self.evidence.canonical_protein_id(requested_id)
        if not canonical_id:
            raise AppError("protein_id_missing", "No concrete protein identifier was provided.", 422)

        rows = self.evidence.known_reactions(canonical_id)
        by_reaction: dict[str, set[str]] = {}
        for row in rows:
            reaction_id = canonical_rhea_id(str(row.reaction_id or ""))
            if not reaction_id:
                continue
            by_reaction.setdefault(reaction_id, set()).add(str(row.source or "integrated_database"))

        items: list[dict[str, Any]] = []
        for reaction_id in sorted(by_reaction):
            meta = self.evidence.reaction_metadata(reaction_id) or {}
            reaction_smiles = str(meta.get("reaction_smiles") or "").strip()
            items.append({
                "candidate_id": reaction_id,
                "name": reaction_smiles or reaction_id,
                "reaction_smiles": reaction_smiles or None,
                "rhea_url": f"https://www.rhea-db.org/rhea/{reaction_id.split(':')[-1]}",
                "source": "integrated_recorded_association",
                "evidence_sources": sorted(by_reaction[reaction_id]),
                "model_score": None,
            })

        protein_meta = self.evidence.protein_metadata(canonical_id) or {}
        local = self.catalog.protein_by_id.get(canonical_id, {})
        protein_name = str(local.get("name") or protein_meta.get("name") or canonical_id)
        species = str(local.get("species") or protein_meta.get("species") or "").strip() or None
        accession = str(protein_meta.get("canonical_accession") or canonical_id).strip()
        protein_url = f"https://www.uniprot.org/uniprotkb/{accession}" if _probable_uniprot(accession) else None
        zh = str(ui_language or "").lower().startswith("zh")
        if items:
            note = (
                f"列出数据库中已记录与 {canonical_id} 关联的反应；这些记录与模型候选分开显示。"
                if zh else
                f"Database-recorded reactions associated with {canonical_id}; these records are kept separate from model-ranked candidates."
            )
        else:
            note = (
                f"当前整合证据库中没有找到 {canonical_id} 的已记录 Rhea 关联。这表示当前证据源没有可核对记录，不等同于证明该蛋白没有催化活性。"
                if zh else
                f"No recorded Rhea association for {canonical_id} was found in the current integrated evidence sources. This is absence of auditable evidence here, not proof of no catalytic activity."
            )

        return {
            "direction": "enzyme_to_reaction",
            "answer_mode": "recorded_protein_reaction_lookup",
            "protein": {
                "id": canonical_id,
                "name": protein_name,
                "species": species,
                "url": protein_url,
                "input_mode": "specific_protein",
            },
            "known_associations": {
                "count": len(items),
                "integrated_database_count": len(items),
                "items": items,
                "truncated": False,
                "source_record_url": protein_url,
                "note": note,
            },
            "candidates": [],
            "ranking": {
                "top_k": 0,
                "ranking_objective": "recorded_protein_reaction_lookup",
                "route_id": "evidence-protein-reaction-lookup-v1",
                "scope": "recorded_evidence",
                "shot_mode": "not_applicable",
                "score_source": "database_evidence",
                "candidate_universe": "recorded_associations",
                "candidate_universe_size": len(items),
                "reliability_status": "not_applicable_database_evidence",
            },
            "discovery_filter": {
                "policy": "retain_recorded_associations_only",
                "result_mode": "known_associations_only",
                "applied": True,
                "recorded_association_count": len(items),
                "integrated_database_association_count": len(items),
                "candidate_universe_recorded_association_count": len(items),
                "excluded_count": 0,
                "known_ids": [item["candidate_id"] for item in items],
                "source": "integrated_database_evidence",
                "scope_note": note,
            },
            "score_note": note,
            "route_view": {
                "direction": "enzyme_to_reaction",
                "route_id": "evidence-protein-reaction-lookup-v1",
                "base_route_id": "evidence-protein-reaction-lookup-v1",
                "active_overlays": [],
                "title": "具体蛋白 · 已记录反应" if zh else "Specific protein · recorded reactions",
                "decision": {"scope": "recorded_evidence", "objective": "protein_reaction_lookup"},
                "nodes": [
                    {
                        "id": "protein-evidence-input",
                        "title": "确认具体蛋白" if zh else "Resolve specific protein",
                        "subtitle": canonical_id,
                        "kind": "input",
                        "detail": protein_name,
                        "metric": canonical_id,
                    },
                    {
                        "id": "protein-evidence-relations",
                        "title": "读取已记录反应" if zh else "Read recorded reactions",
                        "subtitle": "Integrated evidence catalog",
                        "kind": "evidence",
                        "detail": note,
                        "metric": f"{len(items)} recorded reactions",
                    },
                ],
                "edges": [{"from": "protein-evidence-input", "to": "protein-evidence-relations"}],
                "summary": note,
            },
        }
