from __future__ import annotations

import math
import re
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import quote

from scripts.catalyst_finder.errors import AppError
from scripts.catalyst_finder.formatting import (
    lang_text as _lang_text,
    probable_uniprot as _probable_uniprot,
)
from scripts.catalyst_finder.open_world_inputs import ProteinSequenceInput
from scripts.catalyst_finder.protein_resolution import compact_query_terms
from scripts.catalyst_finder.resolution_helpers import (
    candidate_match as _candidate_match,
    explicit_uniprot_accession as _explicit_uniprot_accession,
    fallback_queries as _fallback_queries,
    unique as _unique,
)
from scripts.catalyst_finder.rhea_client import RheaCandidate


class AgentResolutionService:
    """Resolve and verify biological entities used by the scientific harness."""

    def __init__(
        self,
        *,
        catalog: Any,
        evidence: Any,
        rhea: Any,
        deepseek: Any,
        proteins: Any,
        families: Any,
        family_evidence: Any,
        evidence_queries: Any,
        route_design_resolve: Callable[..., dict[str, Any]],
        pathway_resolve: Callable[..., dict[str, Any]],
    ) -> None:
        self.catalog = catalog
        self.evidence = evidence
        self.rhea = rhea
        self.deepseek = deepseek
        self.proteins = proteins
        self.families = families
        self.family_evidence = family_evidence
        self.evidence_queries = evidence_queries
        self.route_design_resolve = route_design_resolve
        self.pathway_resolve = pathway_resolve

    def _resolve_reaction_from_terms(
        self,
        *,
        substrate_terms: list[str],
        product_terms: list[str],
        interpreted_reaction: str = "",
        assumptions: list[str] | None = None,
    ) -> dict[str, Any]:
        working_substrates = list(substrate_terms)
        working_products = list(product_terms)
        working_assumptions = list(assumptions or [])
        merged: dict[str, RheaCandidate] = {}
        hit_counts: dict[str, int] = {}

        def collect(queries: list[str]) -> None:
            for query in _unique(queries)[:14]:
                for candidate in self.rhea.search(query, limit=12):
                    merged.setdefault(candidate.rhea_id, candidate)
                    hit_counts[candidate.rhea_id] = hit_counts.get(candidate.rhea_id, 0) + 1

        collect(_fallback_queries(working_substrates, working_products))
        normalization_source = str(interpreted_reaction or "").strip()
        if not merged:
            if not normalization_source:
                left = " + ".join(working_substrates)
                right = " + ".join(working_products)
                normalization_source = f"{left} -> {right}".strip(" ->")
            try:
                normalized = self.deepseek.parse(normalization_source) if normalization_source else {}
            except AppError:
                normalized = {}
            normalized_substrates = list(normalized.get("substrate_terms") or [])
            normalized_products = list(normalized.get("product_terms") or [])
            if normalized_substrates or normalized_products:
                working_substrates = normalized_substrates or working_substrates
                working_products = normalized_products or working_products
                working_assumptions = _unique(working_assumptions + list(normalized.get("assumptions") or []))
                collect(
                    _fallback_queries(working_substrates, working_products)
                    + list(normalized.get("search_queries") or [])
                )
        if not merged:
            raise AppError(
                "rhea_no_match",
                "Rhea 中没有找到可核对的反应。请尝试更标准的底物/产物名称，或直接输入 RHEA ID。",
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        scored: list[RheaCandidate] = []
        for candidate in merged.values():
            score, orientation = _candidate_match(
                candidate.equation,
                working_substrates,
                working_products,
            )
            hit_count = hit_counts.get(candidate.rhea_id, 0)
            if candidate.enzyme_count and candidate.enzyme_count > 0:
                score += min(0.12, math.log1p(candidate.enzyme_count) * 0.012)
            score += min(0.25, hit_count * 0.05)
            scored.append(RheaCandidate(
                rhea_id=candidate.rhea_id,
                equation=candidate.equation,
                chebi_names=candidate.chebi_names,
                chebi_ids=candidate.chebi_ids,
                enzyme_count=candidate.enzyme_count,
                url=candidate.url,
                orientation=orientation,
                match_score=score,
                hit_count=hit_count,
            ))
        scored.sort(key=lambda row: (row.match_score, row.hit_count, row.enzyme_count or 0), reverse=True)
        top = scored[:5]
        return {
            "mode": "natural_language",
            "interpreted_reaction": interpreted_reaction or normalization_source,
            "assumptions": working_assumptions,
            "normalized": {"substrates": working_substrates, "products": working_products},
            "candidates": [row.as_dict(model_ready=row.rhea_id in self.catalog.reaction_by_id) for row in top],
            "recommended_id": top[0].rhea_id if top else None,
        }


    def resolve_protein(self, text: str) -> dict[str, Any]:
        text = str(text or "").strip()
        if not text:
            raise AppError("empty_protein_input", "请描述一个酶，或输入 UniProt / 本地蛋白 ID。", HTTPStatus.UNPROCESSABLE_ENTITY)
        family = self.families.resolve(text)
        if family is not None:
            return {
                "mode": "protein_family",
                "interpreted_protein": family.label,
                "assumptions": [],
                "normalized": {"family_id": family.family_id},
                "candidates": [],
                "recommended_id": family.family_id,
                "family": family.as_dict(),
            }
        explicit_accession = _explicit_uniprot_accession(text)
        exact = self.proteins.exact_or_search(explicit_accession or text, limit=8)
        if exact:
            return {
                "mode": "protein_id",
                "interpreted_protein": exact[0].name,
                "assumptions": [],
                "normalized": {},
                "candidates": [row.as_dict() for row in exact],
                "recommended_id": exact[0].identifier,
            }
        parsed = self.deepseek.parse_protein(text)
        rows = self.proteins.search(**{**compact_query_terms(parsed), "limit": 8})
        if not rows:
            raise AppError("protein_no_match", "本地模型目录和 UniProt 中都没有找到足够匹配的蛋白。", HTTPStatus.UNPROCESSABLE_ENTITY)
        return {
            "mode": "natural_language",
            "interpreted_protein": parsed.get("interpreted_protein") or text,
            "assumptions": parsed.get("assumptions") or [],
            "normalized": compact_query_terms(parsed),
            "candidates": [row.as_dict() for row in rows],
            "recommended_id": rows[0].identifier,
        }


    def _sequence_candidate_payload(self, item: ProteinSequenceInput) -> dict[str, Any]:
        existing = self.evidence.candidate_protein_for_sequence(item.sequence)
        if existing:
            project_meta = self.catalog.protein_by_id.get(existing, {})
            merged_meta = self.evidence.protein_metadata(existing) or {}
            return {
                "id": existing,
                "accession": _probable_uniprot(existing) or None,
                "name": project_meta.get("name") or item.header or None,
                "organism": project_meta.get("species"),
                "length": int(merged_meta.get("sequence_length") or len(item.sequence)),
                "url": f"https://www.uniprot.org/uniprotkb/{quote(existing, safe='')}" if _probable_uniprot(existing) else None,
                "model_ready": True,
                "input_mode": "general_merged_sequence_match",
                "source": str(merged_meta.get("source_layer") or "general_merged_candidate"),
                "provided_sequence_id": item.query_id,
            }
        return item.as_candidate()


    def resolve(self, text: str) -> dict[str, Any]:
        text = str(text or "").strip()
        if not text:
            raise AppError("empty_input", "请输入底物与产物，或直接输入 RHEA ID。", HTTPStatus.UNPROCESSABLE_ENTITY)
        explicit = re.fullmatch(r"\s*(?:RHEA\s*:\s*)?\d{5}\s*", text, re.IGNORECASE)
        if explicit:
            exact = self.rhea.exact(text)
            return {
                "mode": "rhea_id",
                "interpreted_reaction": exact.equation,
                "assumptions": [],
                "candidates": [exact.as_dict(model_ready=exact.rhea_id in self.catalog.reaction_by_id)],
                "recommended_id": exact.rhea_id,
            }

        parsed = self.deepseek.parse(text)
        substrate_terms = parsed["substrate_terms"]
        product_terms = parsed["product_terms"]
        # The language model only normalizes names/identifiers. Database lookup is built
        # deterministically from those normalized participant terms, so a generated
        # RHEA identifier or free-form query can never become the source of truth.
        queries = _unique(_fallback_queries(substrate_terms, product_terms))[:8]

        merged: dict[str, RheaCandidate] = {}
        hit_counts: dict[str, int] = {}
        for query in queries:
            for candidate in self.rhea.search(query, limit=12):
                merged.setdefault(candidate.rhea_id, candidate)
                hit_counts[candidate.rhea_id] = hit_counts.get(candidate.rhea_id, 0) + 1
        if not merged:
            raise AppError(
                "rhea_no_match",
                "Rhea 中没有找到可核对的反应。请尝试标准英文名称、ChEBI/InChIKey，或直接输入 RHEA ID。",
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )

        scored: list[RheaCandidate] = []
        for candidate in merged.values():
            score, orientation = _candidate_match(candidate.equation, substrate_terms, product_terms)
            hit_count = hit_counts.get(candidate.rhea_id, 0)
            if candidate.enzyme_count and candidate.enzyme_count > 0:
                score += min(0.12, math.log1p(candidate.enzyme_count) * 0.012)
            score += min(0.25, hit_count * 0.05)
            scored.append(
                RheaCandidate(
                    rhea_id=candidate.rhea_id,
                    equation=candidate.equation,
                    chebi_names=candidate.chebi_names,
                    chebi_ids=candidate.chebi_ids,
                    enzyme_count=candidate.enzyme_count,
                    url=candidate.url,
                    orientation=orientation,
                    match_score=score,
                    hit_count=hit_count,
                )
            )
        scored.sort(key=lambda row: (row.match_score, row.hit_count, row.enzyme_count or 0), reverse=True)
        top = scored[:5]
        return {
            "mode": "natural_language",
            "interpreted_reaction": parsed["interpreted_reaction"],
            "assumptions": parsed["assumptions"],
            "normalized": {"substrates": substrate_terms, "products": product_terms},
            "candidates": [row.as_dict(model_ready=row.rhea_id in self.catalog.reaction_by_id) for row in top],
            "recommended_id": top[0].rhea_id if top else None,
        }
