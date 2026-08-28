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
from scripts.catalyst_finder.open_world_inputs import (
    ProteinSequenceInput,
    detect_direct_open_world_inputs,
    strip_structured_payloads,
)
from scripts.catalyst_finder.protein_resolution import compact_query_terms
from scripts.catalyst_finder.resolution_helpers import (
    candidate_match as _candidate_match,
    explicit_uniprot_accession as _explicit_uniprot_accession,
    fallback_queries as _fallback_queries,
    unique as _unique,
)
from scripts.catalyst_finder.rhea_client import RHEA_ID_RE, RheaCandidate
from scripts.catalyst_finder.task_contracts import VALID_TASK_HINTS


class AgentResolutionService:
    """Resolve user intent and biological entities before any neural ranking runs."""

    def __init__(
        self,
        *,
        catalog: Any,
        evidence: Any,
        rhea: Any,
        deepseek: Any,
        proteins: Any,
        families: Any,
        route_design_resolve: Callable[..., dict[str, Any]],
        pathway_resolve: Callable[..., dict[str, Any]],
    ) -> None:
        self.catalog = catalog
        self.evidence = evidence
        self.rhea = rhea
        self.deepseek = deepseek
        self.proteins = proteins
        self.families = families
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
                "Rhea 中没有找到可核对的反应。请尝试更标准的底物/产物名称，或直接输入 RHEA ID。",
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        scored: list[RheaCandidate] = []
        for candidate in merged.values():
            score, orientation = _candidate_match(candidate.equation, substrate_terms, product_terms)
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
            "interpreted_reaction": interpreted_reaction,
            "assumptions": list(assumptions or []),
            "normalized": {"substrates": substrate_terms, "products": product_terms},
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


    def _direct_open_world_resolution(
        self,
        text: str,
        direction_hint: str,
        ui_language: str,
    ) -> dict[str, Any] | None:
        parsed = detect_direct_open_world_inputs(text)
        if not parsed.has_any:
            return None
        hint = direction_hint if direction_hint in VALID_TASK_HINTS else "auto"
        if parsed.reaction is not None and hint in {"auto", "reaction_to_enzyme"}:
            reaction_candidate = parsed.reaction.as_candidate()
            matching_ids = self.evidence.candidate_reactions_for_smiles(parsed.reaction.reaction_smiles)
            reaction_candidate["matched_reaction_ids"] = matching_ids
            positive_groups = []
            for index, item in enumerate(parsed.protein_sequences):
                candidate = self._sequence_candidate_payload(item)
                positive_groups.append(
                    {
                        "mention_index": index,
                        "mention": item.header or _lang_text(ui_language, "Provided known-active sequence", "用户提供的已知有效酶序列"),
                        "normalized": {},
                        "candidates": [candidate],
                        "recommended_id": candidate["id"],
                    }
                )
            return {
                "direction": "reaction_to_enzyme",
                "summary": _lang_text(
                    ui_language,
                    "Use the provided reaction structure directly for enzyme retrieval.",
                    "直接使用你提供的反应结构进行候选酶检索。",
                ),
                "reaction_resolution": {
                    "mode": "raw_reaction_smiles",
                    "interpreted_reaction": parsed.reaction.reaction_smiles,
                    "assumptions": [],
                    "normalized": {},
                    "candidates": [reaction_candidate],
                    "recommended_id": parsed.reaction.query_id,
                    "matched_reaction_ids": matching_ids,
                },
                "positive_enzyme_resolutions": positive_groups,
                "protein_resolution": None,
                "input_provenance": {"parser": "deterministic_structured_input"},
            }
        residual_text = strip_structured_payloads(text)
        if parsed.protein_sequences and hint in {"auto", "enzyme_to_reaction"} and not residual_text:
            item = parsed.protein_sequences[0]
            candidate = self._sequence_candidate_payload(item)
            return {
                "direction": "enzyme_to_reaction",
                "summary": _lang_text(
                    ui_language,
                    "Use the provided protein sequence directly for reaction retrieval.",
                    "直接使用你提供的蛋白序列预测可能反应。",
                ),
                "reaction_resolution": None,
                "positive_enzyme_resolutions": [],
                "protein_resolution": {
                    "mode": str(candidate.get("input_mode") or "raw_protein_sequence"),
                    "interpreted_protein": item.header or _lang_text(ui_language, "Provided protein sequence", "用户提供的蛋白序列"),
                    "assumptions": [],
                    "normalized": {},
                    "candidates": [candidate],
                    "recommended_id": candidate["id"],
                },
                "input_provenance": {"parser": "deterministic_structured_input"},
            }
        return None


    def agent_resolve(
        self,
        text: str,
        direction_hint: str = "auto",
        conversation_context: dict[str, Any] | None = None,
        ui_language: str = "en",
        resolve_reaction: Callable[[str], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        resolver = resolve_reaction or self.resolve
        text = str(text or "").strip()
        if not text:
            raise AppError("empty_input", "告诉我你想从一个反应找酶，或从一个酶找可能的反应。", HTTPStatus.UNPROCESSABLE_ENTITY)
        hint = direction_hint if direction_hint in VALID_TASK_HINTS else "auto"
        structured_inputs = detect_direct_open_world_inputs(text)
        direct_open_world = self._direct_open_world_resolution(text, hint, ui_language)
        if direct_open_world is not None:
            return direct_open_world
        agent_hint = hint
        semantic_context = dict(conversation_context or {})
        semantic_context["ui_language"] = ui_language

        exact_rhea = re.fullmatch(r"\s*(?:RHEA\s*:\s*)?\d{5}\s*", text, re.IGNORECASE)
        exact_protein = self.proteins.exact_or_search(text, limit=4)
        if exact_rhea and agent_hint in {"auto", "reaction_to_enzyme"}:
            return {
                "direction": "reaction_to_enzyme",
                "summary": _lang_text(ui_language, "Find candidate enzymes for the verified Rhea reaction.", "按 Rhea 反应记录寻找候选酶。"),
                "reaction_resolution": resolver(text),
                "positive_enzyme_resolutions": [],
                "protein_resolution": None,
            }
        if exact_protein and agent_hint in {"auto", "enzyme_to_reaction"}:
            return {
                "direction": "enzyme_to_reaction",
                "summary": _lang_text(ui_language, "Find possible reactions for the verified protein record.", "按已确认蛋白记录预测可能反应。"),
                "reaction_resolution": None,
                "positive_enzyme_resolutions": [],
                "protein_resolution": {
                    "mode": "protein_id",
                    "interpreted_protein": exact_protein[0].name,
                    "assumptions": [],
                    "normalized": {},
                    "candidates": [row.as_dict() for row in exact_protein],
                    "recommended_id": exact_protein[0].identifier,
                },
            }

        semantic_text = (
            strip_structured_payloads(text)
            if structured_inputs.protein_sequences
            else text
        )
        parsed = self.deepseek.interpret_agent_request(
            semantic_text or text,
            agent_hint,
            semantic_context,
            ui_language=ui_language,
        )
        if parsed.get("ambiguity") and float(parsed.get("confidence", 0) or 0) < 0.78:
            return {
                "direction": "ambiguous",
                "summary": parsed.get("summary") or _lang_text(ui_language, "Please confirm the intended task.", "需要确认你的目标任务。"),
                "confidence": parsed.get("confidence", 0),
                "alternative_direction": parsed.get("alternative_direction", ""),
                "ambiguity": True,
                "intent_options": [
                    {"direction": parsed.get("direction", ""), "label": _lang_text(ui_language, "Continue with this interpretation", "按当前理解继续")},
                    {"direction": parsed.get("alternative_direction", ""), "label": _lang_text(ui_language, "Use the alternative interpretation", "另一种理解")},
                ],
                "llm_provenance": {**self.deepseek.provenance(), "used_for": "intent_confirmation"},
            }
        direction = parsed["direction"]
        if direction == "route_design":
            return self.route_design_resolve(text, ui_language=ui_language)
        if direction == "pathway_compatibility":
            return self.pathway_resolve(text, ui_language=ui_language)
        if direction == "reaction_to_enzyme":
            rhea_in_text = RHEA_ID_RE.search(text)
            reaction_spec = parsed.get("reaction") or {}
            if rhea_in_text:
                reaction_resolution = resolver(f"RHEA:{rhea_in_text.group(1)}")
            else:
                substrates = list(reaction_spec.get("substrate_terms") or [])
                products = list(reaction_spec.get("product_terms") or [])
                if not substrates and not products:
                    raw = str(reaction_spec.get("raw_text") or "").strip()
                    if not raw:
                        raise AppError("reaction_parse_empty", "已经理解你要找候选酶，但没有识别出目标反应。", HTTPStatus.UNPROCESSABLE_ENTITY)
                    reaction_resolution = resolver(raw)
                else:
                    reaction_resolution = self._resolve_reaction_from_terms(
                        substrate_terms=substrates,
                        product_terms=products,
                        interpreted_reaction=str(reaction_spec.get("raw_text") or "").strip(),
                    )
            positive_groups = []
            for index, spec in enumerate(parsed.get("positive_enzymes") or []):
                terms = compact_query_terms(spec)
                rows = self.proteins.search(**{**terms, "limit": 6})
                positive_groups.append({
                    "mention_index": index,
                    "mention": str(spec.get("raw_text") or "").strip() or f"阳性酶 {index + 1}",
                    "normalized": terms,
                    "candidates": [row.as_dict() for row in rows],
                    "recommended_id": rows[0].identifier if rows else None,
                })
            for item in structured_inputs.protein_sequences:
                candidate = self._sequence_candidate_payload(item)
                positive_groups.append(
                    {
                        "mention_index": len(positive_groups),
                        "mention": item.header
                        or _lang_text(
                            ui_language,
                            "Provided known-active sequence",
                            "用户提供的已知有效酶序列",
                        ),
                        "normalized": {},
                        "candidates": [candidate],
                        "recommended_id": candidate["id"],
                    }
                )
            return {
                "direction": direction,
                "summary": parsed.get("summary") or _lang_text(ui_language, "Find candidate catalysts for the target reaction.", "寻找目标反应的候选催化酶。"),
                "reaction_resolution": reaction_resolution,
                "positive_enzyme_resolutions": positive_groups,
                "protein_resolution": None,
                "llm_provenance": {**self.deepseek.provenance(), "used_for": "agent_interpretation"},
            }

        if structured_inputs.protein_sequences:
            item = structured_inputs.protein_sequences[0]
            candidate = self._sequence_candidate_payload(item)
            return {
                "direction": direction,
                "summary": parsed.get("summary")
                or _lang_text(
                    ui_language,
                    "Use the provided protein sequence to predict possible reactions.",
                    "使用你提供的蛋白序列预测可能反应。",
                ),
                "reaction_resolution": None,
                "positive_enzyme_resolutions": [],
                "protein_resolution": {
                    "mode": str(candidate.get("input_mode") or "raw_protein_sequence"),
                    "interpreted_protein": item.header
                    or _lang_text(ui_language, "Provided protein sequence", "用户提供的蛋白序列"),
                    "assumptions": [],
                    "normalized": {},
                    "candidates": [candidate],
                    "recommended_id": candidate["id"],
                },
                "llm_provenance": {
                    **self.deepseek.provenance(),
                    "used_for": "intent_only_structured_protein_input",
                },
                "input_provenance": {"parser": "deterministic_structured_input"},
            }

        enzyme_spec = parsed.get("enzyme") or {}
        raw = str(enzyme_spec.get("raw_text") or "").strip()
        accession_from_terms = next(
            (
                str(value).strip()
                for value in (enzyme_spec.get("accession_terms") or [])
                if _probable_uniprot(str(value).strip())
                and str(value).strip().upper() in text.upper()
            ),
            "",
        )
        explicit_accession = (
            _explicit_uniprot_accession(raw)
            or _explicit_uniprot_accession(text)
            or accession_from_terms
        )
        if explicit_accession:
            exact = self.proteins.exact_or_search(explicit_accession, limit=8)
            if exact:
                return {
                    "direction": direction,
                    "summary": parsed.get("summary")
                    or _lang_text(
                        ui_language,
                        "Predict possible reactions for the specified protein.",
                        "预测这个指定蛋白可能催化的反应。",
                    ),
                    "reaction_resolution": None,
                    "positive_enzyme_resolutions": [],
                    "protein_resolution": {
                        "mode": "protein_id",
                        "interpreted_protein": raw or str(exact[0].name),
                        "assumptions": [],
                        "normalized": {"accession": explicit_accession},
                        "candidates": [row.as_dict() for row in exact],
                        "recommended_id": exact[0].identifier,
                    },
                    "llm_provenance": {
                        **self.deepseek.provenance(),
                        "used_for": "agent_interpretation+explicit_protein_resolution",
                    },
                }
        family = self.families.resolve(
            raw or text,
            *(enzyme_spec.get("protein_terms") or []),
            *(enzyme_spec.get("accession_terms") or []),
        )
        if family is not None:
            return {
                "direction": direction,
                "summary": parsed.get("summary")
                or _lang_text(
                    ui_language,
                    "Summarize recorded reactions across the resolved protein family.",
                    "汇总这个蛋白家族成员已有数据库记录的反应。",
                ),
                "reaction_resolution": None,
                "positive_enzyme_resolutions": [],
                "protein_resolution": {
                    "mode": "protein_family",
                    "interpreted_protein": family.label,
                    "assumptions": [],
                    "normalized": {"family_id": family.family_id},
                    "candidates": [],
                    "recommended_id": family.family_id,
                    "family": family.as_dict(),
                },
                "llm_provenance": {
                    **self.deepseek.provenance(),
                    "used_for": "agent_interpretation+family_resolution",
                },
            }
        exact_query = explicit_accession or raw
        exact = self.proteins.exact_or_search(exact_query, limit=8) if exact_query else []
        if exact:
            rows = exact
            normalized = {}
        else:
            terms = compact_query_terms(enzyme_spec)
            if not any(terms.values()):
                return {
                    "direction": direction,
                    "summary": parsed.get("summary") or _lang_text(ui_language, "Predict possible reactions for the target enzyme.", "预测目标酶可能催化的反应。"),
                    "reaction_resolution": None,
                    "positive_enzyme_resolutions": [],
                    "protein_resolution": self.resolve_protein(raw or text),
                    "llm_provenance": {**self.deepseek.provenance(), "used_for": "agent_interpretation"},
                }
            rows = self.proteins.search(**{**terms, "limit": 8})
            normalized = terms
        if not rows:
            raise AppError("protein_no_match", "没有找到可核对的蛋白记录。", HTTPStatus.UNPROCESSABLE_ENTITY)
        return {
            "direction": direction,
            "summary": parsed.get("summary") or _lang_text(ui_language, "Predict possible reactions for the target enzyme.", "预测目标酶可能催化的反应。"),
            "reaction_resolution": None,
            "positive_enzyme_resolutions": [],
            "protein_resolution": {
                "mode": "natural_language",
                "interpreted_protein": raw or str(rows[0].name),
                "assumptions": [],
                "normalized": normalized,
                "candidates": [row.as_dict() for row in rows],
                "recommended_id": rows[0].identifier,
            },
            "llm_provenance": {**self.deepseek.provenance(), "used_for": "agent_interpretation"},
        }


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
