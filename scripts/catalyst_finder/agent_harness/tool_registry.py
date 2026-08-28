from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from scripts.catalyst_finder.agent_harness.contracts import TOOL_ARG_MODELS, ToolName, ToolResult
from scripts.catalyst_finder.errors import AppError
from scripts.catalyst_finder.protein_resolution import compact_query_terms
from scripts.catalyst_finder.formatting import probable_uniprot
from scripts.catalyst_finder.resolution_helpers import explicit_uniprot_accession
from scripts.catalyst_finder.open_world_inputs import detect_direct_open_world_inputs


TOOL_CATALOG: list[dict[str, Any]] = [
    {
        "name": "resolve_reaction",
        "purpose": "Resolve a user-described reaction or explicit RHEA ID to verified Rhea records. Use before factual relation lookup when no trusted reaction_ref exists.",
        "args": {"text": "reaction description or explicit RHEA ID from the user/session"},
    },
    {
        "name": "resolve_protein_scope",
        "purpose": "Resolve a concrete protein, explicit protein family/domain, or functional enzyme class. Family/class requests return a scope_ref rather than choosing one representative protein.",
        "args": {"text": "protein/family/class description", "scope_hint": "specific_protein | family_or_class | auto"},
    },
    {
        "name": "lookup_recorded_associations",
        "purpose": "Look up database-recorded proteins for one verified reaction, optionally intersect/filter by a previously resolved protein scope.",
        "args": {"reaction_ref": "ref returned by resolve_reaction", "protein_scope_ref": "optional ref returned by resolve_protein_scope"},
    },
    {
        "name": "lookup_recorded_protein_reactions",
        "purpose": "Look up database-recorded Rhea reactions for one verified concrete protein. Use this for factual specific-protein → recorded-reaction questions; do not misuse family aggregation tools.",
        "args": {"protein_scope_ref": "specific-protein ref returned by resolve_protein_scope"},
    },
    {
        "name": "list_protein_scope_members",
        "purpose": "List concrete auditable members of a previously resolved protein family or functional-class scope. Use for questions like 'which proteins are in this scope?' rather than reaction aggregation.",
        "args": {"protein_scope_ref": "family/class ref returned by resolve_protein_scope or broaden_protein_scope", "limit": "1..30"},
    },
    {
        "name": "resolve_compound",
        "purpose": "Resolve compound names, common biochemical names, or explicit ChEBI identifiers against the local Rhea/ChEBI route index. The model may provide search synonyms, but only the local index assigns ChEBI IDs.",
        "args": {"terms": "0..8 compound names/search synonyms", "compound_ref": "optional verified ref from this run/session", "limit": "1..8"},
    },
    {
        "name": "inspect_verified_entity",
        "purpose": "Inspect one already verified reaction, concrete protein/family scope, or compound without starting a new search workflow. Use for identity/detail follow-ups such as 'what is this record?', 'which organism?', or 'what structure was resolved?'.",
        "args": {"reaction_ref": "one verified reaction ref", "protein_scope_ref": "one verified protein/family ref", "compound_ref": "one verified compound ref; exactly one ref is required"},
    },
    {
        "name": "compare_verified_entities",
        "purpose": "Compare two to six already verified entities of the same kind using only structured fields returned by scientific tools. Use for factual differences between verified reactions, proteins, compounds, or protein scopes; never compare from model memory.",
        "args": {"entity_refs": "2..6 exact refs from current_run_refs or prior tool results; all refs must identify the same entity kind"},
    },
    {
        "name": "summarize_recorded_relations",
        "purpose": "Aggregate database-recorded reactions across a previously resolved protein family or functional-class scope. It does not run a fictitious family-average neural prediction.",
        "args": {"protein_scope_ref": "ref returned by resolve_protein_scope or broaden_protein_scope"},
    },
    {
        "name": "broaden_protein_scope",
        "purpose": "Explicitly broaden a narrow functional class to its nearest annotation-level parent terms when strict search has no useful evidence. This creates a new approximate scope_ref and must not be presented as strict subtype membership.",
        "args": {"protein_scope_ref": "strict functional-class ref returned by resolve_protein_scope"},
    },
    {
        "name": "prepare_candidate_retrieval",
        "purpose": "Prepare a verified model-ranked candidate workflow after YOU have chosen the direction. Copy entity text from the user's message; do not invent database IDs. Reaction SMILES/FASTA are allowed in full_text. Use only for explicit possible/potential/new/unrecorded/model-ranked candidate requests.",
        "args": {
            "direction": "reaction_to_enzyme | enzyme_to_reaction (required; no auto mode)",
            "full_text": "the user's full request, copied verbatim",
            "reaction_text": "reaction/RHEA phrase copied from the user for reaction_to_enzyme",
            "protein_text": "protein/UniProt/family phrase copied from the user for enzyme_to_reaction",
            "positive_enzyme_texts": "optional known-positive enzyme phrases explicitly supplied by the user",
        },
    },
    {
        "name": "prepare_route_design",
        "purpose": "Resolve source/target compounds and prepare biosynthetic route design confirmation.",
        "args": {"text": "full route-design request"},
    },
    {
        "name": "prepare_pathway_compatibility",
        "purpose": "Resolve steps and prepare joint compatibility evaluation for an already specified multi-step pathway.",
        "args": {"text": "full pathway request"},
    },
]


@dataclass
class HarnessRunContext:
    ui_language: str
    conversation_context: dict[str, Any]
    reaction_refs: dict[str, dict[str, Any]] = field(default_factory=dict)
    protein_refs: dict[str, dict[str, Any]] = field(default_factory=dict)
    compound_refs: dict[str, dict[str, Any]] = field(default_factory=dict)
    terminal_resolution: dict[str, Any] | None = None
    _counter: int = 0

    def new_ref(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"


class ScientificToolRegistry:
    def __init__(
        self,
        *,
        agent_resolution: Any,
        deepseek: Any,
        families: Any,
        family_evidence: Any,
        evidence_queries: Any,
        route_design_resolve: Any,
        pathway_resolve: Any,
        compound_resolve: Any | None = None,
    ) -> None:
        self.agent_resolution = agent_resolution
        self.deepseek = deepseek
        self.families = families
        self.family_evidence = family_evidence
        self.evidence_queries = evidence_queries
        self.route_design_resolve = route_design_resolve
        self.pathway_resolve = pathway_resolve
        self.compound_resolve = compound_resolve

    @staticmethod
    def catalog() -> list[dict[str, Any]]:
        catalog: list[dict[str, Any]] = []
        for item in TOOL_CATALOG:
            entry = dict(item)
            model = TOOL_ARG_MODELS.get(str(item.get("name") or ""))
            if model is not None:
                entry["input_schema"] = model.model_json_schema()
            catalog.append(entry)
        return catalog

    def execute(self, tool: ToolName, args: dict[str, Any], ctx: HarnessRunContext) -> ToolResult:
        model = TOOL_ARG_MODELS[str(tool)]
        try:
            parsed = model.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                tool=tool,
                status="error",
                summary="Tool arguments did not satisfy the typed contract.",
                payload={"validation": exc.errors(include_url=False)[:4]},
                recoverable=True,
                error_code="invalid_tool_arguments",
            )
        try:
            handler = getattr(self, f"_tool_{tool}")
            return handler(parsed, ctx)
        except AppError as exc:
            return ToolResult(
                tool=tool,
                status="error",
                summary=str(exc),
                payload={"detail": str(exc.detail or "")[:600]} if getattr(exc, "detail", None) else {},
                recoverable=getattr(exc, "status", 422) < 500,
                error_code=str(getattr(exc, "code", "tool_error") or "tool_error"),
            )
        except Exception as exc:
            return ToolResult(
                tool=tool,
                status="error",
                summary="The scientific tool did not complete.",
                payload={"detail": f"{type(exc).__name__}: {exc}"[:700]},
                recoverable=False,
                error_code="tool_internal_error",
            )

    def _register_specific_protein_ref(self, ctx: HarnessRunContext, protein_id: str) -> str:
        pid = str(protein_id or "").strip()
        ref = ctx.new_ref("protein_scope")
        ctx.protein_refs[ref] = {
            "kind": "specific_protein",
            "label": pid,
            "resolution": {
                "mode": "session_verified_protein",
                "interpreted_protein": pid,
                "assumptions": [],
                "normalized": {},
                "candidates": [],
                "recommended_id": pid,
            },
        }
        return ref

    def _register_reaction_ref(self, ctx: HarnessRunContext, reaction_id: str) -> str:
        rid = str(reaction_id or "").strip()
        evidence = getattr(self.agent_resolution, "evidence", None)
        meta = evidence.reaction_metadata(rid) or {} if rid.startswith("RHEA:") and evidence is not None and hasattr(evidence, "reaction_metadata") else {}
        candidate = {
            "rhea_id": rid,
            "equation": str(meta.get("equation") or meta.get("reaction_smiles") or rid),
            "orientation": "forward",
        }
        ref = ctx.new_ref("reaction")
        ctx.reaction_refs[ref] = {
            "mode": "session_verified_rhea",
            "interpreted_reaction": rid,
            "assumptions": [],
            "normalized": {},
            "candidates": [candidate] if rid else [],
            "recommended_id": rid,
        }
        return ref

    def _tool_resolve_reaction(self, args: Any, ctx: HarnessRunContext) -> ToolResult:
        text = str(args.text or "").strip()
        structured = detect_direct_open_world_inputs(text)
        if structured.reaction is not None:
            item = structured.reaction
            exact_ids = self.agent_resolution.evidence.candidate_reactions_for_smiles(item.reaction_smiles)
            raw_candidate = item.as_candidate()
            raw_candidate["matched_reaction_ids"] = exact_ids
            exact_candidates: list[dict[str, Any]] = []
            for reaction_id in exact_ids[:5]:
                meta = self.agent_resolution.evidence.reaction_metadata(reaction_id) or {}
                exact_candidates.append({
                    "rhea_id": reaction_id,
                    "equation": str(meta.get("equation") or meta.get("reaction_smiles") or item.reaction_smiles)[:420],
                    "orientation": "forward",
                })
            resolution = {
                "mode": "raw_reaction_smiles",
                "interpreted_reaction": item.reaction_smiles,
                "assumptions": [],
                "normalized": {"reaction_smiles": item.reaction_smiles},
                "candidates": [raw_candidate],
                "recommended_id": item.query_id,
                "matched_reaction_ids": exact_ids,
            }
            ref = ctx.new_ref("reaction")
            ctx.reaction_refs[ref] = resolution
            return ToolResult(
                tool="resolve_reaction",
                status="ok",
                summary=(
                    f"Parsed the user-provided Reaction SMILES as a structurally explicit open-world reaction and found {len(exact_ids)} exact Rhea structure match(es). "
                    "No fuzzy Rhea assignment was made. The Reaction SMILES already defines the structure and direction; absence of an exact Rhea match is an evidence-mapping result, not an input ambiguity. "
                    "For a recorded-evidence question, use lookup_recorded_associations with this reaction_ref; for model discovery, reuse this reaction_ref in prepare_candidate_retrieval."
                ),
                payload={
                    "reaction_ref": ref,
                    "recommended_id": item.query_id,
                    "input_mode": "raw_reaction_smiles",
                    "reaction_smiles": item.reaction_smiles,
                    "exact_rhea_ids": exact_ids,
                    "structure_is_explicit": True,
                    "needs_structure_clarification": False,
                    "recorded_lookup_supported": True,
                    "candidates": exact_candidates,
                },
            )

        resolution = self.agent_resolution.resolve(text)
        ref = ctx.new_ref("reaction")
        ctx.reaction_refs[ref] = resolution
        candidates = list(resolution.get("candidates") or [])
        compact = [
            {
                "rhea_id": str(row.get("rhea_id") or ""),
                "equation": str(row.get("equation") or "")[:420],
                "orientation": str(row.get("orientation") or "forward"),
            }
            for row in candidates[:5]
            if isinstance(row, dict)
        ]
        return ToolResult(
            tool="resolve_reaction",
            status="ok",
            summary=f"Resolved reaction to {len(compact)} verified Rhea candidate(s); recommended {resolution.get('recommended_id') or 'none'}.",
            payload={"reaction_ref": ref, "recommended_id": resolution.get("recommended_id"), "candidates": compact},
        )

    def _tool_resolve_protein_scope(self, args: Any, ctx: HarnessRunContext) -> ToolResult:
        text = str(args.text).strip()
        structured = detect_direct_open_world_inputs(text)
        if structured.protein_sequences:
            item = structured.protein_sequences[0]
            candidate = self.agent_resolution._sequence_candidate_payload(item)
            resolution = {
                "mode": str(candidate.get("input_mode") or "raw_protein_sequence"),
                "interpreted_protein": item.header or "Provided protein sequence",
                "assumptions": [],
                "normalized": {},
                "candidates": [candidate],
                "recommended_id": candidate["id"],
            }
            ref = ctx.new_ref("protein_scope")
            ctx.protein_refs[ref] = {
                "kind": "specific_protein",
                "resolution": resolution,
                "label": resolution["interpreted_protein"],
            }
            return ToolResult(
                tool="resolve_protein_scope",
                status="ok",
                summary="Resolved the user-provided protein sequence as one concrete protein query; no family/class inference was forced.",
                payload={
                    "protein_scope_ref": ref,
                    "scope_kind": "specific_protein",
                    "recommended_id": candidate["id"],
                    "input_mode": candidate.get("input_mode"),
                    "model_ready": candidate.get("model_ready"),
                },
            )
        explicit_accession = explicit_uniprot_accession(text) or probable_uniprot(text)
        if args.scope_hint == "specific_protein" and explicit_accession:
            resolution = self.agent_resolution.resolve_protein(explicit_accession)
            ref = ctx.new_ref("protein_scope")
            ctx.protein_refs[ref] = {
                "kind": "specific_protein",
                "resolution": resolution,
                "label": resolution.get("interpreted_protein") or explicit_accession,
            }
            candidates = list(resolution.get("candidates") or [])
            return ToolResult(
                tool="resolve_protein_scope",
                status="ok",
                summary=f"Resolved the explicitly requested protein {resolution.get('recommended_id') or explicit_accession} without broadening it to a family.",
                payload={
                    "protein_scope_ref": ref,
                    "scope_kind": "specific_protein",
                    "recommended_id": resolution.get("recommended_id"),
                    "candidates": [
                        {"id": row.get("id"), "name": row.get("name"), "organism": row.get("organism")}
                        for row in candidates[:6]
                        if isinstance(row, dict)
                    ],
                },
            )

        family = self.families.resolve(text) if args.scope_hint != "specific_protein" else None
        parsed: dict[str, Any] = {}
        if family is None and args.scope_hint == "family_or_class":
            parsed = self.deepseek.parse_protein(text)
            terms = compact_query_terms(parsed)
            family = self.families.resolve(
                text,
                *(terms.get("protein_terms") or []),
                *(terms.get("accession_terms") or []),
            )
        ref = ctx.new_ref("protein_scope")
        if family is not None:
            scope = {
                "kind": "family",
                "family_id": family.family_id,
                "label": family.label,
                "family": family.as_dict(),
                "enzyme_spec": {
                    "raw_text": text,
                    "protein_terms": [family.label],
                    "organism_terms": [],
                    "gene_terms": [],
                    "accession_terms": [],
                },
            }
            ctx.protein_refs[ref] = scope
            return ToolResult(
                tool="resolve_protein_scope",
                status="ok",
                summary=f"Resolved an auditable protein family scope {family.family_id} ({family.label}) with {len(family.member_ids)} current members.",
                payload={"protein_scope_ref": ref, "scope_kind": "family", "family_id": family.family_id, "label": family.label, "member_count": len(family.member_ids)},
            )

        if args.scope_hint == "family_or_class":
            if not parsed:
                parsed = self.deepseek.parse_protein(text)
            normalized = compact_query_terms(parsed)
            expanded = self.deepseek.expand_protein_class_terms(
                raw_text=text,
                protein_terms=list(normalized.get("protein_terms") or []),
            )
            strict_terms = list(expanded.get("strict_terms") or normalized.get("protein_terms") or [])
            broader_terms = list(expanded.get("broader_terms") or [])
            enzyme_spec = {
                "raw_text": text,
                "protein_terms": strict_terms,
                "strict_terms": strict_terms,
                "broader_terms": broader_terms,
                "organism_terms": list(normalized.get("organism_terms") or []),
                "gene_terms": list(normalized.get("gene_terms") or []),
                "accession_terms": [],
                "scope_broadened": False,
            }
            scope = {"kind": "functional_class", "label": text, "enzyme_spec": enzyme_spec}
            ctx.protein_refs[ref] = scope
            return ToolResult(
                tool="resolve_protein_scope",
                status="ok",
                summary="Resolved a strict search-derived functional-class scope; no single protein was selected as a representative.",
                payload={
                    "protein_scope_ref": ref,
                    "scope_kind": "functional_class",
                    "label": text,
                    "strict_terms": strict_terms,
                    "broader_terms_available": broader_terms,
                },
            )

        resolution = self.agent_resolution.resolve_protein(text)
        mode = str(resolution.get("mode") or "")
        if mode == "protein_family":
            family_payload = resolution.get("family") or {}
            scope = {
                "kind": "family",
                "family_id": resolution.get("recommended_id"),
                "label": resolution.get("interpreted_protein"),
                "family": family_payload,
                "enzyme_spec": {"raw_text": text, "protein_terms": [str(resolution.get("interpreted_protein") or text)], "organism_terms": [], "gene_terms": [], "accession_terms": []},
            }
        else:
            scope = {"kind": "specific_protein", "resolution": resolution, "label": resolution.get("interpreted_protein") or text}
        ctx.protein_refs[ref] = scope
        candidates = list(resolution.get("candidates") or [])
        return ToolResult(
            tool="resolve_protein_scope",
            status="ok",
            summary=f"Resolved protein scope as {scope['kind']} with {len(candidates)} candidate record(s).",
            payload={
                "protein_scope_ref": ref,
                "scope_kind": scope["kind"],
                "recommended_id": resolution.get("recommended_id"),
                "candidates": [
                    {"id": row.get("id"), "name": row.get("name"), "organism": row.get("organism")}
                    for row in candidates[:6]
                    if isinstance(row, dict)
                ],
            },
        )

    def _tool_lookup_recorded_associations(self, args: Any, ctx: HarnessRunContext) -> ToolResult:
        reaction = ctx.reaction_refs.get(args.reaction_ref)
        if reaction is None:
            raise AppError("unknown_reaction_ref", "The reaction_ref is not available in this harness run.", 422)
        reaction_id = str(reaction.get("recommended_id") or "").strip()
        if not reaction_id.startswith("RHEA:"):
            exact_ids = [str(x).strip() for x in reaction.get("matched_reaction_ids") or [] if str(x).strip().startswith("RHEA:")]
            if len(exact_ids) == 1:
                reaction_id = exact_ids[0]
            elif not exact_ids:
                zh = str(ctx.ui_language or "").lower().startswith("zh")
                note = (
                    "当前 Reaction SMILES 没有唯一精确匹配的 Rhea 记录，因此不能把任何 Rhea/UniProt 关联断言为这个结构的数据库已记录催化酶。这表示当前证据映射不足，不代表生物学上不存在已知催化酶。"
                    if zh
                    else "This Reaction SMILES has no unique exact Rhea structure match, so Catalyst Finder cannot assert any Rhea/UniProt association as a database-recorded catalyst for this exact structure. This is an evidence-mapping limitation, not proof that no known catalyst exists."
                )
                raw_id = str(reaction.get("recommended_id") or "").strip()
                raw_equation = str(reaction.get("interpreted_reaction") or "").strip()
                result = {
                    "direction": "reaction_to_enzyme",
                    "answer_mode": "recorded_association_lookup",
                    "reaction": {
                        "rhea_id": raw_id,
                        "equation": raw_equation,
                        "url": None,
                        "input_mode": "raw_reaction_smiles",
                        "exact_rhea_ids": [],
                    },
                    "constraint": None,
                    "known_associations": {
                        "count": 0,
                        "items": [],
                        "truncated": False,
                        "source_record_url": None,
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
                        "candidate_universe_size": 0,
                        "reliability_status": "not_applicable_database_evidence",
                    },
                    "discovery_filter": {
                        "policy": "retain_recorded_associations_only",
                        "result_mode": "known_associations_only",
                        "applied": True,
                        "recorded_association_count": 0,
                        "excluded_count": 0,
                        "known_ids": [],
                        "source": "integrated_database_evidence",
                        "scope_note": note,
                    },
                    "score_note": note,
                }
                ctx.terminal_resolution = {
                    "direction": "reaction_to_enzyme",
                    "operation": "lookup_recorded_associations",
                    "enzyme_scope": "unspecified",
                    "summary": note,
                    "reaction_resolution": reaction,
                    "positive_enzyme_resolutions": [],
                    "protein_resolution": None,
                    "immediate_result": result,
                }
                return ToolResult(
                    tool="lookup_recorded_associations",
                    status="ok",
                    summary=note,
                    payload={"reaction_ref": args.reaction_ref, "exact_rhea_ids": [], "recorded_count": 0, "evidence_mapping": "no_unique_exact_rhea"},
                    terminal=False,
                )
            else:
                return ToolResult(
                    tool="lookup_recorded_associations",
                    status="error",
                    summary="The reaction structure maps to multiple exact Rhea records. Resolve the desired recorded context before asserting enzyme associations, or use the raw reaction directly for model candidate retrieval.",
                    payload={"reaction_ref": args.reaction_ref, "exact_rhea_ids": exact_ids},
                    recoverable=True,
                    error_code="raw_reaction_multiple_exact_rhea",
                )
        if not reaction_id:
            raise AppError("unresolved_reaction_ref", "The referenced reaction has no verified recommended Rhea ID.", 422)
        enzyme_spec: dict[str, Any] = {}
        enzyme_scope = "unspecified"
        if args.protein_scope_ref:
            scope = ctx.protein_refs.get(args.protein_scope_ref)
            if scope is None:
                raise AppError("unknown_protein_scope_ref", "The protein_scope_ref is not available in this harness run.", 422)
            kind = str(scope.get("kind") or "")
            if kind in {"family", "functional_class"}:
                enzyme_scope = "family_or_class"
                enzyme_spec = dict(scope.get("enzyme_spec") or {})
            elif kind == "specific_protein":
                enzyme_scope = "specific_protein"
                resolution = scope.get("resolution") or {}
                pid = str(resolution.get("recommended_id") or "")
                enzyme_spec = {"raw_text": pid, "protein_terms": [], "organism_terms": [], "gene_terms": [], "accession_terms": [pid] if pid else []}
        result = self.evidence_queries.lookup_reaction_proteins(
            reaction_id,
            enzyme_spec=enzyme_spec,
            enzyme_scope=enzyme_scope,
            ui_language=ctx.ui_language,
        )
        ctx.terminal_resolution = {
            "direction": "reaction_to_enzyme",
            "operation": "lookup_recorded_associations",
            "enzyme_scope": enzyme_scope,
            "summary": str((result.get("known_associations") or {}).get("note") or ""),
            "reaction_resolution": reaction,
            "positive_enzyme_resolutions": [],
            "protein_resolution": None,
            "immediate_result": result,
        }
        count = int((result.get("known_associations") or {}).get("count") or 0)
        ids = [str(row.get("candidate_id") or "") for row in (result.get("known_associations") or {}).get("items", [])[:12] if isinstance(row, dict)]
        protein_refs = [
            {"ref": self._register_specific_protein_ref(ctx, protein_id), "protein_id": protein_id}
            for protein_id in ids
            if protein_id
        ]
        return ToolResult(
            tool="lookup_recorded_associations",
            status="ok",
            summary=f"Found {count} database-recorded protein association(s) after applying the requested scope.",
            payload={"recorded_count": count, "protein_ids": ids, "protein_refs": protein_refs},
            terminal=False,
        )

    def _tool_lookup_recorded_protein_reactions(self, args: Any, ctx: HarnessRunContext) -> ToolResult:
        scope = ctx.protein_refs.get(args.protein_scope_ref)
        if scope is None:
            raise AppError("unknown_protein_scope_ref", "The protein_scope_ref is not available in this harness run.", 422)
        if str(scope.get("kind") or "") != "specific_protein":
            raise AppError(
                "scope_not_specific_protein",
                "Recorded protein→reaction lookup requires one concrete protein, not a family or functional-class scope.",
                422,
            )
        resolution = dict(scope.get("resolution") or {})
        protein_id = str(resolution.get("recommended_id") or "").strip()
        if not protein_id:
            raise AppError("specific_protein_unresolved", "The verified protein scope has no concrete identifier.", 422)
        result = self.evidence_queries.lookup_protein_reactions(protein_id, ui_language=ctx.ui_language)
        ctx.terminal_resolution = {
            "direction": "enzyme_to_reaction",
            "operation": "lookup_recorded_protein_reactions",
            "enzyme_scope": "specific_protein",
            "summary": str((result.get("known_associations") or {}).get("note") or ""),
            "reaction_resolution": None,
            "positive_enzyme_resolutions": [],
            "protein_resolution": resolution,
            "immediate_result": result,
        }
        count = int((result.get("known_associations") or {}).get("count") or 0)
        ids = [
            str(row.get("candidate_id") or "")
            for row in (result.get("known_associations") or {}).get("items", [])[:20]
            if isinstance(row, dict)
        ]
        reaction_refs = [
            {"ref": self._register_reaction_ref(ctx, reaction_id), "rhea_id": reaction_id}
            for reaction_id in ids
            if reaction_id.startswith("RHEA:")
        ]
        return ToolResult(
            tool="lookup_recorded_protein_reactions",
            status="ok",
            summary=f"Found {count} database-recorded Rhea reaction association(s) for the verified protein.",
            payload={"recorded_count": count, "reaction_ids": ids, "reaction_refs": reaction_refs, "protein_id": protein_id},
            terminal=False,
        )

    def _tool_list_protein_scope_members(self, args: Any, ctx: HarnessRunContext) -> ToolResult:
        scope = ctx.protein_refs.get(args.protein_scope_ref)
        if scope is None:
            raise AppError("unknown_protein_scope_ref", "The protein_scope_ref is not available in this harness run.", 422)
        kind = str(scope.get("kind") or "")
        limit = int(args.limit)
        entities: list[dict[str, Any]] = []
        scope_note = ""
        scope_id = ""
        label = str(scope.get("label") or "Protein scope")
        if kind == "family":
            family = self.families.family(str(scope.get("family_id") or ""))
            if family is None:
                raise AppError("protein_family_not_found", "The verified family scope is no longer available.", 422)
            scope_id = family.family_id
            scope_note = family.scope_note_zh if str(ctx.ui_language).lower().startswith("zh") else family.scope_note
            for protein_id in list(family.member_ids)[:limit]:
                local = self.agent_resolution.catalog.protein_by_id.get(protein_id, {})
                meta = self.agent_resolution.evidence.protein_metadata(protein_id) or {}
                name = str(local.get("name") or meta.get("name") or protein_id)
                organism = str(local.get("species") or meta.get("species") or "").strip()
                accession = str(meta.get("canonical_accession") or protein_id).strip()
                entities.append({
                    "id": protein_id,
                    "name": name,
                    "subtitle": organism,
                    "url": f"https://www.uniprot.org/uniprotkb/{accession}" if probable_uniprot(accession) else None,
                    "source": family.source,
                    "model_ready": self.agent_resolution.evidence.is_candidate_protein(protein_id),
                })
            total = len(family.member_ids)
        elif kind == "functional_class":
            spec = dict(scope.get("enzyme_spec") or {})
            # Functional-class evidence uses a 40-member auditable cohort. Reuse
            # that same cohort here and slice only for presentation; otherwise a
            # follow-up asking for five members would launch a second UniProt search
            # with a different cache key and could subtly change the scope.
            cohort = self.family_evidence.proteins.search_class_members(
                protein_terms=list(spec.get("protein_terms") or []),
                organism_terms=list(spec.get("organism_terms") or []),
                gene_terms=list(spec.get("gene_terms") or []),
                limit=40,
            )
            rows = cohort[:limit]
            for row in rows:
                entities.append({
                    "id": row.identifier,
                    "name": row.name,
                    "subtitle": row.organism or "",
                    "url": row.url,
                    "source": row.source,
                    "model_ready": row.model_ready,
                })
            total = len(cohort)
            scope_id = str(scope.get("scope_id") or "")
            broadened = bool(spec.get("scope_broadened"))
            scope_note = (
                "这是按放宽后的父类术语得到的可核对成员子集，不代表原始窄功能类的严格成员全集。"
                if broadened and str(ctx.ui_language).lower().startswith("zh")
                else "This is an auditable search-derived subset, not a complete membership definition for the functional class."
            )
        else:
            raise AppError("scope_members_require_family", "Member listing requires a family or functional-class scope.", 422)

        zh = str(ctx.ui_language or "").lower().startswith("zh")
        note = scope_note or (
            "这里只列出当前可核对范围内的成员，不把成员身份解释为已验证催化活性。"
            if zh else
            "This lists members in the current auditable scope; membership is not treated as validated catalytic activity."
        )
        result = {
            "answer_mode": "entity_list",
            "entity_kind": "protein",
            "title": f"{label} · {'成员' if zh else 'members'}",
            "scope": {"id": scope_id, "label": label, "kind": kind, "member_count": total},
            "entities": entities,
            "note": note,
        }
        ctx.terminal_resolution = {
            "direction": "conversation",
            "operation": "list_protein_scope_members",
            "summary": note,
            "reaction_resolution": None,
            "positive_enzyme_resolutions": [],
            "protein_resolution": {
                "mode": "protein_family" if kind == "family" else "protein_functional_class",
                "interpreted_protein": label,
                "assumptions": [],
                "normalized": {"scope_id": scope_id},
                "candidates": [],
                "recommended_id": scope_id,
            },
            "immediate_result": result,
        }
        return ToolResult(
            tool="list_protein_scope_members",
            status="ok",
            summary=f"Listed {len(entities)} concrete member(s) from the verified {kind} scope.",
            payload={"entity_count": len(entities), "scope_kind": kind, "scope_id": scope_id},
            terminal=True,
        )

    def _tool_resolve_compound(self, args: Any, ctx: HarnessRunContext) -> ToolResult:
        if self.compound_resolve is None:
            raise AppError("compound_resolver_unavailable", "Compound resolution is not configured.", 503)
        compound_ref = str(args.compound_ref or "").strip()
        if compound_ref:
            verified = ctx.compound_refs.get(compound_ref)
            if verified is None:
                raise AppError("unknown_compound_ref", "The compound_ref is not available in this harness run.", 422)
            chebi_id = str(verified.get("chebi_id") or "").strip()
            rows = list(self.compound_resolve([chebi_id], limit=1) or []) if chebi_id else []
            terms = [chebi_id] if chebi_id else []
        else:
            terms = [str(value).strip() for value in args.terms if str(value).strip()]
            rows = list(self.compound_resolve(terms, limit=int(args.limit)) or [])
        if not rows:
            return ToolResult(
                tool="resolve_compound",
                status="error",
                summary="No compound candidate could be verified in the local ChEBI/Rhea index for the supplied terms.",
                payload={"terms": terms},
                recoverable=True,
                error_code="compound_not_found",
            )
        entities: list[dict[str, Any]] = []
        refs: list[dict[str, str]] = []
        for row in rows:
            chebi_id = str(row.get("chebi_id") or "").strip()
            if not chebi_id:
                continue
            ref = ctx.new_ref("compound")
            normalized = {
                "chebi_id": chebi_id,
                "name": str(row.get("name") or chebi_id),
                "smiles": str(row.get("smiles") or ""),
            }
            ctx.compound_refs[ref] = normalized
            refs.append({"ref": ref, "chebi_id": chebi_id})
            entities.append({
                "id": chebi_id,
                "name": normalized["name"],
                "subtitle": normalized["smiles"],
                "url": f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId={chebi_id}",
                "source": "local_rhea_chebi_index",
            })
        if not entities:
            raise AppError("compound_not_found", "Compound candidates were empty after local-index validation.", 422)
        zh = str(ctx.ui_language or "").lower().startswith("zh")
        note = (
            "ChEBI 编号来自当前本地 Rhea/ChEBI 索引；搜索术语只用于召回候选，模型不会直接指定数据库编号。"
            if zh else
            "ChEBI identifiers come from the local Rhea/ChEBI index; model-provided search terms retrieve candidates but do not assign database IDs."
        )
        result = {
            "answer_mode": "entity_list",
            "entity_kind": "compound",
            "title": "化合物核对" if zh else "Compound resolution",
            "scope": {"terms": terms, "candidate_count": len(entities)},
            "entities": entities,
            "note": note,
        }
        ctx.terminal_resolution = {
            "direction": "conversation",
            "operation": "resolve_compound",
            "summary": note,
            "reaction_resolution": None,
            "positive_enzyme_resolutions": [],
            "protein_resolution": None,
            "compound_resolution": {
                "terms": terms,
                "candidates": rows,
                "recommended_id": entities[0]["id"],
            },
            "immediate_result": result,
        }
        return ToolResult(
            tool="resolve_compound",
            status="ok",
            summary=f"Resolved {len(entities)} ChEBI candidate(s) from the local compound index.",
            payload={"compound_refs": refs, "candidate_ids": [entity["id"] for entity in entities]},
            terminal=False,
        )

    def _tool_inspect_verified_entity(self, args: Any, ctx: HarnessRunContext) -> ToolResult:
        zh = str(ctx.ui_language or "").lower().startswith("zh")
        entity_kind = ""
        entity: dict[str, Any]
        note = ""
        reaction_resolution = None
        protein_resolution = None
        compound_resolution = None

        if args.reaction_ref:
            resolution = ctx.reaction_refs.get(str(args.reaction_ref))
            if resolution is None:
                raise AppError("unknown_reaction_ref", "The reaction_ref is not available in this harness run.", 422)
            reaction_resolution = dict(resolution)
            reaction_id = str(resolution.get("recommended_id") or "").strip()
            interpreted = str(resolution.get("interpreted_reaction") or "").strip()
            candidates = [row for row in resolution.get("candidates") or [] if isinstance(row, dict)]
            selected = next((row for row in candidates if str(row.get("rhea_id") or "") == reaction_id), candidates[0] if candidates else {})
            meta = self.agent_resolution.evidence.reaction_metadata(reaction_id) or {} if reaction_id.startswith("RHEA:") else {}
            equation = str(selected.get("equation") or meta.get("equation") or "").strip()
            raw_fallback = interpreted if str(resolution.get("mode") or "") == "raw_reaction_smiles" else ""
            reaction_smiles = str(meta.get("reaction_smiles") or (resolution.get("normalized") or {}).get("reaction_smiles") or raw_fallback).strip()
            entity_kind = "reaction"
            entity = {
                "id": reaction_id or interpreted,
                "name": equation or reaction_smiles or interpreted or reaction_id,
                "subtitle": reaction_smiles if reaction_smiles and reaction_smiles != equation else "",
                "url": f"https://www.rhea-db.org/rhea/{reaction_id.split(':')[-1]}" if reaction_id.startswith("RHEA:") else None,
                "source": "verified_rhea_record" if reaction_id.startswith("RHEA:") else "verified_open_world_reaction",
            }
            exact_ids = [str(x) for x in resolution.get("matched_reaction_ids") or [] if str(x).startswith("RHEA:")]
            note = (
                (f"这是已核对的 Rhea 反应记录 {reaction_id}。" if reaction_id.startswith("RHEA:") else f"这是当前会话中已核对的开放世界反应结构；精确 Rhea 匹配为 {', '.join(exact_ids) if exact_ids else '无'}。")
                if zh else
                (f"Verified Rhea reaction record {reaction_id}." if reaction_id.startswith("RHEA:") else f"Verified open-world reaction structure in this session; exact Rhea matches: {', '.join(exact_ids) if exact_ids else 'none' }.")
            )
        elif args.protein_scope_ref:
            scope = ctx.protein_refs.get(str(args.protein_scope_ref))
            if scope is None:
                raise AppError("unknown_protein_scope_ref", "The protein_scope_ref is not available in this harness run.", 422)
            kind = str(scope.get("kind") or "")
            if kind == "specific_protein":
                resolution = dict(scope.get("resolution") or {})
                protein_resolution = resolution
                protein_id = str(resolution.get("recommended_id") or "").strip()
                candidates = [row for row in resolution.get("candidates") or [] if isinstance(row, dict)]
                selected = next((row for row in candidates if str(row.get("id") or "") == protein_id), candidates[0] if candidates else {})
                local = self.agent_resolution.catalog.protein_by_id.get(protein_id, {})
                meta = self.agent_resolution.evidence.protein_metadata(protein_id) or {}
                name = str(selected.get("name") or local.get("name") or meta.get("name") or protein_id)
                organism = str(selected.get("organism") or local.get("species") or meta.get("species") or "").strip()
                accession = str(selected.get("accession") or meta.get("canonical_accession") or protein_id).strip()
                detail_source = "verified_protein_record"
                if probable_uniprot(accession) and (not organism or name == protein_id):
                    try:
                        exact = self.agent_resolution.proteins.detail_for(accession)
                    except Exception:
                        exact = None
                    if exact is not None:
                        name = str(exact.name or name or protein_id)
                        organism = str(exact.organism or organism or "").strip()
                        accession = str(exact.accession or accession).strip()
                        detail_source = str(exact.source or detail_source)
                entity_kind = "protein"
                entity = {
                    "id": protein_id,
                    "name": name,
                    "subtitle": organism,
                    "url": f"https://www.uniprot.org/uniprotkb/{accession}" if probable_uniprot(accession) else None,
                    "source": detail_source,
                    "model_ready": self.agent_resolution.evidence.is_candidate_protein(protein_id),
                }
                note = ("这是当前会话中已经核对的具体蛋白记录。" if zh else "This is the concrete protein record already verified in the current session.")
            elif kind in {"family", "functional_class"}:
                scope_id = str(scope.get("family_id") or scope.get("scope_id") or "").strip()
                label = str(scope.get("label") or scope_id or "Protein scope")
                entity_kind = "protein_scope"
                entity = {
                    "id": scope_id,
                    "name": label,
                    "subtitle": "可审计家族范围" if kind == "family" and zh else "检索得到的功能类范围" if zh else "auditable family scope" if kind == "family" else "search-derived functional-class scope",
                    "url": None,
                    "source": "verified_protein_scope",
                }
                protein_resolution = {
                    "mode": "protein_family" if kind == "family" else "protein_functional_class",
                    "interpreted_protein": label,
                    "recommended_id": scope_id,
                    "candidates": [],
                }
                note = ("这是已经核对的查询范围；成员身份本身不等于催化活性证据。" if zh else "This is a verified query scope; membership itself is not catalytic-activity evidence.")
            else:
                raise AppError("unsupported_protein_scope", "The verified protein scope cannot be inspected.", 422)
        else:
            verified = ctx.compound_refs.get(str(args.compound_ref))
            if verified is None:
                raise AppError("unknown_compound_ref", "The compound_ref is not available in this harness run.", 422)
            chebi_id = str(verified.get("chebi_id") or "").strip()
            row = dict(verified)
            if (not str(row.get("name") or "").strip() or not str(row.get("smiles") or "").strip()) and self.compound_resolve is not None and chebi_id:
                hits = list(self.compound_resolve([chebi_id], limit=1) or [])
                if hits:
                    row = dict(hits[0])
            entity_kind = "compound"
            entity = {
                "id": chebi_id,
                "name": str(row.get("name") or chebi_id),
                "subtitle": str(row.get("smiles") or ""),
                "url": f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId={chebi_id}" if chebi_id else None,
                "source": "local_rhea_chebi_index",
            }
            compound_resolution = {"recommended_id": chebi_id, "candidates": [row]}
            note = ("该 ChEBI 身份来自当前会话中已经核对的本地 Rhea/ChEBI 索引结果。" if zh else "This ChEBI identity comes from the locally verified Rhea/ChEBI index result in the current session.")

        result = {
            "answer_mode": "entity_list",
            "entity_kind": entity_kind,
            "title": "实体详情" if zh else "Verified entity details",
            "entities": [entity],
            "note": note,
        }
        ctx.terminal_resolution = {
            "direction": "conversation",
            "operation": "inspect_verified_entity",
            "summary": note,
            "reaction_resolution": reaction_resolution,
            "positive_enzyme_resolutions": [],
            "protein_resolution": protein_resolution,
            "compound_resolution": compound_resolution,
            "immediate_result": result,
        }
        return ToolResult(
            tool="inspect_verified_entity",
            status="ok",
            summary=f"Returned verified details for one {entity_kind} entity.",
            payload={"entity_kind": entity_kind, "entity_id": entity.get("id")},
            terminal=True,
        )

    def _tool_compare_verified_entities(self, args: Any, ctx: HarnessRunContext) -> ToolResult:
        zh = str(ctx.ui_language or "").lower().startswith("zh")
        entities: list[dict[str, Any]] = []
        kinds: list[str] = []
        for ref in args.entity_refs:
            inspect_args: dict[str, str]
            if ref in ctx.reaction_refs:
                inspect_args = {"reaction_ref": ref}
            elif ref in ctx.protein_refs:
                inspect_args = {"protein_scope_ref": ref}
            elif ref in ctx.compound_refs:
                inspect_args = {"compound_ref": ref}
            else:
                raise AppError("unknown_entity_ref", f"The entity ref {ref} is not available in this harness run.", 422)
            temp_ctx = HarnessRunContext(
                ui_language=ctx.ui_language,
                conversation_context=ctx.conversation_context,
                reaction_refs=ctx.reaction_refs,
                protein_refs=ctx.protein_refs,
                compound_refs=ctx.compound_refs,
            )
            inspected = self.execute("inspect_verified_entity", inspect_args, temp_ctx)
            if inspected.status != "ok" or not temp_ctx.terminal_resolution:
                raise AppError("entity_inspection_failed", inspected.summary, 422)
            immediate = temp_ctx.terminal_resolution.get("immediate_result") or {}
            entity_rows = [row for row in immediate.get("entities") or [] if isinstance(row, dict)]
            if not entity_rows:
                raise AppError("entity_inspection_empty", "The verified entity had no structured detail row.", 422)
            entities.append(dict(entity_rows[0]))
            kinds.append(str(immediate.get("entity_kind") or ""))

        unique_kinds = {kind for kind in kinds if kind}
        if len(unique_kinds) != 1:
            raise AppError(
                "comparison_kind_mismatch",
                "Verified entity comparison currently requires refs of the same entity kind.",
                422,
            )
        kind = next(iter(unique_kinds))
        if kind == "reaction":
            field_specs = [("equation", "反应式" if zh else "Equation", "name"), ("reaction_smiles", "Reaction SMILES", "subtitle")]
        elif kind == "protein":
            field_specs = [("name", "蛋白名称" if zh else "Protein name", "name"), ("organism", "物种" if zh else "Organism", "subtitle"), ("model_ready", "模型候选库覆盖" if zh else "Active model coverage", "model_ready")]
        elif kind == "compound":
            field_specs = [("name", "化合物名称" if zh else "Compound name", "name"), ("smiles", "SMILES", "subtitle")]
        else:
            field_specs = [("name", "范围名称" if zh else "Scope name", "name"), ("scope_type", "范围类型" if zh else "Scope type", "subtitle")]

        comparison_rows: list[dict[str, Any]] = []
        for key, label, source_key in field_specs:
            values: list[str] = []
            for entity in entities:
                value = entity.get(source_key)
                if source_key == "model_ready":
                    value = ("是" if bool(value) else "否") if zh else ("yes" if bool(value) else "no")
                values.append(str(value or ""))
            comparison_rows.append({"key": key, "label": label, "values": values})

        note = (
            "这里只比较科学工具已经核对出的结构化字段，不补充模型记忆中的数据库事实。"
            if zh else
            "Only structured fields already verified by scientific tools are compared; no database facts are added from model memory."
        )
        result = {
            "answer_mode": "entity_comparison",
            "entity_kind": kind,
            "title": "已核对实体比较" if zh else "Verified entity comparison",
            "entities": entities,
            "comparison_rows": comparison_rows,
            "note": note,
        }
        ctx.terminal_resolution = {
            "direction": "conversation",
            "operation": "compare_verified_entities",
            "summary": note,
            "reaction_resolution": None,
            "positive_enzyme_resolutions": [],
            "protein_resolution": None,
            "immediate_result": result,
        }
        return ToolResult(
            tool="compare_verified_entities",
            status="ok",
            summary=f"Compared {len(entities)} verified {kind} entities using structured tool fields.",
            payload={"entity_kind": kind, "entity_count": len(entities)},
            terminal=True,
        )

    def _tool_summarize_recorded_relations(self, args: Any, ctx: HarnessRunContext) -> ToolResult:
        scope = ctx.protein_refs.get(args.protein_scope_ref)
        if scope is None:
            raise AppError("unknown_protein_scope_ref", "The protein_scope_ref is not available in this harness run.", 422)
        kind = str(scope.get("kind") or "")
        if kind == "family":
            result = self.family_evidence.summarize(str(scope.get("family_id") or ""), ui_language=ctx.ui_language)
        elif kind == "functional_class":
            result = self.family_evidence.summarize_functional_class(dict(scope.get("enzyme_spec") or {}), ui_language=ctx.ui_language)
        else:
            raise AppError("scope_not_aggregatable", "Recorded-relation summarization requires a family or functional-class scope.", 422)
        count = int((result.get("known_associations") or {}).get("count") or 0)
        evidence_members = int((result.get("family") or {}).get("evidence_member_count") or 0)
        if kind == "functional_class" and count == 0 and not bool((scope.get("enzyme_spec") or {}).get("scope_broadened")):
            broader = list((scope.get("enzyme_spec") or {}).get("broader_terms") or [])
            if broader:
                return ToolResult(
                    tool="summarize_recorded_relations",
                    status="error",
                    summary="The strict functional-class scope produced no recorded reaction evidence; an explicit broaden_protein_scope step is available.",
                    payload={"protein_scope_ref": args.protein_scope_ref, "broader_terms_available": broader},
                    recoverable=True,
                    error_code="strict_scope_no_evidence",
                )
        ctx.terminal_resolution = {
            "direction": "enzyme_to_reaction",
            "operation": "summarize_recorded_relations",
            "enzyme_scope": "family_or_class",
            "summary": str((result.get("known_associations") or {}).get("note") or ""),
            "reaction_resolution": None,
            "positive_enzyme_resolutions": [],
            "protein_resolution": {
                "mode": "protein_family" if kind == "family" else "protein_functional_class",
                "interpreted_protein": scope.get("label"),
                "assumptions": [],
                "normalized": {"scope_id": (result.get("protein") or {}).get("id")},
                "candidates": [],
                "recommended_id": (result.get("protein") or {}).get("id"),
                "family": result.get("family") or {},
            },
            "immediate_result": result,
        }
        return ToolResult(
            tool="summarize_recorded_relations",
            status="ok",
            summary=f"Aggregated {count} recorded Rhea reaction(s) across the resolved scope; {evidence_members} member(s) have evidence.",
            payload={"recorded_reaction_count": count, "evidence_member_count": evidence_members, "scope_broadened": bool((scope.get("enzyme_spec") or {}).get("scope_broadened"))},
            terminal=False,
        )

    def _tool_broaden_protein_scope(self, args: Any, ctx: HarnessRunContext) -> ToolResult:
        scope = ctx.protein_refs.get(args.protein_scope_ref)
        if scope is None:
            raise AppError("unknown_protein_scope_ref", "The protein_scope_ref is not available in this harness run.", 422)
        if str(scope.get("kind") or "") != "functional_class":
            raise AppError("scope_not_broadenable", "Only a search-derived functional-class scope can be broadened.", 422)
        spec = dict(scope.get("enzyme_spec") or {})
        broader = [str(x).strip() for x in spec.get("broader_terms") or [] if str(x).strip()]
        if not broader:
            raise AppError("no_broader_scope", "No validated annotation-level parent terms are available for this scope.", 422)
        strict = [str(x).strip() for x in spec.get("strict_terms") or spec.get("protein_terms") or [] if str(x).strip()]
        new_spec = dict(spec)
        new_spec["protein_terms"] = strict + [x for x in broader if x.casefold() not in {v.casefold() for v in strict}]
        new_spec["scope_broadened"] = True
        ref = ctx.new_ref("protein_scope")
        ctx.protein_refs[ref] = {"kind": "functional_class", "label": scope.get("label"), "enzyme_spec": new_spec, "parent_scope_of": args.protein_scope_ref}
        return ToolResult(
            tool="broaden_protein_scope",
            status="ok",
            summary="Created an explicitly broadened functional-class scope using nearest annotation-level parent terms. Results from this scope are approximate parent-class evidence, not strict subtype membership.",
            payload={"protein_scope_ref": ref, "strict_terms": strict, "broader_terms": broader, "approximate_parent_scope": True},
        )

    def _tool_prepare_candidate_retrieval(self, args: Any, ctx: HarnessRunContext) -> ToolResult:
        full_text = str(args.full_text or "").strip()
        structured = detect_direct_open_world_inputs(full_text)
        direction = str(args.direction)
        zh = str(ctx.ui_language or "").lower().startswith("zh")

        if direction == "reaction_to_enzyme":
            if structured.reaction is not None:
                item = structured.reaction
                reaction_candidate = item.as_candidate()
                matching_ids = self.agent_resolution.evidence.candidate_reactions_for_smiles(item.reaction_smiles)
                reaction_candidate["matched_reaction_ids"] = matching_ids
                reaction_resolution = {
                    "mode": "raw_reaction_smiles",
                    "interpreted_reaction": item.reaction_smiles,
                    "assumptions": [],
                    "normalized": {},
                    "candidates": [reaction_candidate],
                    "recommended_id": item.query_id,
                    "matched_reaction_ids": matching_ids,
                }
            else:
                reaction_ref = str(args.reaction_ref or "").strip()
                if reaction_ref:
                    reaction_resolution = ctx.reaction_refs.get(reaction_ref)
                    if reaction_resolution is None:
                        return ToolResult(
                            tool="prepare_candidate_retrieval",
                            status="error",
                            summary="The supplied reaction_ref is not available in this harness run.",
                            payload={"reaction_ref": reaction_ref},
                            recoverable=True,
                            error_code="unknown_reaction_ref",
                        )
                else:
                    reaction_text = str(args.reaction_text or "").strip()
                    if not reaction_text:
                        sequence_only = bool(structured.protein_sequences) and structured.reaction is None
                        return ToolResult(
                            tool="prepare_candidate_retrieval",
                            status="error",
                            summary=(
                                "This request contains a protein sequence but no reaction target. If the user's goal is to find reactions the protein may catalyze, retry prepare_candidate_retrieval with direction=enzyme_to_reaction; do not treat the amino-acid sequence as a reaction description."
                                if sequence_only
                                else "A verified reaction_ref or reaction phrase copied from the user's request is required for reaction-to-enzyme candidate retrieval."
                            ),
                            payload={
                                "missing": "reaction_ref_or_text",
                                "detected_protein_sequence": sequence_only,
                                "suggested_direction": "enzyme_to_reaction" if sequence_only else "",
                            },
                            recoverable=True,
                            error_code="candidate_reaction_missing",
                        )
                    reaction_resolution = self.agent_resolution.resolve(reaction_text)

            positive_groups: list[dict[str, Any]] = []
            for item in structured.protein_sequences:
                candidate = self.agent_resolution._sequence_candidate_payload(item)
                positive_groups.append({
                    "mention_index": len(positive_groups),
                    "mention": item.header or ("用户提供的已知有效酶序列" if zh else "Provided known-active sequence"),
                    "normalized": {},
                    "candidates": [candidate],
                    "recommended_id": candidate["id"],
                })
            for raw_positive in args.positive_enzyme_texts:
                text = str(raw_positive or "").strip()
                if not text:
                    continue
                resolved = self.agent_resolution.resolve_protein(text)
                positive_groups.append({
                    "mention_index": len(positive_groups),
                    "mention": text,
                    "normalized": dict(resolved.get("normalized") or {}),
                    "candidates": list(resolved.get("candidates") or []),
                    "recommended_id": resolved.get("recommended_id"),
                })

            resolution = {
                "direction": "reaction_to_enzyme",
                "summary": "寻找目标反应的模型候选催化酶。" if zh else "Prepare model-ranked candidate catalysts for the target reaction.",
                "reaction_resolution": reaction_resolution,
                "positive_enzyme_resolutions": positive_groups,
                "protein_resolution": None,
                "llm_provenance": {**self.deepseek.provenance(), "used_for": "model_led_candidate_preparation"},
            }
        else:
            if structured.protein_sequences:
                item = structured.protein_sequences[0]
                candidate = self.agent_resolution._sequence_candidate_payload(item)
                protein_resolution = {
                    "mode": str(candidate.get("input_mode") or "raw_protein_sequence"),
                    "interpreted_protein": item.header or ("用户提供的蛋白序列" if zh else "Provided protein sequence"),
                    "assumptions": [],
                    "normalized": {},
                    "candidates": [candidate],
                    "recommended_id": candidate["id"],
                }
            else:
                protein_scope_ref = str(args.protein_scope_ref or "").strip()
                if protein_scope_ref:
                    scope = ctx.protein_refs.get(protein_scope_ref)
                    if scope is None:
                        return ToolResult(
                            tool="prepare_candidate_retrieval",
                            status="error",
                            summary="The supplied protein_scope_ref is not available in this harness run.",
                            payload={"protein_scope_ref": protein_scope_ref},
                            recoverable=True,
                            error_code="unknown_protein_scope_ref",
                        )
                    if str(scope.get("kind") or "") != "specific_protein":
                        return ToolResult(
                            tool="prepare_candidate_retrieval",
                            status="error",
                            summary="Enzyme-to-reaction neural candidate retrieval requires one concrete protein or sequence, not a family/class scope.",
                            payload={"scope_kind": scope.get("kind")},
                            recoverable=True,
                            error_code="candidate_requires_specific_protein",
                        )
                    protein_resolution = dict(scope.get("resolution") or {})
                else:
                    protein_text = str(args.protein_text or "").strip()
                    if not protein_text:
                        return ToolResult(
                            tool="prepare_candidate_retrieval",
                            status="error",
                            summary="A verified specific-protein ref or protein phrase/accession copied from the user's request is required for enzyme-to-reaction candidate retrieval.",
                            payload={"missing": "protein_scope_ref_or_text"},
                            recoverable=True,
                            error_code="candidate_protein_missing",
                        )
                    protein_resolution = self.agent_resolution.resolve_protein(protein_text)

            resolution = {
                "direction": "enzyme_to_reaction",
                "summary": "为目标蛋白准备模型候选反应检索。" if zh else "Prepare model-ranked candidate reactions for the target protein.",
                "reaction_resolution": None,
                "positive_enzyme_resolutions": [],
                "protein_resolution": protein_resolution,
                "llm_provenance": {**self.deepseek.provenance(), "used_for": "model_led_candidate_preparation"},
            }

        ctx.terminal_resolution = resolution
        return ToolResult(
            tool="prepare_candidate_retrieval",
            status="ok",
            summary=f"Prepared the verified {direction} model-candidate workflow without a second task classifier.",
            payload={
                "direction": direction,
                "reaction_id": (resolution.get("reaction_resolution") or {}).get("recommended_id"),
                "protein_id": (resolution.get("protein_resolution") or {}).get("recommended_id"),
                "positive_seed_count": len(resolution.get("positive_enzyme_resolutions") or []),
            },
            terminal=True,
        )

    def _tool_prepare_route_design(self, args: Any, ctx: HarnessRunContext) -> ToolResult:
        resolution = self.route_design_resolve(args.text, ui_language=ctx.ui_language)
        ctx.terminal_resolution = resolution
        route = resolution.get("route_design_resolution") or {}
        return ToolResult(
            tool="prepare_route_design",
            status="ok",
            summary="Resolved route source/target records and prepared route design confirmation.",
            payload={"source_id": route.get("recommended_source_id"), "target_id": route.get("recommended_target_id")},
            terminal=True,
        )

    def _tool_prepare_pathway_compatibility(self, args: Any, ctx: HarnessRunContext) -> ToolResult:
        resolution = self.pathway_resolve(args.text, ui_language=ctx.ui_language)
        ctx.terminal_resolution = resolution
        pathway = resolution.get("pathway_resolution") or {}
        return ToolResult(
            tool="prepare_pathway_compatibility",
            status="ok",
            summary=f"Resolved and prepared a {len(pathway.get('steps') or [])}-step pathway compatibility workflow.",
            payload={"step_count": len(pathway.get("steps") or [])},
            terminal=True,
        )
