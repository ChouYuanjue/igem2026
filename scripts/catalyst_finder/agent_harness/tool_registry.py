from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from scripts.catalyst_finder.agent_harness.contracts import TOOL_ARG_MODELS, ToolName, ToolResult
from scripts.catalyst_finder.errors import AppError
from scripts.catalyst_finder.protein_resolution import compact_query_terms


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
        "purpose": "Prepare the existing verified R2E/E2R candidate-retrieval flow, including raw Reaction SMILES/FASTA and known-positive inputs. Use for possible/potential/new/model-ranked candidate requests, not simple database fact lookup.",
        "args": {"text": "full user request", "direction_hint": "auto | reaction_to_enzyme | enzyme_to_reaction"},
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
    direction_hint: str
    conversation_context: dict[str, Any]
    reaction_refs: dict[str, dict[str, Any]] = field(default_factory=dict)
    protein_refs: dict[str, dict[str, Any]] = field(default_factory=dict)
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
    ) -> None:
        self.agent_resolution = agent_resolution
        self.deepseek = deepseek
        self.families = families
        self.family_evidence = family_evidence
        self.evidence_queries = evidence_queries
        self.route_design_resolve = route_design_resolve
        self.pathway_resolve = pathway_resolve

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

    def _tool_resolve_reaction(self, args: Any, ctx: HarnessRunContext) -> ToolResult:
        resolution = self.agent_resolution.resolve(args.text)
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
        family = self.families.resolve(text)
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
        return ToolResult(
            tool="lookup_recorded_associations",
            status="ok",
            summary=f"Found {count} database-recorded protein association(s) after applying the requested scope.",
            payload={"recorded_count": count, "protein_ids": ids},
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
            terminal=True,
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
        resolution = self.agent_resolution.agent_resolve(
            args.text,
            direction_hint=args.direction_hint,
            conversation_context=ctx.conversation_context,
            ui_language=ctx.ui_language,
            resolve_reaction=self.agent_resolution.resolve,
        )
        if str(resolution.get("direction") or "") not in {"reaction_to_enzyme", "enzyme_to_reaction", "ambiguous"}:
            raise AppError("candidate_tool_wrong_task", "Candidate retrieval preparation resolved to a different task type; choose the corresponding route/pathway tool instead.", 422)
        ctx.terminal_resolution = resolution
        return ToolResult(
            tool="prepare_candidate_retrieval",
            status="ok",
            summary=f"Prepared the verified {resolution.get('direction') or 'candidate'} workflow; user confirmation is retained when needed.",
            payload={"direction": resolution.get("direction"), "has_immediate_result": bool(resolution.get("immediate_result"))},
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
