from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ToolName = Literal[
    "resolve_reaction",
    "resolve_protein_scope",
    "lookup_relations",
    "list_scope_members",
    "resolve_compound",
    "resolve_literature",
    "inspect_entity",
    "compare_entities",
    "research_workspace",
    "broaden_scope",
    "candidate_search",
    "route_design",
    "pathway_compatibility",
]


class HarnessAction(BaseModel):
    kind: Literal["tool", "respond", "ask_user", "return_result"]
    tool: ToolName | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    question: str = ""
    message: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_provider_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "kind" not in normalized and isinstance(normalized.get("agent_action"), dict):
            wrapped = normalized.get("agent_action")
            if set(normalized).issubset({"agent_action"}):
                normalized = dict(wrapped)
        if "kind" not in normalized:
            alias = str(normalized.get("action") or normalized.get("action_kind") or normalized.get("type") or "").strip()
            if alias in {"tool", "respond", "ask_user", "return_result"}:
                normalized["kind"] = alias
        if "kind" not in normalized:
            tool = str(normalized.get("tool") or "").strip()
            question = str(normalized.get("question") or "").strip()
            message = str(normalized.get("message") or "").strip()
            if tool:
                normalized["kind"] = "tool"
            elif question:
                normalized["kind"] = "ask_user"
            elif message:
                normalized["kind"] = "respond"
        if not str(normalized.get("tool") or "").strip():
            normalized["tool"] = None
        if not isinstance(normalized.get("args"), dict):
            normalized["args"] = {}
        for key in ("reason", "question", "message"):
            if normalized.get(key) is None:
                normalized[key] = ""
        return normalized

    @model_validator(mode="after")
    def validate_shape(self) -> "HarnessAction":
        if self.kind == "tool" and self.tool is None:
            raise ValueError("tool action requires a tool")
        if self.kind != "tool" and self.tool is not None:
            raise ValueError("non-tool action must not specify a tool")
        if self.kind == "ask_user" and not self.question.strip():
            raise ValueError("ask_user requires a question")
        if self.kind == "respond" and not self.message.strip():
            raise ValueError("respond requires a message")
        return self


class ToolResult(BaseModel):
    tool: ToolName
    status: Literal["ok", "error"]
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    terminal: bool = False
    recoverable: bool = True
    error_code: str = ""

    def model_view(self) -> dict[str, Any]:
        """Small tool result safe to feed back to the controller model."""
        return {
            "tool": self.tool,
            "status": self.status,
            "summary": self.summary[:1600],
            "payload": self.payload,
            "terminal": self.terminal,
            "recoverable": self.recoverable,
            "error_code": self.error_code,
        }


class HarnessTraceStep(BaseModel):
    turn: int
    action_kind: str
    tool: str = ""
    status: str = ""
    summary: str = ""


class ResolveReactionArgs(BaseModel):
    text: str = Field(min_length=1, max_length=1200)


class ResolveProteinScopeArgs(BaseModel):
    text: str = Field(min_length=1, max_length=1200)
    scope_hint: Literal["specific_protein", "family_or_class", "auto"] = "auto"


class LookupRelationsArgs(BaseModel):
    reaction_ref: str = Field(default="", max_length=80)
    protein_scope_ref: str = Field(default="", max_length=80)

    @model_validator(mode="after")
    def require_relation_anchor(self) -> "LookupRelationsArgs":
        if not self.reaction_ref.strip() and not self.protein_scope_ref.strip():
            raise ValueError("lookup_relations requires a reaction_ref, protein_scope_ref, or both")
        return self


class ListProteinScopeMembersArgs(BaseModel):
    protein_scope_ref: str = Field(min_length=1, max_length=80)
    limit: int = Field(default=12, ge=1, le=30)


class ResolveCompoundArgs(BaseModel):
    terms: list[str] = Field(default_factory=list, max_length=8)
    compound_ref: str = Field(default="", max_length=80)
    limit: int = Field(default=5, ge=1, le=8)

    @model_validator(mode="after")
    def require_terms_or_ref(self) -> "ResolveCompoundArgs":
        self.terms = [str(value).strip() for value in self.terms if str(value).strip()]
        if not self.terms and not self.compound_ref.strip():
            raise ValueError("resolve_compound requires terms or a verified compound_ref")
        return self


class ResolveLiteratureArgs(BaseModel):
    text: str = Field(min_length=1, max_length=1600)
    limit: int = Field(default=6, ge=1, le=12)


class InspectVerifiedEntityArgs(BaseModel):
    reaction_ref: str = Field(default="", max_length=80)
    protein_scope_ref: str = Field(default="", max_length=80)
    compound_ref: str = Field(default="", max_length=80)
    literature_ref: str = Field(default="", max_length=80)

    @model_validator(mode="after")
    def require_one_ref(self) -> "InspectVerifiedEntityArgs":
        refs = [self.reaction_ref.strip(), self.protein_scope_ref.strip(), self.compound_ref.strip(), self.literature_ref.strip()]
        if sum(bool(value) for value in refs) != 1:
            raise ValueError("inspect_entity requires exactly one verified entity ref")
        return self




class CompareVerifiedEntitiesArgs(BaseModel):
    entity_refs: list[str] = Field(min_length=2, max_length=6)
    comparison_goal: str = Field(default="", max_length=1200)

    @model_validator(mode="after")
    def normalize_refs(self) -> "CompareVerifiedEntitiesArgs":
        refs = [str(value).strip() for value in self.entity_refs if str(value).strip()]
        if len(refs) < 2:
            raise ValueError("compare_entities requires at least two verified refs")
        if len(set(refs)) != len(refs):
            raise ValueError("compare_entities requires distinct verified refs")
        self.entity_refs = refs
        return self


ResearchSection = Literal[
    "annotations",
    "structures",
    "literature",
    "recorded_relations",
    "model",
    "next_steps",
]


class BuildResearchWorkspaceArgs(BaseModel):
    reaction_ref: str = Field(default="", max_length=80)
    protein_scope_ref: str = Field(default="", max_length=80)
    sections: list[ResearchSection] = Field(default_factory=lambda: ["recorded_relations", "model"], min_length=1, max_length=6)
    primary_section: ResearchSection | None = None
    literature_limit: int = Field(default=10, ge=1, le=20)

    @model_validator(mode="after")
    def require_one_supported_ref(self) -> "BuildResearchWorkspaceArgs":
        refs = [self.reaction_ref.strip(), self.protein_scope_ref.strip()]
        if sum(bool(value) for value in refs) != 1:
            raise ValueError("research_workspace requires exactly one reaction_ref or protein_scope_ref")
        normalized: list[str] = []
        for section in self.sections:
            value = str(section).strip()
            if value and value not in normalized:
                normalized.append(value)
        self.sections = normalized  # type: ignore[assignment]
        if self.primary_section and str(self.primary_section) not in self.sections:
            raise ValueError("primary_section must be one of the requested sections")
        return self




class BroadenProteinScopeArgs(BaseModel):
    protein_scope_ref: str = Field(min_length=1, max_length=80)


class PrepareCandidateRetrievalArgs(BaseModel):
    direction: Literal["reaction_to_enzyme", "enzyme_to_reaction"]
    full_text: str = Field(min_length=1, max_length=12000)
    reaction_text: str = Field(default="", max_length=2400)
    protein_text: str = Field(default="", max_length=2400)
    reaction_ref: str = Field(default="", max_length=80)
    protein_scope_ref: str = Field(default="", max_length=80)
    positive_enzyme_texts: list[str] = Field(default_factory=list, max_length=8)
    positive_reaction_texts: list[str] = Field(default_factory=list, max_length=8)
    positive_reaction_refs: list[str] = Field(default_factory=list, max_length=8)
    required_substrate_ref_groups: list[list[str]] = Field(default_factory=list, max_length=6)
    required_product_ref_groups: list[list[str]] = Field(default_factory=list, max_length=6)
    # Backward-compatible singleton groups. New controller actions should prefer
    # *_ref_groups so one user constraint can preserve multiple verified database
    # representations without turning them into a conjunction.
    required_substrate_refs: list[str] = Field(default_factory=list, max_length=8)
    required_product_refs: list[str] = Field(default_factory=list, max_length=8)
    top_k: Literal[3, 5, 10, 20] | None = None
    seed_policy: Literal["default", "none"] | None = None
    known_association_policy: Literal[
        "separate_known", "rank_with_known", "known_only", "exclude_known"
    ] = "separate_known"
    retrieval_scope: Literal["broad", "application_domain"] | None = None
    analysis_depth: Literal["standard", "deep"] | None = None
    enzyme_taxonomy_scope: Literal["all", "eukaryote", "prokaryote"] | None = None
    homology_policy: Literal["allow", "cross_cluster"] | None = None
    target_ph: float | None = Field(default=None, ge=0.0, le=14.0)
    target_temperature_c: float | None = Field(default=None, ge=-20.0, le=150.0)
    target_cofactors: list[str] = Field(default_factory=list, max_length=12)


class PrepareRouteDesignArgs(BaseModel):
    text: str = Field(min_length=1, max_length=6000)


class PreparePathwayCompatibilityArgs(BaseModel):
    text: str = Field(min_length=1, max_length=12000)


TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    "resolve_reaction": ResolveReactionArgs,
    "resolve_protein_scope": ResolveProteinScopeArgs,
    "lookup_relations": LookupRelationsArgs,
    "list_scope_members": ListProteinScopeMembersArgs,
    "resolve_compound": ResolveCompoundArgs,
    "resolve_literature": ResolveLiteratureArgs,
    "inspect_entity": InspectVerifiedEntityArgs,
    "compare_entities": CompareVerifiedEntitiesArgs,
    "research_workspace": BuildResearchWorkspaceArgs,
    "broaden_scope": BroadenProteinScopeArgs,
    "candidate_search": PrepareCandidateRetrievalArgs,
    "route_design": PrepareRouteDesignArgs,
    "pathway_compatibility": PreparePathwayCompatibilityArgs,
}
