from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ToolName = Literal[
    "resolve_reaction",
    "resolve_protein_scope",
    "lookup_recorded_associations",
    "lookup_recorded_protein_reactions",
    "list_protein_scope_members",
    "resolve_compound",
    "inspect_verified_entity",
    "compare_verified_entities",
    "summarize_recorded_relations",
    "broaden_protein_scope",
    "prepare_candidate_retrieval",
    "prepare_route_design",
    "prepare_pathway_compatibility",
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


class LookupRecordedAssociationsArgs(BaseModel):
    reaction_ref: str = Field(min_length=1, max_length=80)
    protein_scope_ref: str = Field(default="", max_length=80)


class LookupRecordedProteinReactionsArgs(BaseModel):
    protein_scope_ref: str = Field(min_length=1, max_length=80)


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


class InspectVerifiedEntityArgs(BaseModel):
    reaction_ref: str = Field(default="", max_length=80)
    protein_scope_ref: str = Field(default="", max_length=80)
    compound_ref: str = Field(default="", max_length=80)

    @model_validator(mode="after")
    def require_one_ref(self) -> "InspectVerifiedEntityArgs":
        refs = [self.reaction_ref.strip(), self.protein_scope_ref.strip(), self.compound_ref.strip()]
        if sum(bool(value) for value in refs) != 1:
            raise ValueError("inspect_verified_entity requires exactly one verified entity ref")
        return self




class CompareVerifiedEntitiesArgs(BaseModel):
    entity_refs: list[str] = Field(min_length=2, max_length=6)

    @model_validator(mode="after")
    def normalize_refs(self) -> "CompareVerifiedEntitiesArgs":
        refs = [str(value).strip() for value in self.entity_refs if str(value).strip()]
        if len(refs) < 2:
            raise ValueError("compare_verified_entities requires at least two verified refs")
        if len(set(refs)) != len(refs):
            raise ValueError("compare_verified_entities requires distinct verified refs")
        self.entity_refs = refs
        return self


class SummarizeRecordedRelationsArgs(BaseModel):
    protein_scope_ref: str = Field(min_length=1, max_length=80)


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


class PrepareRouteDesignArgs(BaseModel):
    text: str = Field(min_length=1, max_length=6000)


class PreparePathwayCompatibilityArgs(BaseModel):
    text: str = Field(min_length=1, max_length=12000)


TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    "resolve_reaction": ResolveReactionArgs,
    "resolve_protein_scope": ResolveProteinScopeArgs,
    "lookup_recorded_associations": LookupRecordedAssociationsArgs,
    "lookup_recorded_protein_reactions": LookupRecordedProteinReactionsArgs,
    "list_protein_scope_members": ListProteinScopeMembersArgs,
    "resolve_compound": ResolveCompoundArgs,
    "inspect_verified_entity": InspectVerifiedEntityArgs,
    "compare_verified_entities": CompareVerifiedEntitiesArgs,
    "summarize_recorded_relations": SummarizeRecordedRelationsArgs,
    "broaden_protein_scope": BroadenProteinScopeArgs,
    "prepare_candidate_retrieval": PrepareCandidateRetrievalArgs,
    "prepare_route_design": PrepareRouteDesignArgs,
    "prepare_pathway_compatibility": PreparePathwayCompatibilityArgs,
}
