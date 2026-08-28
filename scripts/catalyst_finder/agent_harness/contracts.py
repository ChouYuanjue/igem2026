from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ToolName = Literal[
    "resolve_reaction",
    "resolve_protein_scope",
    "lookup_recorded_associations",
    "summarize_recorded_relations",
    "broaden_protein_scope",
    "prepare_candidate_retrieval",
    "prepare_route_design",
    "prepare_pathway_compatibility",
]


class HarnessAction(BaseModel):
    kind: Literal["tool", "ask_user", "final"]
    tool: ToolName | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    question: str = ""

    @model_validator(mode="after")
    def validate_shape(self) -> "HarnessAction":
        if self.kind == "tool" and self.tool is None:
            raise ValueError("tool action requires a tool")
        if self.kind != "tool" and self.tool is not None:
            raise ValueError("non-tool action must not specify a tool")
        if self.kind == "ask_user" and not self.question.strip():
            raise ValueError("ask_user requires a question")
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


class SummarizeRecordedRelationsArgs(BaseModel):
    protein_scope_ref: str = Field(min_length=1, max_length=80)


class BroadenProteinScopeArgs(BaseModel):
    protein_scope_ref: str = Field(min_length=1, max_length=80)


class PrepareCandidateRetrievalArgs(BaseModel):
    text: str = Field(min_length=1, max_length=12000)
    direction_hint: Literal["auto", "reaction_to_enzyme", "enzyme_to_reaction"] = "auto"


class PrepareRouteDesignArgs(BaseModel):
    text: str = Field(min_length=1, max_length=6000)


class PreparePathwayCompatibilityArgs(BaseModel):
    text: str = Field(min_length=1, max_length=12000)


TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    "resolve_reaction": ResolveReactionArgs,
    "resolve_protein_scope": ResolveProteinScopeArgs,
    "lookup_recorded_associations": LookupRecordedAssociationsArgs,
    "summarize_recorded_relations": SummarizeRecordedRelationsArgs,
    "broaden_protein_scope": BroadenProteinScopeArgs,
    "prepare_candidate_retrieval": PrepareCandidateRetrievalArgs,
    "prepare_route_design": PrepareRouteDesignArgs,
    "prepare_pathway_compatibility": PreparePathwayCompatibilityArgs,
}
