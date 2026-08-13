from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage1Signals:
    public_exact: bool
    best_pe_level: int | None
    reviewed: bool
    has_pdb: bool
    structured_experiment: bool
    force_stage2: bool
    force_stage3: bool


def stage1_decision(s: Stage1Signals, promote_pe_at_most: int = 2) -> tuple[str, str]:
    """Cheap pass: finalize obvious low-evidence rows; promote only high-recall signals/focus rows."""
    if s.force_stage3:
        return "PROMOTE", "forced by high-priority focus list for deep verification"
    if s.force_stage2:
        return "PROMOTE", "forced by model-priority focus list"
    if not s.public_exact:
        return "FINALIZE", "no public exact sequence and not focus-prioritized"
    if s.has_pdb:
        return "PROMOTE", "PDB signal requires structured verification"
    if s.structured_experiment:
        return "PROMOTE", "structured experimental annotation signal"
    if s.best_pe_level is not None and s.best_pe_level <= promote_pe_at_most:
        return "PROMOTE", f"protein-existence signal PE{int(s.best_pe_level)}"
    if s.reviewed:
        return "PROMOTE", "reviewed exact-sequence entry merits structured evidence pass"
    return "FINALIZE", "no escalation signal after cheap registry pass"


def stage2_decision(
    touch_level: str,
    has_publication: bool,
    force_stage3: bool,
    unresolved_identity: bool = False,
) -> tuple[str, str]:
    if force_stage3:
        return "PROMOTE", "high-priority focus candidate requires deep verification"
    if unresolved_identity:
        return "PROMOTE", "identity/evidence mapping unresolved after structured pass"
    if has_publication and touch_level in {"T0", "T1", "T2", "T3"}:
        return "PROMOTE", "linked publication may contain unstructured experimental evidence"
    return "FINALIZE", "structured evidence pass sufficient under current policy"
