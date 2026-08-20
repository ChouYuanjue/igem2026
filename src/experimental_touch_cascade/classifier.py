from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SummaryEvidence:
    public_exact: bool
    best_pe_level: int | None = None
    reviewed: bool = False
    has_pdb: bool = False
    functional_experimental: bool = False
    catalytic_experimental: bool = False
    kinetics_present: bool = False
    mass_spec_present: bool = False


def classify(e: SummaryEvidence, stage: int) -> tuple[str, str]:
    """Return simple T0-T5 only; stage depth is stored separately, never encoded in the T label."""
    if not e.public_exact:
        return "T0", "exact sequence absent from current public UniParc snapshot"
    if e.functional_experimental and (e.has_pdb or e.kinetics_present):
        return "T5", "experimental function/catalysis plus structure and/or kinetics"
    if e.functional_experimental:
        return "T4", "published experimental function/catalysis evidence"
    if e.has_pdb:
        return "T3", "experimental structure linked to exact-sequence public accession"
    if e.best_pe_level == 1 or e.mass_spec_present:
        return "T2", "positive protein-level experimental existence evidence"
    return "T1", "public exact sequence but no positive protein-level experiment established"
