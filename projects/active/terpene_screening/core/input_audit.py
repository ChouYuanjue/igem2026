from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
TOLERATED_AA = frozenset("BXZJUO")

@dataclass(frozen=True)
class ProteinInputAudit:
    status: str
    sequence_sha256: str
    sequence_length: int
    invalid_characters: str
    ambiguous_fraction: float
    low_complexity_fraction: float
    warning: str
    def as_columns(self) -> dict[str, object]:
        return {f"protein_input_{key}": value for key, value in asdict(self).items()}

@dataclass(frozen=True)
class ReactionInputAudit:
    status: str
    reaction_sha256: str
    canonical_reaction: str
    drfp_status: str
    fallback_used: bool
    warning: str
    def as_columns(self) -> dict[str, object]:
        return {f"reaction_input_{key}": value for key, value in asdict(self).items()}

def clean_protein_sequence(value: object) -> str:
    return "".join(str(value).upper().split()).rstrip("*")

def audit_protein_sequence(sequence: str, policy: str = "warn") -> tuple[str, ProteinInputAudit]:
    if policy not in {"strict", "warn", "fallback"}:
        raise ValueError(f"Unsupported protein input policy: {policy}")
    cleaned = clean_protein_sequence(sequence)
    invalid = sorted(set(cleaned) - CANONICAL_AA - TOLERATED_AA)
    ambiguous = sum(character in TOLERATED_AA for character in cleaned)
    counts = {character: cleaned.count(character) for character in set(cleaned)}
    dominant = max(counts.values(), default=0)
    low_complexity_fraction = dominant / max(len(cleaned), 1)
    warnings = []
    if not cleaned: warnings.append("empty_sequence")
    if invalid: warnings.append("invalid_characters")
    if len(cleaned) < 20: warnings.append("sequence_too_short")
    if len(cleaned) > 5000: warnings.append("sequence_extremely_long")
    if ambiguous / max(len(cleaned), 1) > 0.05: warnings.append("high_ambiguous_fraction")
    if low_complexity_fraction > 0.5: warnings.append("low_complexity")
    if warnings and policy == "strict":
        raise ValueError(f"Protein input failed validation: {', '.join(warnings)}")
    audit = ProteinInputAudit(
        status="valid" if not warnings else "warning",
        sequence_sha256=hashlib.sha256(cleaned.encode()).hexdigest(),
        sequence_length=len(cleaned),
        invalid_characters="".join(invalid),
        ambiguous_fraction=ambiguous / max(len(cleaned), 1),
        low_complexity_fraction=low_complexity_fraction,
        warning=";".join(warnings),
    )
    return cleaned, audit

def normalize_reaction_text(value: object) -> str:
    return re.sub(r"\s+", "", str(value).strip())

def initial_reaction_audit(reaction: str, canonical: str) -> ReactionInputAudit:
    normalized = normalize_reaction_text(reaction)
    warnings = []
    if not normalized: warnings.append("empty_reaction")
    if ">>" not in normalized: warnings.append("missing_reaction_arrow")
    return ReactionInputAudit(
        status="valid" if not warnings else "warning",
        reaction_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
        canonical_reaction=canonical,
        drfp_status="pending",
        fallback_used=False,
        warning=";".join(warnings),
    )
