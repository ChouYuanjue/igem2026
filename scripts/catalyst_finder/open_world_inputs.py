from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from projects.active.terpene_screening.core.input_audit import (
    CANONICAL_AA,
    TOLERATED_AA,
    audit_protein_sequence,
    normalize_reaction_text,
)

AA_ALPHABET = CANONICAL_AA | TOLERATED_AA | frozenset("*")
REACTION_LABEL_RE = re.compile(
    r"(?:reaction\s*smiles|rxn\s*smiles|reaction_smarts|反应\s*smiles|反应结构)\s*[:：]\s*([^\n]+)",
    re.IGNORECASE,
)
SEQUENCE_LABEL_RE = re.compile(
    r"(?:protein\s+sequence|amino[- ]?acid\s+sequence|fasta|蛋白(?:质)?序列|氨基酸序列)\s*[:：]?\s*(.*)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ProteinSequenceInput:
    query_id: str
    sequence: str
    header: str = ""
    source: str = "raw_sequence"

    def as_candidate(self) -> dict[str, object]:
        return {
            "id": self.query_id,
            "accession": None,
            "name": self.header or "Provided protein sequence",
            "organism": None,
            "length": len(self.sequence),
            "url": None,
            "model_ready": True,
            "input_mode": "raw_protein_sequence",
            "sequence": self.sequence,
            "source": self.source,
        }


@dataclass(frozen=True)
class ReactionStructureInput:
    query_id: str
    reaction_smiles: str
    source: str = "raw_reaction_smiles"

    def as_candidate(self) -> dict[str, object]:
        return {
            "rhea_id": self.query_id,
            "equation": self.reaction_smiles,
            "orientation": "forward",
            "enzyme_count": None,
            "url": None,
            "model_ready": True,
            "input_mode": "raw_reaction_smiles",
            "reaction_smiles": self.reaction_smiles,
            "source": self.source,
        }


@dataclass(frozen=True)
class DirectOpenWorldInputs:
    reaction: ReactionStructureInput | None
    protein_sequences: tuple[ProteinSequenceInput, ...]

    @property
    def has_any(self) -> bool:
        return self.reaction is not None or bool(self.protein_sequences)


def strip_structured_payloads(text: str) -> str:
    """Remove FASTA/sequence payloads while preserving surrounding task prose.

    This is used only for deciding whether a protein sequence is the whole target or
    an attachment to a larger natural-language request. It never supplies biological
    facts and therefore deliberately keeps all non-sequence prose unchanged.
    """
    lines = str(text or "").splitlines()
    kept: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        label = SEQUENCE_LABEL_RE.search(line)
        if stripped.startswith(">"):
            index += 1
            while index < len(lines) and _line_is_sequence(lines[index]):
                index += 1
            continue
        if label:
            prefix = line[: label.start()].strip()
            if prefix:
                kept.append(prefix)
            same_line = str(label.group(1) or "").strip()
            index += 1
            if same_line and not _line_is_sequence(same_line) and not same_line.startswith(">"):
                kept.append(same_line)
            if same_line.startswith(">"):
                pass
            while index < len(lines):
                current = lines[index].strip()
                if current.startswith(">") or _line_is_sequence(current):
                    index += 1
                    continue
                break
            continue
        kept.append(line)
        index += 1
    return "\n".join(kept).strip()


def _digest(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16].upper()
    return f"{prefix}-{digest}"


def stable_protein_query_id(sequence: str) -> str:
    cleaned, _ = audit_protein_sequence(sequence, policy="warn")
    return _digest("EXT-PROT", cleaned)


def stable_reaction_query_id(reaction_smiles: str) -> str:
    return _digest("EXT-RXN", normalize_reaction_text(reaction_smiles))


def _clean_sequence_candidate(value: str) -> str | None:
    compact = "".join(str(value or "").upper().split()).rstrip("*")
    if len(compact) < 20 or len(compact) > 5000:
        return None
    if any(character not in AA_ALPHABET for character in compact):
        return None
    cleaned, audit = audit_protein_sequence(compact, policy="warn")
    if audit.invalid_characters or not cleaned:
        return None
    return cleaned


def _line_is_sequence(line: str) -> bool:
    stripped = "".join(str(line or "").split()).upper().rstrip("*")
    if not stripped:
        return False
    return all(character in AA_ALPHABET for character in stripped)


def _extract_fasta_records(text: str) -> list[tuple[str, str]]:
    lines = str(text or "").splitlines()
    records: list[tuple[str, str]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line.startswith(">"):
            index += 1
            continue
        header = line[1:].strip()
        index += 1
        chunks: list[str] = []
        while index < len(lines):
            current = lines[index].strip()
            if current.startswith(">"):
                break
            if not current:
                if chunks:
                    break
                index += 1
                continue
            if not _line_is_sequence(current):
                break
            chunks.append(current)
            index += 1
        sequence = _clean_sequence_candidate("".join(chunks))
        if sequence:
            records.append((header, sequence))
        if index < len(lines) and lines[index].strip().startswith(">"):
            continue
        index += 1
    return records


def _extract_labeled_sequences(text: str) -> list[tuple[str, str]]:
    lines = str(text or "").splitlines()
    records: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        match = SEQUENCE_LABEL_RE.search(line)
        if not match:
            continue
        same_line = str(match.group(1) or "").strip()
        chunks: list[str] = []
        if same_line and not same_line.startswith(">") and _line_is_sequence(same_line):
            chunks.append(same_line)
        cursor = index + 1
        header = ""
        if same_line.startswith(">"):
            header = same_line[1:].strip()
        while cursor < len(lines):
            current = lines[cursor].strip()
            if current.startswith(">") and not chunks:
                header = current[1:].strip()
                cursor += 1
                continue
            if not current or not _line_is_sequence(current):
                break
            chunks.append(current)
            cursor += 1
        sequence = _clean_sequence_candidate("".join(chunks))
        if sequence:
            records.append((header, sequence))
    return records


def _extract_bare_sequence(text: str) -> list[tuple[str, str]]:
    value = str(text or "").strip()
    if not value or ">" in value or ":" in value:
        return []
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    # Bare sequence mode is intentionally strict: each line must be a compact
    # sequence token. Natural-language sentences contain inter-word whitespace and
    # must use an explicit "protein sequence:" / FASTA label if mixed with prose.
    if not lines or any(re.search(r"\s", line) for line in lines):
        return []
    if not all(_line_is_sequence(line) for line in lines):
        return []
    sequence = _clean_sequence_candidate(value)
    return [("", sequence)] if sequence else []


def extract_protein_sequences(text: str, limit: int = 5) -> tuple[ProteinSequenceInput, ...]:
    seen: set[str] = set()
    result: list[ProteinSequenceInput] = []
    candidates = _extract_fasta_records(text) + _extract_labeled_sequences(text)
    if not candidates:
        candidates = _extract_bare_sequence(text)
    for header, sequence in candidates:
        if sequence in seen:
            continue
        seen.add(sequence)
        result.append(
            ProteinSequenceInput(
                query_id=stable_protein_query_id(sequence),
                sequence=sequence,
                header=header,
                source="fasta" if header else "raw_sequence",
            )
        )
        if len(result) >= limit:
            break
    return tuple(result)


def _reaction_candidate_from_value(value: str) -> str | None:
    normalized = normalize_reaction_text(value)
    if not normalized or normalized.count(">>") != 1:
        return None
    left, right = normalized.split(">>", 1)
    if not left or not right:
        return None
    if len(normalized) > 20000:
        return None
    return normalized


def extract_reaction_smiles(text: str) -> ReactionStructureInput | None:
    value = str(text or "").strip()
    labelled = REACTION_LABEL_RE.search(value)
    candidates: Iterable[str]
    if labelled:
        candidates = [labelled.group(1)]
    else:
        # A raw reaction SMILES is normally a compact token. This deliberately does
        # not treat natural-language arrows (A -> B) as molecular structure.
        compact_lines = [line.strip() for line in value.splitlines() if ">>" in line]
        candidates = compact_lines if compact_lines else ([value] if ">>" in value else [])
    for candidate in candidates:
        # Strip a short prose prefix such as "reaction: " only when the structural
        # token itself remains intact.
        if " " in candidate.strip():
            tokens = [token for token in candidate.strip().split() if ">>" in token]
            if len(tokens) == 1:
                candidate = tokens[0]
        normalized = _reaction_candidate_from_value(candidate)
        if normalized:
            return ReactionStructureInput(
                query_id=stable_reaction_query_id(normalized),
                reaction_smiles=normalized,
            )
    return None


def detect_direct_open_world_inputs(text: str) -> DirectOpenWorldInputs:
    return DirectOpenWorldInputs(
        reaction=extract_reaction_smiles(text),
        protein_sequences=extract_protein_sequences(text),
    )
