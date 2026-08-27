from __future__ import annotations

import csv
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

RHEA_RE = re.compile(r"(?:RHEA\s*:\s*)?(\d{5})", re.IGNORECASE)


@dataclass(frozen=True)
class AssociationEvidence:
    protein_id: str
    reaction_id: str
    source: str
    evidence_type: str
    canonical_protein_id: str | None = None


class IntegratedEvidenceCatalog:
    """Known enzyme↔reaction facts kept separate from neural candidate coverage.

    Sources are unioned rather than ranked against one another. A relation being absent
    from the neural model catalog never makes it "novel" when a database source records
    it. Likewise, model readiness never changes entity identity or evidence priority.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.merged_root = self.root / "data/catalyst_candidate_universes/general_merged"
        self.cached_uniprot_rhea = self.root / "data/external/reactzyme/cleaned_uniprot_rhea.tsv"
        self._loaded = False
        self._by_reaction: dict[str, list[AssociationEvidence]] = defaultdict(list)
        self._by_protein: dict[str, list[AssociationEvidence]] = defaultdict(list)
        self._alias_to_canonical: dict[str, str] = {}
        self._metadata_by_canonical: dict[str, dict[str, str]] = {}
        self._reaction_metadata: dict[str, dict[str, str]] = {}
        self._sequence_sha_to_canonical: dict[str, str] = {}
        self._reaction_smiles_to_ids: dict[str, list[str]] = defaultdict(list)

    @staticmethod
    def canonical_rhea(value: str) -> str | None:
        match = RHEA_RE.search(str(value or ""))
        return f"RHEA:{match.group(1)}" if match else None

    @staticmethod
    def _protein_key(value: str) -> str:
        return str(value or "").strip().upper()

    def _load_aliases(self) -> None:
        path = self.merged_root / "protein_metadata.csv"
        if not path.is_file():
            return
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                canonical = self._protein_key(row.get("protein_id") or row.get("canonical_accession") or "")
                if not canonical:
                    continue
                self._metadata_by_canonical[canonical] = {str(k): str(v or "") for k, v in row.items()}
                self._alias_to_canonical[canonical] = canonical
                sequence_sha = str(row.get("sequence_sha256") or "").strip().lower()
                if sequence_sha:
                    self._sequence_sha_to_canonical[sequence_sha] = canonical
                for alias in str(row.get("aliases") or "").split(";"):
                    alias = self._protein_key(alias)
                    if alias:
                        self._alias_to_canonical[alias] = canonical

    def _load_reactions(self) -> None:
        path = self.merged_root / "reactions.csv"
        if not path.is_file():
            return
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                reaction_id = str(row.get("reaction_id") or "").strip()
                if reaction_id:
                    metadata = {
                        str(key): str(value or "") for key, value in row.items()
                    }
                    self._reaction_metadata[reaction_id] = metadata
                    smiles = self._normalize_reaction_smiles(metadata.get("reaction_smiles") or "")
                    if smiles and reaction_id not in self._reaction_smiles_to_ids[smiles]:
                        self._reaction_smiles_to_ids[smiles].append(reaction_id)

    @staticmethod
    def _sequence_sha256(sequence: str) -> str:
        cleaned = "".join(str(sequence or "").upper().split()).rstrip("*")
        return hashlib.sha256(cleaned.encode("utf-8")).hexdigest() if cleaned else ""

    @staticmethod
    def _normalize_reaction_smiles(value: str) -> str:
        return re.sub(r"\s+", "", str(value or "").strip())

    def canonical_protein_id(self, value: str) -> str:
        self._ensure_loaded()
        key = self._protein_key(value)
        return self._alias_to_canonical.get(key, key)

    def protein_metadata(self, value: str) -> dict[str, str] | None:
        self._ensure_loaded()
        return self._metadata_by_canonical.get(self.canonical_protein_id(value))

    def candidate_protein_ids(self) -> set[str]:
        """Canonical protein IDs present in the merged neural candidate universe."""
        self._ensure_loaded()
        return set(self._metadata_by_canonical)

    def candidate_protein_count(self) -> int:
        self._ensure_loaded()
        return len(self._metadata_by_canonical)

    def is_candidate_protein(self, value: str) -> bool:
        self._ensure_loaded()
        canonical = self.canonical_protein_id(value)
        return canonical in self._metadata_by_canonical

    def candidate_protein_for_sequence(self, sequence: str) -> str | None:
        self._ensure_loaded()
        digest = self._sequence_sha256(sequence)
        return self._sequence_sha_to_canonical.get(digest) if digest else None

    def candidate_reaction_ids(self) -> set[str]:
        self._ensure_loaded()
        return set(self._reaction_metadata)

    def candidate_reaction_count(self) -> int:
        self._ensure_loaded()
        return len(self._reaction_metadata)

    def is_candidate_reaction(self, value: str) -> bool:
        self._ensure_loaded()
        reaction_id = str(value or "").strip()
        return reaction_id in self._reaction_metadata

    def reaction_metadata(self, value: str) -> dict[str, str] | None:
        self._ensure_loaded()
        return self._reaction_metadata.get(str(value or "").strip())

    def candidate_reactions_for_smiles(self, reaction_smiles: str) -> list[str]:
        self._ensure_loaded()
        normalized = self._normalize_reaction_smiles(reaction_smiles)
        return list(self._reaction_smiles_to_ids.get(normalized, ())) if normalized else []

    def _add(self, evidence: AssociationEvidence, seen: set[tuple[str, str, str]]) -> None:
        protein = self._protein_key(evidence.protein_id)
        reaction = self.canonical_rhea(evidence.reaction_id)
        if not protein or not reaction:
            return
        canonical = self._alias_to_canonical.get(protein, protein)
        marker = (protein, reaction, evidence.source)
        if marker in seen:
            return
        seen.add(marker)
        normalized = AssociationEvidence(
            protein_id=protein,
            reaction_id=reaction,
            source=evidence.source,
            evidence_type=evidence.evidence_type,
            canonical_protein_id=canonical,
        )
        self._by_reaction[reaction].append(normalized)
        # Index both the reported accession and the canonical exact-sequence representative.
        self._by_protein[protein].append(normalized)
        if canonical != protein:
            self._by_protein[canonical].append(normalized)

    def _load_merged_associations(self, seen: set[tuple[str, str, str]]) -> bool:
        path = self.merged_root / "associations.csv"
        if not path.is_file():
            return False
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                self._add(
                    AssociationEvidence(
                        protein_id=str(row.get("protein_id") or ""),
                        reaction_id=str(row.get("reaction_id") or ""),
                        source=str(row.get("source") or "merged_catalog"),
                        evidence_type=str(row.get("evidence_type") or "recorded_association"),
                    ),
                    seen,
                )
        return True

    def _load_cached_uniprot_rhea(self, seen: set[tuple[str, str, str]]) -> None:
        if not self.cached_uniprot_rhea.is_file():
            return
        with self.cached_uniprot_rhea.open(encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                protein = str(row.get("Entry") or "").strip()
                for token in str(row.get("Rhea ID") or "").split(";"):
                    reaction = self.canonical_rhea(token)
                    if not reaction:
                        continue
                    self._add(
                        AssociationEvidence(
                            protein_id=protein,
                            reaction_id=reaction,
                            source="uniprot_rhea_cached",
                            evidence_type="database_recorded_association",
                        ),
                        seen,
                    )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._load_aliases()
        self._load_reactions()
        seen: set[tuple[str, str, str]] = set()
        # The merged artifact contains project associations plus the cached general
        # UniProt/Rhea corpus. During a first build or degraded deployment, fall back
        # to the source snapshot so evidence lookup remains broader than the model.
        loaded_merged = self._load_merged_associations(seen)
        if not loaded_merged:
            self._load_cached_uniprot_rhea(seen)
        self._loaded = True

    def known_proteins(self, reaction_id: str) -> list[AssociationEvidence]:
        self._ensure_loaded()
        reaction = self.canonical_rhea(reaction_id)
        if not reaction:
            return []
        return list(self._by_reaction.get(reaction, ()))

    def known_reactions(self, protein_id: str) -> list[AssociationEvidence]:
        self._ensure_loaded()
        key = self._protein_key(protein_id)
        if not key:
            return []
        canonical = self._alias_to_canonical.get(key, key)
        rows = list(self._by_protein.get(key, ()))
        if canonical != key:
            rows.extend(self._by_protein.get(canonical, ()))
        dedup: dict[tuple[str, str, str], AssociationEvidence] = {}
        for row in rows:
            dedup[(row.protein_id, row.reaction_id, row.source)] = row
        return list(dedup.values())

    def summary(self) -> dict[str, int | str]:
        self._ensure_loaded()
        unique = {
            (row.protein_id, row.reaction_id, row.source)
            for rows in self._by_reaction.values()
            for row in rows
        }
        return {
            "proteins_with_recorded_reactions": len(self._by_protein),
            "reactions_with_recorded_proteins": len(self._by_reaction),
            "recorded_associations": len(unique),
            "alias_count": len(self._alias_to_canonical),
            "candidate_proteins": len(self._metadata_by_canonical),
            "candidate_reactions": len(self._reaction_metadata),
            "scope": "database_evidence_independent_of_model_candidate_universe",
        }
