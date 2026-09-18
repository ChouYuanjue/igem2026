from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TAXONOMY_SCOPE_REGISTRY = ROOT / "data/terpene_taxonomy_scope/protein_taxonomy_scope.csv"
TAXONOMY_SCOPE_VERSION = "terpene-enzyme-taxonomy-scope-v1"
SUPPORTED_ENZYME_TAXONOMY_SCOPES = {"all", "eukaryote", "prokaryote"}


@dataclass(frozen=True)
class ProteinTaxonomyRecord:
    protein_id: str
    taxonomy_scope: str
    kingdom: str
    species: str
    taxonomy_source: str
    taxonomy_confidence: str


@lru_cache(maxsize=4)
def load_taxonomy_registry(path: str = str(DEFAULT_TAXONOMY_SCOPE_REGISTRY)) -> dict[str, ProteinTaxonomyRecord]:
    registry_path = Path(path)
    if not registry_path.is_file():
        raise FileNotFoundError(
            f"Missing enzyme taxonomy scope registry: {registry_path}. "
            "Run scripts/prepare_terpene_taxonomy_scope.py."
        )
    records: dict[str, ProteinTaxonomyRecord] = {}
    with registry_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            protein_id = str(row.get("protein_id", "")).strip()
            if not protein_id:
                continue
            records[protein_id] = ProteinTaxonomyRecord(
                protein_id=protein_id,
                taxonomy_scope=str(row.get("taxonomy_scope", "unknown") or "unknown").strip(),
                kingdom=str(row.get("kingdom", "") or "").strip(),
                species=str(row.get("species", "") or "").strip(),
                taxonomy_source=str(row.get("taxonomy_source", "unknown") or "unknown").strip(),
                taxonomy_confidence=str(row.get("taxonomy_confidence", "unknown") or "unknown").strip(),
            )
    return records


def validate_scope(scope: str) -> str:
    value = str(scope or "all").strip().lower()
    if value not in SUPPORTED_ENZYME_TAXONOMY_SCOPES:
        raise ValueError(
            f"Unsupported enzyme taxonomy scope {scope!r}; choose from "
            f"{sorted(SUPPORTED_ENZYME_TAXONOMY_SCOPES)}"
        )
    return value


def taxonomy_record(protein_id: str, *, registry_path: Path = DEFAULT_TAXONOMY_SCOPE_REGISTRY) -> ProteinTaxonomyRecord:
    record = load_taxonomy_registry(str(registry_path.resolve())).get(str(protein_id))
    if record is not None:
        return record
    return ProteinTaxonomyRecord(
        protein_id=str(protein_id),
        taxonomy_scope="unknown",
        kingdom="",
        species="",
        taxonomy_source="not_in_registry",
        taxonomy_confidence="unknown",
    )


def filter_candidate_ids(
    candidate_ids: Iterable[str],
    scope: str,
    *,
    registry_path: Path = DEFAULT_TAXONOMY_SCOPE_REGISTRY,
) -> tuple[list[int], dict[str, int]]:
    normalized = validate_scope(scope)
    ids = [str(value) for value in candidate_ids]
    registry = load_taxonomy_registry(str(registry_path.resolve()))
    classes = [registry.get(value).taxonomy_scope if value in registry else "unknown" for value in ids]
    counts = Counter(classes)
    if normalized == "all":
        keep = list(range(len(ids)))
    else:
        keep = [index for index, value in enumerate(classes) if value == normalized]
    audit = {
        "pre_filter_size": len(ids),
        "post_filter_size": len(keep),
        "eukaryote_count": int(counts.get("eukaryote", 0)),
        "prokaryote_count": int(counts.get("prokaryote", 0)),
        "other_count": int(counts.get("other", 0)),
        "unknown_count": int(counts.get("unknown", 0)),
        "excluded_count": len(ids) - len(keep),
    }
    return keep, audit


def validate_seed_scope(
    seed_ids: Iterable[str],
    scope: str,
    *,
    registry_path: Path = DEFAULT_TAXONOMY_SCOPE_REGISTRY,
) -> None:
    normalized = validate_scope(scope)
    if normalized == "all":
        return
    registry = load_taxonomy_registry(str(registry_path.resolve()))
    incompatible: list[str] = []
    unknown: list[str] = []
    for value in [str(item) for item in seed_ids]:
        record = registry.get(value)
        if record is None or record.taxonomy_scope == "unknown":
            unknown.append(value)
        elif record.taxonomy_scope != normalized:
            incompatible.append(f"{value}:{record.taxonomy_scope}")
    if unknown or incompatible:
        fragments: list[str] = []
        if incompatible:
            fragments.append("incompatible=" + ",".join(incompatible))
        if unknown:
            fragments.append("unclassified=" + ",".join(unknown))
        raise ValueError(
            f"Few-shot enzyme seeds must belong to taxonomy scope {normalized!r}; "
            + "; ".join(fragments)
        )


def taxonomy_summary(registry_path: Path = DEFAULT_TAXONOMY_SCOPE_REGISTRY) -> dict[str, int | str]:
    registry = load_taxonomy_registry(str(registry_path.resolve()))
    counts = Counter(record.taxonomy_scope for record in registry.values())
    return {
        "version": TAXONOMY_SCOPE_VERSION,
        "total": len(registry),
        "eukaryote": int(counts.get("eukaryote", 0)),
        "prokaryote": int(counts.get("prokaryote", 0)),
        "other": int(counts.get("other", 0)),
        "unknown": int(counts.get("unknown", 0)),
    }
