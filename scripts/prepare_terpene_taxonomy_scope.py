from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_REGISTRY = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu/protein_registry.csv"
MARTS_ENZYMES = ROOT / "data/terpene_marts/marts_enzymes.tsv"
UNIPROT_NORMALIZED = ROOT / "data/terpene_uniprot_expansion/uniprot_tps_normalized.tsv"
DEFAULT_OUT = ROOT / "data/terpene_taxonomy_scope/protein_taxonomy_scope.csv"
VERSION = "terpene-enzyme-taxonomy-scope-v1"


def rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [{key: value or "" for key, value in row.items()} for row in csv.DictReader(handle, delimiter=delimiter)]


def normalized_genus(value: str) -> str:
    text = re.sub(r"\s*\([^)]*\).*?$", "", str(value).strip()).strip()
    return text.split()[0].lower() if text else ""


def taxonomy_scope(kingdom: str) -> str:
    value = str(kingdom or "").strip().lower()
    if not value:
        return "unknown"
    if value == "viruses":
        return "other"
    if value in {"bacteria", "archaea", "cyanobacteria"}:
        return "prokaryote"
    if value.startswith(("plantae", "fungi", "animalia", "amoebozoa")):
        return "eukaryote"
    return "other"


def build() -> tuple[list[dict[str, str]], dict[str, object]]:
    production = rows(PRODUCTION_REGISTRY)
    marts = rows(MARTS_ENZYMES, "\t")
    uniprot = rows(UNIPROT_NORMALIZED, "\t")

    direct: dict[str, dict[str, str]] = {}
    for row in marts:
        kingdom = row.get("kingdom", "").strip()
        species = row.get("species", "").strip()
        if not kingdom:
            continue
        payload = {"kingdom": kingdom, "species": species}
        for column in ("enzyme_id", "uniprot_id", "genbank_id"):
            identifier = row.get(column, "").strip()
            if identifier:
                direct[identifier] = payload

    genus_to_kingdom: dict[str, str] = {}
    genus_conflicts: set[str] = set()
    for row in marts:
        genus = normalized_genus(row.get("species", ""))
        kingdom = row.get("kingdom", "").strip()
        if not genus or not kingdom:
            continue
        previous = genus_to_kingdom.get(genus)
        if previous and previous != kingdom:
            genus_conflicts.add(genus)
        else:
            genus_to_kingdom[genus] = kingdom
    for genus in genus_conflicts:
        genus_to_kingdom.pop(genus, None)

    uniprot_by_id = {row.get("accession", "").strip(): row for row in uniprot if row.get("accession", "").strip()}
    output: list[dict[str, str]] = []
    for row in production:
        protein_id = row.get("protein_id", "").strip()
        production_source = row.get("source", "").strip() or "unknown"
        kingdom = ""
        species = ""
        taxonomy_source = "unknown"
        confidence = "unknown"

        if protein_id in direct:
            kingdom = direct[protein_id]["kingdom"]
            species = direct[protein_id]["species"]
            taxonomy_source = "marts_kingdom_direct"
            confidence = "direct_kingdom"
        else:
            uniprot_row = uniprot_by_id.get(protein_id)
            if uniprot_row is not None:
                species = uniprot_row.get("organism_name", "").strip()
                genus = normalized_genus(species)
                inferred = genus_to_kingdom.get(genus)
                if inferred:
                    kingdom = inferred
                    taxonomy_source = "local_uniprot_organism_genus_to_marts_kingdom"
                    confidence = "conservative_genus_inference"
                else:
                    taxonomy_source = "local_uniprot_organism_unresolved"
                    confidence = "unknown"

        output.append({
            "protein_id": protein_id,
            "production_source": production_source,
            "taxonomy_scope": taxonomy_scope(kingdom),
            "kingdom": kingdom,
            "species": species,
            "taxonomy_source": taxonomy_source,
            "taxonomy_confidence": confidence,
        })

    scope_counts = Counter(row["taxonomy_scope"] for row in output)
    source_counts = Counter(row["taxonomy_source"] for row in output)
    summary: dict[str, object] = {
        "version": VERSION,
        "production_registry": str(PRODUCTION_REGISTRY.relative_to(ROOT)),
        "marts_enzymes": str(MARTS_ENZYMES.relative_to(ROOT)),
        "local_uniprot_snapshot": str(UNIPROT_NORMALIZED.relative_to(ROOT)),
        "total": len(output),
        "scope_counts": dict(sorted(scope_counts.items())),
        "taxonomy_source_counts": dict(sorted(source_counts.items())),
        "genus_conflicts_excluded": sorted(genus_conflicts),
        "policy": {
            "eukaryote": "Plantae/Fungi/Animalia/Amoebozoa kingdom labels and conservative same-genus inference",
            "prokaryote": "Bacteria/Archaea/Cyanobacteria kingdom labels and conservative same-genus inference",
            "other": "Viruses or non-eukaryote/non-prokaryote resolved labels",
            "unknown": "No supported local classification; excluded from restricted modes",
        },
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local enzyme taxonomy-scope registry used by production retrieval filters.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output, summary = build()
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n", fieldnames=[
            "protein_id", "production_source", "taxonomy_scope", "kingdom", "species",
            "taxonomy_source", "taxonomy_confidence",
        ])
        writer.writeheader()
        writer.writerows(output)
    summary_path = target.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(target), "summary": str(summary_path), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
