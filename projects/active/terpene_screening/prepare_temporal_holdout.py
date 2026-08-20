from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAIRS = ROOT / "data/terpene_marts/marts_reaction_pairs.tsv"
DEFAULT_OUTPUT = ROOT / "results/terpene_temporal_holdout_readiness"
STRICT_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def extract_year(publication: str, maximum_year: int) -> tuple[int | None, str]:
    matches = [int(value) for value in STRICT_YEAR.findall(str(publication))]
    valid = sorted({value for value in matches if 1900 <= value <= maximum_year})
    future = sorted({value for value in matches if value > maximum_year})
    if len(valid) == 1:
        return valid[0], "strict_text_year"
    if len(valid) > 1:
        return None, "ambiguous_multiple_years"
    if future:
        return None, "future_numeric_token"
    return None, "year_unresolved"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit publication dates and build a temporal holdout only when metadata is adequate.")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--publication-metadata", type=Path, default=None, help="Optional CSV with publication and publication_year columns.")
    parser.add_argument("--cutoff-year", type=int, default=2020)
    parser.add_argument("--minimum-coverage", type=float, default=0.80)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pairs, sep="\t", dtype=str).fillna("")
    if "publication" not in pairs:
        raise ValueError("Temporal holdout requires a publication column")
    maximum_year = datetime.now(timezone.utc).year
    metadata_map: dict[str, int] = {}
    if args.publication_metadata is not None:
        metadata = pd.read_csv(args.publication_metadata, dtype=str).fillna("")
        required = {"publication", "publication_year"}
        if required - set(metadata.columns):
            raise ValueError("Publication metadata requires publication and publication_year")
        years = pd.to_numeric(metadata["publication_year"], errors="coerce")
        metadata_map = {
            str(publication): int(year)
            for publication, year in zip(metadata["publication"], years)
            if pd.notna(year) and 1900 <= int(year) <= maximum_year
        }
    audit_rows = []
    for publication in pairs["publication"].astype(str):
        if publication in metadata_map:
            year, source = metadata_map[publication], "provided_metadata"
        else:
            year, source = extract_year(publication, maximum_year)
        audit_rows.append({"publication": publication, "publication_year": year, "year_source": source})
    audit = pd.DataFrame(audit_rows)
    enriched = pairs.copy()
    enriched["publication_year"] = audit["publication_year"]
    enriched["publication_year_source"] = audit["year_source"]
    coverage = float(enriched["publication_year"].notna().mean()) if len(enriched) else 0.0
    unresolved = enriched[enriched["publication_year"].isna()].copy()
    audit.to_csv(output / "publication_year_audit.csv", index=False)
    unresolved.to_csv(output / "unresolved_publications.csv", index=False)
    ready = coverage >= args.minimum_coverage
    split_created = ready or args.allow_incomplete
    outputs: dict[str, str] = {
        "audit": str(output / "publication_year_audit.csv"),
        "unresolved": str(output / "unresolved_publications.csv"),
    }
    if split_created:
        known = enriched[enriched["publication_year"].notna()].copy()
        known["publication_year"] = known["publication_year"].astype(int)
        train = known[known["publication_year"] <= args.cutoff_year].copy()
        test = known[known["publication_year"] > args.cutoff_year].copy()
        unknown = enriched[enriched["publication_year"].isna()].copy()
        train.to_csv(output / "temporal_train.tsv", sep="\t", index=False)
        test.to_csv(output / "temporal_test.tsv", sep="\t", index=False)
        unknown.to_csv(output / "temporal_unknown.tsv", sep="\t", index=False)
        outputs.update({
            "train": str(output / "temporal_train.tsv"),
            "test": str(output / "temporal_test.tsv"),
            "unknown": str(output / "temporal_unknown.tsv"),
        })
    summary = {
        "status": "ready" if ready else "insufficient_publication_metadata",
        "split_created": split_created,
        "allow_incomplete": args.allow_incomplete,
        "cutoff_year": args.cutoff_year,
        "minimum_coverage": args.minimum_coverage,
        "pair_rows": len(enriched),
        "resolved_year_rows": int(enriched["publication_year"].notna().sum()),
        "year_coverage_fraction": coverage,
        "year_source_counts": enriched["publication_year_source"].value_counts().to_dict(),
        "warning": (
            "No temporal performance claim is allowed until coverage reaches the configured threshold."
            if not ready else "Temporal split metadata threshold passed."
        ),
        "outputs": outputs,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
