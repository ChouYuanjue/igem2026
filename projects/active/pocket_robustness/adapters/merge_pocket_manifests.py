from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projects.active.pocket_robustness.adapters.pocket_manifest import (
    PocketRecord,
    make_pocket_global_id,
    read_manifest,
    write_manifest,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _center_distance(a: PocketRecord, b: PocketRecord) -> float | None:
    coords_a = [a.pocket_center_x, a.pocket_center_y, a.pocket_center_z]
    coords_b = [b.pocket_center_x, b.pocket_center_y, b.pocket_center_z]
    if any(value is None for value in coords_a + coords_b):
        return None
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(coords_a, coords_b)))


def _residue_set(record: PocketRecord) -> set[str]:
    return {token.strip() for token in record.pocket_residues.split(",") if token.strip()}


def _jaccard(a: PocketRecord, b: PocketRecord) -> float | None:
    set_a = _residue_set(a)
    set_b = _residue_set(b)
    if not set_a or not set_b:
        return None
    return len(set_a & set_b) / len(set_a | set_b)


def _is_duplicate(
    candidate: PocketRecord,
    kept: PocketRecord,
    center_distance_threshold: float,
    residue_jaccard_threshold: float,
) -> bool:
    if candidate.enzyme_id != kept.enzyme_id:
        return False
    center_distance = _center_distance(candidate, kept)
    residue_jaccard = _jaccard(candidate, kept)
    center_match = center_distance is not None and center_distance < center_distance_threshold
    residue_match = residue_jaccard is not None and residue_jaccard > residue_jaccard_threshold
    return center_match or residue_match


def merge_manifests(
    manifest_csvs: list[Path],
    output_csv: Path,
    run_id: str,
    deduplicate: bool = False,
    center_distance_threshold: float = 4.0,
    residue_jaccard_threshold: float = 0.5,
) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    input_counts: dict[str, int] = {}
    all_records: list[PocketRecord] = []
    for manifest_csv in manifest_csvs:
        records = read_manifest(manifest_csv)
        input_counts[str(manifest_csv)] = len(records)
        for record in records:
            source_counts[record.pocket_source] += 1
            all_records.append(record)

    kept_records: list[PocketRecord] = []
    duplicates_removed = 0
    for record in all_records:
        if deduplicate and any(
            _is_duplicate(record, kept, center_distance_threshold, residue_jaccard_threshold)
            for kept in kept_records
        ):
            duplicates_removed += 1
            continue
        kept_records.append(record)

    used_ids: set[str] = set()
    source_rank_counters: Counter[tuple[str, str]] = Counter()
    merged_records: list[PocketRecord] = []
    for record in kept_records:
        key = (record.enzyme_id, record.pocket_source)
        source_rank_counters[key] += 1
        rank = source_rank_counters[key]
        global_id = make_pocket_global_id(record.enzyme_id, record.pocket_source, rank)
        while global_id in used_ids:
            source_rank_counters[key] += 1
            rank = source_rank_counters[key]
            global_id = make_pocket_global_id(record.enzyme_id, record.pocket_source, rank)
        used_ids.add(global_id)
        merged_records.append(
            PocketRecord(
                run_id=run_id,
                enzyme_id=record.enzyme_id,
                structure_path=record.structure_path,
                pocket_method=record.pocket_method,
                pocket_source=record.pocket_source,
                pocket_rank=rank,
                pocket_global_id=global_id,
                pocket_score=record.pocket_score,
                pocket_center_x=record.pocket_center_x,
                pocket_center_y=record.pocket_center_y,
                pocket_center_z=record.pocket_center_z,
                pocket_residues=record.pocket_residues,
                pocket_pdb_path=record.pocket_pdb_path,
                source_raw_dir=record.source_raw_dir,
                pocket_pdb_mode=record.pocket_pdb_mode,
            )
        )

    write_manifest(merged_records, output_csv)
    summary = {
        "timestamp": _now(),
        "run_id": run_id,
        "manifest_csvs": [str(path) for path in manifest_csvs],
        "output_csv": str(output_csv),
        "deduplicate": deduplicate,
        "center_distance_threshold": center_distance_threshold,
        "residue_jaccard_threshold": residue_jaccard_threshold,
        "input_counts": input_counts,
        "source_counts": dict(source_counts),
        "n_input_records": len(all_records),
        "n_output_records": len(merged_records),
        "duplicates_removed": duplicates_removed,
    }
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] Wrote merged manifest to {output_csv}")
    print(f"[done] Wrote merge summary to {summary_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge pocket manifest CSVs across pocket sources.")
    parser.add_argument("--manifest_csvs", nargs="+", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--center_distance_threshold", type=float, default=4.0)
    parser.add_argument("--residue_jaccard_threshold", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    merge_manifests(
        manifest_csvs=[Path(path) for path in args.manifest_csvs],
        output_csv=Path(args.output_csv),
        run_id=args.run_id,
        deduplicate=args.deduplicate,
        center_distance_threshold=args.center_distance_threshold,
        residue_jaccard_threshold=args.residue_jaccard_threshold,
    )


if __name__ == "__main__":
    main()
