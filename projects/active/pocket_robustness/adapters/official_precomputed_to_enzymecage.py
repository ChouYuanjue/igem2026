from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projects.active.pocket_robustness.adapters.pocket_manifest import (  # noqa: E402
    PocketRecord,
    make_pocket_global_id,
    write_manifest,
)


ENZYME_COLUMNS = ["UniprotID", "uniprot_id", "uid", "protein_id", "enzyme_id"]


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {_normalize(column): column for column in columns}
    for candidate in candidates:
        key = _normalize(candidate)
        if key in normalized:
            return normalized[key]
    return None


def build_manifest(input_csv: Path, pocket_dir: Path, output_dir: Path, run_id: str) -> dict[str, Any]:
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / "pocket_manifest.csv"

    summary: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "initialized",
        "input_csv": str(input_csv),
        "pocket_dir": str(pocket_dir),
        "output_dir": str(output_dir),
        "warnings": [],
        "generated_files": [],
    }

    df = pd.read_csv(input_csv)
    enzyme_col = _find_column(list(df.columns), ENZYME_COLUMNS)
    if enzyme_col is None:
        summary["status"] = "failed_input_parse"
        summary["warnings"].append(f"No enzyme id column found in {input_csv}")
        write_manifest([], manifest_path)
        summary["generated_files"].append(str(manifest_path))
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    records: list[PocketRecord] = []
    enzyme_ids = list(dict.fromkeys(df[enzyme_col].dropna().astype(str).str.strip()))
    for enzyme_id in enzyme_ids:
        pocket_path = pocket_dir / f"{enzyme_id}.pdb"
        if not pocket_path.exists():
            warning = f"Official precomputed pocket missing for {enzyme_id}: {pocket_path}"
            print(f"[warning] {warning}")
            summary["warnings"].append(warning)
            continue
        records.append(
            PocketRecord(
                run_id=run_id,
                enzyme_id=enzyme_id,
                structure_path="",
                pocket_method="official_precomputed_pocket",
                pocket_source="official_precomputed",
                pocket_rank=1,
                pocket_global_id=make_pocket_global_id(enzyme_id, "official_precomputed", 1),
                pocket_score=None,
                pocket_center_x=None,
                pocket_center_y=None,
                pocket_center_z=None,
                pocket_residues="",
                pocket_pdb_path=str(pocket_path),
                source_raw_dir=str(pocket_dir),
                pocket_pdb_mode="external_original",
            )
        )

    write_manifest(records, manifest_path)
    summary["status"] = "completed" if records else "blocked_no_official_precomputed_pockets"
    summary["n_enzymes"] = len(enzyme_ids)
    summary["n_records"] = len(records)
    summary["generated_files"].append(str(manifest_path))
    summary_path = output_dir / "summary.json"
    summary["generated_files"].append(str(summary_path))
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create manifest for official precomputed EnzymeCAGE pockets.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--pocket_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_id", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_manifest(
        input_csv=Path(args.input_csv),
        pocket_dir=Path(args.pocket_dir),
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
