from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def _center_distance(row_a: pd.Series, row_b: pd.Series) -> float | None:
    fields = ["pocket_center_x", "pocket_center_y", "pocket_center_z"]
    values = []
    for field in fields:
        a = pd.to_numeric(row_a.get(field), errors="coerce")
        b = pd.to_numeric(row_b.get(field), errors="coerce")
        if pd.isna(a) or pd.isna(b):
            return None
        values.append((float(a), float(b)))
    return math.sqrt(sum((a - b) ** 2 for a, b in values))


def _residue_set(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {token.strip() for token in str(value).split(",") if token.strip()}


def _jaccard(a: object, b: object) -> float | None:
    set_a = _residue_set(a)
    set_b = _residue_set(b)
    if not set_a or not set_b:
        return None
    return len(set_a & set_b) / len(set_a | set_b)


def compare_sources(p2rank_manifest: Path, fpocket_manifest: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    p2rank = pd.read_csv(p2rank_manifest)
    fpocket = pd.read_csv(fpocket_manifest)
    rows = []
    for enzyme_id in sorted(set(p2rank["enzyme_id"]).intersection(set(fpocket["enzyme_id"]))):
        p_rows = p2rank[p2rank["enzyme_id"] == enzyme_id]
        f_rows = fpocket[fpocket["enzyme_id"] == enzyme_id]
        for _, p_row in p_rows.iterrows():
            for _, f_row in f_rows.iterrows():
                rows.append(
                    {
                        "enzyme_id": enzyme_id,
                        "p2rank_pocket_global_id": p_row.get("pocket_global_id"),
                        "fpocket_pocket_global_id": f_row.get("pocket_global_id"),
                        "p2rank_rank": p_row.get("pocket_rank"),
                        "fpocket_rank": f_row.get("pocket_rank"),
                        "center_distance": _center_distance(p_row, f_row),
                        "residue_jaccard": _jaccard(
                            p_row.get("pocket_residues"),
                            f_row.get("pocket_residues"),
                        ),
                    }
                )

    output_csv = output_dir / "pocket_source_overlap.csv"
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    summary = {
        "p2rank_manifest": str(p2rank_manifest),
        "fpocket_manifest": str(fpocket_manifest),
        "n_pairs": int(len(df)),
        "mean_center_distance": None,
        "high_overlap_fraction": None,
        "different_pocket_fraction": None,
    }
    if not df.empty:
        distances = pd.to_numeric(df["center_distance"], errors="coerce").dropna()
        jaccard = pd.to_numeric(df["residue_jaccard"], errors="coerce").dropna()
        if not distances.empty:
            summary["mean_center_distance"] = float(distances.mean())
            summary["different_pocket_fraction"] = float((distances > 10.0).mean())
        if not jaccard.empty:
            summary["high_overlap_fraction"] = float((jaccard >= 0.5).mean())

    summary_json = output_dir / "pocket_source_overlap_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "pocket_source_overlap_csv": str(output_csv),
        "pocket_source_overlap_summary": str(summary_json),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare P2Rank and fpocket pocket hypotheses.")
    parser.add_argument("--p2rank_manifest", required=True)
    parser.add_argument("--fpocket_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = compare_sources(Path(args.p2rank_manifest), Path(args.fpocket_manifest), Path(args.output_dir))
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
