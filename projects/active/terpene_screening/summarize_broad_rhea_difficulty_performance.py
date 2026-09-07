from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_source(value: str) -> tuple[str, Path]:
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise ValueError("--source must use NAME=PATH_TO_QUERY_METRICS.csv")
    return name, Path(path).resolve()


def aggregate(group: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = {
        "n_queries": int(len(group)),
        "mean_candidate_count": float(group.candidate_count.mean()),
        "mean_positive_count": float(group.positive_count.mean()),
        "mrr": float(group.reciprocal_rank.mean()),
        "map": float(group.average_precision.mean()),
        "macro_roc_auc": float(group.roc_auc.dropna().mean()) if group.roc_auc.notna().any() else None,
        "median_best_positive_rank": float(group.best_positive_rank.dropna().median()),
        "mean_best_positive_rank_fraction": float(group.best_positive_rank_fraction.dropna().mean()),
        "mean_positive_rank": float(group.mean_positive_rank.dropna().mean()),
        "mean_positive_reciprocal_rank": float(group.mean_positive_reciprocal_rank.dropna().mean()),
    }
    for k in (1, 3, 5, 10, 20, 50):
        for column in (f"hit_at_{k}", f"precision_at_{k}", f"positive_recall_at_{k}", f"ndcg_at_{k}"):
            if column in group:
                row[column] = float(group[column].mean())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate strict-cold retrieval metrics by nearest-train difficulty slices.")
    parser.add_argument("--cell", required=True)
    parser.add_argument("--slices-root", type=Path, default=Path("results/broad_rhea_difficulty_slices_v1"))
    parser.add_argument("--source", action="append", required=True, help="Repeat NAME=path/query_metrics.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    slice_dir = args.slices_root.resolve() / args.cell
    protein = pd.read_csv(slice_dir / "protein_slices.csv", dtype={"protein_id": str})
    reaction = pd.read_csv(slice_dir / "reaction_slices.csv", dtype={"reaction_id": str})
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    joined_outputs: list[str] = []
    for source_arg in args.source:
        model, path = parse_source(source_arg)
        frame = pd.read_csv(path, dtype={"query_id": str})
        required = {"direction", "query_id", "candidate_count", "positive_count", "reciprocal_rank", "average_precision"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns {sorted(missing)}")
        parts: list[pd.DataFrame] = []
        for direction, group in frame.groupby("direction", sort=True):
            if direction == "enzyme_to_reaction":
                merged = group.merge(protein, left_on="query_id", right_on="protein_id", how="left", validate="many_to_one")
                axes = ["protein_identity_bucket"]
            elif direction == "reaction_to_enzyme":
                merged = group.merge(reaction, left_on="query_id", right_on="reaction_id", how="left", validate="many_to_one")
                axes = ["reaction_similarity_bucket"]
            else:
                raise ValueError(f"unknown direction {direction}")
            merged["model"] = model
            parts.append(merged)
            for axis in axes:
                if merged[axis].isna().any():
                    missing_ids = merged.loc[merged[axis].isna(), "query_id"].astype(str).unique().tolist()
                    raise ValueError(f"Difficulty slices missing {len(missing_ids)} {direction} queries; examples={missing_ids[:5]}")
                for value, sliced in merged.groupby(axis, sort=True):
                    rows.append({
                        "model": model, "cell": args.cell, "direction": direction,
                        "slice_name": axis, "slice_value": str(value), **aggregate(sliced),
                    })
            rows.append({
                "model": model, "cell": args.cell, "direction": direction,
                "slice_name": "all", "slice_value": "all", **aggregate(merged),
            })
        joined = pd.concat(parts, ignore_index=True)
        joined_path = output / f"{model}_query_metrics_with_difficulty.csv"
        joined.to_csv(joined_path, index=False); joined_outputs.append(str(joined_path))

    summary = pd.DataFrame(rows)
    summary.to_csv(output / "difficulty_metrics.csv", index=False)
    payload = {
        "cell": args.cell,
        "sources": {parse_source(value)[0]: str(parse_source(value)[1]) for value in args.source},
        "difficulty_slices": str(slice_dir),
        "joined_outputs": joined_outputs,
        "metrics": str(output / "difficulty_metrics.csv"),
        "note": "R2E is stratified by nearest-train reaction DRFP Tanimoto; E2R by nearest-train protein MMseqs2 identity. No target labels define difficulty bins.",
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
