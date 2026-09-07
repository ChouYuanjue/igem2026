#!/usr/bin/env python3
"""Collate broad Rhea full-candidate benchmark summaries without losing protocol provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


CORE_METRICS = (
    "hit_at_1",
    "hit_at_5",
    "hit_at_10",
    "hit_at_20",
    "hit_at_50",
    "mrr",
    "map",
    "macro_roc_auc",
    "ndcg_at_10",
    "ndcg_at_50",
    "success_at_0.01_fraction",
    "success_at_0.02_fraction",
    "median_best_positive_rank",
    "mean_best_positive_rank_fraction",
)


def markdown_table(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    columns = list(columns)

    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            text = f"{value:.6g}"
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=RESULT_ROOT")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("run label cannot be empty")
    return label, Path(path).expanduser()


def summary_rows(label: str, path: Path) -> list[dict[str, object]]:
    summaries = [path] if path.is_file() else sorted(path.glob("*/summary.json"))
    rows: list[dict[str, object]] = []
    for summary_path in summaries:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = payload.get("cell_manifest", {})
        audit = manifest.get("audit", {})
        expected = manifest.get("expected", {})
        for direction, metrics in payload.get("metrics", {}).items():
            row: dict[str, object] = {
                "run": label,
                "cell": payload.get("cell"),
                "direction": direction,
                "claim_tier": manifest.get("claim_tier"),
                "source_protocol": manifest.get("source_protocol"),
                "protein_unseen": bool(expected.get("protein_unseen", False)),
                "reaction_unseen": bool(expected.get("reaction_unseen", False)),
                "train_pairs": audit.get("train_pairs"),
                "test_pairs": audit.get("test_pairs"),
                "query_count": metrics.get("query_count"),
                "candidate_pool_size": metrics.get("mean_candidate_pool_size"),
                "positive_rows": metrics.get("positive_rows"),
                "manifest_sha256": payload.get("cell_manifest_sha256"),
                "train_pairs_sha256": payload.get("train_pairs_sha256"),
                "test_pairs_sha256": payload.get("test_pairs_sha256"),
                "r2e_model_dir": payload.get("r2e_model_dir"),
                "e2r_model_dir": payload.get("e2r_model_dir"),
                "summary_path": str(summary_path.resolve()),
            }
            for metric in CORE_METRICS:
                row[metric] = metrics.get(metric)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=parse_run)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for label, path in args.run:
        rows.extend(summary_rows(label, path))
    if not rows:
        raise SystemExit("No benchmark summary.json files found")

    frame = pd.DataFrame(rows).sort_values(["claim_tier", "cell", "direction", "run"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        visible = [
            "run", "cell", "direction", "claim_tier", "query_count", "candidate_pool_size",
            "hit_at_1", "hit_at_10", "hit_at_50", "mrr", "map", "macro_roc_auc",
            "ndcg_at_10", "success_at_0.01_fraction",
        ]
        args.output_markdown.write_text(markdown_table(frame, visible), encoding="utf-8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
