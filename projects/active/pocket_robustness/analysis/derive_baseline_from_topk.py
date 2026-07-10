from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from projects.active.pocket_robustness.analysis.aggregate_pocket_scores import aggregate_scores
from projects.active.pocket_robustness.analysis.evaluate_topk import evaluate_topk


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_commands(commands_jsonl: Path, records: list[dict[str, Any]]) -> None:
    commands_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with commands_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _merge_labels(
    predictions: pd.DataFrame,
    label_csv: Path | None,
    group_col: str = "reaction_id",
    enzyme_col: str = "enzyme_id",
    label_col: str = "label",
) -> pd.DataFrame:
    if label_csv is None or not label_csv.exists() or label_col in predictions.columns:
        return predictions

    labels = pd.read_csv(label_csv)
    if label_col not in labels.columns and "Label" in labels.columns:
        labels = labels.rename(columns={"Label": label_col})
    merge_cols = [group_col, enzyme_col]
    missing_prediction = [column for column in merge_cols if column not in predictions.columns]
    missing_label = [column for column in merge_cols + [label_col] if column not in labels.columns]
    if missing_prediction or missing_label:
        missing = missing_prediction or missing_label
        raise ValueError(f"Cannot merge labels; missing columns: {missing}")
    return predictions.merge(labels[merge_cols + [label_col]], on=merge_cols, how="left")


def derive_baseline(
    source_run_dir: Path,
    target_run_dir: Path,
    baseline_name: str,
    aggregation_method: str,
    top1_only: bool = False,
    temperature: float = 0.2,
    source_weights: str | None = None,
    label_csv: Path | None = None,
) -> dict[str, Any]:
    source_summary = _read_json(source_run_dir / "run_summary.json")
    source_manifest = pd.read_csv(source_run_dir / "manifests/pocket_manifest.csv")
    source_predictions = pd.read_csv(source_run_dir / "predictions/pocket_level_predictions.csv")

    if top1_only:
        source_manifest = source_manifest[source_manifest["pocket_rank"] == 1].copy()
        source_predictions = source_predictions[source_predictions["pocket_rank"] == 1].copy()

    target_run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["logs", "manifests", "predictions", "aggregation", "metrics", "analysis"]:
        (target_run_dir / sub).mkdir(parents=True, exist_ok=True)

    manifest_path = target_run_dir / "manifests/pocket_manifest.csv"
    prediction_path = target_run_dir / "predictions/pocket_level_predictions.csv"
    aggregated_path = target_run_dir / "aggregation" / f"enzyme_level_{aggregation_method}.csv"
    metrics_path = target_run_dir / "metrics" / "metrics_top5_top10.json"
    summary_path = target_run_dir / "run_summary.json"
    commands_jsonl = target_run_dir / "commands.jsonl"
    config_path = target_run_dir / "config.yaml"

    if top1_only:
        source_manifest = source_manifest.copy()
        source_manifest["pocket_method"] = "p2rank_top1"
        source_manifest["pocket_source"] = "p2rank"
        source_manifest["pocket_rank"] = 1
        source_manifest["pocket_global_id"] = [
            f"{enzyme}__p2rank__rank1" for enzyme in source_manifest["enzyme_id"].astype(str)
        ]

    source_manifest.to_csv(manifest_path, index=False)
    source_predictions.to_csv(prediction_path, index=False)

    aggregation_kwargs: dict[str, Any] = {"method": aggregation_method, "temperature": temperature}
    if source_weights:
        aggregation_kwargs["source_weights"] = json.loads(source_weights)
    aggregated = aggregate_scores(source_predictions, **aggregation_kwargs)
    aggregated.to_csv(aggregated_path, index=False)

    metrics: dict[str, Any] = {"status": "skipped"}
    if label_csv is not None and label_csv.exists():
        evaluated = _merge_labels(
            aggregated,
            label_csv,
            group_col="reaction_id" if "reaction_id" in aggregated.columns else "enzyme_id",
            enzyme_col="enzyme_id",
            label_col="label",
        )
        metrics = evaluate_topk(
            predictions=evaluated,
            topk_values=[5, 10],
            group_col="reaction_id" if "reaction_id" in aggregated.columns else "enzyme_id",
            score_col="aggregated_score",
            label_col="label",
        )
        metrics["status"] = "ok"
        _write_json(metrics_path, metrics)

    source_config = source_run_dir / "config.yaml"
    if source_config.exists():
        shutil.copy2(source_config, config_path)
    else:
        config_path.write_text("", encoding="utf-8")

    commands = [
        {
            "step": "derive_baseline",
            "command": [
                "derive_baseline_from_topk",
                "--source_run_dir",
                str(source_run_dir),
                "--target_run_dir",
                str(target_run_dir),
                "--baseline_name",
                baseline_name,
                "--aggregation_method",
                aggregation_method,
            ],
            "status": "derived",
            "dry_run": False,
            "note": "Derived from a real top-k pocket-level prediction table; no inference rerun.",
        }
    ]
    _write_commands(commands_jsonl, commands)

    summary = {
        "run_id": baseline_name,
        "baseline_name": baseline_name,
        "timestamp": _now(),
        "status": "completed",
        "failed_step": None,
        "error": None,
        "data_mode": source_summary.get("data_mode", "derived_smallset"),
        "derived_from_run": source_run_dir.name,
        "warnings": [
            "Aggregation-only baseline derived from the same real top-k pocket-level predictions; no predictions were fabricated."
        ],
        "commands": commands,
        "generated_files": [
            str(config_path),
            str(manifest_path),
            str(prediction_path),
            str(aggregated_path),
            str(commands_jsonl),
        ],
    }
    if metrics_path.exists():
        summary["generated_files"].append(str(metrics_path))
    _write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Derive a baseline from a real top-k pocket-level run.")
    parser.add_argument("--source_run_dir", required=True)
    parser.add_argument("--target_run_dir", required=True)
    parser.add_argument("--baseline_name", required=True)
    parser.add_argument("--aggregation_method", required=True)
    parser.add_argument("--top1_only", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--source_weights")
    parser.add_argument("--label_csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    label_csv = Path(args.label_csv) if args.label_csv else None
    summary = derive_baseline(
        source_run_dir=Path(args.source_run_dir),
        target_run_dir=Path(args.target_run_dir),
        baseline_name=args.baseline_name,
        aggregation_method=args.aggregation_method,
        top1_only=args.top1_only,
        temperature=args.temperature,
        source_weights=args.source_weights,
        label_csv=label_csv,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
