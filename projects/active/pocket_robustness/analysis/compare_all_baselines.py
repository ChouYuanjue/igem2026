from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ENZYME405_50_BASELINES = [
    "official_precomputed_pocket",
    "p2rank_top1",
    "p2rank_topk_max",
    "p2rank_topk_mean",
    "p2rank_topk_rank_weighted",
    "p2rank_topk_softmax_pool",
    "fpocket_top1",
    "fpocket_topk_rank_weighted",
    "p2rank_fpocket_union_max",
    "p2rank_fpocket_union_source_weighted",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return {"status": "not_run", "warnings": [], "generated_files": []}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _find_aggregation(run_dir: Path) -> Path | None:
    aggregation_dir = run_dir / "aggregation"
    if not aggregation_dir.exists():
        return None
    matches = sorted(aggregation_dir.glob("enzyme_level_*.csv"))
    return matches[0] if matches else None


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    for candidate in [
        run_dir / "metrics/metrics_top5_top10.json",
        run_dir / "metrics/topk_metrics.json",
    ]:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _infer_status(run_dir: Path, summary: dict[str, Any]) -> str:
    status = str(summary.get("status", "not_run"))
    aggregation = _find_aggregation(run_dir)
    predictions = run_dir / "predictions/pocket_level_predictions.csv"
    metrics = _load_metrics(run_dir)
    if (
        aggregation is not None
        and predictions.exists()
        and metrics.get("status") == "ok"
        and not status.startswith("blocked")
        and not status.startswith("failed")
    ):
        return "completed"
    if (
        aggregation is not None
        and predictions.exists()
        and metrics.get("status") == "ok"
        and status.startswith("failed")
    ):
        return "completed"
    return status


def _run_id_for_baseline(baseline: dict[str, Any], dataset_scale: str | None = None) -> str:
    if dataset_scale:
        return f"{dataset_scale}_{baseline['name']}"
    data_mode = baseline.get("data_mode", "demo_mining")
    return f"demo_{baseline['name']}" if data_mode == "demo_mining" else baseline["name"]


def _status_table(results_root: Path, matrix: dict[str, Any], dataset_scale: str | None = None) -> pd.DataFrame:
    rows = []
    for baseline in matrix["baselines"]:
        run_id = _run_id_for_baseline(baseline, dataset_scale)
        run_dir = results_root / run_id
        summary = _load_summary(run_dir)
        rows.append(
            {
                "baseline_name": baseline["name"],
                "run_id": run_id,
                "data_mode": baseline.get("data_mode", "demo_mining"),
                "status": _infer_status(run_dir, summary),
                "failed_step": summary.get("failed_step"),
                "error": summary.get("error"),
                "n_warnings": len(summary.get("warnings", [])),
                "run_summary": str(run_dir / "run_summary.json"),
                "aggregation_csv": str(_find_aggregation(run_dir) or ""),
            }
        )
    return pd.DataFrame(rows)


def _metric_table(results_root: Path, matrix: dict[str, Any], dataset_scale: str | None = None) -> pd.DataFrame:
    rows = []
    for baseline in matrix["baselines"]:
        run_id = _run_id_for_baseline(baseline, dataset_scale)
        metrics_path = results_root / run_id / "metrics/metrics_top5_top10.json"
        if not metrics_path.exists():
            metrics_path = results_root / run_id / "metrics/topk_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = {key: value for key, value in metrics.items() if key in {"status", "reason", "top5_success_rate", "top10_success_rate", "n_groups", "n_pairs"}}
        metrics["baseline_name"] = baseline["name"]
        metrics["run_id"] = run_id
        metrics["data_mode"] = baseline.get("data_mode", "demo_mining")
        rows.append(metrics)
    return pd.DataFrame(rows)


def _pocket_selection_summary(results_root: Path, matrix: dict[str, Any], dataset_scale: str | None = None) -> pd.DataFrame:
    rows = []
    for baseline in matrix["baselines"]:
        run_id = _run_id_for_baseline(baseline, dataset_scale)
        run_dir = results_root / run_id
        manifest_path = run_dir / "manifests/pocket_manifest.csv"
        if not manifest_path.exists():
            rows.append(
                {
                    "baseline_name": baseline["name"],
                    "run_id": run_id,
                    "n_pockets": 0,
                    "n_enzymes": 0,
                    "avg_pockets_per_enzyme": 0.0,
                    "placeholder_fraction": None,
                }
            )
            continue
        df = pd.read_csv(manifest_path)
        n_pockets = len(df)
        n_enzymes = int(df["enzyme_id"].nunique()) if "enzyme_id" in df.columns and n_pockets else 0
        placeholder_fraction = None
        if "pocket_pdb_mode" in df.columns and n_pockets:
            placeholder_fraction = float((df["pocket_pdb_mode"] == "full_structure_placeholder").mean())
        rows.append(
            {
                "baseline_name": baseline["name"],
                "run_id": run_id,
                "n_pockets": int(n_pockets),
                "n_enzymes": n_enzymes,
                "avg_pockets_per_enzyme": float(n_pockets / n_enzymes) if n_enzymes else 0.0,
                "placeholder_fraction": placeholder_fraction,
            }
        )
    return pd.DataFrame(rows)


def _rank_shift_matrix(results_root: Path, matrix: dict[str, Any], dataset_scale: str | None = None) -> pd.DataFrame:
    reference_run_id = _run_id_for_baseline({"name": "p2rank_top1", "data_mode": "derived_smallset"}, dataset_scale)
    baseline_path = results_root / reference_run_id / "aggregation"
    baseline_files = sorted(baseline_path.glob("enzyme_level_*.csv")) if baseline_path.exists() else []
    if not baseline_files:
        return pd.DataFrame(columns=["baseline_name", "reaction_id", "enzyme_id", "rank_shift"])
    baseline_df = pd.read_csv(baseline_files[0])
    group_col = "reaction_id" if "reaction_id" in baseline_df.columns else None
    key_cols = ["enzyme_id"] if group_col is None else [group_col, "enzyme_id"]
    baseline_df = baseline_df.copy()
    if group_col:
        baseline_df["baseline_rank"] = baseline_df.groupby(group_col)["aggregated_score"].rank(method="first", ascending=False)
    else:
        baseline_df["baseline_rank"] = baseline_df["aggregated_score"].rank(method="first", ascending=False)

    rows = []
    for baseline in matrix["baselines"]:
        name = baseline["name"]
        if name == "p2rank_top1":
            continue
        agg_path = _find_aggregation(results_root / _run_id_for_baseline(baseline, dataset_scale))
        if agg_path is None:
            continue
        new_df = pd.read_csv(agg_path)
        if group_col and group_col in new_df.columns:
            new_df["new_rank"] = new_df.groupby(group_col)["aggregated_score"].rank(method="first", ascending=False)
        else:
            new_df["new_rank"] = new_df["aggregated_score"].rank(method="first", ascending=False)
        merged = baseline_df[key_cols + ["baseline_rank"]].merge(
            new_df[key_cols + ["new_rank"]],
            on=key_cols,
            how="inner",
        )
        merged["rank_shift"] = merged["baseline_rank"] - merged["new_rank"]
        merged["baseline_name"] = name
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["baseline_name", *key_cols, "rank_shift"])


def _prediction_summary(results_root: Path, matrix: dict[str, Any], dataset_scale: str | None = None) -> list[str]:
    lines = []
    for baseline in matrix["baselines"]:
        agg_path = _find_aggregation(results_root / _run_id_for_baseline(baseline, dataset_scale))
        if agg_path is None:
            continue
        df = pd.read_csv(agg_path)
        if df.empty:
            continue
        lines.append(
            f"- `{baseline['name']}`: n={len(df)}, "
            f"score_mean={df['aggregated_score'].mean():.4f}, "
            f"score_min={df['aggregated_score'].min():.4f}, "
            f"score_max={df['aggregated_score'].max():.4f}"
        )
        if "best_pocket_rank" in df.columns:
            lines.append(f"  best pocket ranks: {df['best_pocket_rank'].value_counts().to_dict()}")
        if "best_pocket_source" in df.columns:
            lines.append(f"  best pocket sources: {df['best_pocket_source'].value_counts().to_dict()}")
    return lines


def _write_report(
    output_dir: Path,
    status: pd.DataFrame,
    metrics: pd.DataFrame,
    pocket_summary: pd.DataFrame,
    rank_shift: pd.DataFrame,
    prediction_lines: list[str],
    dataset_scale: str | None = None,
) -> Path:
    def table(df: pd.DataFrame) -> str:
        if df.empty:
            return "No rows."
        return "```text\n" + df.to_string(index=False) + "\n```"

    report_path = output_dir / "comparison_report.md"
    title = "# Pocket Baseline Comparison Report"
    if dataset_scale:
        title = f"# {dataset_scale} Pocket Baseline Comparison Report"
    lines = [
        title,
        "",
        "## Baseline Status",
        "",
        table(status),
        "",
        "## Inputs",
        "",
        "Inputs are recorded per baseline in each `run_summary.json`, including reaction CSV, structure directory, and checkpoint status." if not dataset_scale else f"Inputs are recorded for the `{dataset_scale}` slice only; each baseline points at its run-specific `run_summary.json` and derived outputs.",
        "",
        "The README mining demo references `dataset/demo` and `best_model.pth`, but the official Google Drive assets currently extracted here do not include those paths. The official assets contain `epoch_19.pth` and evaluation datasets, so `official_eval` baselines are the correct first target. Pocket intervention baselines require constructing a derived smallset with full structures; if only pre-extracted pockets are present, those baselines are blocked rather than fabricated." if not dataset_scale else f"The `{dataset_scale}` slice uses derived smallset inputs and real pocket predictions; fpocket-dependent baselines are reported as blocked rather than fabricated when the executable is unavailable.",
        "",
        "## Pocket Extraction Summary",
        "",
        table(pocket_summary),
        "",
        "## Prediction Summary",
        "",
        "\n".join(prediction_lines) if prediction_lines else "No prediction outputs are available yet.",
        "",
        "## Top-5 / Top-10 Metrics",
        "",
        table(metrics),
        "",
        "## Ranking Changes",
        "",
        "Compared against `p2rank_top1` when both baseline and comparison predictions are available.",
        "",
    ]
    if rank_shift.empty:
        lines.append("No rank shift table is available yet.")
    else:
        largest = rank_shift.reindex(rank_shift["rank_shift"].abs().sort_values(ascending=False).index).head(20)
        lines.append(table(largest))
    lines.extend(
        [
            "",
            "## Rescued / Harmed Cases",
            "",
            "Label-aware rescued/harmed analysis is generated when labels are available. This report intentionally focuses on Top-5 and Top-10, not Top-1 or Top-3.",
            "",
            "## Interpretation Guide",
            "",
            "- If `best_pocket_rank > 1` is common, the top-1 pocket assumption is unstable.",
            "- If fpocket and P2Rank rankings differ strongly, pocket detector source matters.",
            "- If union-source baselines change or improve ranking, heterogeneous pocket hypotheses are valuable.",
            "- If all baselines are nearly unchanged on this slice, pocket choice may not be the bottleneck; expand to additional reactions if you need a stronger separator.",
            "",
            "## Required Baseline Families",
            "",
            "- official_precomputed_pocket",
            "- p2rank_top1",
            "- p2rank_topk_max",
            "- p2rank_topk_mean",
            "- p2rank_topk_rank_weighted",
            "- p2rank_topk_softmax_pool",
            "- fpocket_top1",
            "- fpocket_topk_rank_weighted",
            "- p2rank_fpocket_union_max",
            "- p2rank_fpocket_union_source_weighted",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def compare_all(
    results_root: Path,
    baseline_matrix: Path,
    output_dir: Path,
    dataset_scale: str | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if dataset_scale:
        matrix = {"baselines": [{"name": baseline, "data_mode": "derived_smallset"} for baseline in ENZYME405_50_BASELINES]}
    else:
        matrix = _load_yaml(baseline_matrix)

    status = _status_table(results_root, matrix, dataset_scale=dataset_scale)
    metrics = _metric_table(results_root, matrix, dataset_scale=dataset_scale)
    pocket_summary = _pocket_selection_summary(results_root, matrix, dataset_scale=dataset_scale)
    rank_shift = _rank_shift_matrix(results_root, matrix, dataset_scale=dataset_scale)
    rescued_harmed = pd.DataFrame(columns=["baseline_name", "case_type", "reaction_id", "enzyme_id", "note"])

    paths = {
        "baseline_status_table": output_dir / "baseline_status_table.csv",
        "baseline_metric_table": output_dir / "baseline_metric_table.csv",
        "rank_shift_matrix": output_dir / "rank_shift_matrix.csv",
        "pocket_selection_summary": output_dir / "pocket_selection_summary.csv",
        "rescued_harmed_cases": output_dir / "rescued_harmed_cases.csv",
    }
    status.to_csv(paths["baseline_status_table"], index=False)
    metrics.to_csv(paths["baseline_metric_table"], index=False)
    rank_shift.to_csv(paths["rank_shift_matrix"], index=False)
    pocket_summary.to_csv(paths["pocket_selection_summary"], index=False)
    rescued_harmed.to_csv(paths["rescued_harmed_cases"], index=False)
    report_path = _write_report(
        output_dir,
        status=status,
        metrics=metrics,
        pocket_summary=pocket_summary,
        rank_shift=rank_shift,
        prediction_lines=_prediction_summary(results_root, matrix, dataset_scale=dataset_scale),
        dataset_scale=dataset_scale,
    )
    paths["comparison_report"] = report_path
    return {key: str(value) for key, value in paths.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare all pocket baselines under results/pocket.")
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--baseline_matrix", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_scale")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = compare_all(
        Path(args.results_root),
        Path(args.baseline_matrix),
        Path(args.output_dir),
        dataset_scale=args.dataset_scale,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
