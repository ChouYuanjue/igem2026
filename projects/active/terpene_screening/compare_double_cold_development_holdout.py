from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "results/terpene_double_cold_development_holdout"
DEFAULT_BUDGETS = (3, 10, 20)


def parse_source(value: str) -> tuple[str, Path, str | None]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("Each source must be LABEL=PATH[:METHOD]")
    label, payload = parts
    method = None
    path_text = payload
    if ":" in payload:
        possible_path, possible_method = payload.rsplit(":", 1)
        if Path(possible_path).exists():
            path_text = possible_path
            method = possible_method
    path = Path(path_text)
    if path.is_dir():
        path = path / "query_metrics.csv"
    return label, path.resolve(), method


def partition_from_split(split_id: str, development_fold: int) -> str:
    pieces = str(split_id).replace("p", "").split("_r")
    if len(pieces) != 2:
        raise ValueError(f"Invalid split_id: {split_id}")
    protein_fold, reaction_fold = map(int, pieces)
    return (
        "development_9_cells"
        if protein_fold == development_fold or reaction_fold == development_fold
        else "frozen_16_cells"
    )


def aggregate(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n_query_cells": ("query_id", "size"),
        "n_unique_queries": ("query_id", "nunique"),
        "mean_reciprocal_rank": ("reciprocal_rank", "mean"),
        "median_best_positive_rank": ("best_positive_rank", "median"),
    }
    for budget in budgets:
        aggregations[f"hit_probability_at_{budget}"] = (f"hit_at_{budget}", "mean")
        aggregations[f"positive_recall_at_{budget}"] = (
            f"positive_recall_at_{budget}",
            "mean",
        )
    return (
        frame.groupby(["partition", "model_label", "direction"])
        .agg(**aggregations)
        .reset_index()
    )


def clustered_bootstrap_difference(
    frame: pd.DataFrame,
    selected_label: str,
    reference_label: str,
    direction: str,
    budget: int,
    samples: int,
    seed: int,
) -> dict[str, float | int | str]:
    keys = ["split_id", "direction", "query_id"]
    column = f"hit_at_{budget}"
    selected = frame[
        frame["model_label"].eq(selected_label)
        & frame["direction"].eq(direction)
    ][keys + [column]].rename(columns={column: "selected_hit"})
    reference = frame[
        frame["model_label"].eq(reference_label)
        & frame["direction"].eq(direction)
    ][keys + [column]].rename(columns={column: "reference_hit"})
    paired = selected.merge(reference, on=keys, how="inner", validate="one_to_one")
    if paired.empty:
        raise ValueError(
            f"No paired frozen rows for {selected_label} vs {reference_label}, "
            f"direction={direction}, budget={budget}"
        )
    paired["difference"] = paired["selected_hit"] - paired["reference_hit"]
    cells = sorted(paired["split_id"].unique())
    cell_values = {
        cell: paired.loc[paired["split_id"].eq(cell), "difference"].to_numpy(dtype=float)
        for cell in cells
    }
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled_cells = rng.choice(cells, size=len(cells), replace=True)
        values = np.concatenate([cell_values[str(cell)] for cell in sampled_cells])
        bootstrap[index] = values.mean()
    return {
        "selected_model": selected_label,
        "reference_model": reference_label,
        "direction": direction,
        "selection_budget": int(budget),
        "n_paired_query_cells": int(len(paired)),
        "n_frozen_cells": int(len(cells)),
        "selected_hit_probability": float(paired["selected_hit"].mean()),
        "reference_hit_probability": float(paired["reference_hit"].mean()),
        "absolute_hit_delta": float(paired["difference"].mean()),
        "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
        "selected_only_hits": int(
            ((paired["selected_hit"] == 1) & (paired["reference_hit"] == 0)).sum()
        ),
        "reference_only_hits": int(
            ((paired["selected_hit"] == 0) & (paired["reference_hit"] == 1)).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare strict double-cold models using nine development cells for "
            "selection and sixteen frozen cells for unchanged evaluation."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="LABEL=RESULT_DIR_OR_QUERY_METRICS[:METHOD]",
    )
    parser.add_argument("--development-fold", type=int, default=4)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--reference-label", default="base_pu")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    source_summary: list[dict[str, object]] = []
    for source in args.source:
        label, path, method = parse_source(source)
        frame = pd.read_csv(path, dtype=str).fillna("")
        if "split_id" not in frame.columns and "fold" in frame.columns:
            frame = frame.rename(columns={"fold": "split_id"})
        if "split_id" not in frame.columns:
            raise ValueError(f"Source lacks split_id/fold column: {path}")
        numeric_columns = [
            "reciprocal_rank",
            "best_positive_rank",
            *[f"hit_at_{budget}" for budget in budgets],
            *[f"positive_recall_at_{budget}" for budget in budgets],
        ]
        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if method is not None:
            if "method" not in frame.columns:
                raise ValueError(f"Method requested for source without method column: {path}")
            frame = frame[frame["method"].astype(str).eq(method)].copy()
        if frame.empty:
            raise ValueError(f"No rows loaded for {label}: {path}, method={method}")
        frame["model_label"] = label
        frame["partition"] = frame["split_id"].map(
            lambda value: partition_from_split(value, args.development_fold)
        )
        frames.append(frame)
        source_summary.append(
            {
                "label": label,
                "path": str(path),
                "method": method,
                "rows": len(frame),
            }
        )

    query_metrics = pd.concat(frames, ignore_index=True)
    query_metrics.to_csv(output_dir / "query_metrics_partitioned.csv", index=False)
    metrics = aggregate(query_metrics, budgets)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    development = metrics[metrics["partition"].eq("development_9_cells")]
    frozen = metrics[metrics["partition"].eq("frozen_16_cells")]
    if args.reference_label not in set(metrics["model_label"]):
        raise ValueError(f"Reference label not found: {args.reference_label}")

    selection_rows: list[pd.DataFrame] = []
    frozen_rows: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, float | int | str]] = []
    for direction in sorted(development["direction"].unique()):
        group = development[development["direction"].eq(direction)]
        for budget in budgets:
            selected = group.sort_values(
                [f"hit_probability_at_{budget}", "mean_reciprocal_rank", "model_label"],
                ascending=[False, False, True],
            ).head(1).copy()
            selected.insert(3, "selection_budget", budget)
            selection_rows.append(selected)
            selected_label = str(selected.iloc[0]["model_label"])
            frozen_row = frozen[
                frozen["direction"].eq(direction)
                & frozen["model_label"].eq(selected_label)
            ].copy()
            if len(frozen_row) != 1:
                raise ValueError(
                    f"Expected one frozen metric row for {selected_label}, {direction}; "
                    f"found {len(frozen_row)}"
                )
            frozen_row.insert(3, "selection_budget", budget)
            frozen_row.insert(4, "selected_on", "development_9_cells")
            frozen_rows.append(frozen_row)
            bootstrap_rows.append(
                clustered_bootstrap_difference(
                    query_metrics[query_metrics["partition"].eq("frozen_16_cells")],
                    selected_label,
                    args.reference_label,
                    direction,
                    budget,
                    args.bootstrap_samples,
                    args.bootstrap_seed
                    + budget
                    + (0 if direction == "reaction_to_enzyme" else 1000),
                )
            )

    development_selection = pd.concat(selection_rows, ignore_index=True)
    frozen_evaluation = pd.concat(frozen_rows, ignore_index=True)
    bootstrap_comparison = pd.DataFrame(bootstrap_rows)
    development_selection.to_csv(output_dir / "development_selection.csv", index=False)
    frozen_evaluation.to_csv(
        output_dir / "development_selected_frozen_evaluation.csv", index=False
    )
    bootstrap_comparison.to_csv(output_dir / "frozen_paired_bootstrap.csv", index=False)

    summary = {
        "development_fold": args.development_fold,
        "development_partition": "protein_fold==4 OR reaction_fold==4",
        "frozen_partition": "protein_fold in 0..3 AND reaction_fold in 0..3",
        "selection_rule": (
            "Select model labels only on development_9_cells, then evaluate the "
            "unchanged labels on frozen_16_cells."
        ),
        "reference_label": args.reference_label,
        "bootstrap_samples": args.bootstrap_samples,
        "budgets": budgets,
        "sources": source_summary,
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "development_selection": str(output_dir / "development_selection.csv"),
            "development_selected_frozen_evaluation": str(
                output_dir / "development_selected_frozen_evaluation.csv"
            ),
            "frozen_paired_bootstrap": str(output_dir / "frozen_paired_bootstrap.csv"),
            "query_metrics_partitioned": str(
                output_dir / "query_metrics_partitioned.csv"
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("\nDEVELOPMENT SELECTION")
    print(development_selection.to_string(index=False))
    print("\nFROZEN EVALUATION OF DEVELOPMENT-SELECTED MODELS")
    print(frozen_evaluation.to_string(index=False))
    print("\nPAIRED FROZEN COMPARISON AGAINST REFERENCE")
    print(bootstrap_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
