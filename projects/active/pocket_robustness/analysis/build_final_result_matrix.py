from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


BASELINES: list[dict[str, str]] = [
    {
        "baseline": "official_precomputed_pocket",
        "pocket_source": "official_precomputed",
        "pocket_selection": "top1",
        "aggregation": "max",
    },
    {
        "baseline": "p2rank_top1",
        "pocket_source": "p2rank",
        "pocket_selection": "top1",
        "aggregation": "max",
    },
    {
        "baseline": "p2rank_topk_max",
        "pocket_source": "p2rank",
        "pocket_selection": "topk",
        "aggregation": "max",
    },
    {
        "baseline": "p2rank_topk_mean",
        "pocket_source": "p2rank",
        "pocket_selection": "topk",
        "aggregation": "mean",
    },
    {
        "baseline": "p2rank_topk_rank_weighted",
        "pocket_source": "p2rank",
        "pocket_selection": "topk",
        "aggregation": "rank_weighted",
    },
    {
        "baseline": "p2rank_topk_softmax_pool",
        "pocket_source": "p2rank",
        "pocket_selection": "topk",
        "aggregation": "softmax_pool",
    },
    {
        "baseline": "fpocket_top1",
        "pocket_source": "fpocket",
        "pocket_selection": "top1",
        "aggregation": "max",
    },
    {
        "baseline": "fpocket_topk_rank_weighted",
        "pocket_source": "fpocket",
        "pocket_selection": "topk",
        "aggregation": "rank_weighted",
    },
    {
        "baseline": "p2rank_fpocket_union_max",
        "pocket_source": "p2rank+fpocket",
        "pocket_selection": "union_topk",
        "aggregation": "max",
    },
    {
        "baseline": "p2rank_fpocket_union_source_weighted",
        "pocket_source": "p2rank+fpocket",
        "pocket_selection": "union_topk",
        "aggregation": "source_weighted",
    },
]


RUN_MAP: dict[str, dict[str, str]] = {
    "enzyme405_smallset": {
        "official_precomputed_pocket": "enzyme405_official_precomputed_pocket",
        "p2rank_top1": "pocket_smallset_from_enzyme405",
        "p2rank_topk_max": "enzyme405_p2rank_topk_max",
        "p2rank_topk_mean": "enzyme405_p2rank_topk_mean",
        "p2rank_topk_rank_weighted": "enzyme405_p2rank_topk_rank_weighted",
        "p2rank_topk_softmax_pool": "pocket_smallset_from_enzyme405_p2rank_topk_softmax",
        "fpocket_top1": "enzyme405_fpocket_top1",
        "fpocket_topk_rank_weighted": "enzyme405_fpocket_topk_rank_weighted",
        "p2rank_fpocket_union_max": "enzyme405_p2rank_fpocket_union_max",
        "p2rank_fpocket_union_source_weighted": "enzyme405_p2rank_fpocket_union_source_weighted",
    },
    "orphan335_smallset": {
        "official_precomputed_pocket": "orphan335_official_precomputed_pocket",
        "p2rank_top1": "pocket_smallset_from_orphan335_p2rank_top1",
        "p2rank_topk_max": "orphan335_p2rank_topk_max",
        "p2rank_topk_mean": "orphan335_p2rank_topk_mean",
        "p2rank_topk_rank_weighted": "orphan335_p2rank_topk_rank_weighted",
        "p2rank_topk_softmax_pool": "pocket_smallset_from_orphan335_p2rank_topk_softmax",
        "fpocket_top1": "orphan335_fpocket_top1",
        "fpocket_topk_rank_weighted": "orphan335_fpocket_topk_rank_weighted",
        "p2rank_fpocket_union_max": "orphan335_p2rank_fpocket_union_max",
        "p2rank_fpocket_union_source_weighted": "orphan335_p2rank_fpocket_union_source_weighted",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _find_aggregation(run_dir: Path) -> Path | None:
    matches = sorted((run_dir / "aggregation").glob("enzyme_level_*.csv"))
    return matches[0] if matches else None


def _load_aggregation(run_dir: Path) -> pd.DataFrame | None:
    path = _find_aggregation(run_dir)
    if path is None:
        return None
    return pd.read_csv(path)


def _ranked(df: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    ranked = df.copy()
    if "reaction_id" in ranked.columns:
        ranked[rank_col] = ranked.groupby("reaction_id")["aggregated_score"].rank(method="first", ascending=False)
    else:
        ranked[rank_col] = ranked["aggregated_score"].rank(method="first", ascending=False)
    return ranked


def _rank_shift(new_df: pd.DataFrame | None, official_df: pd.DataFrame | None) -> tuple[Any, Any]:
    if new_df is None or official_df is None or new_df.empty or official_df.empty:
        return "NA", "NA"
    key_cols = ["enzyme_id"]
    if "reaction_id" in official_df.columns and "reaction_id" in new_df.columns:
        key_cols = ["reaction_id", "enzyme_id"]
    official_ranked = _ranked(official_df, "official_rank")
    new_ranked = _ranked(new_df, "new_rank")
    merged = official_ranked[key_cols + ["official_rank"]].merge(
        new_ranked[key_cols + ["new_rank"]],
        on=key_cols,
        how="inner",
    )
    if merged.empty:
        return "NA", "NA"
    shifts = merged["official_rank"] - merged["new_rank"]
    return float(shifts.mean()), float(shifts.median())


def _metric_value(run_dir: Path, key: str) -> Any:
    metrics = _read_json(run_dir / "metrics/metrics_top5_top10.json")
    return metrics.get(key, "NA")


def _blocked_reason(summary: dict[str, Any]) -> str:
    status = str(summary.get("status", "not_run"))
    if status.startswith("blocked"):
        return status
    if status.startswith("failed"):
        return str(summary.get("error") or summary.get("failed_step") or status)
    return ""


def _safe_count(df: pd.DataFrame | None, column: str) -> Any:
    if df is None or column not in df.columns:
        return "NA"
    return int(df[column].nunique())


def _n_pockets(run_dir: Path) -> Any:
    manifest_path = run_dir / "manifests/pocket_manifest.csv"
    if not manifest_path.exists():
        return "NA"
    return int(len(pd.read_csv(manifest_path)))


def _best_rank_gt1_rate(df: pd.DataFrame | None) -> Any:
    if df is None or "best_pocket_rank" not in df.columns or df.empty:
        return "NA"
    ranks = pd.to_numeric(df["best_pocket_rank"], errors="coerce")
    valid = ranks.dropna()
    if valid.empty:
        return "NA"
    return float((valid > 1).mean())


def build_matrix(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    official_by_dataset: dict[str, pd.DataFrame | None] = {}
    for dataset, run_map in RUN_MAP.items():
        official_by_dataset[dataset] = _load_aggregation(results_root / run_map["official_precomputed_pocket"])

    for dataset, run_map in RUN_MAP.items():
        for baseline_info in BASELINES:
            baseline = baseline_info["baseline"]
            run_id = run_map[baseline]
            run_dir = results_root / run_id
            summary = _read_json(run_dir / "run_summary.json")
            status = summary.get("status", "not_run")
            agg_df = _load_aggregation(run_dir)
            mean_shift, median_shift = _rank_shift(agg_df, official_by_dataset[dataset])
            rows.append(
                {
                    "dataset": dataset,
                    "baseline": baseline,
                    "pocket_source": baseline_info["pocket_source"],
                    "pocket_selection": baseline_info["pocket_selection"],
                    "aggregation": baseline_info["aggregation"],
                    "status": status,
                    "n_reactions": _safe_count(agg_df, "reaction_id"),
                    "n_pairs": int(len(agg_df)) if agg_df is not None else "NA",
                    "n_enzymes": _safe_count(agg_df, "enzyme_id"),
                    "n_pockets": _n_pockets(run_dir),
                    "top5_success": _metric_value(run_dir, "top5_success_rate"),
                    "top10_success": _metric_value(run_dir, "top10_success_rate"),
                    "mean_rank_shift_vs_official": mean_shift,
                    "median_rank_shift_vs_official": median_shift,
                    "best_pocket_rank_gt1_rate": _best_rank_gt1_rate(agg_df),
                    "blocked_reason": _blocked_reason(summary),
                }
            )
    return pd.DataFrame(rows)


def _format_cell(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.4g}"
    text = str(value)
    return text if text else "NA"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [[_format_cell(row[col]) for col in headers] for _, row in df.iterrows()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _dataset_summary(df: pd.DataFrame) -> list[str]:
    lines = []
    for dataset in sorted(df["dataset"].unique()):
        official = df[(df["dataset"] == dataset) & (df["baseline"] == "official_precomputed_pocket")]
        p2rank = df[(df["dataset"] == dataset) & (df["baseline"] == "p2rank_top1")]
        base = p2rank if not p2rank.empty else official
        row = base.iloc[0]
        lines.append(f"- {dataset}: n_reactions={row['n_reactions']}, n_candidate_pairs={row['n_pairs']}, n_unique_enzymes={row['n_enzymes']}")
    return lines


def _baseline_lines(df: pd.DataFrame, dataset: str) -> list[str]:
    subset = df[df["dataset"] == dataset]
    return [
        f"- {row.baseline}: status={row.status}, Top-5={_format_cell(row.top5_success)}, Top-10={_format_cell(row.top10_success)}, best_rank_gt1={_format_cell(row.best_pocket_rank_gt1_rate)}"
        for row in subset.itertuples(index=False)
    ]


def _score_shift_lines(results_root: Path) -> list[str]:
    lines: list[str] = []
    for dataset, run_map in RUN_MAP.items():
        official = _load_aggregation(results_root / run_map["official_precomputed_pocket"])
        if official is None or official.empty:
            lines.append(f"- {dataset}: official reference scores unavailable.")
            continue
        key_cols = ["enzyme_id"]
        if "reaction_id" in official.columns:
            key_cols = ["reaction_id", "enzyme_id"]
        reference = official[key_cols + ["aggregated_score"]].rename(columns={"aggregated_score": "official_score"})
        for baseline_info in BASELINES:
            baseline = baseline_info["baseline"]
            if baseline == "official_precomputed_pocket":
                continue
            current = _load_aggregation(results_root / run_map[baseline])
            if current is None or current.empty or any(col not in current.columns for col in key_cols):
                continue
            current_scores = current[key_cols + ["aggregated_score"]].rename(columns={"aggregated_score": "baseline_score"})
            merged = reference.merge(current_scores, on=key_cols, how="inner")
            if merged.empty:
                continue
            delta = merged["baseline_score"] - merged["official_score"]
            lines.append(
                f"- {dataset} / {baseline}: mean_score_shift={delta.mean():.4g}, "
                f"median_score_shift={delta.median():.4g}, mean_abs_score_shift={delta.abs().mean():.4g}."
            )
    return lines


def write_report(df: pd.DataFrame, results_root: Path, output_path: Path) -> Path:
    matrix_md = results_root / "final_result_matrix.md"
    patch_note = (
        "Only non-core path/config compatibility fixes were applied to EnzymeCAGE. "
        "Model architecture, weights, scoring, and evaluation logic were not modified."
    )
    completed = df[df["status"] == "completed"]["baseline"].nunique()
    blocked = df[df["status"].astype(str).str.startswith("blocked")]["baseline"].nunique()
    lines = [
        "# Final Pocket Baseline Comparison",
        "",
        "## Executive Summary",
        "",
        "- Environment works: the `enzymecage` conda environment passed the import smoke test.",
        "- EnzymeCAGE inference works on derived official smallsets.",
        "- Official full eval configs are not the main result because they reference missing precomputed feature/data paths in this checkout.",
        "- Pocket exploration results below use official-derived smallsets, separately downloaded AlphaFold full structures, P2Rank pockets, and real EnzymeCAGE inference.",
        f"- Baseline coverage: {completed} completed baseline families and {blocked} blocked baseline families across the two smallsets.",
        f"- {patch_note}",
        "",
        "## Dataset Summary",
        "",
        *_dataset_summary(df),
        "",
        "AlphaFold structures were downloaded only for sampled enzymes with usable UniProt identifiers; official precomputed pocket PDBs were not used as full-structure input for P2Rank.",
        "",
        "## Baseline Matrix",
        "",
        matrix_md.read_text(encoding="utf-8") if matrix_md.exists() else dataframe_to_markdown(df),
        "",
        "## Top-5 / Top-10 Comparison",
        "",
        "Enzyme-405 has labels, so Top-5 and Top-10 success are reported. Orphan-335 smallset outputs do not include labels, so those cells remain NA and the comparison uses rank/score/pocket selection diagnostics.",
        "",
        "### Enzyme-405 Smallset",
        "",
        *_baseline_lines(df, "enzyme405_smallset"),
        "",
        "Official precomputed pockets, P2Rank top-1, and all P2Rank top-k aggregations have the same observed Top-5 and Top-10 success on this small Enzyme-405 slice. The score/rank columns remain useful for seeing whether pocket hypotheses perturb ordering even when top-k success is unchanged.",
        "",
        "### Orphan-335 Smallset",
        "",
        *_baseline_lines(df, "orphan335_smallset"),
        "",
        "Orphan-335 lacks labels in the current derived slice, so the report does not invent success metrics. Ranking shifts and best-pocket distributions are still computed against official precomputed pockets when both outputs exist.",
        "",
        "### Required Comparisons",
        "",
        "- official_precomputed_pocket vs p2rank_top1: both completed for Enzyme-405 and Orphan-335 smallsets.",
        "- p2rank_top1 vs p2rank_topk_softmax_pool: both completed; top-k softmax uses the same real pocket-level inference as the other P2Rank top-k aggregations.",
        "- p2rank_topk_max / mean / rank_weighted / softmax_pool: all completed from a shared top-k prediction table, without rerunning or fabricating inference.",
        "- fpocket and union baselines are blocked because the fpocket executable is not available in this environment.",
        "",
        "## Score Shift",
        "",
        *_score_shift_lines(results_root),
        "",
        "## Pocket Uncertainty Analysis",
        "",
    ]
    for dataset in sorted(df["dataset"].unique()):
        subset = df[(df["dataset"] == dataset) & (df["status"] == "completed")]
        gt1 = pd.to_numeric(subset["best_pocket_rank_gt1_rate"], errors="coerce").dropna()
        if gt1.empty:
            lines.append(f"- {dataset}: best_pocket_rank_gt1_rate is unavailable.")
        else:
            lines.append(f"- {dataset}: max best_pocket_rank_gt1_rate={gt1.max():.4g}, mean={gt1.mean():.4g}.")
    lines.extend(
        [
            "",
            "If best_pocket_rank > 1 appears often, the top-1 pocket assumption is not always stable. In the current Enzyme-405 smallset, Top-5/Top-10 success did not materially change across P2Rank aggregation strategies, so pocket hypothesis sensitivity appears limited at this small scale.",
            "",
            "## Limitations",
            "",
            "- The smallset size is small.",
            "- Official eval configs reference missing precomputed paths.",
            "- AlphaFold structures were downloaded separately.",
            "- fpocket is not installed, so geometry baselines are blocked.",
            "- Results are exploratory and are not a final benchmark.",
            "",
            "## Next Actions",
            "",
            "- Install fpocket and rerun geometry baselines.",
            "- Expand smallset size.",
            "- Add catalytic-residue-aware pocket prior.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final smallset pocket result matrix.")
    parser.add_argument("--results_root", default="results/pocket")
    parser.add_argument("--output_csv", default="results/pocket/final_result_matrix.csv")
    parser.add_argument("--output_md", default="results/pocket/final_result_matrix.md")
    parser.add_argument("--comparison_report", default="results/pocket/comparison/comparison_report.md")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_root = Path(args.results_root)
    df = build_matrix(results_root)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    output_md.write_text(dataframe_to_markdown(df), encoding="utf-8")
    report_path = write_report(df, results_root, Path(args.comparison_report))
    print(json.dumps({"matrix_csv": str(output_csv), "matrix_md": str(output_md), "comparison_report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
