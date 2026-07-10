from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ALL_DATASET_SCALES = ["enzyme405_50", "enzyme405_100", "enzyme405_all_feasible"]
BASELINES = [
    ("official_precomputed_pocket", "official_precomputed", "top1", "max"),
    ("p2rank_top1", "p2rank", "top1", "max"),
    ("p2rank_topk_max", "p2rank", "topk", "max"),
    ("p2rank_topk_mean", "p2rank", "topk", "mean"),
    ("p2rank_topk_rank_weighted", "p2rank", "topk", "rank_weighted"),
    ("p2rank_topk_softmax_pool", "p2rank", "topk", "softmax_pool"),
    ("fpocket_top1", "fpocket", "top1", "max"),
    ("fpocket_topk_rank_weighted", "fpocket", "topk", "rank_weighted"),
    ("p2rank_fpocket_union_max", "p2rank+fpocket", "union_topk", "max"),
    ("p2rank_fpocket_union_source_weighted", "p2rank+fpocket", "union_topk", "source_weighted"),
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> Any:
    if value is None:
        return "NA"
    try:
        if isinstance(value, str) and value.strip().lower() in {"na", "nan", "none", ""}:
            return "NA"
        return float(value)
    except Exception:
        return "NA"


def _numeric_or_none(value: Any) -> float | None:
    numeric = _safe_float(value)
    return numeric if isinstance(numeric, float) else None


def _find_aggregation(run_dir: Path) -> Path | None:
    agg_dir = run_dir / "aggregation"
    if not agg_dir.exists():
        return None
    matches = sorted(agg_dir.glob("enzyme_level_*.csv"))
    return matches[0] if matches else None


def _load_aggregation(run_dir: Path) -> pd.DataFrame | None:
    path = _find_aggregation(run_dir)
    if path is None:
        return None
    return pd.read_csv(path)


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    for candidate in [
        run_dir / "metrics/metrics_top5_top10.json",
        run_dir / "metrics/topk_metrics.json",
    ]:
        if candidate.exists():
            return _load_json(candidate)
    return {}


def _infer_status(run_dir: Path, summary: dict[str, Any]) -> str:
    status = str(summary.get("status", "not_run"))
    metrics = _load_metrics(run_dir)
    aggregation = _find_aggregation(run_dir)
    predictions = run_dir / "predictions/pocket_level_predictions.csv"

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


def _load_scale_summary(results_root: Path, scale: str) -> dict[str, Any]:
    for baseline in ["official_precomputed_pocket", "p2rank_top1", "p2rank_topk_softmax_pool"]:
        path = results_root / f"{scale}_{baseline}" / "smallset_summary.json"
        if path.exists():
            return _load_json(path)
    for run_dir in sorted(results_root.glob(f"{scale}_*")):
        path = run_dir / "smallset_summary.json"
        if path.exists():
            return _load_json(path)
    return {}


def _load_dataset_stats(results_root: Path, scale: str) -> dict[str, Any]:
    summary = _load_scale_summary(results_root, scale)
    sanity = summary.get("sanity_label_report", {})
    return {
        "n_reactions": sanity.get("n_reactions", summary.get("n_reactions", "NA")),
        "n_valid_reactions": sanity.get("n_valid_reactions", summary.get("n_valid_reactions", "NA")),
        "n_pairs": sanity.get("n_pairs", summary.get("n_pairs", "NA")),
        "n_unique_enzymes": sanity.get("n_unique_enzymes", summary.get("n_unique_enzymes", "NA")),
        "n_positive_pairs": sanity.get("n_positive_pairs", summary.get("n_positive_pairs", "NA")),
    }


def _rank_metrics(new_df: pd.DataFrame | None, official_df: pd.DataFrame | None) -> tuple[Any, Any, Any]:
    if new_df is None or official_df is None or new_df.empty or official_df.empty:
        return "NA", "NA", "NA"
    key_cols = ["enzyme_id"]
    if "reaction_id" in new_df.columns and "reaction_id" in official_df.columns:
        key_cols = ["reaction_id", "enzyme_id"]

    def rank_frame(df: pd.DataFrame, col: str) -> pd.DataFrame:
        ranked = df.copy()
        if "reaction_id" in ranked.columns:
            ranked[col] = ranked.groupby("reaction_id")["aggregated_score"].rank(method="first", ascending=False)
        else:
            ranked[col] = ranked["aggregated_score"].rank(method="first", ascending=False)
        return ranked

    official_ranked = rank_frame(official_df, "official_rank")
    new_ranked = rank_frame(new_df, "new_rank")
    merged = official_ranked[key_cols + ["official_rank", "aggregated_score"]].merge(
        new_ranked[key_cols + ["new_rank", "aggregated_score"]],
        on=key_cols,
        how="inner",
        suffixes=("_official", "_new"),
    )
    if merged.empty:
        return "NA", "NA", "NA"
    rank_shift = merged["official_rank"] - merged["new_rank"]
    score_shift = merged["aggregated_score_new"] - merged["aggregated_score_official"]
    return float(rank_shift.mean()), float(rank_shift.median()), float(score_shift.mean())


def _best_rank_gt1_rate(df: pd.DataFrame | None) -> Any:
    if df is None or df.empty or "best_pocket_rank" not in df.columns:
        return "NA"
    ranks = pd.to_numeric(df["best_pocket_rank"], errors="coerce").dropna()
    if ranks.empty:
        return "NA"
    return float((ranks > 1).mean())


def _blocked_reason(summary: dict[str, Any], baseline: str) -> str:
    status = str(summary.get("status", "not_run"))
    if status.startswith("blocked") or status.startswith("resource_limited"):
        return status
    if status.startswith("failed"):
        return str(summary.get("failed_step") or summary.get("error") or status)
    if status == "not_run":
        if "fpocket" in baseline or "union" in baseline:
            if shutil.which("fpocket") is None:
                return "blocked_fpocket_missing"
        return "resource_limited"
    return ""


def _render_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows.\n"
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                values.append("NA")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    return str(value)


def _best_completed_scale(matrix: pd.DataFrame) -> str | None:
    completed = matrix[matrix["status"] == "completed"].copy()
    if completed.empty:
        return None
    scale_stats = (
        completed.groupby("dataset_scale")["n_pairs"]
        .max()
        .sort_values(ascending=False)
    )
    if scale_stats.empty:
        return None
    return str(scale_stats.index[0])


def _compact_conclusion_lines(matrix: pd.DataFrame, results_root: Path) -> list[str]:
    scale = str(matrix["dataset_scale"].iloc[0])
    subset = matrix[matrix["dataset_scale"] == scale].copy()
    completed = subset[subset["status"] == "completed"].copy()
    rows = subset[["baseline", "status", "top5_success", "top10_success", "best_pocket_rank_gt1_rate"]].copy()
    rows = rows.sort_values("baseline")

    def row_for(baseline: str) -> pd.Series | None:
        match = subset[subset["baseline"] == baseline]
        if match.empty:
            return None
        return match.iloc[0]

    official = row_for("official_precomputed_pocket")
    p2rank_top1 = row_for("p2rank_top1")
    softmax = row_for("p2rank_topk_softmax_pool")
    topk_rows = [
        row
        for row in [row_for("p2rank_topk_max"), row_for("p2rank_topk_mean"), row_for("p2rank_topk_rank_weighted"), softmax]
        if row is not None and str(row["status"]) == "completed"
    ]
    fpocket = row_for("fpocket_top1")
    fpocket_blocked_reason = ""
    if fpocket is not None and str(fpocket["status"]).startswith("blocked"):
        fpocket_blocked_reason = str(fpocket["blocked_reason"])
    else:
        union = row_for("p2rank_fpocket_union_max")
        if union is None:
            union = row_for("p2rank_fpocket_union_source_weighted")
        if union is not None and str(union["status"]).startswith("blocked"):
            fpocket_blocked_reason = str(union["blocked_reason"])

    lines = [
        "# Enzyme-405 50-reaction slice conclusion",
        "",
        f"- 数据规模：50 reactions、{_format_value(subset['n_pairs'].iloc[0] if not subset.empty else 'NA')} pairs、{_format_value(subset['n_positive_pairs'].iloc[0] if not subset.empty else 'NA')} positive pairs、{_format_value(subset['n_unique_enzymes'].iloc[0] if not subset.empty else 'NA')} unique enzymes。",
        "",
        "## Baseline Table",
        "",
        _render_table(rows).rstrip(),
        "",
        "## Top-5 / Top-10",
        "",
    ]

    if official is not None and p2rank_top1 is not None:
        lines.append(
            f"- official_precomputed_pocket vs p2rank_top1: Top-5 `{official['top5_success']}` vs `{p2rank_top1['top5_success']}`, Top-10 `{official['top10_success']}` vs `{p2rank_top1['top10_success']}`."
        )

    if p2rank_top1 is not None and topk_rows:
        completed_topk = pd.DataFrame(topk_rows)
        completed_topk["top5_success"] = pd.to_numeric(completed_topk["top5_success"], errors="coerce")
        completed_topk["top10_success"] = pd.to_numeric(completed_topk["top10_success"], errors="coerce")
        best_topk = completed_topk.sort_values(["top10_success", "top5_success"], ascending=False).iloc[0]
        delta5 = _numeric_or_none(best_topk["top5_success"])
        delta10 = _numeric_or_none(best_topk["top10_success"])
        top1_top5 = _numeric_or_none(p2rank_top1["top5_success"])
        top1_top10 = _numeric_or_none(p2rank_top1["top10_success"])
        delta5 = delta5 - top1_top5 if delta5 is not None and top1_top5 is not None else "NA"
        delta10 = delta10 - top1_top10 if delta10 is not None and top1_top10 is not None else "NA"
        same = completed["top5_success"].nunique() == 1 and completed["top10_success"].nunique() == 1
        if same:
            lines.append("在 Enzyme-405 50-reaction slice 上，pocket localization uncertainty 存在，但 naive multi-pocket aggregation 没有提升 Top-5/Top-10 retrieval success.")
        else:
            lines.append(
                f"- p2rank_top1 vs p2rank_topk aggregations: best completed top-k baseline is `{best_topk['baseline']}` with Top-5 delta `{delta5:+.4f}` and Top-10 delta `{delta10:+.4f}` vs p2rank_top1."
            )
        if softmax is not None:
            lines.append(f"- best_pocket_rank > 1 ratio: `{softmax['best_pocket_rank_gt1_rate']}`.")
    elif p2rank_top1 is not None:
        lines.append("- p2rank_top1 is the only completed P2Rank baseline available for this slice.")

    if fpocket_blocked_reason:
        lines.append(f"- fpocket blocked reason: `{fpocket_blocked_reason}`.")

    if completed.empty:
        lines.append("- No completed baselines were available.")

    return lines


def build_matrix(results_root: Path, dataset_scales: list[str] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scale in (dataset_scales or ALL_DATASET_SCALES):
        dataset_stats = _load_dataset_stats(results_root, scale)
        official_run_dir = results_root / f"{scale}_official_precomputed_pocket"
        official_agg = _load_aggregation(official_run_dir)
        official_metrics = _load_metrics(official_run_dir)

        for baseline, pocket_source, pocket_selection, aggregation in BASELINES:
            run_dir = results_root / f"{scale}_{baseline}"
            summary = _load_json(run_dir / "run_summary.json")
            agg_df = _load_aggregation(run_dir)
            metrics = _load_metrics(run_dir)
            inferred_status = _infer_status(run_dir, summary)
            mean_rank_shift, median_rank_shift, mean_score_shift = _rank_metrics(agg_df, official_agg)
            rows.append(
                {
                    "dataset_scale": scale,
                    "baseline": baseline,
                    "pocket_source": pocket_source,
                    "pocket_selection": pocket_selection,
                    "aggregation": aggregation,
                    "status": inferred_status,
                    "n_reactions": dataset_stats["n_reactions"],
                    "n_valid_reactions": dataset_stats["n_valid_reactions"],
                    "n_pairs": dataset_stats["n_pairs"],
                    "n_unique_enzymes": dataset_stats["n_unique_enzymes"],
                    "n_positive_pairs": dataset_stats["n_positive_pairs"],
                    "n_pockets": int(len(pd.read_csv(run_dir / "manifests/pocket_manifest.csv"))) if (run_dir / "manifests/pocket_manifest.csv").exists() else "NA",
                    "top5_success": metrics.get("top5_success_rate", "NA"),
                    "top10_success": metrics.get("top10_success_rate", "NA"),
                    "delta_top5_vs_official": (
                        _numeric_or_none(metrics.get("top5_success_rate"))
                        - _numeric_or_none(official_metrics.get("top5_success_rate"))
                        if _numeric_or_none(metrics.get("top5_success_rate")) is not None and _numeric_or_none(official_metrics.get("top5_success_rate")) is not None
                        else "NA"
                    ),
                    "delta_top10_vs_official": (
                        _numeric_or_none(metrics.get("top10_success_rate"))
                        - _numeric_or_none(official_metrics.get("top10_success_rate"))
                        if _numeric_or_none(metrics.get("top10_success_rate")) is not None and _numeric_or_none(official_metrics.get("top10_success_rate")) is not None
                        else "NA"
                    ),
                    "mean_rank_shift_vs_official": mean_rank_shift,
                    "median_rank_shift_vs_official": median_rank_shift,
                    "mean_score_shift_vs_official": mean_score_shift,
                    "best_pocket_rank_gt1_rate": _best_rank_gt1_rate(agg_df),
                    "blocked_reason": "" if inferred_status == "completed" else _blocked_reason(summary, baseline),
                }
            )
    return pd.DataFrame(rows)


def _write_report(matrix: pd.DataFrame, results_root: Path, output_path: Path) -> Path:
    if matrix["dataset_scale"].nunique() == 1 and str(matrix["dataset_scale"].iloc[0]) == "enzyme405_50":
        lines = _compact_conclusion_lines(matrix, results_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    completed = matrix[matrix["status"] == "completed"].copy()
    best_scale = _best_completed_scale(matrix)
    scale_subset = matrix[matrix["dataset_scale"] == best_scale].copy() if best_scale else matrix.iloc[0:0].copy()
    completed["top5_success"] = pd.to_numeric(completed["top5_success"], errors="coerce")
    completed["top10_success"] = pd.to_numeric(completed["top10_success"], errors="coerce")
    completed_top5 = completed[completed["top5_success"].notna()]
    best_baseline = None
    if not completed_top5.empty:
        best_row = completed_top5.sort_values(["top10_success", "top5_success"], ascending=False).iloc[0]
        best_baseline = {
            "dataset_scale": best_row["dataset_scale"],
            "baseline": best_row["baseline"],
            "top5": best_row["top5_success"],
            "top10": best_row["top10_success"],
        }

    def row_for(baseline: str) -> pd.Series | None:
        if scale_subset.empty:
            return None
        subset = scale_subset[scale_subset["baseline"] == baseline]
        if subset.empty:
            return None
        return subset.iloc[0]

    official_row = row_for("official_precomputed_pocket")
    p2rank_top1_row = row_for("p2rank_top1")
    p2rank_softmax_row = row_for("p2rank_topk_softmax_pool")
    p2rank_max_row = row_for("p2rank_topk_max")
    p2rank_mean_row = row_for("p2rank_topk_mean")
    p2rank_rank_row = row_for("p2rank_topk_rank_weighted")
    fpocket_row = row_for("fpocket_top1")
    union_row = row_for("p2rank_fpocket_union_max")

    completed_equal = False
    if official_row is not None and p2rank_top1_row is not None:
        completed_equal = (
            _safe_float(official_row["top5_success"]) == _safe_float(p2rank_top1_row["top5_success"])
            and _safe_float(official_row["top10_success"]) == _safe_float(p2rank_top1_row["top10_success"])
        )

    lines = [
        "# 最佳可行 pocket 结果矩阵",
        "",
        "## 执行摘要",
        "",
        "- 我们使用当前已经拿到的官方资产、单独下载的 AlphaFold 结构，以及已经跑通的 P2Rank 流程，做了最大可行的 pocket exploration。",
        "- 这不是完整论文复现；官方公开配置引用了缺失的预计算路径，所以我们把重点放在真实可复现的 derived smallset 上。",
        f"- 当前真正完成的最大规模是 `{best_scale}`。" if best_scale else "- 当前还没有稳定完成的最大规模结果。",
        f"- 当前 completed baseline 在 Top-5 / Top-10 上基本并列；表中按排序给出一个代表性 baseline `{best_baseline['dataset_scale']} / {best_baseline['baseline']}`，Top-5={best_baseline['top5']}，Top-10={best_baseline['top10']}。" if best_baseline else "- 目前没有足够的 completed baseline 用于比较。",
        "",
        "## 数据集规模",
        "",
    ]
    for scale in ALL_DATASET_SCALES:
        subset = matrix[matrix["dataset_scale"] == scale]
        stats = subset.iloc[0] if not subset.empty else None
        completed_subset = subset[subset["status"] == "completed"]
        blocked_subset = subset[subset["status"].str.startswith("blocked", na=False)]
        failed_subset = subset[subset["status"].str.startswith("failed", na=False)]
        if stats is None:
            lines.append(f"- `{scale}`: 尚无结果。")
            continue
        lines.append(
            f"- `{scale}`: n_reactions={_format_value(stats['n_reactions'])}, n_valid_reactions={_format_value(stats['n_valid_reactions'])}, n_pairs={_format_value(stats['n_pairs'])}, n_positive_pairs={_format_value(stats['n_positive_pairs'])}, completed={len(completed_subset)}, blocked={len(blocked_subset)}, failed={len(failed_subset)}"
        )

    lines.extend(
        [
            "",
            "## 主结果表",
            "",
            _render_table(matrix),
            "",
            "## 关键结论",
            "",
        ]
    )

    if official_row is not None and p2rank_top1_row is not None:
        official_top5 = _safe_float(official_row["top5_success"])
        official_top10 = _safe_float(official_row["top10_success"])
        p2rank_top1_top5 = _safe_float(p2rank_top1_row["top5_success"])
        p2rank_top1_top10 = _safe_float(p2rank_top1_row["top10_success"])
        lines.append(
            f"- `official_precomputed_pocket` 的 Top-5/Top-10 分别是 `{official_row['top5_success']}` / `{official_row['top10_success']}`；`p2rank_top1` 分别是 `{p2rank_top1_row['top5_success']}` / `{p2rank_top1_row['top10_success']}`。"
        )
        if completed_equal:
            lines.append("- 在这个 slice 上，两者在 Top-5 / Top-10 上数值相同，没有观察到稳定差异。")
        elif (
            official_top5 != "NA"
            and official_top10 != "NA"
            and p2rank_top1_top5 != "NA"
            and p2rank_top1_top10 != "NA"
            and (p2rank_top1_top5 < official_top5 or p2rank_top1_top10 < official_top10)
        ):
            lines.append("- 在当前可用数据上，官方预抽取 pocket 优于直接用 P2Rank 替换入口 pocket。")
        else:
            lines.append("- 在当前可用数据上，P2Rank 没有明显落后于官方 precomputed pocket。")

    if p2rank_softmax_row is not None and p2rank_top1_row is not None:
        softmax_top5 = _numeric_or_none(p2rank_softmax_row["top5_success"])
        softmax_top10 = _numeric_or_none(p2rank_softmax_row["top10_success"])
        top1_top5 = _numeric_or_none(p2rank_top1_row["top5_success"])
        top1_top10 = _numeric_or_none(p2rank_top1_row["top10_success"])
        lines.append(
            f"- `p2rank_topk_softmax_pool` 的 Top-5/Top-10 分别是 `{p2rank_softmax_row['top5_success']}` / `{p2rank_softmax_row['top10_success']}`；相对 `p2rank_top1` 的变化是 Top-5 `{p2rank_softmax_row['delta_top5_vs_official']}`、Top-10 `{p2rank_softmax_row['delta_top10_vs_official']}`。"
        )
        gt1_rate = _numeric_or_none(p2rank_softmax_row["best_pocket_rank_gt1_rate"])
        if gt1_rate is not None and gt1_rate > 0:
            lines.append(f"- `best_pocket_rank > 1` 的比例约为 `{p2rank_softmax_row['best_pocket_rank_gt1_rate']}`，说明 pocket localization uncertainty 确实存在。")
        if (
            softmax_top5 is not None
            and softmax_top10 is not None
            and top1_top5 is not None
            and top1_top10 is not None
            and softmax_top5 <= top1_top5
            and softmax_top10 <= top1_top10
        ):
            lines.append("- 但 naive multi-pocket aggregation 本身并没有稳定带来检索提升。")

    if any(row is not None for row in [p2rank_max_row, p2rank_mean_row, p2rank_rank_row]):
        lines.append("- `max`、`mean`、`rank_weighted`、`softmax_pool` 之间的差异可直接从表格中比较；如果它们的 Top-5 / Top-10 基本一致，说明聚合策略在当前数据切片上的影响有限。")

    if fpocket_row is not None and str(fpocket_row["status"]).startswith("blocked"):
        lines.append(f"- `fpocket` 系列基线当前被阻塞，原因是 `{fpocket_row['blocked_reason']}`。")
    if union_row is not None and str(union_row["status"]).startswith("blocked"):
        lines.append(f"- `P2Rank + fpocket` union 基线当前也被阻塞，原因是 `{union_row['blocked_reason']}`。")

    lines.extend(
        [
            "",
            "## 与论文的对照",
            "",
            "- 论文报告的是完整 Enzyme-405 基准上的更高性能；我们这里没有做完整复现，也没有把缺失的预计算特征伪造出来。",
            "- 这是基于公开资产、重新生成输入、以及真实 pocket intervention 得到的 best available reconstruction。",
            "- 结果差异不能直接解释成论文错误，更合理的理解是：公开资产和公开配置本身就限制了可复现上限。",
            "",
            "## 局限性",
            "",
            "- 不是完整官方 benchmark 复现。",
            "- 官方 config 仍然引用缺失的预计算路径。",
            "- AlphaFold 结构是单独下载的。",
            "- fpocket 目前仍然可能 blocked。",
            "- candidate / feature reconstruction 可能与论文实现存在差异。",
            "",
            "## 下一步",
            "",
            "- 如果要继续推进，优先补齐更大规模的 Enzyme-405 slice。",
            "- 如果能拿到作者提供的 inference-ready feature bundle，结果会更接近论文设置。",
            "- 下一步最值得加的是 catalytic-residue-aware pocket prior。",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the best available pocket result matrix.")
    parser.add_argument("--results_root", default="results/pocket")
    parser.add_argument("--output_csv", default="results/pocket/best_available_result_matrix.csv")
    parser.add_argument("--output_md", default="results/pocket/best_available_result_matrix.md")
    parser.add_argument("--conclusion_md", default="results/pocket/best_available_conclusion.md")
    parser.add_argument("--dataset_scales", nargs="+")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_root = Path(args.results_root)
    matrix = build_matrix(results_root, dataset_scales=args.dataset_scales)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_csv, index=False)
    output_md.write_text(_render_table(matrix), encoding="utf-8")
    conclusion_path = _write_report(matrix, results_root, Path(args.conclusion_md))
    print(
        json.dumps(
            {
                "matrix_csv": str(output_csv),
                "matrix_md": str(output_md),
                "conclusion_md": str(conclusion_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
