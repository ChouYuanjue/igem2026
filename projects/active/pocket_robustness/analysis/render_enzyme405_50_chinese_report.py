from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


BASELINE_GROUPS = {
    "official": ["official_precomputed_pocket"],
    "p2rank_top1": ["p2rank_top1"],
    "p2rank_topk": [
        "p2rank_topk_max",
        "p2rank_topk_mean",
        "p2rank_topk_rank_weighted",
        "p2rank_topk_softmax_pool",
    ],
    "fpocket": ["fpocket_top1", "fpocket_topk_rank_weighted"],
    "union": ["p2rank_fpocket_union_max", "p2rank_fpocket_union_source_weighted"],
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else "NA"
    try:
        if pd.isna(value):
            return "NA"
    except Exception:  # noqa: BLE001
        pass
    try:
        number = float(value)
    except Exception:  # noqa: BLE001
        return str(value)
    if number.is_integer():
        return str(int(number))
    text = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _row_for(matrix: pd.DataFrame, baseline: str) -> pd.Series | None:
    match = matrix[matrix["baseline"] == baseline]
    if match.empty:
        return None
    return match.iloc[0]


def _best_completed_row(matrix: pd.DataFrame, baselines: list[str]) -> pd.Series | None:
    subset = matrix[(matrix["baseline"].isin(baselines)) & (matrix["status"] == "completed")].copy()
    if subset.empty:
        return None
    subset["top5_success"] = pd.to_numeric(subset["top5_success"], errors="coerce")
    subset["top10_success"] = pd.to_numeric(subset["top10_success"], errors="coerce")
    subset = subset.dropna(subset=["top5_success", "top10_success"])
    if subset.empty:
        return None
    return subset.sort_values(["top10_success", "top5_success"], ascending=False).iloc[0]


def _all_completed_same(matrix: pd.DataFrame) -> bool:
    completed = matrix[matrix["status"] == "completed"].copy()
    if completed.empty:
        return False
    pairs = {
        (str(row["top5_success"]), str(row["top10_success"]))
        for _, row in completed.iterrows()
    }
    return len(pairs) == 1


def _candidate_stats(predictions_path: Path) -> dict[str, Any]:
    if not predictions_path.exists():
        return {
            "avg_candidates_per_reaction": "NA",
            "min_candidates_per_reaction": "NA",
            "n_reactions_with_lt10_candidates": "NA",
        }
    df = pd.read_csv(predictions_path)
    if "reaction_id" not in df.columns or "enzyme_id" not in df.columns:
        return {
            "avg_candidates_per_reaction": "NA",
            "min_candidates_per_reaction": "NA",
            "n_reactions_with_lt10_candidates": "NA",
        }
    counts = df.groupby("reaction_id")["enzyme_id"].nunique()
    return {
        "avg_candidates_per_reaction": float(counts.mean()),
        "min_candidates_per_reaction": int(counts.min()),
        "n_reactions_with_lt10_candidates": int((counts < 10).sum()),
    }


def _completion_phrase(matrix: pd.DataFrame) -> str:
    if _all_completed_same(matrix):
        return "在当前 50-reaction slice 上，不同 pocket baseline 没有改变 Top-5 / Top-10 success。"
    return ""


def _section_baseline_design() -> list[str]:
    return [
        "- `official_precomputed_pocket`：官方预抽取 pocket，作为 anchor。",
        "- `p2rank_top1`：从 AlphaFold full structure 重新抽取 P2Rank top1 pocket。",
        "- `p2rank_topk_max`：P2Rank top5 pockets，取最大 CAGE score。",
        "- `p2rank_topk_mean`：P2Rank top5 pockets，取均值。",
        "- `p2rank_topk_rank_weighted`：按 P2Rank rank 加权。",
        "- `p2rank_topk_softmax_pool`：按 CAGE score softmax pooling。",
        "- `fpocket_top1`：几何 pocket detector 的 top1 pocket。",
        "- `fpocket_topk_rank_weighted`：fpocket top5 pockets 的 rank-weighted 版本。",
        "- `p2rank_fpocket_union_max`：P2Rank + fpocket pocket hypotheses 的并集后取最大分数。",
        "- `p2rank_fpocket_union_source_weighted`：按 source weight 融合的并集版本。",
    ]


def build_report(matrix: pd.DataFrame, matrix_md: Path, results_root: Path) -> str:
    scale = str(matrix["dataset_scale"].iloc[0]) if not matrix.empty else "enzyme405_50"
    stats_row = matrix.iloc[0] if not matrix.empty else None
    official_row = _row_for(matrix, "official_precomputed_pocket")
    p2rank_top1 = _row_for(matrix, "p2rank_top1")
    p2rank_best_topk = _best_completed_row(matrix, BASELINE_GROUPS["p2rank_topk"])
    fpocket_best = _best_completed_row(matrix, BASELINE_GROUPS["fpocket"])
    union_best = _best_completed_row(matrix, BASELINE_GROUPS["union"])

    official_run_dir = results_root / f"{scale}_official_precomputed_pocket"
    candidate_stats = _candidate_stats(official_run_dir / "predictions/pocket_level_predictions.csv")
    official_metrics = _load_json(official_run_dir / "metrics/metrics_top5_top10.json")
    completions_phrase = _completion_phrase(matrix)

    completed = matrix[matrix["status"] == "completed"].copy()
    if not completed.empty:
        completed["top5_success"] = pd.to_numeric(completed["top5_success"], errors="coerce")
        completed["top10_success"] = pd.to_numeric(completed["top10_success"], errors="coerce")
    best_top5_rows = []
    best_top10_rows = []
    if not completed.empty:
        best_top5_value = completed["top5_success"].max()
        best_top10_value = completed["top10_success"].max()
        best_top5_rows = completed[completed["top5_success"] == best_top5_value]["baseline"].tolist()
        best_top10_rows = completed[completed["top10_success"] == best_top10_value]["baseline"].tolist()

    lines: list[str] = [
        "# EnzymeCAGE Pocket Baseline 实验报告：Enzyme-405 50-reaction slice",
        "",
        "## 1. 实验目的",
        "",
        "- 本实验不是完整复现论文 full benchmark。",
        "- 目标是在公开可用资产和当前可运行条件下，比较不同 pocket 方案对 EnzymeCAGE 排序结果的影响。",
        "- 统一使用 Enzyme-405 的 50 reaction slice。",
        "- 只看 Top-5 和 Top-10。",
        "",
        "## 2. 数据与规模",
        "",
        f"- n_reactions: `{_fmt(stats_row['n_reactions']) if stats_row is not None else 'NA'}`",
        f"- n_valid_reactions: `{_fmt(stats_row['n_valid_reactions']) if stats_row is not None else 'NA'}`",
        f"- n_pairs: `{_fmt(stats_row['n_pairs']) if stats_row is not None else 'NA'}`",
        f"- n_positive_pairs: `{_fmt(stats_row['n_positive_pairs']) if stats_row is not None else 'NA'}`",
        f"- n_unique_enzymes: `{_fmt(stats_row['n_unique_enzymes']) if stats_row is not None else 'NA'}`",
        f"- 每个 reaction 平均 candidate 数: `{_fmt(candidate_stats['avg_candidates_per_reaction'])}`",
        f"- 是否有 reaction 无 positive: `{_fmt(official_metrics.get('reactions_without_positive', 'NA'))}`",
        f"- 是否存在少于 10 candidates 的 reaction: `{candidate_stats['n_reactions_with_lt10_candidates']}` 个 reaction",
        "",
        "## 3. Baseline 设计",
        "",
    ]
    lines.extend(_section_baseline_design())
    lines.extend(
        [
            "",
            "## 4. 结果总表",
            "",
            matrix_md.read_text(encoding="utf-8").rstrip(),
            "",
            "## 5. Top-5 / Top-10 结果解读",
            "",
        ]
    )

    if official_row is not None and p2rank_top1 is not None:
        lines.append(
            f"- `official_precomputed_pocket` 的 Top-5/Top-10 分别是 `{_fmt(official_row['top5_success'])}` / `{_fmt(official_row['top10_success'])}`。"
        )
        lines.append(
            f"- `p2rank_top1` 的 Top-5/Top-10 分别是 `{_fmt(p2rank_top1['top5_success'])}` / `{_fmt(p2rank_top1['top10_success'])}`。"
        )
        if _fmt(official_row["top5_success"]) == _fmt(p2rank_top1["top5_success"]) and _fmt(official_row["top10_success"]) == _fmt(p2rank_top1["top10_success"]):
            lines.append("- P2Rank top1 与 official_precomputed_pocket 在当前 slice 上几乎一致，没有观察到 Top-5 / Top-10 差异。")
        else:
            delta5 = float(p2rank_top1["top5_success"]) - float(official_row["top5_success"])
            delta10 = float(p2rank_top1["top10_success"]) - float(official_row["top10_success"])
            lines.append(f"- P2Rank top1 相对 official_precomputed_pocket 的变化为 Top-5 `{_fmt(delta5)}`、Top-10 `{_fmt(delta10)}`。")

    if p2rank_best_topk is not None and p2rank_top1 is not None:
        delta5 = float(p2rank_best_topk["top5_success"]) - float(p2rank_top1["top5_success"])
        delta10 = float(p2rank_best_topk["top10_success"]) - float(p2rank_top1["top10_success"])
        lines.append(
            f"- P2Rank top-k 聚合里当前最好的 baseline 是 `{p2rank_best_topk['baseline']}`，相对 `p2rank_top1` 的变化为 Top-5 `{_fmt(delta5)}`、Top-10 `{_fmt(delta10)}`。"
        )
        lines.append(
            f"- `best_pocket_rank > 1` 比例约为 `{_fmt(p2rank_best_topk['best_pocket_rank_gt1_rate'])}`，说明 pocket localization uncertainty 确实存在。"
        )
        if float(p2rank_best_topk["top5_success"]) <= float(p2rank_top1["top5_success"]) and float(p2rank_best_topk["top10_success"]) <= float(p2rank_top1["top10_success"]):
            lines.append("- 但 naive multi-pocket aggregation 没有把这种不确定性转化为检索命中率收益。")

    if fpocket_best is not None and p2rank_top1 is not None:
        delta5 = float(fpocket_best["top5_success"]) - float(p2rank_top1["top5_success"])
        delta10 = float(fpocket_best["top10_success"]) - float(p2rank_top1["top10_success"])
        lines.append(
            f"- fpocket 最好的完成结果是 `{fpocket_best['baseline']}`，相对 `p2rank_top1` 的变化为 Top-5 `{_fmt(delta5)}`、Top-10 `{_fmt(delta10)}`。"
        )
        if delta5 > 0 or delta10 > 0:
            lines.append("- fpocket 带来了可见提升。")
        else:
            lines.append("- fpocket 没有带来可见提升。")
    else:
        fpocket_row = _row_for(matrix, "fpocket_top1")
        if fpocket_row is not None:
            lines.append(
                f"- fpocket baseline 当前状态为 `{fpocket_row['status']}`，原因是 `{fpocket_row['blocked_reason'] or fpocket_row['status']}`。"
            )

    if union_best is not None and p2rank_top1 is not None:
        delta5 = float(union_best["top5_success"]) - float(p2rank_top1["top5_success"])
        delta10 = float(union_best["top10_success"]) - float(p2rank_top1["top10_success"])
        lines.append(
            f"- union baseline 的最优完成结果是 `{union_best['baseline']}`，相对 `p2rank_top1` 的变化为 Top-5 `{_fmt(delta5)}`、Top-10 `{_fmt(delta10)}`。"
        )
        if delta5 > 0 or delta10 > 0:
            lines.append("- union 方案带来了可见提升。")
        else:
            lines.append("- union 方案没有带来可见提升。")
    else:
        union_row = _row_for(matrix, "p2rank_fpocket_union_max")
        if union_row is None:
            union_row = _row_for(matrix, "p2rank_fpocket_union_source_weighted")
        if union_row is not None:
            lines.append(
                f"- union baseline 当前状态为 `{union_row['status']}`，原因是 `{union_row['blocked_reason'] or union_row['status']}`。"
            )

    if completions_phrase:
        lines.append(f"- {completions_phrase}")

    lines.extend(
        [
            "",
            "## 6. Pocket uncertainty 分析",
            "",
            f"- 当前较好的 P2Rank top-k baseline 中，`best_pocket_rank > 1` 的比例约为 `{_fmt(p2rank_best_topk['best_pocket_rank_gt1_rate']) if p2rank_best_topk is not None else 'NA'}`。",
            "- 这说明不少样本的最高 CAGE score pocket 并不是 rank1，pocket localization uncertainty 的确存在。",
            "- 如果这种现象并没有带来 Top-5 / Top-10 的提升，更合理的解释是：单纯扩大 pocket 搜索空间并不能自动修复检索排序。",
            "",
            "## 7. 与论文结果的关系",
            "",
            "- 论文在完整 Enzyme-405 benchmark 上报告更高表现。",
            "- 当前实验不是 full benchmark 复现，因为公开 config 里引用了缺失的预计算 feature 路径，我们没有把这些缺失路径伪造成可用结果。",
            "- 当前结果是基于公开资产、derived 50-reaction slice、重新组织输入和真实 EnzymeCAGE inference 的 pocket exploration。",
            "- 这些结果只能说明当前可复现条件下 pocket 选择策略的相对表现，不能据此判断论文错误。",
            "",
            "## 8. 结论",
            "",
        ]
    )

    best_completed = completed.copy()
    if not best_completed.empty:
        best_completed["top5_success"] = pd.to_numeric(best_completed["top5_success"], errors="coerce")
        best_completed["top10_success"] = pd.to_numeric(best_completed["top10_success"], errors="coerce")
        best_completed = best_completed.dropna(subset=["top5_success", "top10_success"])
    if not best_completed.empty:
        if _all_completed_same(matrix):
            lines.append("在当前 Enzyme-405 50-reaction slice 上，替换或扩展 pocket hypothesis 并未提升 Top-5/Top-10；单纯扩大 pocket 搜索空间不是有效改进方向。后续应转向 catalytic-residue-aware 或 mechanism-aware pocket reranking。")
        else:
            best_row = best_completed.sort_values(["top10_success", "top5_success"], ascending=False).iloc[0]
            delta5 = float(best_row["top5_success"]) - float(p2rank_top1["top5_success"]) if p2rank_top1 is not None else float("nan")
            delta10 = float(best_row["top10_success"]) - float(p2rank_top1["top10_success"]) if p2rank_top1 is not None else float("nan")
            lines.append(
                f"当前最佳 completed baseline 是 `{best_row['baseline']}`，相对 `p2rank_top1` 的提升为 Top-5 `{_fmt(delta5)}`、Top-10 `{_fmt(delta10)}`。"
            )
            lines.append("这意味着 pocket 设计仍然有影响，但收益取决于具体 detector / aggregation 组合。")
    else:
        lines.append("当前没有足够的 completed baseline 来给出稳健结论。")

    lines.extend(
        [
            "",
            "## 9. 局限性",
            "",
            "- 只用了 50 reaction slice。",
            "- 不是完整论文 benchmark。",
            "- 可能存在 candidate pool / feature reconstruction 差异。",
            "- AlphaFold structures 是单独下载的。",
        ]
    )

    fpocket_row = _row_for(matrix, "fpocket_top1")
    union_row = _row_for(matrix, "p2rank_fpocket_union_max")
    if union_row is None:
        union_row = _row_for(matrix, "p2rank_fpocket_union_source_weighted")
    if fpocket_row is not None and not str(fpocket_row["status"]).startswith("completed"):
        lines.append(f"- fpocket 失败或被阻塞时的具体类型是 `{fpocket_row['blocked_reason'] or fpocket_row['status']}`。")
    if union_row is not None and not str(union_row["status"]).startswith("completed"):
        lines.append(f"- union 失败或被阻塞时的具体类型是 `{union_row['blocked_reason'] or union_row['status']}`。")

    lines.extend(
        [
            "",
            "## 10. 下一步建议",
            "",
            "1. 扩展到更大 Enzyme-405 slice。",
            "2. 请求作者提供 inference-ready feature bundle。",
            "3. 加入 catalytic-residue-aware pocket prior，而不是继续盲目 top-k aggregation。",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Chinese enzyme405_50 report.")
    parser.add_argument("--matrix_csv", required=True)
    parser.add_argument("--matrix_md", required=True)
    parser.add_argument("--results_root", default="results/pocket")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    matrix = pd.read_csv(args.matrix_csv)
    report = build_report(matrix, Path(args.matrix_md), Path(args.results_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(str(output))


if __name__ == "__main__":
    main()
