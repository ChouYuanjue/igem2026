from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projects.active.terpene_screening.common import (
    TERPENE_RESULTS_DIR,
    read_table,
    safe_json_dump,
    write_markdown,
)


DATA_AUDIT_JSON = TERPENE_RESULTS_DIR / "data_audit.json"
STRUCTURE_REPORT_CSV = TERPENE_RESULTS_DIR / "structure_download_report.csv"
P2RANK_MANIFEST_CSV = TERPENE_RESULTS_DIR / "p2rank_pocket_manifest.csv"
METRICS_JSON = TERPENE_RESULTS_DIR / "metrics" / "topk_metrics.json"
REACTION_RESULTS_CSV = TERPENE_RESULTS_DIR / "reaction_level_results.csv"
FAILED_ID_MAPPING = TERPENE_RESULTS_DIR / "failed_id_mapping.tsv"
FAILED_P2RANK = TERPENE_RESULTS_DIR / "failed_p2rank_pockets.csv"
REPORT_PATH = TERPENE_RESULTS_DIR / "terpene_screen_report.md"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_table(path)


def _fmt_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0/0 (0.00%)"
    return f"{numerator}/{denominator} ({numerator / denominator:.2%})"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df.empty:
        return ["(无)"]
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    rows = df[columns].fillna("").astype(str).values.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_report() -> list[str]:
    audit = _load_json(DATA_AUDIT_JSON)
    metrics = _load_json(METRICS_JSON)
    reaction_df = _load_csv(REACTION_RESULTS_CSV)
    structure_df = _load_csv(STRUCTURE_REPORT_CSV)
    manifest_df = _load_csv(P2RANK_MANIFEST_CSV)
    failed_id_df = _load_csv(FAILED_ID_MAPPING)
    failed_p2rank_df = _load_csv(FAILED_P2RANK)

    source_files = audit.get("source_files", {})
    summary = audit.get("summary", {})

    n_structures_success = int(
        ((structure_df["status"] == "downloaded") | (structure_df["status"] == "existing")).sum()
    ) if not structure_df.empty and "status" in structure_df.columns else 0
    n_structures_total = int(len(structure_df))
    n_pockets_success = int(len(manifest_df)) if not manifest_df.empty else 0
    n_p2rank_fail = int(len(failed_p2rank_df)) if not failed_p2rank_df.empty else 0
    n_pairs_expected = int(summary.get("expected_pair_count", 0))
    n_pairs_scored = int(metrics.get("n_pairs_scored", 0))

    lines: list[str] = [
        "# Terpene CAGE 全酶库检索实验报告",
        "",
        "## 实验目的",
        "给定 10 条萜类 Rhea 反应，把所有萜类合酶作为候选酶，不做候选池筛选，直接构造 10 reactions × all terpene synthases 的 pair，使用 EnzymeCAGE 对所有 pair 打分并做排序，随后计算 Top-1 / Top-5 / Top-10 recall、MRR，以及每条 reaction 的真实酶排名与 Top-10 推荐列表。",
        "",
        "## 数据来源和数据规模",
    ]

    for name, item in source_files.items():
        lines.append(
            f"- `{name}`: `{item['rows']}` 行，列名 `{', '.join(item['columns'])}`"
        )
    lines.extend(
        [
            f"- 候选酶总数: `{summary.get('candidate_total', 0)}`",
            f"- 10 条 reaction 中有 positive label 的数量: `{summary.get('n_selected_reactions_with_positive_label', 0)}`",
            f"- 10 条 reaction 中没有 positive label 的数量: `{summary.get('n_selected_reactions_without_positive_label', 0)}`",
            f"- 预计 pair 数: `{n_pairs_expected}`",
            "",
            "## 为什么不做候选池筛选",
            "这次实验目标是做全库检索，而不是候选池内重排序。直接对所有 terpene synthase 构造 pair 可以避免先验筛选把真正的正例提前排除，也更接近“检索”场景本身，能更真实地衡量 EnzymeCAGE 在大候选空间里的区分能力。",
            "",
            "## 为什么只用 P2Rank top1",
            "本实验要求单 pocket 设置。使用 P2Rank top1 可以把 pocket 选择固定下来，避免多 pocket 聚合带来的额外设计自由度，同时也更省计算成本，更适合先做一版全库 screening 基线。",
            "",
            "## 结构下载成功率",
            f"- 成功结构数: `{n_structures_success}` / `{n_structures_total}`",
        ]
    )
    lines.append(f"- 成功率: `{_fmt_ratio(n_structures_success, n_structures_total)}`")
    lines.extend(
        [
            "",
            "## P2Rank 成功率",
            f"- 成功 pocket 数: `{n_pockets_success}` / `{n_structures_success}`",
            f"- 成功率: `{_fmt_ratio(n_pockets_success, n_structures_success)}`",
            f"- P2Rank 失败数: `{n_p2rank_fail}`",
            "",
            "## CAGE 成功打分 pair 数",
            f"- 成功打分 pair 数: `{n_pairs_scored}` / `{n_pairs_expected}`",
            f"- 覆盖率: `{_fmt_ratio(n_pairs_scored, n_pairs_expected)}`",
            "",
            "## Top-1 / Top-5 / Top-10 recall",
            f"- Top-1 recall: `{metrics.get('top1_recall', 0.0):.4f}`",
            f"- Top-5 recall: `{metrics.get('top5_recall', 0.0):.4f}`",
            f"- Top-10 recall: `{metrics.get('top10_recall', 0.0):.4f}`",
            f"- MRR: `{metrics.get('mean_reciprocal_rank', 0.0):.4f}`",
            f"- median best positive rank: `{metrics.get('median_best_positive_rank')}`",
            "",
            "## 每条 reaction 的命中情况",
        ]
    )

    if reaction_df.empty:
        lines.append("(暂无 reaction 结果)")
    else:
        lines.extend(
            _markdown_table(
                reaction_df,
                [
                    "reaction_id",
                    "rhea_id",
                    "status",
                    "n_candidates",
                    "n_positive_enzymes",
                    "best_positive_rank",
                    "top1_hit",
                    "top5_hit",
                    "top10_hit",
                ],
            )
        )

    lines.extend(["", "## Top-10 推荐酶列表"])
    if reaction_df.empty:
        lines.append("(暂无推荐列表)")
    else:
        for _, row in reaction_df.iterrows():
            top10_enzyme_ids = []
            top10_scores = []
            if "top10_enzyme_ids" in row.index and pd.notna(row["top10_enzyme_ids"]) and str(row["top10_enzyme_ids"]).strip():
                top10_enzyme_ids = json.loads(str(row["top10_enzyme_ids"]))
            if "top10_scores" in row.index and pd.notna(row["top10_scores"]) and str(row["top10_scores"]).strip():
                top10_scores = json.loads(str(row["top10_scores"]))
            best_rank = row.get("best_positive_rank")
            if pd.isna(best_rank) or str(best_rank).strip() == "":
                best_rank_text = "NA"
            else:
                try:
                    best_rank_text = str(int(float(best_rank)))
                except Exception:
                    best_rank_text = str(best_rank)
            lines.append(
                f"- `{row['reaction_id']}` / `{row['rhea_id']}`: best positive rank `{best_rank_text}`; "
                f"top10 enzymes `{', '.join(map(str, top10_enzyme_ids))}`; "
                f"scores `{', '.join(f'{float(score):.4f}' for score in top10_scores)}`"
            )

    lines.extend(
        [
            "",
            "## 失败记录",
            f"- ID 解析失败记录数: `{len(failed_id_df)}`",
            f"- P2Rank 失败记录数: `{len(failed_p2rank_df)}`",
        ]
    )
    if not failed_id_df.empty:
        lines.extend(
            [
                "",
                "### failed_id_mapping.tsv 前几行",
                "```tsv",
                failed_id_df.head(10).to_csv(sep="\t", index=False).rstrip(),
                "```",
            ]
        )
    if not failed_p2rank_df.empty:
        lines.extend(
            [
                "",
                "### failed_p2rank_pockets.csv 前几行",
                "```csv",
                failed_p2rank_df.head(10).to_csv(index=False).rstrip(),
                "```",
            ]
        )

    lines.extend(
        [
            "",
            "## 结果解释",
            "这次实验把搜索空间完全展开到全体 terpene synthase，因此指标更能反映模型在大候选空间中的排序能力。若 Top-k 指标较高，说明模型不仅能分辨反应类型，还能把真实酶稳定推到候选前列；若 MRR 偏低，则说明正确酶虽然有机会进入前列，但整体排序仍不够稳定。",
            "",
            "## 局限性",
            "1. 依赖 AlphaFold 结构和 P2Rank top1 pocket，结构误差或 pocket 选择误差都会影响最终分数。",
            "2. 只看单 pocket，无法利用多 pocket 的互补信息。",
            "3. 全库检索会放大数据覆盖问题，如果某些酶没有结构或 pocket，相关 pair 会被排除出打分集合。",
            "4. 真实标签来自已知 Rhea 注释，未标注正例并不等价于真正的负例。",
        ]
    )

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the Chinese terpene screening report.")
    args = parser.parse_args()
    write_markdown(REPORT_PATH, build_report())
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
