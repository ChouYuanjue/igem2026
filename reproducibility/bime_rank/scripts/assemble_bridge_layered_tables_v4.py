from __future__ import annotations

import json
import pandas as pd
from projects.active.bridge.model.assets import ROOT

R2E_BASE = ROOT / "results/bridge_layered_v4_r2e_cage"
E2R_BASE = ROOT / "results/bridge_layered_v4_e2r_cage"
R2E_EDGE = ROOT / "results/bridge_current_edgewise_r2e_v1/edge_metrics.csv.gz"
E2R_EDGE = ROOT / "results/bridge_current_edgewise_e2r_v1/edge_metrics.csv.gz"
R2E_ABL = ROOT / "results/bridge_r2e_four_group_ablation_v1/edge_metrics.csv.gz"
E2R_ABL = ROOT / "results/bridge_e2r_four_group_ablation_v1/edge_metrics.csv.gz"
GEN = ROOT / "reproducibility/bime_rank/records/BRIDGE_DIFFICULTY_BALANCED_MAIN_V1_RESULT.json"
DOC = ROOT / "projects/active/bridge/docs/evaluation_tables_v4.md"
RECORD = ROOT / "reproducibility/bime_rank/records/BRIDGE_LAYERED_MAIN_TABLES_V4_RESULT.json"
CAND_SUMMARY = ROOT / "results/bridge_layered_v4_candidates/summary.json"
SUPPORT_SUMMARY = ROOT / "results/bridge_layered_v4_cage_support/summary.json"
R2E_PREP = R2E_BASE / "prepare_summary.json"
E2R_PREP = E2R_BASE / "prepare_summary.json"
KEYS = ["protein_id", "reaction_id"]


def native_rows(direction: str, queries: set[str]) -> list[dict[str, object]]:
    edge_path = R2E_EDGE if direction == "r2e" else E2R_EDGE
    abl_path = R2E_ABL if direction == "r2e" else E2R_ABL
    qcol = "reaction_id" if direction == "r2e" else "protein_id"
    edge = pd.read_csv(edge_path, dtype=str).fillna("")
    edge = edge[edge[qcol].isin(queries)].copy()
    abl = pd.read_csv(abl_path, dtype=str).fillna("")
    abl = abl[abl[qcol].isin(queries)].copy()
    cols = [
        "minus_functional_rank",
        "minus_structure_mechanism_rank",
        "minus_relational_memory_rank",
        "minus_family_domain_rank",
    ]
    frame = edge.merge(abl[KEYS + cols], on=KEYS, validate="one_to_one")
    rank_cols = ["broad_rank", "full_rank", *cols]
    for col in rank_cols:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    grouped = frame.groupby(qcol).agg({col: "min" for col in rank_cols})
    missing = sorted(queries - set(grouped.index.astype(str)))
    if missing:
        raise RuntimeError(f"{direction} full-space ranks miss queries: {missing[:10]}")
    definitions = [
        ("Broad Retrieval", "broad_rank", "原生完整候选空间"),
        ("BRIDGE", "full_rank", "原生生产路径"),
        ("BRIDGE - Functional", "minus_functional_rank", "去功能证据"),
        ("BRIDGE - Structure/Mechanism", "minus_structure_mechanism_rank", "去结构与机制证据"),
        ("BRIDGE - Long-term Relation Context", "minus_relational_memory_rank", "去 clean2023 长期关系上下文"),
        ("BRIDGE - Family/Domain", "minus_family_domain_rank", "去家族与领域专家"),
    ]
    rows = []
    for name, col, note in definitions:
        ranks = grouped[col]
        rows.append({
            "model": name,
            "queries": int(len(ranks)),
            "mrr": float((1.0 / ranks).mean()),
            "hit3": float((ranks <= 3).mean()),
            "hit10": float((ranks <= 10).mean()),
            "hit100": float((ranks <= 100).mean()),
            "median_rank": float(ranks.median()),
            "note": note,
        })
    return rows


def gate_stats(frame: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    recall = pd.to_numeric(frame[f"{prefix}_gate_positive_recall"], errors="raise")
    positives = pd.to_numeric(frame["positive_count"], errors="raise")
    return {
        "queries": int(len(frame)),
        "query_hit": float((recall > 0).mean()),
        "macro_positive_recall": float(recall.mean()),
        "edge_recall": float((recall * positives).sum() / positives.sum()),
    }


def cage_rows(summary: dict) -> list[dict[str, object]]:
    return [
        {
            "model": "EnzymeCAGE",
            **summary["metrics"]["enzymecage"],
            "note": "原生/对称低预算门控后用同一 CAGE epoch-19 scorer；unsupported 计 miss",
        },
        {
            "model": "Broad Retrieval + CAGE Reranking",
            **summary["metrics"]["broad_cage"],
            "note": "Broad 仅为 CAGE 路径取相同 Nq，再用同一 CAGE epoch-19 scorer；unsupported 计 miss",
        },
    ]


def main() -> None:
    r2e_summary = json.loads((R2E_BASE / "summary.json").read_text())
    e2r_summary = json.loads((E2R_BASE / "summary.json").read_text())
    cand_summary = json.loads(CAND_SUMMARY.read_text())
    support_summary = json.loads(SUPPORT_SUMMARY.read_text())
    r2e_prep = json.loads(R2E_PREP.read_text())
    e2r_prep = json.loads(E2R_PREP.read_text())
    for direction in ("r2e", "e2r"):
        if cand_summary[direction]["native_candidate_rows"] != cand_summary[direction]["broad_candidate_rows"]:
            raise RuntimeError(f"{direction} matched-budget candidate rows diverged")
    if cand_summary["r2e"]["native_candidate_rows"] != r2e_prep["native_candidate_rows"]:
        raise RuntimeError("R2E prepare rows diverged from frozen candidate registry")
    if cand_summary["e2r"]["native_candidate_rows"] != e2r_prep["native_candidate_rows"]:
        raise RuntimeError("E2R prepare rows diverged from frozen candidate registry")
    r2e_q = pd.read_csv(R2E_BASE / "query_metrics.csv", dtype={"reaction_id": str})
    e2r_q = pd.read_csv(E2R_BASE / "query_metrics.csv", dtype={"protein_id": str})
    if len(r2e_q) != 1923:
        raise RuntimeError(f"R2E query count changed: {len(r2e_q)}")
    if len(e2r_q) != 17486:
        raise RuntimeError(f"E2R query count changed: {len(e2r_q)}")
    r2e_queries = set(r2e_q.reaction_id.astype(str))
    e2r_queries = set(e2r_q.protein_id.astype(str))
    result = {
        "schema": "bridge-layered-main-tables-v4",
        "status": "completed",
        "mother_benchmark": {
            "edges": 23773,
            "r2e_queries": 1923,
            "e2r_queries": 17486,
            "strict_relation_unseen_relative_to_clean2023": True,
        },
        "protocol": {
            "matched_budget_only_for_cage_paths": True,
            "known_relation_mask_precedes_budget": True,
            "broad_bridge_full_space_unchanged": True,
            "unsupported_cage_query_or_pair_is_miss": True,
            "cage_feature_cache_does_not_select_test_queries": True,
        },
        "audit": {
            "candidate_registry": cand_summary,
            "cage_support_registry": support_summary,
            "r2e_prepare": r2e_prep,
            "e2r_prepare": e2r_prep,
        },
        "r2e": {
            "gate": {
                "enzymecage": gate_stats(r2e_q, "cage"),
                "broad_equal_budget": gate_stats(r2e_q, "broad_cage"),
            },
            "cage_rows": cage_rows(r2e_summary),
            "native_rows": native_rows("r2e", r2e_queries),
            "support": r2e_summary.get("support", {}),
        },
        "e2r": {
            "gate": {
                "similar_enzyme_gate": gate_stats(e2r_q, "native"),
                "broad_equal_budget": gate_stats(e2r_q, "broad"),
            },
            "cage_rows": cage_rows(e2r_summary),
            "native_rows": native_rows("e2r", e2r_queries),
            "support": e2r_summary.get("support", {}),
        },
        "broad_generalization_reference": json.loads(GEN.read_text()),
    }
    RECORD.write_text(json.dumps(result, indent=2) + "\n")

    def pct(x: float) -> str:
        return f"{100*x:.2f}%"

    def f4(x: float) -> str:
        return f"{x:.4f}"

    lines = [
        "# BRIDGE 分层主实验 V4",
        "",
        "母集固定为 23,773 条相对 clean2023 严格 relation-unseen 关系。R2E 共 1,923 个反应 query，E2R 共 17,486 个蛋白 query。测试 query 不再按本地 CAGE 特征缓存筛选。",
        "",
        "只有 EnzymeCAGE 与 Broad→CAGE 两条路径共享逐 query 候选预算。Broad Retrieval、BRIDGE 与四组消融始终保留原生完整候选空间。CAGE 缺少蛋白结构/特征或反应资产时保留该 query/pair，并在最终排序指标中按 miss 处理。",
        "",
    ]
    for title, key in (("R2E：反应 → 酶", "r2e"), ("E2R：酶 → 反应", "e2r")):
        block = result[key]
        lines += [f"## {title}", "", "### 候选召回", "", "| 路径 | Query hit | Macro positive recall | Edge recall |", "|---|---:|---:|---:|"]
        for name, stats in block["gate"].items():
            lines.append(f"| {name} | {pct(stats['query_hit'])} | {pct(stats['macro_positive_recall'])} | {pct(stats['edge_recall'])} |")
        lines += ["", "### CAGE 同预算路径", "", "| 模型/路径 | Queries | MRR | Hit@3 | Hit@10 | Hit@100 | Macro positive recall |", "|---|---:|---:|---:|---:|---:|---:|"]
        for row in block["cage_rows"]:
            lines.append(f"| {row['model']} | {row['queries']} | {f4(row['mrr'])} | {pct(row['hit3'])} | {pct(row['hit10'])} | {pct(row['hit100'])} | {pct(row['macro_positive_recall'])} |")
        lines += ["", "### 原生完整空间", "", "| 模型 | Queries | MRR | Hit@3 | Hit@10 | Hit@100 | Median rank |", "|---|---:|---:|---:|---:|---:|---:|"]
        for row in block["native_rows"]:
            lines.append(f"| {row['model']} | {row['queries']} | {f4(row['mrr'])} | {pct(row['hit3'])} | {pct(row['hit10'])} | {pct(row['hit100'])} | {row['median_rank']:.1f} |")
        lines.append("")
    lines += [
        "## 解释边界",
        "",
        "- CAGE 同预算表回答候选生成与 CAGE scorer 组合后的端到端能力；召回失败和特征/结构缺失都不会从分母删除。",
        "- Broad Retrieval、BRIDGE 与消融表回答生产系统在完整候选宇宙中的 query-macro 排名能力，不共享 CAGE 的小候选池。",
        "- BRIDGE - Long-term Relation Context 对应 clean2023 长期关系上下文；运行时 Seed/Homology 情景证据不混入本零样本主表。",
        "- 作者 Orphan-335 / Enzyme-405 只作为补充外部基准，不替代本 23,773 条母集。",
        "",
    ]
    DOC.write_text("\n".join(lines) + "\n")
    print(json.dumps({"record": str(RECORD), "doc": str(DOC)}, indent=2))


if __name__ == "__main__":
    main()
