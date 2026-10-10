from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from projects.active.bridge.model.assets import ROOT

R2E_RETRIEVAL = ROOT / "results/bridge_relation_unseen_r2e_retrieval_v1/summary.json"
E2R_RETRIEVAL = ROOT / "results/bridge_relation_unseen_e2r_similarity_gate_v1/summary.json"
R2E_CAGE = ROOT / "results/bridge_relation_unseen_r2e_cage_shared_support_v1/summary.json"
R2E_Q = ROOT / "results/bridge_relation_unseen_r2e_cage_shared_support_v1/query_metrics.csv"
E2R_Q = ROOT / "results/bridge_relation_unseen_e2r_similarity_cage_v1/query_metrics_main.csv"
R2E_EDGE = ROOT / "results/bridge_current_edgewise_r2e_v1/edge_metrics.csv.gz"
E2R_EDGE = ROOT / "results/bridge_current_edgewise_e2r_v1/edge_metrics.csv.gz"
R2E_ABL = ROOT / "results/bridge_r2e_four_group_ablation_v1/edge_metrics.csv.gz"
E2R_ABL = ROOT / "results/bridge_e2r_four_group_ablation_v1/edge_metrics.csv.gz"
GEN_V2 = ROOT / "reproducibility/bime_rank/records/BRIDGE_DIFFICULTY_BALANCED_MAIN_V1_RESULT.json"

DOC = ROOT / "projects/active/bridge/docs/evaluation_tables_v3.md"
RECORD = ROOT / "reproducibility/bime_rank/records/BRIDGE_LAYERED_MAIN_TABLES_V3_RESULT.json"

KEYS = ["protein_id", "reaction_id"]


def native_rows(direction: str, queries: set[str]) -> list[dict]:
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
    frame = edge.merge(
        abl[KEYS + cols],
        on=KEYS,
        validate="one_to_one",
    )
    rank_cols = [
        "broad_rank",
        "full_rank",
        *cols,
    ]
    for col in rank_cols:
        frame[col] = pd.to_numeric(frame[col])

    grouped = frame.groupby(qcol).agg({c: "min" for c in rank_cols})
    definitions = [
        ("Broad Retrieval", "broad_rank", "原生完整候选空间"),
        ("BRIDGE", "full_rank", "原生生产路径"),
        ("BRIDGE - Functional", "minus_functional_rank", "去功能证据"),
        (
            "BRIDGE - Structure/Mechanism",
            "minus_structure_mechanism_rank",
            "去结构/机制证据",
        ),
        (
            "BRIDGE - Relational Memory",
            "minus_relational_memory_rank",
            "去长期关系记忆",
        ),
        (
            "BRIDGE - Family/Domain",
            "minus_family_domain_rank",
            "去家族/领域专家",
        ),
    ]
    rows = []
    for name, col, note in definitions:
        r = grouped[col]
        rows.append(
            {
                "model": name,
                "queries": int(len(r)),
                "mrr": float((1.0 / r).mean()),
                "hit3": float((r <= 3).mean()),
                "hit10": float((r <= 10).mean()),
                "hit100": float((r <= 100).mean()),
                "median_rank": float(r.median()),
                "note": note,
            }
        )
    return rows


def main() -> None:
    r2e_ret = json.loads(R2E_RETRIEVAL.read_text())
    e2r_ret = json.loads(E2R_RETRIEVAL.read_text())
    r2e_cage = json.loads(R2E_CAGE.read_text())

    r2e_q = pd.read_csv(R2E_Q, dtype={"reaction_id": str})
    e2r_q = pd.read_csv(E2R_Q, dtype={"protein_id": str})

    r2e_queries = set(r2e_q.reaction_id.astype(str))
    e2r_queries = set(e2r_q.protein_id.astype(str))

    r2e_native = native_rows("r2e", r2e_queries)
    e2r_native = native_rows("e2r", e2r_queries)

    e2r_cage_rows = []
    for name, prefix, note in [
        (
            "EnzymeCAGE",
            "cage",
            "Top-10相似训练酶→已知反应的小候选池，再用CAGE重排",
        ),
        (
            "Broad Retrieval + CAGE Reranking",
            "broad_cage",
            "Broad仅为CAGE路径取相同Nq候选，再用同一CAGE重排",
        ),
    ]:
        e2r_cage_rows.append(
            {
                "model": name,
                "queries": int(len(e2r_q)),
                "mrr": float(pd.to_numeric(e2r_q[f"{prefix}_rr"]).mean()),
                "hit3": float(pd.to_numeric(e2r_q[f"{prefix}_h3"]).mean()),
                "hit10": float(pd.to_numeric(e2r_q[f"{prefix}_h10"]).mean()),
                "positive_recall": float(
                    pd.to_numeric(e2r_q[f"{prefix}_recall"]).mean()
                ),
                "note": note,
            }
        )

    r2e_cage_rows = [
        {
            "model": "EnzymeCAGE",
            **r2e_cage["metrics"]["enzymecage"],
            "note": "CAGE原生门控；神经重排只在CAGE可评分严格子集上统计",
        },
        {
            "model": "Broad Retrieval + CAGE Reranking",
            **r2e_cage["metrics"]["broad_cage"],
            "note": "Broad为CAGE路径取同预算候选，再由CAGE重排；仅统计CAGE可评分严格子集",
        },
    ]

    result = {
        "schema": "bridge-layered-main-tables-v3",
        "status": "completed",
        "mother_benchmark": {
            "edges": 23773,
            "strict_relation_unseen_relative_to_clean2023": True,
        },
        "retrieval": {
            "r2e": {
                "queries": r2e_ret["protocol"]["queries"],
                "candidate_budget": r2e_ret["candidate_budget"],
                "enzymecage_native_gate": r2e_ret["enzymecage"],
                "broad_equal_budget": r2e_ret["broad_equal_budget"],
            },
            "e2r": {
                "queries": e2r_ret["protocol"]["queries"],
                "candidate_budget": e2r_ret["candidate_budget"],
                "similar_enzyme_gate": {
                    "query_hit": e2r_ret["query_hit"],
                    "macro_positive_recall": e2r_ret["macro_positive_recall"],
                    "edge_recall": e2r_ret["edge_recall"],
                },
            },
        },
        "ranking_tables": {
            "r2e": {
                "cage_scoreable_queries": int(len(r2e_q)),
                "cage_rows": r2e_cage_rows,
                "native_rows": r2e_native,
                "protocol": (
                    "Only EnzymeCAGE and Broad→CAGE use matched candidate budgets. "
                    "Broad Retrieval, BRIDGE and all ablations keep their native "
                    "185,918-protein full-space path."
                ),
            },
            "e2r": {
                "cage_scoreable_queries": int(len(e2r_q)),
                "cage_rows": e2r_cage_rows,
                "native_rows": e2r_native,
                "protocol": (
                    "Only EnzymeCAGE and Broad→CAGE use matched small candidate "
                    "budgets. Broad Retrieval, BRIDGE and all ablations keep their "
                    "native 11,081-reaction full-space path."
                ),
            },
        },
        "broad_generalization_reference": json.loads(GEN_V2.read_text()),
    }
    RECORD.write_text(json.dumps(result, indent=2) + "\n")

    def pct(x: float) -> str:
        return f"{100*x:.2f}%"

    def f4(x: float) -> str:
        return f"{x:.4f}"

    lines = [
        "# BRIDGE 最终分层主实验 V3",
        "",
        "母集始终为 23,773 条相对 clean2023 严格 relation-unseen 关系。只有涉及 EnzymeCAGE 的两条路径共享候选预算；Broad Retrieval、BRIDGE 与四组消融始终保留各自原生完整候选空间。",
        "",
        "## R2E：召回层（全 1,923 query）",
        "",
        "| 召回器 | Query hit | Macro positive recall | Edge recall |",
        "|---|---:|---:|---:|",
        f"| EnzymeCAGE 原生门控 | {pct(r2e_ret['enzymecage']['query_hit'])} | {pct(r2e_ret['enzymecage']['macro_positive_recall'])} | {pct(r2e_ret['enzymecage']['edge_recall'])} |",
        f"| Broad 同预算召回 | {pct(r2e_ret['broad_equal_budget']['query_hit'])} | {pct(r2e_ret['broad_equal_budget']['macro_positive_recall'])} | {pct(r2e_ret['broad_equal_budget']['edge_recall'])} |",
        "",
        "## R2E：排序层（当前已完成 CAGE 神经评分的 212 个严格 query）",
        "",
        "| 模型/路径 | 候选路径 | MRR | Hit@3 | Hit@10 | Hit@100/正例召回 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in r2e_cage_rows:
        lines.append(
            f"| {row['model']} | CAGE 同预算路径 | {f4(row['mrr'])} | "
            f"{pct(row['hit3'])} | {pct(row['hit10'])} | "
            f"{pct(row['macro_positive_recall'])} 正例召回 |"
        )
    for row in r2e_native:
        lines.append(
            f"| {row['model']} | {row['note']} | {f4(row['mrr'])} | "
            f"{pct(row['hit3'])} | {pct(row['hit10'])} | {pct(row['hit100'])} |"
        )

    lines += [
        "",
        "## E2R：召回层（全 17,486 query）",
        "",
        "E2R 的低预算门控采用与 CAGE R2E 对称的规则：Top-10 ESM-C 相似 clean2023 酶 → 汇总其已知反应。候选池中位数 6，均值约 6.86。",
        "",
        "| 召回器 | Query hit | Macro positive recall | Edge recall |",
        "|---|---:|---:|---:|",
        f"| 相似酶门控 | {pct(e2r_ret['query_hit'])} | {pct(e2r_ret['macro_positive_recall'])} | {pct(e2r_ret['edge_recall'])} |",
        "",
        "## E2R：排序层（当前已完成 CAGE 神经评分的 284 个严格 query）",
        "",
        "| 模型/路径 | 候选路径 | MRR | Hit@3 | Hit@10 | Hit@100/正例召回 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in e2r_cage_rows:
        lines.append(
            f"| {row['model']} | CAGE 同预算路径 | {f4(row['mrr'])} | "
            f"{pct(row['hit3'])} | {pct(row['hit10'])} | "
            f"{pct(row['positive_recall'])} 正例召回 |"
        )
    for row in e2r_native:
        lines.append(
            f"| {row['model']} | {row['note']} | {f4(row['mrr'])} | "
            f"{pct(row['hit3'])} | {pct(row['hit10'])} | {pct(row['hit100'])} |"
        )

    lines += [
        "",
        "## 口径",
        "",
        "- EnzymeCAGE 与 Broad→CAGE：只为比较 CAGE 门控/重排而共享逐 query 候选预算。",
        "- Broad Retrieval：始终使用完整候选空间的 Broad 原生排名。",
        "- BRIDGE 与四组消融：始终使用完整候选空间的生产路径，不因 CAGE 的候选预算缩小。",
        "- 212 / 284 只是当前本地已经完成完整 CAGE 特征物化与神经评分的严格 query 交集，不代表 EnzymeCAGE 的模型支持范围。作者完整 dataset 含更广的 AlphaFill/pocket 资产，主实验将继续补齐所需特征。",
        "- R2E 召回层已覆盖全部 1,923 query；E2R 低预算门控已覆盖全部 17,486 query。",
        "",
    ]
    DOC.write_text("\n".join(lines) + "\n")
    print(json.dumps({"record": str(RECORD), "doc": str(DOC)}, indent=2))


if __name__ == "__main__":
    main()
