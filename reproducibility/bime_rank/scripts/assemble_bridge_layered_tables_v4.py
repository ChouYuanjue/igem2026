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
R2E_FALLBACK = ROOT / "results/bridge_layered_v4_cage_features/r2e_fallback"
CAGE_GATE_BRIDGE = ROOT / "results/bridge_layered_v4_cage_gate_bridge/summary.json"
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


def bridge_gate_row(summary: dict, direction: str) -> dict[str, object]:
    return {
        "model": "EnzymeCAGE Gate + BRIDGE Reranking",
        **summary[direction]["metrics"],
        "note": (
            "冻结 EnzymeCAGE gate 只提供候选集合；忽略 gate 原顺序，"
            "集合内由 Broad 建立基础序，再施加当前 BRIDGE 上层有限修正"
        ),
    }


def main_rows(cage: list[dict[str, object]], gate_bridge: dict[str, object], native: list[dict[str, object]]) -> list[dict[str, object]]:
    by_name = {str(row["model"]): row for row in native}
    return [
        cage[0],
        by_name["Broad Retrieval"],
        cage[1],
        gate_bridge,
        by_name["BRIDGE"],
        by_name["BRIDGE - Functional"],
        by_name["BRIDGE - Structure/Mechanism"],
        by_name["BRIDGE - Long-term Relation Context"],
        by_name["BRIDGE - Family/Domain"],
    ]


def main() -> None:
    r2e_summary = json.loads((R2E_BASE / "summary.json").read_text())
    e2r_summary = json.loads((E2R_BASE / "summary.json").read_text())
    gate_bridge_summary = json.loads(CAGE_GATE_BRIDGE.read_text())
    cand_summary = json.loads(CAND_SUMMARY.read_text())
    support_summary = json.loads(SUPPORT_SUMMARY.read_text())
    r2e_prep = json.loads(R2E_PREP.read_text())
    e2r_prep = json.loads(E2R_PREP.read_text())
    r2e_fallback_audit = {}
    for name in ("structure_summary.json", "p2rank_summary.json", "esm_seed_summary.json"):
        path = R2E_FALLBACK / name
        if path.exists():
            r2e_fallback_audit[name.removesuffix(".json")] = json.loads(path.read_text())
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
    r2e_cage = cage_rows(r2e_summary)
    e2r_cage = cage_rows(e2r_summary)
    r2e_native = native_rows("r2e", r2e_queries)
    e2r_native = native_rows("e2r", e2r_queries)
    r2e_gate_bridge = bridge_gate_row(gate_bridge_summary, "r2e")
    e2r_gate_bridge = bridge_gate_row(gate_bridge_summary, "e2r")

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
            "r2e_fallback": r2e_fallback_audit,
        },
        "r2e": {
            "gate": {
                "enzymecage": gate_stats(r2e_q, "cage"),
                "broad_equal_budget": gate_stats(r2e_q, "broad_cage"),
            },
            "cage_rows": r2e_cage,
            "cage_gate_bridge_row": r2e_gate_bridge,
            "native_rows": r2e_native,
            "main_rows": main_rows(r2e_cage, r2e_gate_bridge, r2e_native),
            "support": r2e_summary.get("support", {}),
        },
        "e2r": {
            "gate": {
                "similar_enzyme_gate": gate_stats(e2r_q, "native"),
                "broad_equal_budget": gate_stats(e2r_q, "broad"),
            },
            "cage_rows": e2r_cage,
            "cage_gate_bridge_row": e2r_gate_bridge,
            "native_rows": e2r_native,
            "main_rows": main_rows(e2r_cage, e2r_gate_bridge, e2r_native),
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
    structure_audit = r2e_fallback_audit.get("structure_summary", {})
    pocket_audit = r2e_fallback_audit.get("p2rank_summary", {})
    if structure_audit or pocket_audit:
        lines += [
            "### R2E CAGE fallback 支持覆盖",
            "",
            f"- fallback UID：{structure_audit.get('fallback_uids', 'NA')}；取得结构：{structure_audit.get('structures_ok', 'NA')}；结构不可得：{structure_audit.get('structures_failed', 'NA')}。",
            f"- P2Rank pocket：{pocket_audit.get('pocket_uids', 'NA')}；pocket 失败：{pocket_audit.get('pocket_failures', 'NA')}；最终进入 CAGE extra feature：{r2e_prep.get('fallback_extra_uids', 'NA')}。",
            f"- 最终 unsupported logical candidate rows：{r2e_prep.get('unsupported_logical_rows', 'NA')}，全部继续留在母集分母中按 miss 处理。",
            "",
        ]

    for title, key in (("R2E：反应 → 酶", "r2e"), ("E2R：酶 → 反应", "e2r")):
        block = result[key]
        lines += [
            f"## {title}",
            "",
            "| 方法 | MRR | Hit@3 | Hit@10 | Hit@100 |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in block["main_rows"]:
            lines.append(
                f"| {row['model']} | {f4(row['mrr'])} | {pct(row['hit3'])} | "
                f"{pct(row['hit10'])} | {pct(row['hit100'])} |"
            )
        lines += [
            "",
            "> EnzymeCAGE gate 相关组合沿用冻结的逐-query gate 候选集合；gate 本身不提供基础序。"
            " `EnzymeCAGE Gate + BRIDGE Reranking` 在该集合内先由 Broad 建立基础序，再应用 BRIDGE 上层有限修正。"
            " EnzymeCAGE 与 Broad Retrieval + CAGE Reranking 使用 CAGE scorer；Broad Retrieval、BRIDGE 与四组消融保持原生完整候选空间。",
            "",
        ]
    lines += [
        "## 解释边界",
        "",
        "- EnzymeCAGE gate 只定义候选集合，不携带可用于 BRIDGE 的基础顺序；新组合行始终从 Broad 基础分开始。",
        "- CAGE gate 相关组合的召回失败不会从分母删除；Broad Retrieval、BRIDGE 与消融仍在原生完整候选宇宙中评测。",
        "- BRIDGE - Long-term Relation Context 对应 clean2023 长期关系上下文；运行时 Seed/Homology 情景证据不混入本零样本主表。",
        "- 作者 Orphan-335 / Enzyme-405 只作为补充外部基准，不替代本 23,773 条母集。",
        "",
    ]
    DOC.write_text("\n".join(lines) + "\n")
    print(json.dumps({"record": str(RECORD), "doc": str(DOC)}, indent=2))


if __name__ == "__main__":
    main()
