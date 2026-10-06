from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.bridge.model.assets import ROOT
from reproducibility.bime_rank.scripts.analyze_bridge_difficulty_balanced_main_v1 import (
    annotate,
    summarize,
)

TRAIN = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
R2E_PROD = ROOT / "results/bridge_current_edgewise_r2e_v1/edge_metrics.csv.gz"
E2R_PROD = ROOT / "results/bridge_current_edgewise_e2r_v1/edge_metrics.csv.gz"
R2E_ABL = ROOT / "results/bridge_r2e_four_group_ablation_v1/edge_metrics.csv.gz"
E2R_ABL = ROOT / "results/bridge_e2r_four_group_ablation_v1/edge_metrics.csv.gz"
CAGE_GATE = ROOT / "results/bridge_relation_unseen_cage_gate_v1/edge_gate_recall.csv.gz"
CAGE_GATE_SUMMARY = ROOT / "results/bridge_relation_unseen_cage_gate_v1/summary.json"
BROAD_CAGE = ROOT / "results/bridge_relation_unseen_broad_cage_v1/edge_metrics.csv.gz"
BROAD_CAGE_SUMMARY = ROOT / "results/bridge_relation_unseen_broad_cage_v1/summary.json"
UNIFIED = ROOT / "reproducibility/bime_rank/records/FIBRE_UNIFIED_GENERAL_BENCHMARK_MATRIX_V1_RESULT.json"
ORPHAN = ROOT / "results/orphan335_fixed_pool_v1/summary.json"
ENZYME405_LAYERED = ROOT / "results/bime_rank_unified_v1/enzyme405_complete226_augmented_v1/summary.json"
OUT = ROOT / "results/bridge_layered_tables_v1"
RECORD = ROOT / "reproducibility/bime_rank/records/BRIDGE_LAYERED_MAIN_TABLES_V1_RESULT.json"
DOC = ROOT / "projects/active/bridge/docs/evaluation_tables_v1.md"

KEYS = ["protein_id", "reaction_id"]


def fmt(x: float) -> str:
    return f"{x:.4f}"


def pct(x: float) -> str:
    return f"{100*x:.2f}%"


def table_metrics(frame: pd.DataFrame, rank_col: str) -> dict:
    s = summarize(frame, rank_col)
    return {
        "mrr": float(s["difficulty_macro"]["mrr"]),
        "hit10": float(s["difficulty_macro"]["hit10"]),
        "hit100": float(s["difficulty_macro"]["hit100"]),
        "hit1000": float(s["difficulty_macro"]["hit1000"]),
        "micro_mrr": float(s["micro"]["mrr"]),
        "micro_hit10": float(s["micro"]["hit10"]),
    }


def optimistic_cage_rank(frame: pd.DataFrame) -> np.ndarray:
    # Strict upper bound after the published EnzymeCAGE candidate-retrieval gate:
    # if the positive relation enters the gate, grant rank 1; otherwise it cannot
    # be returned and is placed after the full filtered candidate universe.
    return np.where(
        frame["cage_gate_hit"].astype(bool).to_numpy(),
        1,
        frame["candidate_count_filtered"].astype(int).to_numpy() + 1,
    ).astype(np.int64)


def build_direction(direction: str, pdeg: dict, rdeg: dict) -> pd.DataFrame:
    prod_path = R2E_PROD if direction == "r2e" else E2R_PROD
    abl_path = R2E_ABL if direction == "r2e" else E2R_ABL
    prod = pd.read_csv(prod_path, dtype={"protein_id": str, "reaction_id": str})
    abl = pd.read_csv(abl_path, dtype={"protein_id": str, "reaction_id": str})
    keep = KEYS + [
        "minus_functional_rank",
        "minus_structure_mechanism_rank",
        "minus_relational_memory_rank",
        "minus_family_domain_rank",
    ]
    frame = prod.merge(abl[keep], on=KEYS, validate="one_to_one")
    if direction == "e2r":
        # No family/TPS member exists on the current E2R production path.
        # This ablation is method-identical to Full, so reuse the exact
        # production ranks instead of exposing independent-GPU tie noise.
        frame["minus_family_domain_rank"] = frame["full_rank"].astype(int)
    frame = annotate(frame, pdeg, rdeg)

    gate = pd.read_csv(CAGE_GATE, dtype={"protein_id": str, "reaction_id": str}).fillna("")
    gate["cage_gate_hit"] = gate["cage_gate_hit"].astype(str).str.lower().eq("true")
    frame = frame.merge(gate[KEYS + ["cage_gate_hit"]], on=KEYS, validate="one_to_one")
    frame["enzymecage_gate_upper_rank"] = optimistic_cage_rank(frame)
    if direction == "r2e" and BROAD_CAGE.exists():
        bc = pd.read_csv(BROAD_CAGE, dtype={"protein_id": str, "reaction_id": str})
        frame = frame.merge(bc[KEYS + ["broad_cage_rank"]], on=KEYS, validate="one_to_one")
    return frame


def row(name: str, metrics: dict | None, note: str = "") -> dict:
    return {"model": name, "metrics": metrics, "note": note}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN, dtype=str).fillna("").drop_duplicates(KEYS)
    pdeg = train.groupby("protein_id").size().astype(int).to_dict()
    rdeg = train.groupby("reaction_id").size().astype(int).to_dict()

    r2e = build_direction("r2e", pdeg, rdeg)
    e2r = build_direction("e2r", pdeg, rdeg)
    if len(r2e) != 23773 or len(e2r) != 23773:
        raise RuntimeError("main mother-set drift")

    rank_cols = [
        ("Broad Retrieval", "broad_rank", ""),
        (
            "Broad Retrieval + CAGE Reranking",
            "broad_cage_rank",
            "由 Broad 负责候选召回，再用 EnzymeCAGE 神经分数重排已有完整 CAGE 证据的候选槽位；未物化证据保持 Broad 原位，不作负证据",
        ),
        ("BRIDGE", "full_rank", "当前冻结 FinalBridgeRuntime"),
        ("BRIDGE - Functional", "minus_functional_rank", "去除 EnzGFM 功能证据"),
        (
            "BRIDGE - Structure/Mechanism",
            "minus_structure_mechanism_rank",
            "去除结构与机制证据",
        ),
        (
            "BRIDGE - Relational Memory",
            "minus_relational_memory_rank",
            "去除 clean2023 训练图长期关系记忆；本测试不提供运行时情景记忆",
        ),
        (
            "BRIDGE - Family/Domain",
            "minus_family_domain_rank",
            "R2E 去除 family-CAGE 与 TPS；当前 E2R 没有对应专项成员",
        ),
    ]

    main_r2e = [
        row(
            "EnzymeCAGE（候选门上界）",
            table_metrics(r2e, "enzymecage_gate_upper_rank"),
            "作者 Top-10 相似反应候选门之后的最乐观上界；1,917/1,923 个主测试反应可执行，神经重排真实结果只能低于或等于此上界",
        )
    ]
    for name, col, note in rank_cols:
        if col in r2e:
            main_r2e.append(row(name, table_metrics(r2e, col), note))

    main_e2r = [
        row(
            "EnzymeCAGE",
            None,
            "N/A：EnzymeCAGE 没有与当前 11,081 反应全空间同口径的原生酶到反应全局检索入口",
        ),
        row(
            "Broad Retrieval + CAGE Reranking",
            None,
            "N/A：当前没有与 11,081 反应全空间同口径的 CAGE 神经重排；固定 pair reservoir 的反向统计不视为全局 E2R 检索",
        ),
    ]
    for name, col, note in rank_cols:
        if name == "Broad Retrieval + CAGE Reranking":
            continue
        main_e2r.append(row(name, table_metrics(e2r, col), note))

    unified = json.loads(UNIFIED.read_text())
    native_rows = [
        x for x in unified["rows"]
        if x.get("benchmark") == "Orphan-335"
        and x.get("task") == "reaction_to_protein"
    ]
    orphan = json.loads(ORPHAN.read_text())
    e405_layered = json.loads(ENZYME405_LAYERED.read_text())
    e405_rows = [
        x for x in unified["rows"]
        if x.get("benchmark") == "Enzyme-405 complete226"
        and x.get("task") == "reaction_to_protein"
        and x.get("model") in ("FIBRE Broad Core", "EnzymeCAGE generic pretrain")
    ]
    e405_by_model = {x["model"]: x for x in e405_rows}

    native = {
        "enzyme405_complete226": {
            "queries": 226,
            "rows": [
                {
                    "model": "EnzymeCAGE generic pretrain",
                    "mrr": e405_by_model["EnzymeCAGE generic pretrain"]["mrr"],
                    "hit10": e405_by_model["EnzymeCAGE generic pretrain"]["hit_at_10"],
                    "hit20": e405_by_model["EnzymeCAGE generic pretrain"]["hit_at_20"],
                    "note": "同一 immutable complete226 candidate pool；固定官方 seeds 40–44 均值",
                },
                {
                    "model": "Broad Retrieval",
                    "mrr": e405_by_model["FIBRE Broad Core"]["mrr"],
                    "hit10": e405_by_model["FIBRE Broad Core"]["hit_at_10"],
                    "hit20": e405_by_model["FIBRE Broad Core"]["hit_at_20"],
                    "note": "同一 complete226 candidate pool 上的冻结 Broad Core",
                },
                {
                    "model": "Layered mainline",
                    "mrr": e405_layered["metrics"]["reaction_to_enzyme"]["mrr"],
                    "hit10": e405_layered["metrics"]["reaction_to_enzyme"]["hit_at_10"],
                    "hit20": e405_layered["metrics"]["reaction_to_enzyme"]["hit_at_20"],
                    "note": "仓库已有 complete226 augmented-support 分层结果；没有针对 benchmark 重调参数",
                },
            ],
        },
        "orphan335": {
            "queries": 335,
            "rows": [
                {
                    "model": "Selenzyme (author retrieval stage)",
                    "mrr": orphan["selenzyme_metrics"]["reaction_to_enzyme"]["mrr"],
                    "hit10": orphan["selenzyme_metrics"]["reaction_to_enzyme"]["hit_at_10"],
                    "hit20": orphan["selenzyme_metrics"]["reaction_to_enzyme"]["hit_at_20"],
                    "note": "作者固定候选池上的原始检索分数",
                },
                {
                    "model": "FIBRE Broad Core",
                    "mrr": next(
                        x["mrr"] for x in native_rows
                        if x["benchmark"] == "Orphan-335" and x["model"] == "FIBRE Broad Core"
                    ),
                    "hit10": next(
                        x["hit_at_10"] for x in native_rows
                        if x["benchmark"] == "Orphan-335" and x["model"] == "FIBRE Broad Core"
                    ),
                    "hit20": next(
                        x["hit_at_20"] for x in native_rows
                        if x["benchmark"] == "Orphan-335" and x["model"] == "FIBRE Broad Core"
                    ),
                    "note": "作者固定候选池上的冻结 Broad Core",
                },
                {
                    "model": "Catalyst layered",
                    "mrr": orphan["catalyst_metrics"]["reaction_to_enzyme"]["mrr"],
                    "hit10": orphan["catalyst_metrics"]["reaction_to_enzyme"]["hit_at_10"],
                    "hit20": orphan["catalyst_metrics"]["reaction_to_enzyme"]["hit_at_20"],
                    "note": "作者固定候选池上的已有冻结分层模型",
                },
            ],
            "enzymecage_neural_note": (
                "The repository does not contain a complete Orphan-335 generic-pretrain EnzymeCAGE "
                "neural score file over the full 44,889-UID author pool; Selenzyme is kept under its "
                "correct name and is not relabeled as EnzymeCAGE."
            ),
        },
    }

    cage_gate_summary = json.loads(CAGE_GATE_SUMMARY.read_text())
    broad_cage_summary = (
        json.loads(BROAD_CAGE_SUMMARY.read_text())
        if BROAD_CAGE_SUMMARY.exists()
        else None
    )

    result = {
        "schema": "bridge-layered-main-tables-v1",
        "status": "completed",
        "protocol": {
            "main_edges": 23773,
            "r2e_queries": int(r2e.reaction_id.nunique()),
            "e2r_queries": int(e2r.protein_id.nunique()),
            "difficulty_macro": "equal novelty-group weight with within-group train-degree strata equal weight",
            "all_edges_retained": True,
            "r2e_candidate_universe": 185918,
            "e2r_candidate_universe": 11081,
            "episodic_runtime_support": "none",
            "cage_capability_boundary": (
                "CAGE candidate reachability is evaluated with the published Top-10 similar-reaction "
                "retrieval over the 2023 association graph. Missing local structural feature caches are "
                "not treated as method incapability."
            ),
        },
        "cage_audit": {
            "candidate_gate": cage_gate_summary,
            "broad_plus_cage": broad_cage_summary,
            "separate_compatible_subset_reported": False,
        },
        "main_tables": {"r2e": main_r2e, "e2r": main_e2r},
        "native_benchmarks": native,
        "reproduction": {
            "r2e_four_group_ablation": "reproducibility/bime_rank/scripts/evaluate_bridge_r2e_four_group_ablation_v1.py",
            "e2r_four_group_ablation": "reproducibility/bime_rank/scripts/evaluate_bridge_e2r_four_group_ablation_v1.py",
            "cage_candidate_gate": "reproducibility/bime_rank/scripts/evaluate_bridge_relation_unseen_cage_gate_v1.py",
            "broad_retrieval_plus_cage_prepare": "reproducibility/bime_rank/scripts/prepare_bridge_relation_unseen_broad_cage_v1.py",
            "broad_retrieval_plus_cage_evaluate": "reproducibility/bime_rank/scripts/evaluate_bridge_relation_unseen_broad_cage_v1.py",
            "table_assembler": "reproducibility/bime_rank/scripts/assemble_bridge_layered_tables_v1.py"
        },
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    RECORD.write_text(json.dumps(result, indent=2) + "\n")

    def md_table(rows: list[dict]) -> list[str]:
        out = [
            "| 模型 | MRR | Hit@10 | Hit@100 | Hit@1000 | 说明 |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for rec in rows:
            m = rec["metrics"]
            if m is None:
                vals = ["—", "—", "—", "—"]
            else:
                vals = [fmt(m["mrr"]), pct(m["hit10"]), pct(m["hit100"]), pct(m["hit1000"])]
            out.append(
                f"| {rec['model']} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {rec['note']} |"
            )
        return out

    lines = [
        "# BRIDGE 分层主评测表",
        "",
        "主表使用全部 23,773 条相对 clean2023 严格未见关系。难度宏平均对实体新旧类别等权，并在类别内部对训练图度数分层等权；不删除难样本。",
        "",
        "EnzymeCAGE 作者原始候选检索可处理 1,917/1,923 个 R2E query，因此不另设“CAGE-compatible”公平子集。其候选门只召回 3.43% 的 held-out relation edge，这是检索能力结果，不是本地特征缓存覆盖率。",
        "",
        "## 主表：反应到酶",
        "",
        *md_table(main_r2e),
        "",
        "## 主表：酶到反应",
        "",
        *md_table(main_e2r),
        "",
        "## EnzymeCAGE 作者原生基准",
        "",
        "### Enzyme-405（226-query aligned comparison）",
        "",
        "| 模型 | MRR | Hit@10 | Hit@20 |",
        "|---|---:|---:|---:|",
    ]
    for x in native["enzyme405_complete226"]["rows"]:
        lines.append(f"| {x['model']} | {fmt(x['mrr'])} | {pct(x['hit10'])} | {pct(x['hit20'])} |")
    lines += [
        "",
        "### Orphan-335",
        "",
        "| 模型 | MRR | Hit@10 | Hit@20 |",
        "|---|---:|---:|---:|",
    ]
    for x in native["orphan335"]["rows"]:
        lines.append(f"| {x['model']} | {fmt(x['mrr'])} | {pct(x['hit10'])} | {pct(x['hit20'])} |")
    lines += [
        "",
        "> Orphan-335 当前仓库未保存覆盖完整作者候选池的 EnzymeCAGE generic-pretrain 神经分数，因此保留作者检索阶段 Selenzyme 的真实名称，不作替换。",
        "",
        "## 证据组定义",
        "",
        "- 功能证据：EnzGFM 功能兼容性。",
        "- 结构与机制证据：CLIPZyme、反应中心/机制以及 R2E pocket 局部重排。",
        "- 关系记忆证据：clean2023 训练图长期记忆；本主实验不提供运行时情景记忆。",
        "- 家族与领域专家：家族 CAGE 与 TPS；当前 E2R 生产路径没有对应专项成员。",
        "",
    ]
    DOC.write_text("\n".join(lines))
    print(json.dumps({
        "record": str(RECORD.relative_to(ROOT)),
        "doc": str(DOC.relative_to(ROOT)),
        "main_r2e_rows": len(main_r2e),
        "main_e2r_rows": len(main_e2r),
    }, indent=2))


if __name__ == "__main__":
    main()
