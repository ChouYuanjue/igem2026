from __future__ import annotations

import json

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

R2E_CONTROL = ROOT / "results/bridge_current_r2e_cage_shared_pool_v1/summary.json"
R2E_CAGE_NATIVE = ROOT / "results/fibre_vs_enzymecage_shared_pool_v1/summary.json"
E2R_CONTROL = ROOT / "results/bridge_cage_e2r_fixed465_v1/summary.json"

UNIFIED = ROOT / "reproducibility/bime_rank/records/FIBRE_UNIFIED_GENERAL_BENCHMARK_MATRIX_V1_RESULT.json"
ORPHAN = ROOT / "results/orphan335_fixed_pool_v1/summary.json"
ENZYME405_LAYERED = ROOT / "results/bime_rank_unified_v1/enzyme405_complete226_augmented_v1/summary.json"

OUT = ROOT / "results/bridge_layered_tables_v2"
RECORD = ROOT / "reproducibility/bime_rank/records/BRIDGE_LAYERED_MAIN_TABLES_V2_RESULT.json"
DOC = ROOT / "projects/active/bridge/docs/evaluation_tables_v2.md"
KEYS = ["protein_id", "reaction_id"]


def fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100*x:.2f}%"


def generalization_metrics(frame: pd.DataFrame, rank_col: str) -> dict:
    value = summarize(frame, rank_col)
    return {
        "mrr": float(value["difficulty_macro"]["mrr"]),
        "hit10": float(value["difficulty_macro"]["hit10"]),
        "hit100": float(value["difficulty_macro"]["hit100"]),
        "hit1000": float(value["difficulty_macro"]["hit1000"]),
        "micro_mrr": float(value["micro"]["mrr"]),
        "micro_hit10": float(value["micro"]["hit10"]),
    }


def build_generalization(
    direction: str,
    pdeg: dict[str, int],
    rdeg: dict[str, int],
) -> pd.DataFrame:
    prod_path = R2E_PROD if direction == "r2e" else E2R_PROD
    abl_path = R2E_ABL if direction == "r2e" else E2R_ABL
    prod = pd.read_csv(
        prod_path,
        dtype={"protein_id": str, "reaction_id": str},
    )
    abl = pd.read_csv(
        abl_path,
        dtype={"protein_id": str, "reaction_id": str},
    )
    frame = prod.merge(
        abl[
            KEYS
            + [
                "minus_functional_rank",
                "minus_structure_mechanism_rank",
                "minus_relational_memory_rank",
                "minus_family_domain_rank",
            ]
        ],
        on=KEYS,
        validate="one_to_one",
    )
    if direction == "e2r":
        frame["minus_family_domain_rank"] = frame["full_rank"].astype(int)
    return annotate(frame, pdeg, rdeg)


def control_row(
    model: str,
    metrics: dict,
    *,
    coverage_key: str,
    note: str,
) -> dict:
    return {
        "model": model,
        "mrr": metrics.get("mrr"),
        "map": metrics.get("map"),
        "hit3": metrics.get("hit_at_3", metrics.get("hit3")),
        "hit10": metrics.get("hit_at_10", metrics.get("hit10")),
        "hit20": metrics.get("hit_at_20", metrics.get("hit20")),
        "coverage": metrics.get(coverage_key),
        "note": note,
    }


def generalization_rows(frame: pd.DataFrame, direction: str) -> list[dict]:
    definitions = [
        ("Broad Retrieval", "broad_rank", "冻结 Broad 全候选排序"),
        ("BRIDGE", "full_rank", "当前冻结 FinalBridgeRuntime"),
        (
            "BRIDGE - Functional",
            "minus_functional_rank",
            "去除 EnzGFM 功能证据",
        ),
        (
            "BRIDGE - Structure/Mechanism",
            "minus_structure_mechanism_rank",
            "去除结构/机制证据；R2E 同时去除 pocket 重排",
        ),
        (
            "BRIDGE - Relational Memory",
            "minus_relational_memory_rank",
            "去除 clean2023 长期关系记忆；不含运行时情景记忆",
        ),
        (
            "BRIDGE - Family/Domain",
            "minus_family_domain_rank",
            (
                "去除 family-CAGE 与 TPS"
                if direction == "r2e"
                else "当前 E2R 没有 family/TPS 成员，因此与完整 BRIDGE 相同"
            ),
        ),
    ]
    return [
        {
            "model": model,
            "metrics": generalization_metrics(frame, col),
            "note": note,
        }
        for model, col, note in definitions
    ]


def native_benchmarks() -> dict:
    unified = json.loads(UNIFIED.read_text())
    orphan = json.loads(ORPHAN.read_text())
    e405_layered = json.loads(ENZYME405_LAYERED.read_text())

    e405_rows = [
        row
        for row in unified["rows"]
        if row.get("benchmark") == "Enzyme-405 complete226"
        and row.get("task") == "reaction_to_protein"
        and row.get("model")
        in ("FIBRE Broad Core", "EnzymeCAGE generic pretrain")
    ]
    e405 = {row["model"]: row for row in e405_rows}

    orphan_rows = [
        row
        for row in unified["rows"]
        if row.get("benchmark") == "Orphan-335"
        and row.get("task") == "reaction_to_protein"
    ]
    orphan_broad = next(
        row for row in orphan_rows if row["model"] == "FIBRE Broad Core"
    )

    return {
        "enzyme405_complete226": {
            "queries": 226,
            "rows": [
                {
                    "model": "EnzymeCAGE generic pretrain",
                    "mrr": e405["EnzymeCAGE generic pretrain"]["mrr"],
                    "hit10": e405["EnzymeCAGE generic pretrain"]["hit_at_10"],
                    "hit20": e405["EnzymeCAGE generic pretrain"]["hit_at_20"],
                },
                {
                    "model": "Broad Retrieval",
                    "mrr": e405["FIBRE Broad Core"]["mrr"],
                    "hit10": e405["FIBRE Broad Core"]["hit_at_10"],
                    "hit20": e405["FIBRE Broad Core"]["hit_at_20"],
                },
                {
                    "model": "Layered mainline",
                    "mrr": e405_layered["metrics"]["reaction_to_enzyme"]["mrr"],
                    "hit10": e405_layered["metrics"]["reaction_to_enzyme"]["hit_at_10"],
                    "hit20": e405_layered["metrics"]["reaction_to_enzyme"]["hit_at_20"],
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
                },
                {
                    "model": "Broad Retrieval",
                    "mrr": orphan_broad["mrr"],
                    "hit10": orphan_broad["hit_at_10"],
                    "hit20": orphan_broad["hit_at_20"],
                },
                {
                    "model": "Catalyst layered",
                    "mrr": orphan["catalyst_metrics"]["reaction_to_enzyme"]["mrr"],
                    "hit10": orphan["catalyst_metrics"]["reaction_to_enzyme"]["hit_at_10"],
                    "hit20": orphan["catalyst_metrics"]["reaction_to_enzyme"]["hit_at_20"],
                },
            ],
            "enzymecage_neural_note": (
                "仓库没有覆盖完整 Orphan-335 作者候选池的 EnzymeCAGE "
                "generic-pretrain 神经分数；Selenzyme 保持真实名称，不替换为 EnzymeCAGE。"
            ),
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = (
        pd.read_csv(TRAIN, dtype=str)
        .fillna("")
        .drop_duplicates(KEYS)
    )
    pdeg = train.groupby("protein_id").size().astype(int).to_dict()
    rdeg = train.groupby("reaction_id").size().astype(int).to_dict()

    r2e_gen = build_generalization("r2e", pdeg, rdeg)
    e2r_gen = build_generalization("e2r", pdeg, rdeg)
    if len(r2e_gen) != 23773 or len(e2r_gen) != 23773:
        raise RuntimeError("generalization mother-set drift")

    r2e_control = json.loads(R2E_CONTROL.read_text())
    r2e_native = json.loads(R2E_CAGE_NATIVE.read_text())
    e2r_control = json.loads(E2R_CONTROL.read_text())

    r2e_native_metrics = r2e_native["operational_common459"][
        "cage_original_gate_and_ranking"
    ]
    r2e_rows = [
        control_row(
            "EnzymeCAGE",
            r2e_native_metrics,
            coverage_key="query_positive_coverage",
            note="作者原生相似反应候选门 + generic EnzymeCAGE 排序",
        ),
        control_row(
            "Broad Retrieval",
            r2e_control["broad"],
            coverage_key="query_positive_coverage",
            note="Broad 召回；每个 query 的 K 等于 CAGE 原生候选预算",
        ),
        control_row(
            "Broad Retrieval + CAGE Reranking",
            r2e_control["broad_plus_cage"],
            coverage_key="query_positive_coverage",
            note="Broad 同预算召回后，仅用 EnzymeCAGE pred_logit 重排同一批候选",
        ),
        control_row(
            "BRIDGE",
            r2e_control["bridge"],
            coverage_key="query_positive_coverage",
            note="与 Broad→CAGE 完全相同的 Broad 候选池，由当前 BRIDGE 重排",
        ),
    ]
    for key, label, note in (
        ("minus_functional", "BRIDGE - Functional", "去功能证据"),
        (
            "minus_structure_mechanism",
            "BRIDGE - Structure/Mechanism",
            "去结构/机制与 pocket 证据",
        ),
        (
            "minus_relational_memory",
            "BRIDGE - Relational Memory",
            "去长期关系记忆",
        ),
        (
            "minus_family_domain",
            "BRIDGE - Family/Domain",
            "去 family-CAGE 与 TPS 专家",
        ),
    ):
        r2e_rows.append(
            control_row(
                label,
                r2e_control[key],
                coverage_key="query_positive_coverage",
                note=note,
            )
        )

    e2r_metrics = e2r_control["metrics"]
    e2r_rows = [
        control_row(
            "EnzymeCAGE",
            e2r_metrics["cage"],
            coverage_key="macro_positive_recall",
            note="在固定 465 反应域内由 CAGE 选/排 Top155",
        ),
        control_row(
            "Broad Retrieval",
            e2r_metrics["broad"],
            coverage_key="macro_positive_recall",
            note="在同一 465 反应域内由 Broad 选/排 Top155",
        ),
        control_row(
            "Broad Retrieval + CAGE Reranking",
            e2r_metrics["broad_cage"],
            coverage_key="macro_positive_recall",
            note="Broad Top155 召回后用同一 CAGE pred_logit 重排",
        ),
        control_row(
            "BRIDGE",
            e2r_metrics["bridge"],
            coverage_key="macro_positive_recall",
            note="当前 BRIDGE 在同一 465 反应域内选/排 Top155",
        ),
    ]
    for key, label, note in (
        ("minus_functional", "BRIDGE - Functional", "去功能证据"),
        (
            "minus_structure_mechanism",
            "BRIDGE - Structure/Mechanism",
            "去结构证据",
        ),
        (
            "minus_relational_memory",
            "BRIDGE - Relational Memory",
            "去长期关系记忆",
        ),
        (
            "minus_family_domain",
            "BRIDGE - Family/Domain",
            "当前 E2R 无 family/TPS 成员，与完整 BRIDGE 相同",
        ),
    ):
        e2r_rows.append(
            control_row(
                label,
                e2r_metrics[key],
                coverage_key="macro_positive_recall",
                note=note,
            )
        )

    generalization = {
        "r2e": generalization_rows(r2e_gen, "r2e"),
        "e2r": generalization_rows(e2r_gen, "e2r"),
    }
    native = native_benchmarks()

    result = {
        "schema": "bridge-layered-main-tables-v2",
        "status": "completed",
        "controlled_main_tables": {
            "status": "diagnostic_only_due_to_clean2023_exact_pair_overlap",
            "overlap_audit": {
                "r2e": {
                    "positive_pairs": 1526,
                    "clean2023_exact_pair_overlap": 1305,
                    "overlap_fraction": 0.8551769331585846,
                    "zero_exact_overlap_queries": 71,
                    "total_queries": 459,
                },
                "e2r": {
                    "positive_pairs": 1529,
                    "clean2023_exact_pair_overlap": 1308,
                    "overlap_fraction": 0.855461085676913,
                    "zero_exact_overlap_queries": 88,
                    "total_queries": 852,
                },
            },
            "r2e": {
                "protocol": {
                    "queries": 459,
                    "candidate_domain": "1,379 proteins with real EnzymeCAGE scores",
                    "budget": "per-query K = EnzymeCAGE native candidate_gate_size",
                    "purpose": "diagnose recall/reranking only; not a fair performance benchmark",
                    "generalization_claim": False,
                },
                "rows": r2e_rows,
            },
            "e2r": {
                "protocol": {
                    **e2r_control["protocol"],
                    "purpose": "diagnose recall/reranking only; not a fair performance benchmark",
                    "generalization_claim": False,
                },
                "rows": e2r_rows,
            },
        },
        "generalization_tables": {
            "protocol": {
                "edges": 23773,
                "all_edges_retained": True,
                "r2e_queries": int(r2e_gen.reaction_id.nunique()),
                "e2r_queries": int(e2r_gen.protein_id.nunique()),
                "r2e_candidate_universe": 185918,
                "e2r_candidate_universe": 11081,
                "difficulty_macro": (
                    "novelty groups equal weighted; train-degree strata "
                    "equal weighted within novelty groups"
                ),
                "episodic_runtime_support": "none",
            },
            **generalization,
        },
        "native_benchmarks": native,
        "evidence_groups": {
            "functional": "EnzGFM functional compatibility",
            "structure_mechanism": (
                "CLIPZyme + reaction-center/mechanism + R2E pocket reorder"
            ),
            "relational_memory": (
                "clean2023 long-term relation memory; episodic memory absent "
                "unless runtime support is supplied"
            ),
            "family_domain": "family CAGE + TPS specialists",
        },
        "reproduction": {
            "r2e_controlled": (
                "reproducibility/bime_rank/scripts/"
                "evaluate_bridge_current_r2e_cage_shared_pool_v1.py"
            ),
            "e2r_controlled": (
                "reproducibility/bime_rank/scripts/"
                "evaluate_bridge_cage_e2r_fixed465_v1.py"
            ),
            "r2e_generalization_ablation": (
                "reproducibility/bime_rank/scripts/"
                "evaluate_bridge_r2e_four_group_ablation_v1.py"
            ),
            "e2r_generalization_ablation": (
                "reproducibility/bime_rank/scripts/"
                "evaluate_bridge_e2r_four_group_ablation_v1.py"
            ),
            "assembler": (
                "reproducibility/bime_rank/scripts/"
                "assemble_bridge_layered_tables_v2.py"
            ),
        },
    }

    RECORD.write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# BRIDGE 主实验与分层评测表",
        "",
        "前两张表仅保留为训练重叠诊断：R2E/E2R controlled 正例与 clean2023 的精确关系重叠均约 85.5%，因此不能用于证明 BRIDGE 性能。后两张 23,773 条严格 relation-unseen 表才承担广域泛化结论。",
        "",
        "## 诊断表一：反应到酶——同预算召回 × 重排",
        "",
        "459 个 query；1,526 条正例中 1,305 条（85.52%）已在 clean2023 出现，仅 71 个 query 没有任何测试正例与训练图精确重叠。下表只用于审计召回/重排行为。",
        "",
        "| 模型 | MRR | MAP | Hit@3 | Hit@10 | Hit@20 | Query 正例覆盖 | 说明 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in r2e_rows:
        lines.append(
            f"| {row['model']} | {fmt(row['mrr'])} | {fmt(row['map'])} | "
            f"{pct(row['hit3'])} | {pct(row['hit10'])} | {pct(row['hit20'])} | "
            f"{pct(row['coverage'])} | {row['note']} |"
        )

    lines += [
        "",
        "## 诊断表二：酶到反应——固定域同预算排序",
        "",
        "852 个有效酶 query、1,529 条正例中 1,308 条（85.55%）已在 clean2023 出现，仅 88 个 query 没有任何测试正例与训练图精确重叠。固定 465 反应、K=155 的数值仅作为泄漏诊断保留。",
        "",
        "| 模型 | MRR | MAP | Hit@3 | Hit@10 | Hit@20 | 宏正例召回 | 说明 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in e2r_rows:
        lines.append(
            f"| {row['model']} | {fmt(row['mrr'])} | {fmt(row['map'])} | "
            f"{pct(row['hit3'])} | {pct(row['hit10'])} | {pct(row['hit20'])} | "
            f"{pct(row['coverage'])} | {row['note']} |"
        )

    def add_generalization(title: str, rows: list[dict]) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| 模型 | MRR | Hit@10 | Hit@100 | Hit@1000 | 说明 |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in rows:
            m = row["metrics"]
            lines.append(
                f"| {row['model']} | {fmt(m['mrr'])} | {pct(m['hit10'])} | "
                f"{pct(m['hit100'])} | {pct(m['hit1000'])} | {row['note']} |"
            )

    add_generalization(
        "广域泛化：反应到酶（23,773 relation-unseen）",
        generalization["r2e"],
    )
    add_generalization(
        "广域泛化：酶到反应（23,773 relation-unseen）",
        generalization["e2r"],
    )

    lines += [
        "",
        "## EnzymeCAGE 作者原生基准",
        "",
        "### Enzyme-405（226-query aligned comparison）",
        "",
        "| 模型 | MRR | Hit@10 | Hit@20 |",
        "|---|---:|---:|---:|",
    ]
    for row in native["enzyme405_complete226"]["rows"]:
        lines.append(
            f"| {row['model']} | {fmt(row['mrr'])} | "
            f"{pct(row['hit10'])} | {pct(row['hit20'])} |"
        )

    lines += [
        "",
        "### Orphan-335",
        "",
        "| 模型 | MRR | Hit@10 | Hit@20 |",
        "|---|---:|---:|---:|",
    ]
    for row in native["orphan335"]["rows"]:
        lines.append(
            f"| {row['model']} | {fmt(row['mrr'])} | "
            f"{pct(row['hit10'])} | {pct(row['hit20'])} |"
        )
    lines += [
        "",
        "> Orphan-335 当前仓库没有覆盖完整作者候选池的 EnzymeCAGE generic-pretrain 神经分数；Selenzyme 保留真实名称，不冒充 EnzymeCAGE。",
        "",
        "## 四组证据定义",
        "",
        "- 功能证据：EnzGFM 功能兼容性。",
        "- 结构与机制证据：CLIPZyme、反应中心/机制，以及 R2E pocket 局部重排。",
        "- 关系记忆证据：clean2023 长期训练图记忆；只有提供训练后/用户确认 support 时才额外启用情景记忆。",
        "- 家族与领域专家：family CAGE 与 TPS。",
        "",
        "controlled 表中长期关系记忆几乎直接回放训练图，因此完整 BRIDGE 的高分无泛化含义。专家的广域独立价值只读取 23,773 条严格 relation-unseen 泛化表。",
        "",
    ]
    DOC.write_text("\n".join(lines) + "\n")

    print(
        json.dumps(
            {
                "record": str(RECORD.relative_to(ROOT)),
                "doc": str(DOC.relative_to(ROOT)),
                "controlled_r2e_rows": len(r2e_rows),
                "controlled_e2r_rows": len(e2r_rows),
                "generalization_r2e_rows": len(generalization["r2e"]),
                "generalization_e2r_rows": len(generalization["e2r"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
