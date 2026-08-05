from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "docs/terpene_candidate_retrieval_comprehensive_report_zh.md"
OUTPUT_SUMMARY = ROOT / "results/terpene_candidate_retrieval_comprehensive_report_summary.json"


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def pp(value: float, digits: int = 2) -> str:
    return f"{100 * float(value):+.{digits}f} pp"


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" if i == 0 else "---:" for i in range(len(headers))) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def heading(title: str, level: int = 2) -> str:
    return f"{'#' * level} {title}\n"


def code(value: str) -> str:
    return f"`{value}`"


def main() -> None:
    scheme = load_json("docs/terpene_candidate_retrieval_scheme_comparison_metrics.json")
    iteration = load_json("results/terpene_research_iteration_summary.json")
    finalization = load_json("results/terpene_finalization_summary.json")
    dual_route = load_json("results/terpene_marts_dual_kernel_rescue_route_v1/summary.json")
    dual_confirm = load_json(
        "results/terpene_marts_dual_kernel_confirmatory20260726/locked_confirmatory_summary.json"
    )
    dual_asset = load_json(
        "results/terpene_production_models/marts_dual_kernel_e2r_top20/summary.json"
    )
    dual_validation = load_json(
        "results/terpene_deployment_validation_e2r_top20_dual_kernel.json"
    )
    calibration_merge = load_json(
        "results/terpene_open_world_uncertainty_rrf_routing/dual_kernel_top20_merge_summary.json"
    )
    registry = load_json("results/terpene_registry_batch/summary.json")
    registry_audit = load_json(
        "results/terpene_registry_batch/dual_kernel_top20_change_audit.json"
    )
    uniprot = load_json("results/terpene_uniprot_expansion_report_summary.json")
    wetlab = load_json("results/terpene_combined_wetlab_campaign/summary.json")
    canonical_plate = load_json("results/terpene_wetlab_plate_manifest/summary.json")
    rescue = load_json("results/terpene_uniprot_rescue_campaign/summary.json")
    plate_balance = load_json("results/terpene_wetlab_plate_balanced/compact_balance_summary.json")
    randomization = load_json("results/terpene_wetlab_randomized_layout/summary.json")
    calibrators = load_csv(
        "results/terpene_open_world_uncertainty_rrf_routing/calibration_summary.csv"
    )
    exact_entity_protocols = load_csv(
        "results/terpene_exact_entity_protocols/metrics.csv"
    )
    exact_entity_visibility = load_csv(
        "results/terpene_protocol_reassessment/exact_entity_visibility_matrix.csv"
    )
    pfam_exact = load_csv(
        "results/terpene_current_pfam_on_locked_me8_v1/frozen_metrics.csv"
    ).iloc[0]
    pfam_hier = load_csv(
        "results/terpene_current_pfam_hierarchical_full_v1/frozen_metrics.csv"
    ).iloc[0]
    registry_concentration = load_csv(
        "results/terpene_registry_batch/discovery_concentration_summary.csv"
    )
    protocol_taxonomy = load_csv("results/terpene_protocol_reassessment/protocol_taxonomy.csv")
    protocol_capability = load_csv("results/terpene_protocol_reassessment/capability_spectrum.csv")
    protocol_same_model = load_csv("results/terpene_protocol_reassessment/same_model_cold_protocol_matrix.csv")
    protocol_fewshot = load_csv("results/terpene_protocol_reassessment/fewshot_protocol_matrix.csv")
    tests_text = (ROOT / "results/terpene_full_test_suite_dual_kernel.log").read_text(
        encoding="utf-8"
    )

    legacy = scheme["legacy_exact_reaction_protocol"]
    common = scheme["common_double_cold_25cell"]
    production_external = iteration["selected_routes"]
    top20_cal = calibrators[
        calibrators["calibrator"].eq("enzyme_to_reaction_top20")
    ].iloc[0]
    exact_protocol_names = {
        "protein_exact": "Exact 蛋白未见；同簇同源物允许可见",
        "reaction_exact": "Exact 反应未见；同簇相似反应允许可见",
    }
    direction_names = {
        "reaction_to_enzyme": "R2E",
        "enzyme_to_reaction": "E2R",
    }
    exact_entity_rows: list[list[object]] = []
    for protocol in ["protein_exact", "reaction_exact"]:
        for direction in ["reaction_to_enzyme", "enzyme_to_reaction"]:
            row = exact_entity_protocols[
                exact_entity_protocols["protocol"].eq(protocol)
                & exact_entity_protocols["direction"].eq(direction)
            ].iloc[0]
            exact_entity_rows.append(
                [
                    exact_protocol_names[protocol],
                    direction_names[direction],
                    int(row["n_query_cells"]),
                    pct(row["hit_probability_at_3"]),
                    pct(row["hit_probability_at_5"]),
                    pct(row["hit_probability_at_10"]),
                    pct(row["hit_probability_at_20"]),
                    f"{row['mean_reciprocal_rank']:.3f}",
                ]
            )
    exact_visibility_rows: list[list[object]] = []
    for protocol in ["protein_exact", "reaction_exact"]:
        for direction in ["reaction_to_enzyme", "enzyme_to_reaction"]:
            for visibility in ["visible", "not_visible"]:
                row = exact_entity_visibility[
                    exact_entity_visibility["protocol"].eq(protocol)
                    & exact_entity_visibility["direction"].eq(direction)
                    & exact_entity_visibility["same_cluster_evidence"].eq(visibility)
                ].iloc[0]
                exact_visibility_rows.append(
                    [
                        exact_protocol_names[protocol],
                        direction_names[direction],
                        "有同簇训练邻居" if visibility == "visible" else "无同簇训练邻居",
                        int(row["n_query_cells"]),
                        pct(row["hit_probability_at_3"]),
                        pct(row["hit_probability_at_10"]),
                        pct(row["hit_probability_at_20"]),
                    ]
                )

    parts: list[str] = []
    parts.append(
        """# 萜类合酶双向检索、开放发现与湿实验候选推荐系统完整技术报告

**副标题：从旧版 Gate / Reaction Similarity / CAGE 元排序，到多场景评测、MARTS 域适配、目标分预算路由、双核协同检索、受控 UniProt 扩展与六块板实验落地**

- 报告状态：当前正式主报告
- 报告日期：2026-07-24
- 项目目录：`/home/s241850073/igem2026`
- 适用范围：萜类反应与候选酶的双向检索、外部开放发现、少样本扩展、候选库扩展、可靠性评估和湿实验执行
- 不包含：位点突变设计、蛋白从头设计、口袋改造、酶动力学参数预测或对单个候选“必然有活性”的保证

> **最重要的阅读原则**：本文中的 Hit@K、MRR、ROC-AUC 等是“检索与排序指标”，不是湿实验阳性率，也不是酶活概率。本系统没有一个可以覆盖所有用途的单一总分：数据库补全、允许同源的 few-shot 扩展、reaction-cold、protein-cold 和 double-cold 分别回答不同问题。相似蛋白在实际筛选中是合法且高价值的证据；双冷只是完整开放外推的压力测试，不能取代同源扩展和数据库补全指标。只有在同一候选集合、同一训练信息边界和同一拆分协议下的数字才具有严格可比性。

---
"""
    )

    parts.append(heading("一、执行摘要：现在到底完成了什么", 2))
    parts.append(
        """本项目已经从一个以规则门控、反应相似度和 CAGE 辅助元排序为主的数据库内补全流程，升级为一个可部署的、双向的、开放世界候选检索系统。系统同时支持：

1. **Reaction → Enzyme（R2E）**：给定目标反应，寻找值得验证的候选酶；
2. **Enzyme → Reaction（E2R）**：给定未知或外部酶，预测其最可能对应的反应；
3. **Zero-shot**：查询时没有已知阳性 seed；实体/簇是否未见由 exact、reaction-cold、protein-cold 或 double-cold 协议另行说明；
4. **Few-shot**：已有 1–5 个阳性种子酶时扩展同类候选；
5. **开放候选库**：当前库、MARTS 外部库与受控 UniProt rescue 层；
6. **可靠性分层**：给出排序证据等级，而不是伪造生化活性概率；
7. **湿实验落地**：反应选择、候选选择、架构约束、板间平衡、孔位随机化、构建与采购清单。

当前生产候选宇宙为 2,085 条 canonical 蛋白、753 个 canonical 反应和 3,439 条去重训练关联；另有 5,672 条具名称且通过证据分级的 UniProt TPS 作为受控尾部救援层。正式注册表包含 694 个外部酶查询和 240 个外部反应查询。

本报告采用多轨能力谱，而不把最难的 double-cold 当作唯一总指标。对已有阳性 seed 的同源扩展，Top-10 为 73.7%（1 seed）到 92.8%（5 seeds）；对从未进入训练的 exact 新蛋白，在允许同簇同源物作为合法证据时，E2R Top-10 为 72.4%，其中训练中实际存在同簇同源物的查询达到 82.6%；对当前数据库 exact-reaction 补全，嵌套 Top-10/20 为 48.1%/57.5%；对同一个 multiview 双塔，只隔离反应簇且允许复用蛋白空间时，R2E Top-10/20 为 28.7%/36.3%；只有当蛋白簇与反应簇同时未见时，才进入最难的 double-cold 口径。详细任务矩阵见第六章及 `docs/terpene_retrieval_protocol_reassessment_zh.md`。

本轮新增并完成生产化的核心改进，是 **E2R Top-20 双核协同 RRF 路由**。它不再单纯依赖一个神经模型，而是把“查询蛋白与训练酶的序列相似性”“候选反应与训练反应的化学相似性”和“训练反应—酶关联图”三者相乘，形成非参数协同证据，再以 70% 原生产路线 + 30% 双核路线的 RRF 融合。该路由在独立锁定切分上把 Hit@20 从 34.77% 提高到 43.37%，绝对增加 8.60 个百分点，配对 bootstrap 95% 置信区间为 +5.02 到 +12.54 个百分点。

生产接入已经完成，可靠性校准器已重训，694 个外部酶的 Top-20 注册表已重排。R2E 全部结果以及 E2R Top-3/10 的候选与分数均保持逐字节一致；仅授权变化的 E2R Top-20 被更新。全部 30,822 行注册表排名的已知关联泄漏仍为 0。五个神经部署目录和一个双核稀疏资产包均验证为 `valid`；TPS 测试套件为 79 passed。
"""
    )

    parts.append(heading("二、直接回答最容易产生疑惑的几个问题", 2))
    parts.append(
        """### 2.1 新版是不是把旧版推倒重来？

不是。新版保留了旧版真正有效的思想：

- 反应相似度是强基线；
- 蛋白序列相似度对同源家族扩展非常有效；
- 高召回候选池与最终 Top-K 排序应分离；
- 结构或催化证据更适合作为辅助证据，而不是无条件替代主排序；
- 零样本首轮筛选后，湿实验阳性可以转化为 few-shot 种子继续扩展。

新版修正的是旧版的评测口径、候选宇宙、可复现性和部署边界，而不是否认所有旧方法。

### 2.2 旧版 39.6% / 45.2% 是不是假的？

不是“假的”，也不是简单答案泄漏。它主要回答的是“在该折整条 exact reaction ID 不进入模型训练、但相似反应簇和已知蛋白家族仍可利用时，能否补全这个反应的候选酶”。这本来就是一个有现实价值的任务。它不能被解释成完整开放发现，但也不应因为不是双冷就被降格为无效指标。

### 2.3 为什么双冷指标明显更低？

因为它刻意删除了两类在实际筛选中通常允许使用的强证据：相似反应簇和近同源蛋白簇。双冷回答的是“当两侧都没有可直接迁移的邻域时，模型还能否工作”，而不是系统在日常候选筛选中的总体成功率。

### 2.4 为什么不要求所有评测都同时隔离两侧？

是否隔离哪一侧取决于任务。已有阳性酶时，同源蛋白就是核心证据；给一个新反应筛选已知家族时，应使用 reaction-cold；给一个新蛋白注释到已知反应目录时，应使用 protein-cold E2R。只有当研究目标明确是“新反应家族 × 新蛋白家族同时外推”时，double-cold 才是主指标。正确做法是多轨并列报告，而不是用最难一轨覆盖其他任务。

### 2.5 新版是否已经直接修改生产代码，而不是旁边另写一套实验？

是。新 Top-20 路由已经接入 `rank_open_world.py` 和 `rank_registry_batch.py`；单查询与批处理公共入口在同一注册酶上实现候选、顺序和浮点 RRF 分数完全一致。新资产在 `results/terpene_production_models/marts_dual_kernel_e2r_top20/`，并由独立部署验证器检查。

### 2.6 为什么湿实验六块板没有因为本轮 E2R 提升而全部重做？

六块板的主要上游是 **R2E：给定反应找酶**。本轮生产更新只影响 **E2R Top-20：给定酶找反应**。最终审计证明 R2E Top-3/10/20 的候选、排名和分数完全不变，因此重做板布局不仅没有依据，还会破坏已经锁定的实验设计。正式报告和注册表审计已更新，但物理板清单保持不变。
"""
    )

    parts.append(heading("三、生物学与任务背景", 2))
    parts.append(
        """### 3.1 萜类与萜类合酶

萜类化合物由异戊二烯单元构成，是天然产物中结构多样性最丰富的类别之一。不同前体长度、环化级联、碳正离子重排、终止方式、后续氧化或修饰，会产生大量不同骨架和产物。萜类合酶（terpene synthase, TPS）负责其中关键的成键、环化或重排步骤。

在工程应用中，两个方向都重要：

- **给定目标产物反应找酶**：用于合成生物学路线搭建、候选酶筛选和湿实验验证；
- **给定未知酶找反应**：用于功能注释、发现新底物/产物关系、扩大反应注册表。

### 3.2 Class I、Class II 与其他相关架构

TPS 并非单一同源家族。常见机制和结构域包括：

- Class I TPS：典型金属依赖离去基团活化，常关注 DDxxD 和 NSE/DTE 等基序；
- Class II cyclase：常通过酸催化启动，可能含 DxDD 等基序；
- 植物双功能 TPS：可能同时包含 Class II 和 Class I 相关结构域；
- Oxidosqualene cyclase（OSC）：完整架构通常要求 PF13243 + PF13249，而单域命中可能只是片段；
- Prenyltransferase、P450 和其他修饰酶：可出现在关联数据中，但不能简单等同于经典 TPS。

因此“只要有一个 TPS Pfam 就保留”或“非经典 TPS 一律删除”都过于粗糙。生产系统采用反应条件化架构证据和受控 rescue，而不是全局硬门控。

### 3.3 本项目不是什么

本项目解决的是候选检索与实验优先级，而不是：

- 预测突变后活性提高多少；
- 设计催化口袋；
- 预测绝对 kcat/Km；
- 证明候选在特定宿主中可表达；
- 证明某个计算高分候选一定产生目标产物。

这些仍需要表达、底物供给、GC/LC-MS、产物结构确认和必要的动力学实验。
"""
    )

    parts.append(heading("四、任务形式化：R2E、E2R、Zero-shot 与 Few-shot", 2))
    parts.append(
        r"""令反应集合为 \(\mathcal{R}\)，蛋白集合为 \(\mathcal{P}\)，已知关联为 \(A\subseteq\mathcal{R}\times\mathcal{P}\)。系统学习或构造一个打分函数 \(s(r,p)\)。

- R2E：固定 \(r\)，按 \(s(r,p)\) 对所有 \(p\in\mathcal{P}\) 排序；
- E2R：固定 \(p\)，按 \(s(r,p)\) 对所有 \(r\in\mathcal{R}\) 排序。

**Zero-shot** 只表示查询时没有已知阳性 seed，不等于自动要求实体簇未见。一个 zero-shot 查询可以属于 current exact holdout、reaction-cold、protein-cold 或 double-cold。**Few-shot** 表示查询反应已有少量已知阳性酶种子，可利用种子序列邻域扩展候选；是否允许近同源阳性同样由 random-positive 或 protein-cluster-cold 协议另行定义。

评测实际上有三个独立轴：

1. 是否给出阳性 seed；
2. exact reaction ID / 相似反应簇是否可见；
3. 正确蛋白及其 50% identity cluster 是否可见。

前两个 cold 轴构成无 seed 条件下的二维新颖性平面；few-shot 是第三个独立轴。double-cold 只是二维平面的一个角，不是 zero-shot 的同义词。

需要特别区分三种 few-shot 难度：

1. 随机隐藏同一反应的部分阳性：实用同源扩展，主要测同源家族补全；
2. seed 与 hidden positives 按 50% identity cluster 隔离：更难，测跨簇扩展；
3. 外部 MARTS 查询的 few-shot：查询反应与候选宇宙更开放，但未必同时满足 seed-hidden cluster-cold。
"""
    )

    parts.append(heading("五、旧方案的完整结构与合理部分", 2))
    parts.append(
        """旧方案大体由四层组成：

1. **Gate matrix / 候选门控**：根据反应、底物、产物类型和已知关系缩小候选池；
2. **Reaction similarity backbone**：从相似反应转移候选酶；
3. **CAGE 或结构相关分数**：提供蛋白—反应配对的辅助证据；
4. **RF/HGB meta-ranker 与 rescue slots**：在候选池内部融合多种信号。

其优点是直观、可解释、能快速建立高召回 reservoir，并在数据库内 exact-reaction holdout 上取得较高 Top-K。旧 few-shot 结果也稳定说明，序列相似度对已有阳性附近的同源扩展很强。

旧方案需要修正的不是“允许使用相似反应和相似蛋白”，因为这些在数据库补全与同源扩展中是合法证据；真正的问题是旧报告没有把不同任务分轨，容易把数据库内补全成绩误读为开放世界泛化。具体包括：

- 只报告 exact reaction ID 分组，未同时给出 reaction-cold、protein-cold 和 double-cold 能力谱；
- 候选蛋白可通过其他反应关系出现在训练中，因此旧数字只能解释为数据库补全，不能解释为未见蛋白家族发现；
- 多种 gate、融合权重和救援槽在同一批 OOF 结果上反复选择；
- 旧报告中的部分 tune/test 过程缺少当前服务器上的正式脚本和结果目录；
- CAGE sigmoid 概率存在饱和和大量并列，直接 probability fusion 会放大错误顺序；
- 旧脚本使用 Python 内置 `hash()` 生成部分随机种子，跨进程不能保证稳定复现。

新版不再把旧方案整体“降级”。reaction similarity、seed sequence similarity 和 current-library 路线继续服务于高成功率的 exploitation；双塔、RRF 和双核路线同时承担无 seed、外部实体和开放泛化。生产系统按场景路由，而不是强迫所有查询使用双冷逻辑。
"""
    )

    exact_entity_table = md_table(
        ["协议", "方向", "查询单元", "Hit@3", "Hit@5", "Hit@10", "Hit@20", "MRR"],
        exact_entity_rows,
    )
    exact_visibility_table = md_table(
        ["Exact 协议", "方向", "训练中同簇证据", "查询单元", "Hit@3", "Hit@10", "Hit@20"],
        exact_visibility_rows,
    )
    parts.append(heading("六、评测协议必须按真实任务分轨", 2))
    parts.append(
        f"""### 6.1 三个独立轴，而不是一条“从宽松到严格”的直线

没有理由要求所有评测都同时隔离相似反应簇与 50% 蛋白序列簇。在真实酶筛选中，相似蛋白通常是合法且最有价值的证据；只有在研究远缘发现或完整开放外推时，才需要主动移除这条路径。

需要分别声明：

- **Seed 轴**：zero-shot 还是给出 1–5 个阳性 seed；
- **反应新颖性轴**：只留出 exact reaction ID，还是整个反应簇均未见；
- **蛋白新颖性轴**：允许复用同源蛋白簇，还是正确蛋白簇未见。

因此不能把“没有 seed”“新反应”“新蛋白家族”混成一个开放世界标签。

### 6.2 无 seed 条件下的二维任务矩阵

| 反应侧 | 蛋白同源空间可用 | 正确蛋白簇不可用 |
|---|---|---|
| exact reaction ID 留出，但相似反应簇可用 | Current exact completion：数据库补全 | Protein-cluster-cold：已知反应空间中的远缘酶发现 |
| 整个反应簇不可用 | Reaction-cluster-cold：新反应映射到已有蛋白家族 | Double-cold：新反应簇 × 新蛋白簇同时外推 |

Few-shot 不属于这个二维表：有 seed 后，还要另外区分 seed 与 hidden positives 是否允许处于同一蛋白簇。

本项目正式并列报告以下场景：

| 场景 | 是否给 seed | 允许相似蛋白 | 允许已知/相似反应空间 | 主要用途 |
|---|---:|---:|---:|---|
| Current exact-reaction holdout | 否 | 是 | 相似簇可见，exact ID 留出 | 数据库内关联补全 |
| Exact-protein holdout | 否 | 是；测试蛋白本身及其关联未见 | 是 | 新序列注释、已知反应的新候选扩展 |
| Exact-reaction holdout | 否 | 是 | 相似簇可见，测试反应本身及其关联未见 | 新 exact 反应在相似化学空间中的检索 |
| Seeded homolog expansion | 是 | 是，且是核心证据 | 是 | 找更多同源或近同源替代酶 |
| Reaction-cluster-cold R2E | 否 | 是 | 否 | 新反应映射到已有蛋白家族 |
| Protein-cluster-cold R2E | 否 | 否 | 是 | 远缘或不同家族同功能酶发现 |
| Protein-cluster-cold E2R | 否 | 否 | 是 | 新蛋白注释到已有反应目录 |
| Reaction-cluster-cold E2R | 否 | 是 | 否 | 已知/相似酶的新反应发现 |
| Double-cold | 否 | 否 | 否 | 两侧同时外推的压力测试 |

### 6.3 同一个模型、只改变隔离条件

下表固定为同一个 current-only multiview dual tower、相同 1,391 条蛋白和 513 个反应，只改变 split，因此可以直接看出每种隔离带来的难度。这是协议消融表；current exact 与 few-shot 使用各自更合适的实用路线，不能与本表混作同模型性能差值。

| 协议 | 方向 | Hit@3 | Hit@5 | Hit@10 | Hit@20 |
|---|---:|---:|---:|---:|---:|
| 只隔离反应簇，蛋白空间可复用 | R2E | 15.4% | 20.1% | 28.7% | 36.3% |
| 只隔离蛋白簇，反应空间可复用 | R2E | 9.0% | 12.0% | 18.1% | 24.5% |
| 双侧同时隔离 | R2E | 2.1% | 3.5% | 6.4% | 12.8% |
| 只隔离反应簇，蛋白空间可复用 | E2R | 14.1% | 20.5% | 31.3% | 40.1% |
| 只隔离蛋白簇，反应空间可复用 | E2R | 24.0% | 29.1% | 36.1% | 48.0% |
| 双侧同时隔离 | E2R | 2.2% | 5.2% | 14.1% | 30.1% |

因此，R2E double-cold Top-10 的 6.4% 不能被称为“系统找酶的总体能力”；它只代表两侧邻域同时不可用的最难情形。允许利用已有蛋白家族、只测试新反应簇时，同一模型的 R2E Top-10 是 28.7%。

### 6.4 Exact 实体未见，但同源或相似簇允许可见

“测试实体未见”不等于必须把整个相似簇删除。下面固定同一个 current-only multiview 双塔：测试蛋白或反应本身及其全部关联不进入训练，但同一 50% 蛋白簇中的其他同源物、或同一化学簇中的其他反应仍可作为合法训练证据。

{exact_entity_table}

其中，exact 新蛋白注释到已有反应目录的 E2R Top-10 为 **72.4%**；已知反应寻找 exact 新蛋白的 R2E Top-10 为 **51.2%**。这两项更贴近“新序列但同源家族已知”的常见实际场景。

### 6.5 同簇证据实际存在与否

{exact_visibility_table}

最关键的 exact-protein E2R 对比是：训练中有同簇同源物时 Top-10 为 **82.6%**，没有同簇同源物时为 **38.4%**；整个蛋白簇未见的 protein-cluster-cold E2R 为 **36.1%**。R2E 同样明显：有同簇训练同源物时 Top-10 为 **62.6%**，没有时为 **15.5%**。

这直接证明，同源可见能力与远缘发现能力是两项不同能力。前者在实际筛选中应主动利用；cluster-cold 用于测量这条证据不可用时的后备能力，而不是取代前者。

### 6.6 Legacy exact-reaction holdout

每个目标折中的整条 exact reaction ID 都不进入该折模型训练；但训练中仍可包含高度相似的其他反应簇成员，候选蛋白也可通过其他反应关系出现。它回答数据库补全问题，不是“只隐藏单条 pair”，也不能被解释为 reaction-cluster-cold 或远缘开放发现。

### 6.7 Seeded homolog expansion 与 seeded cross-cluster expansion

已有阳性 seed 时，random-positive 允许隐藏阳性来自同一同源家族，是实际同源扩展指标；protein-cluster-cold 强制 seed 与隐藏阳性跨 50% identity cluster，是远缘扩展指标。二者应同时报告，不是一个取代另一个。

### 6.8 Common current-only 与 MARTS external double-cold

双冷协议用于评估查询蛋白簇和反应簇均未见时的外推能力，也用于构建保守的可靠性校准器。它是方法学压力测试和探索能力指标，而不是所有生产查询的默认口径。

### 6.9 Development / frozen / independent confirmation

- development 9 cells：只用于选择极少量超参数或路线；
- frozen 16 cells：参数锁定后一次性验证；
- independent locked fold seed：重新随机平衡簇分配，模型与切分重新训练，所有参数保持不变。

这个流程的核心是防止“在冻结集上继续调参数”。标识 `20260726` 是切分 seed，不是运行日期。

### 6.10 Query-cell 与 unique query

同一查询可能因其正例分布跨越多个笛卡尔单元而形成多个 query-cell。报告同时给 query-cell 数和 unique query 数。Hit@K 的统计单元通常是 query-cell，因此不能把 279 个 query-cell 误读为 279 条不同蛋白。

更完整的协议重评和机器可读矩阵见 `docs/terpene_retrieval_protocol_reassessment_zh.md` 与 `results/terpene_protocol_reassessment/`。
"""
    )

    parts.append(heading("七、数据、候选宇宙与证据来源", 2))
    data_rows = [
        ["当前蛋白", "1,391", "current TPS/相关蛋白库"],
        ["注册 MARTS 外部蛋白", "694", "开放 E2R 查询与 canonical 候选"],
        ["canonical 蛋白总数", "2,085", "生产神经模型与双核资产"],
        ["当前反应", "513", "Rhea/current registry"],
        ["注册 MARTS 外部反应", "240", "开放 R2E 查询与 E2R 候选"],
        ["canonical 反应总数", "753", "生产排名宇宙"],
        ["训练关联", "3,439", "current + MARTS 去重关联"],
        ["UniProt raw", f"{uniprot['candidate_expansion']['raw_rows']:,}", "五个 TPS 相关 Pfam 查询"],
        ["UniProt novel exact-sequence unique", f"{uniprot['candidate_expansion']['novel_sequence_unique_rows']:,}", "移除 current/MARTS 重复后"],
        ["UniProt 50% identity clusters", f"{uniprot['candidate_expansion']['novel_clusters']:,}", "MMseqs 代表"],
        ["启用的 named UniProt", f"{uniprot['candidate_expansion']['primary_named_embedding_candidates']:,}", "A–D 证据层"],
        ["暂不启用 domain-only", f"{uniprot['candidate_expansion']['domain_only_rescue_candidates']:,}", "E 层"],
    ]
    parts.append(md_table(["对象", "数量", "用途/解释"], data_rows) + "\n")
    parts.append(
        """数据治理遵循三个原则：

1. exact accession、exact sequence 和 cluster ID 分开管理；
2. 已知关联在 discovery 排名中默认屏蔽，但在训练和回归审计中保留；
3. 未标注不等于负例，同簇未标注候选不应轻易进入对比学习分母。
"""
    )

    parts.append(heading("八、表示学习与主模型", 2))
    parts.append(
        """### 8.1 蛋白表示：ESM-C 600M mean embedding

每条蛋白序列编码为 1,152 维全局向量。它概括序列上下文和进化/结构相关模式，但不是显式结构预测，也不会直接输出催化机制。

### 8.2 反应表示：DRFP + 前体/产物骨架类别

DRFP 把反应物到产物的原子环境变化编码为 2,048 维差分指纹，再拼接前体类别和产物骨架类别，总维度为 2,115。其优点是无需三维结构，且可以处理多种反应变换；局限是哈希碰撞、无法完整表达立体电子机制和复杂多步级联。

### 8.3 Multi-positive dual tower

蛋白塔和反应塔分别把输入映射到 256 维共享空间。与普通一对一 InfoNCE 不同，一个反应可有多个阳性酶，一个酶也可对应多个反应。训练时保留所有已知正例。

### 8.4 PU cluster mask

PU 表示 positive–unlabeled。未标注候选不能简单视为真负例。系统把同一 50% 蛋白簇或同一反应簇中的未标注样本从对比学习分母中排除，降低把潜在同功能同源物当作负例的风险。

### 8.5 MARTS domain adaptation 与三 seed ensemble

生产模型以 current 模型为起点，在 current + MARTS 关联上适配。三个不同随机 seed 的模型共同输出，既提高稳定性，也提供 seed disagreement、Top-K Jaccard、边界 margin 等可靠性特征。
"""
    )

    parts.append(heading("九、旧新方案在同一口径下的公平比较", 2))
    parts.append(
        """本章只比较**同一协议、同一候选集合**下的旧新方法。第六章的能力谱则包含不同场景的最佳实用路线，例如同源扩展用 3-mer/ESM-C seed similarity，current exact 用嵌套融合；那些数字用于回答“这个场景现在能做到什么”，不能被解释成同一个模型从 6% 提升到 70%。

"""
    )
    exact_rows = [
        ["旧 reaction similarity", pct(legacy["old_reaction_similarity_backbone"]["hit_at_5"]), pct(legacy["old_reaction_similarity_backbone"]["hit_at_10"]), pct(legacy["old_reaction_similarity_backbone"]["hit_at_20"])],
        ["旧 RF/CAGE rescue", pct(legacy["old_final_rf_rescue"]["hit_at_5"]), pct(legacy["old_final_rf_rescue"]["hit_at_10"]), pct(legacy["old_final_rf_rescue"]["hit_at_20"])],
        ["新 controlled dual tower", pct(legacy["new_controlled_dual_tower"]["hit_at_5"]), pct(legacy["new_controlled_dual_tower"]["hit_at_10"]), pct(legacy["new_controlled_dual_tower"]["hit_at_20"])],
    ]
    parts.append("### 9.1 Legacy exact-reaction protocol\n\n" + md_table(["方法", "Hit@5", "Hit@10", "Hit@20"], exact_rows) + "\n")
    parts.append(
        """在旧协议上，新双塔并没有因为改用严格方法就失去数据库补全能力：Top-10 与旧最终 RF 基本持平，Top-20略高；Top-5 略低。这说明新版改进不是靠牺牲旧任务获得的。
"""
    )
    cold_rows = [
        ["旧 fold-local RF rescue", pct(common["old_fold_local_rf_rescue"]["hit_at_5"]), pct(common["old_fold_local_rf_rescue"]["hit_at_10"]), pct(common["old_fold_local_rf_rescue"]["hit_at_20"])],
        ["新 controlled dual tower", pct(common["new_controlled_dual_tower"]["hit_at_5"]), pct(common["new_controlled_dual_tower"]["hit_at_10"]), pct(common["new_controlled_dual_tower"]["hit_at_20"])],
        ["绝对差值", pp(common["new_minus_old"]["hit_at_5"]), pp(common["new_minus_old"]["hit_at_10"]), pp(common["new_minus_old"]["hit_at_20"])],
    ]
    parts.append("### 9.2 Common current-only double-cold\n\n" + md_table(["方法", "Hit@5", "Hit@10", "Hit@20"], cold_rows) + "\n")
    parts.append(
        f"""在真正隔离蛋白簇与反应簇后，旧 fold-local RF rescue 基本失效，而新双塔仍维持可用检索能力。Top-10 的配对 bootstrap 区间为 {pp(common['paired_bootstrap'][1]['bootstrap_ci_low'])} 到 {pp(common['paired_bootstrap'][1]['bootstrap_ci_high'])}；Top-20 为 {pp(common['paired_bootstrap'][2]['bootstrap_ci_low'])} 到 {pp(common['paired_bootstrap'][2]['bootstrap_ci_high'])}。这比只比较两个均值更有说服力。
"""
    )

    parts.append(heading("十、当前生产目标分预算路由", 2))
    route_rows = [
        ["E2R", "Top-3", "freeze-reaction + 5-neighbor, direct 0.75", "7.8%"],
        ["E2R", "Top-10", "0.35 freeze-route + 0.65 hard-negative route RRF, c=60", "25.4%"],
        ["E2R", "Top-20", "0.70 freeze-route + 0.30 dual-kernel RRF, c=60", pct(top20_cal["base_hit_rate"])],
        ["R2E", "Top-3", "reaction-loss weight 0.75 direct", "4.6%"],
        ["R2E", "Top-10", "Horizyn exact-residual direct", "13.5%"],
        ["R2E", "Top-20", "Horizyn exact-residual direct", "19.0%"],
    ]
    parts.append(md_table(["方向", "预算", "生产路线", "严格外部指标"], route_rows) + "\n")
    parts.append(
        """分预算路由的原因是 Top-3、Top-10 和 Top-20 的错误代价不同。Top-3 更重视非常早期的精度；Top-20 更重视覆盖和互补证据。一个在 Top-20 有帮助的辅助源可能破坏 Top-3，因此不能用同一融合权重覆盖所有预算。
"""
    )

    parts.append(heading("十一、E2R Top-10 神经双路线 RRF", 2))
    parts.append(
        """Top-10 使用两个独立训练的神经路线：

- primary：freeze-reaction tower，5 个蛋白邻居，direct weight 0.5；
- secondary：hard-negative K=128，3 个蛋白邻居，direct weight 0.9；
- RRF 权重：primary 0.35，secondary 0.65，constant 60。

该参数在确认切分前锁定，并在两个确认 fold seed 上保持正向：22.7% vs 19.4%，以及 21.9% vs 17.7%。Top-10 的成功说明，模型间排序差异可以比原始 cosine/sigmoid 分数更稳定地融合。
"""
    )

    parts.append(heading("十二、E2R Top-20 双核协同方法：原理与推导", 2))
    parts.append(
        r"""### 12.1 核心公式

设训练关联矩阵为 \(A\in\mathbb{R}^{|R|\times|P|}\)，反应核为 \(K_R\)，蛋白核为 \(K_P\)。对于查询蛋白 \(p_q\)，双核分数可写为：

\[
S_{DK}(r,p_q)=\sum_{(r_i,p_j)\in A_{train}}
K_R(r,r_i)\,\tilde A_{ij}\,K_P(p_j,p_q).
\]

直观地说，只有当“候选反应接近某个训练反应”且“查询蛋白接近该训练反应的已知酶”同时成立时，训练正例才提供高支持。

### 12.2 反应核

对每个候选反应只保留最相似的 50 个训练反应；相似度经温度 0.03 的 softmax 转成局部权重。小温度使最相似反应获得更集中权重，但仍允许多个邻居贡献。

### 12.3 蛋白核

查询蛋白在训练蛋白中选取 Top-5 ESM-C cosine 邻居，同样使用温度 0.03。若查询蛋白本身存在于训练注册表，运行时强制排除 exact query self-neighbor，防止已知酶通过自身标签造成伪发现。

### 12.4 度归一化

原始关联图中某些反应或蛋白关联度较高。若直接求和，高度节点会天然获得高分。系统使用 degree power 1 的归一化邻接，降低 hub bias。

### 12.5 为什么使用 RRF 而不是直接相加分数

神经模型 cosine、邻居 tied-rank 和双核支持分数的尺度不同。直接加权容易被分数方差和饱和问题支配。Reciprocal Rank Fusion 只使用名次：

\[
S_{RRF}(x)=\frac{w}{c+rank_1(x)}+\frac{1-w}{c+rank_2(x)}.
\]

生产参数为 primary 0.70、dual-kernel 0.30、\(c=60\)。

### 12.6 为什么它适合 Top-20 而不是 Top-3

双核协同是覆盖型证据。它擅长把神经模型漏掉但有“化学邻域 × 序列邻域”共同支持的反应拉入较深候选列表，却不一定能把唯一正确反应稳定推到前三。因此它在 Top-20 有显著互补性，但单独用于 Top-3/5 在冻结切分上失败。
"""
    )

    asset_rows = [
        ["反应数", f"{dual_asset['n_reactions']:,}"],
        ["蛋白数", f"{dual_asset['n_proteins']:,}"],
        ["训练关联", f"{dual_asset['n_training_pairs']:,}"],
        ["进入训练图的蛋白", f"{dual_asset['n_train_proteins']:,}"],
        ["稀疏支持矩阵", f"{dual_asset['support_shape'][0]} × {dual_asset['support_shape'][1]}"],
        ["非零项", f"{dual_asset['support_nnz']:,}"],
        ["空反应行", str(dual_asset['support_zero_rows'])],
    ]
    parts.append("### 12.7 生产稀疏资产\n\n" + md_table(["项目", "值"], asset_rows) + "\n")

    parts.append(heading("十三、双核路线的开发、冻结与独立确认", 2))
    confirm_rows = [
        ["开发 9 cells（选参）", "—", pct(dual_route["selected_development"]["hit_at_20"]), "—", f"MRR {dual_route['selected_development']['mrr']:.3f}"],
        ["原冻结 16 cells", dual_route["frozen"]["n"], pct(dual_route["frozen"]["selected_hit"]), pct(dual_route["frozen"]["production_hit"]), f"{pp(dual_route['frozen']['difference'])}; CI [{pp(dual_route['frozen']['bootstrap_ci_low'])}, {pp(dual_route['frozen']['bootstrap_ci_high'])}]"],
        ["独立锁定 fold seed 20260726", dual_confirm["n_query_cells"], pct(dual_confirm["fused_hit"]), pct(dual_confirm["production_hit"]), f"{pp(dual_confirm['difference'])}; CI [{pp(dual_confirm['bootstrap_ci_low'])}, {pp(dual_confirm['bootstrap_ci_high'])}]"],
    ]
    parts.append(md_table(["阶段", "Query-cells", "融合 Hit@20", "原生产 Hit@20", "差值/诊断"], confirm_rows) + "\n")
    parts.append(
        f"""独立确认中新增命中 {dual_confirm['new_hits']} 个、丢失 {dual_confirm['lost_hits']} 个，净增加 {dual_confirm['new_hits'] - dual_confirm['lost_hits']} 个 query-cell；MRR 从 {dual_confirm['production_mrr']:.4f} 提升到 {dual_confirm['fused_mrr']:.4f}。置信区间下界大于 0，说明提升不只是单个切分上的偶然方向。

原冻结切分的区间跨 0，因此当时没有直接生产化，而是继续做独立确认。这一决策过程本身比最终高分更重要：开发集高分不足以部署，冻结集小幅正向也不足以声称稳定成功。
"""
    )

    parts.append(heading("十四、可靠性校准：它是什么、又不是什么", 2))
    rel_rows: list[list[object]] = []
    for row in calibrators.itertuples(index=False):
        rel_rows.append([
            row.calibrator,
            "是" if bool(row.deployable) else "否",
            f"{float(row.roc_auc):.3f}",
            f"[{float(row.roc_auc_ci_low):.3f}, {float(row.roc_auc_ci_high):.3f}]",
            f"{float(row.average_precision):.3f}",
            pct(float(row.base_hit_rate)),
            f"{float(row.brier_score):.3f}",
        ])
    parts.append(md_table(["校准器", "部署", "ROC-AUC", "95% CI", "AP", "Base hit", "Brier"], rel_rows) + "\n")
    parts.append(
        f"""E2R Top-20 新校准器 ROC-AUC 为 {top20_cal['roc_auc']:.3f}，bootstrap 95% CI 为 [{top20_cal['roc_auc_ci_low']:.3f}, {top20_cal['roc_auc_ci_high']:.3f}]。最高可靠性四分位的 Hit@20 为 65.7%，总体为 {pct(top20_cal['base_hit_rate'])}。

可靠性分数表达“这次排序是否像严格双冷中较容易成功的查询”，不是候选酶的生化活性概率。它主要使用查询与训练库的最近相似性、三 seed 排名一致性、Top-K Jaccard、边界 margin 等特征。当前校准器是保守的开放外推可靠性校准，不能解释为 homolog-visible 或 few-shot 场景的实际成功概率；后续若要估计同源扩展成功率，应单独建立对应协议的校准器。只有 ROC-AUC bootstrap 下界超过 0.5 的校准器才允许部署。

生产合并采用最小替换：只替换 `enzyme_to_reaction_top20`，其余五个校准器均断言保持不变。替换了 268 条对应严格双冷样本和 4 条 selective-performance 分位统计。
"""
    )

    parts.append(heading("十五、生产代码接入与边界控制", 2))
    parts.append(
        """### 15.1 单查询入口

`rank_open_world.py rank-reactions` 在满足以下全部条件时启用双核 Top-20：

- objective 为 Top-20；
- 查询酶不是 current-library 实体；
- 没有 few-shot reaction seeds；
- `retrieval_mode=auto`；
- 未手动覆盖 model directory；
- 使用默认 E2R 生产模型；
- 没有临时外部反应；
- 使用默认注册反应表。

任何一项不满足都回退旧路线。Top-3、Top-10、R2E、current entity、few-shot 和 manual override 不受影响。

### 15.2 屏蔽与种子的接口分离

旧公共接口把 `known-reaction-ids` 同时当作“few-shot 种子”和“需要从输出中屏蔽的已知标签”。这会让注册表零样本查询错误地绕开自动路由。现在：

- `--known-reaction-ids`：作为 few-shot seed，并屏蔽输出；
- `--mask-reaction-ids`：只屏蔽，不提供 seed。

### 15.3 单查询—批处理一致性

对注册酶 `7S5L_A`，公共单查询与向量化批处理输出的 Top-20 候选、名次和浮点 RRF 分数完全一致，11 条已知关联均被屏蔽。

### 15.4 部署包

- 五个神经模型部署目录：全部 `valid`；
- 双核稀疏资产：`valid`；
- 候选反应集合按 ID 对齐，允许注册外部反应在文件中的顺序不同；
- 双核运行时明确排除 exact query self-neighbor；
- 13,880 条 E2R Top-20 注册表行全部记录正确 `score_source` 与 `auxiliary_score_directory`。
"""
    )

    parts.append(heading("十六、注册表更新的真实影响", 2))
    turnover = registry_audit["e2r_top20_turnover"]
    conc_old = turnover["old_concentration"]
    conc_new = turnover["new_concentration"]
    turnover_rows = [
        ["查询数", turnover["n_queries"]],
        ["平均保留旧 Top-20 候选", f"{turnover['mean_overlap']:.2f}/20"],
        ["中位保留候选", f"{turnover['median_overlap']:.0f}/20"],
        ["最小保留候选", f"{turnover['minimum_overlap']}/20"],
        ["平均新增候选", f"{turnover['mean_new_candidates']:.2f}"],
        ["平均 Jaccard", f"{turnover['mean_jaccard']:.3f}"],
        ["Top-1 改变", f"{turnover['top1_changed_queries']} ({pct(turnover['top1_changed_fraction'])})"],
        ["完全未变查询", turnover["queries_with_no_candidate_change"]],
        ["最大 Top-1 候选份额", f"{pct(conc_old['top1_top_candidate_share'])} → {pct(conc_new['top1_top_candidate_share'])}"],
        ["有效 Top-1 候选数", f"{conc_old['effective_top1_candidates']:.1f} → {conc_new['effective_top1_candidates']:.1f}"],
        ["不同 Top-1 候选数", f"{conc_old['unique_top1']} → {conc_new['unique_top1']}"],
        ["外部反应成为 Top-1 的比例", f"{pct(conc_old['top1_external_share'])} → {pct(conc_new['top1_external_share'])}"],
    ]
    parts.append(md_table(["审计项", "结果"], turnover_rows) + "\n")
    parts.append(
        """最终注册表采用“旧 Top-3/10 原样保留 + 新 Top-20 合入”的最小变更策略。R2E 三档和 E2R Top-3/10 的 candidate/rank/score 均逐字节相同。Top-20 的变化不是简单把同一个热门反应推给所有酶：最大 Top-1 候选份额下降，有效 Top-1 候选数增加，说明候选分布更分散而非更集中。30,822 行排名的已知关联泄漏为 0。
"""
    )

    parts.append(heading("十七、完整探索实验：成功、失败与它们分别说明什么", 2))
    parts.append(
        """本节刻意保留失败结果。开发集改善但冻结失败的路线，不应从记录中删除；它们揭示了任务瓶颈，也防止未来重复踩坑。

### 17.1 催化 motif 通道

**假设**：DDxxD、NSE/DTE、DxDD、QW 等局部上下文比全局 ESM-C 更直接反映 TPS 机制。

**实现**：对 2,085 条蛋白提取 motif 周围 ±24 aa 的 ESM-C 上下文，加上 motif presence、距离和序列描述符，形成 5,774 维局部表示。

**结果**：独立 motif 模型在冻结 current-only 上 Hit@3/5/10/20 仅约 1.06/1.27/1.91/4.67%，不能承担全库召回。开发阶段 15% motif RRF 曾把 Top-20 从 19.34% 提到 20.99%，但冻结确认不稳定。

**解释**：催化 motif 能帮助局部辨别，却无法单独编码产物骨架特异性；许多相关蛋白共享相似 motif。它适合作为人工解释或深预算救援，而不是主召回器。

### 17.2 P2Rank / pocket-local 通道

**假设**：局部口袋序列比全序列更接近底物与产物选择性。

**结果**：独立口袋模型冻结 Hit@10 约 2.55%、Hit@20 约 4.67%。开发 RRF 曾让 Top-10 从 11.52% 到 12.35%，但冻结不成立。

**解释**：口袋预测误差、结构覆盖不完整、局部片段对远程构象和整体折叠信息缺失，使其难以跨蛋白簇泛化。真正有效的口袋重排可能需要高质量结构、配体姿态与大量配对监督。

### 17.3 同前体、不同骨架的结构化难负例

**假设**：TPS 最难的错误不是任意负例，而是同 GPP/FPP/GGPP 前体、不同产物骨架的近邻蛋白。

**实现**：构造同前体、不同骨架、不同 50% 蛋白簇且非已知正例的 pairwise 三元组；一个开发折中 1,058 个训练正 pair 可生成 8,372 个安全三元组。

**结果**：开发 Top-5/10 有小幅提升，冻结 specialized 模型 Hit@3/5/10/20 为 3.82/5.31/9.55/18.47%，没有超过锁定基线。

**解释**：难负例语义正确，但现有骨架标签和局部模型不足以稳定学习未见簇选择性。

### 17.4 粗骨架、Morgan 化学簇与碳连接图监督

依次测试了：粗粒度 `C15_bicyclic_hydrocarbon` 等标签、Morgan 反应簇、去杂原子和弱化键级后的碳连接图签名。标签从 57 个粗类细化到 289 个反应化学簇或约 205 种碳骨架。

开发集上偶尔增加 1–3 个命中，但冻结均选择回到 scale 0。结论不是“碳骨架没有意义”，而是现有样本量不足以让一个新参数化局部模型在双冷环境中泛化。

### 17.5 两阶段 motif residual reranker

**思路**：主模型先召回 Top-100，motif 模型只纠正真实假阳性，不承担全库检索。

开发 scale 0.05 时 Hit@10/20 为 11.93/20.16%，MRR 0.0509；冻结最优 scale 退回 0，说明开发收益不可迁移。

### 17.6 两阶段 carbon-graph residual

开发 scale 0.025 时 Hit@10/20 为 12.35/20.16%，但冻结同样退回 scale 0。

### 17.7 Pfam 精确架构辅助

current 专用 Pfam v2 实际覆盖 1,332/1,391，不是早先误用 MARTS 表时看到的 628。完整 Pfam combination 以 5% 权重重排，在冻结 Top-10 从 11.89% 提到 12.31%，新增 3、丢 1。这是小但真实的辅助增益。

### 17.8 层次化 Pfam：单域 + 完整组合

开发 Top-10 可到 12.76%，冻结却降到 11.68%，新增 7、丢 8。单个 Pfam 域跨组合共享会把功能不同但共享域的蛋白混在一起，因此淘汰。

### 17.9 current-only 双核与三源融合

current-only 双核在开发 Top-3/5 曾明显升高，但冻结不复现；Pfam、双核与 8-expert 三源配额开发可到 13.99%，冻结降到 11.46%。这说明“互补命中”并不自动等于存在稳定的固定融合规则。

### 17.10 为什么 MARTS E2R Top-20 双核最终成功

与失败路线相比，它具备四个差异：

1. 不学习新的局部参数，降低小样本过拟合；
2. 同时利用反应和蛋白两侧邻域，而非单侧传播；
3. 只在 Top-20 使用，不承担早期精排；
4. 经过原冻结和全新 fold seed 两次确认，并在独立切分上得到正的置信区间下界。
"""
    )

    parts.append(heading("十八、Few-shot：同源扩展与远缘扩展是两项并列能力", 2))
    fewshot_rows = [
        ["Random-positive，1 seed，3-mer max", "73.7%", "实际同源/近同源扩展"],
        ["Random-positive，2 seeds，3-mer max", "82.8%", "实际同源/近同源扩展"],
        ["Random-positive，3 seeds，3-mer max", "87.1%", "实际同源/近同源扩展"],
        ["Random-positive，5 seeds，3-mer max", "92.8%", "实际同源/近同源扩展"],
        ["Protein-cluster-cold，1 seed，ESM-C centroid", "27.6%", "seed/hidden 跨 50% 簇"],
        ["Protein-cluster-cold，2 seeds，ESM-C centroid", "25.5%", "远缘家族扩展"],
        ["Protein-cluster-cold，3 seeds，ESM-C centroid", "29.6%", "远缘家族扩展"],
        ["Protein-cluster-cold，5 seeds，ESM-C centroid", "27.0%", "远缘家族扩展"],
        ["MARTS external reaction，1 seed，ESM-C max", "50.9%", "外部库实用 few-shot；不强制跨簇"],
        ["MARTS external reaction，2 seeds，ESM-C max", "60.4%", "外部库实用 few-shot"],
        ["MARTS external reaction，3 seeds，ESM-C max", "59.3%", "外部库实用 few-shot"],
    ]
    parts.append(md_table(["协议与方法", "Hit@10", "对应能力"], fewshot_rows) + "\n")
    parts.append(
        """Random-positive 的 73%–93% 不是需要被“纠正掉”的虚高数字，而是系统在已有阳性 seed、允许寻找同源替代酶时的正式实用指标。对于这一任务，把近同源候选排除反而会改变用户需求。

Protein-cluster-cold 回答另一件事：当同一 50% 序列簇内的候选不可用时，系统还能否找到远缘家族。它是探索能力、家族多样性和鲁棒性指标，不能替代同源扩展指标。

两类结果都应进入报告和实验设计：若目标是尽快获得阳性，优先使用同源扩展；若目标是发现新家族、提高机制多样性或避免知识产权重叠，则为跨簇候选保留专门配额。种子增加在跨簇场景中也不保证单调改善，因为多个 seed 可能集中在同一偏置子家族。
"""
    )

    parts.append(heading("十九、CAGE 与结构证据的当前定位", 2))
    parts.append(
        """CAGE/结构证据不再参与生产主排序。当前结论是：

- raw CAGE sigmoid 分数大量饱和或并列；
- 直接 probability fusion 会制造伪精细顺序；
- 旧 RF/HGB CAGE-aware rescue 尚未在 train-only reservoir、raw logit、tie-aware、nested selection 的严格 25-cell 双冷下完整重建；
- 当前服务器缺少旧报告对应的正式 meta-ranker 输出目录，不能把历史数字当作可重现生产资产。

CAGE 仍可用于：

- 结构证据展示；
- 主模型与结构模型 disagreement 的人工复核；
- 非经典 TPS 或边界候选的解释；
- 将来有足够严格训练数据时的局部 learned rescue。
"""
    )

    parts.append(heading("二十、UniProt 扩展：为什么不能把 5,672 条序列直接混入主排名", 2))
    uni_rows = [
        ["Top-3", "3 + 0", pct(uniprot["selected_quota"][0]["hit_retention_fraction"]), pct(uniprot["free_merge_paired_retention"][0]["hit_retention_fraction"])],
        ["Top-10", "9 + 1", pct(uniprot["selected_quota"][1]["hit_retention_fraction"]), pct(uniprot["free_merge_paired_retention"][1]["hit_retention_fraction"])],
        ["Top-20", "18 + 2", pct(uniprot["selected_quota"][2]["hit_retention_fraction"]), pct(uniprot["free_merge_paired_retention"][2]["hit_retention_fraction"])],
    ]
    parts.append(md_table(["预算", "canonical + UniProt", "受控保留率", "自由合并保留率"], uni_rows) + "\n")
    parts.append(
        """自由合并失败的原因不是 UniProt 序列一定差，而是它们在 benchmark 中大多没有标签。新增 5,672 个候选后，许多未标注候选会挤占已知正例名次；评测无法判断这些新候选是真阳性还是噪声。因此部署采用 canonical prefix + tail quota，保证主排名的已验证能力不被大规模未标注库吞没。

证据层：

- A：reviewed；
- B：实验或转录层面、具名称；
- C：同源推断、具名称；
- D：预测、具名称；
- E：仅结构域、未表征，暂不启用。
"""
    )

    parts.append(heading("二十一、Reaction-specific Pfam architecture contract", 2))
    parts.append(
        f"""240 个注册反应中，{uniprot['architecture_contracts']['rescue_supported_reactions']} 个支持五-Pfam rescue，{uniprot['architecture_contracts']['unsupported_or_unresolved_reactions']} 个保持 canonical-only。支持状态包括 {uniprot['architecture_contracts']['contract_status']['reference_supported']} 个 reference-supported 和 {uniprot['architecture_contracts']['contract_status']['multi_architecture_reference']} 个 multi-architecture 反应。

架构合同不是根据“C10/C15/C20”粗略猜测，而是由 known-positive accession、exact sequence 和高覆盖 MMseqs nearest neighbor 建立。关键约束包括：

- complete OSC 必须 PF13243 + PF13249；
- PF13243-only / PF13249-only 片段不作为完整 OSC；
- 植物 TPS 完整架构与单域片段分开；
- 未能映射到五个扩展 Pfam 的反应不强行安排 UniProt rescue。
"""
    )

    parts.append(heading("二十二、湿实验执行系统", 2))
    wet_rows = [
        ["板数", wetlab["n_plates"]],
        ["总孔数", wetlab["n_wells"]],
        ["蛋白 assay wells", wetlab["protein_assay_wells"]],
        ["不同反应", wetlab["n_reactions"]],
        ["canonical discovery wells", wetlab["campaign_wells"]["canonical_discovery"]],
        ["UniProt rescue wells", wetlab["campaign_wells"]["uniprot_rescue"]],
        ["候选 ID 构建", wetlab["candidate_id_constructs"]],
        ["exact-sequence 去重构建", wetlab["sequence_deduplicated_constructs"]],
        ["总氨基酸", f"{wetlab['total_amino_acids']:,}"],
        ["总 coding nt（无 stop）", f"{wetlab['total_coding_nucleotides_without_stop']:,}"],
    ]
    parts.append(md_table(["项目", "当前正式值"], wet_rows) + "\n")
    parts.append(
        f"""canonical 四块板包含 {canonical_plate['n_discovery_assays']} 个发现 assay、{canonical_plate['n_positive_control_wells']} 个阳性对照、{canonical_plate['n_empty_vector_negative_wells']} 个空载体阴性和 {canonical_plate['n_substrate_process_blank_wells']} 个底物/流程空白。canonical 去重构建为 {canonical_plate['n_sequence_deduplicated_constructs']} 条。

rescue 两块板覆盖 {rescue['n_reactions']} 个反应、{rescue['n_selected_candidates']} 个候选 assignment、{rescue['n_unique_selected_candidates']} 个不同 UniProt 候选；高置信序列风险为 {rescue['selected_high_confidence_sequence_risks']}。证据层分布为 A={rescue['evidence_tiers'].get('A_reviewed', 0)}、B={rescue['evidence_tiers'].get('B_experimental_or_transcript_named', 0)}、C={rescue['evidence_tiers'].get('C_homology_named', 0)}、D={rescue['evidence_tiers'].get('D_named_predicted', 0)}。

### 22.1 MILP 板间平衡

目标是在每块板容量固定的条件下平衡萜类型、TPS class、底物、对照、候选长度、外部候选比例、证据层和 Pfam 架构。求解器状态为 optimal。

- canonical candidate median-length mean 板间 range：{plate_balance['canonical_discovery']['candidate_median_length_mean']['before_range']:.3f} → {plate_balance['canonical_discovery']['candidate_median_length_mean']['after_range']:.3f} aa；
- canonical q90-length mean range：{plate_balance['canonical_discovery']['candidate_q90_length_mean']['before_range']:.3f} → {plate_balance['canonical_discovery']['candidate_q90_length_mean']['after_range']:.3f} aa；
- rescue median-length mean range：{plate_balance['uniprot_rescue']['candidate_median_length_mean']['before_range']:.3f} → {plate_balance['uniprot_rescue']['candidate_median_length_mean']['after_range']:.3f} aa；
- rescue q90-length mean range：{plate_balance['uniprot_rescue']['candidate_q90_length_mean']['before_range']:.3f} → {plate_balance['uniprot_rescue']['candidate_q90_length_mean']['after_range']:.3f} aa；
- rescue A/B/C/D、bacterial class-I、plant TPS full 和 OSC full 的板间 range 全部降到 0。

### 22.2 Hungarian 孔位随机化

在不移动阳性对照、空载体和流程空白的前提下，对候选角色与相对孔位做顺序化 Hungarian 分配：

- mean normalized slot entropy：{randomization['mean_normalized_entropy_before']:.3f} → {randomization['mean_normalized_entropy_after']:.3f}；
- maximum single-slot role share：{pct(randomization['maximum_slot_share_before'])} → {pct(randomization['maximum_slot_share_after'])}；
- maximum role slot-count range：{randomization['maximum_role_slot_count_range_before']} → {randomization['maximum_role_slot_count_range_after']}；
- control / blank moved：{randomization['control_and_blank_wells_moved']}。

这些设计降低了行列效应、边缘效应和角色—孔位混杂，但不能消除批次、表达量、底物稳定性和检测灵敏度差异。
"""
    )

    parts.append(heading("二十三、历史数字纠错与版本关系", 2))
    version_rows = [
        ["六块板不同反应数", "早期总结 29", f"当前正式 {wetlab['n_reactions']}", "以 combined campaign summary 为准"],
        ["主构建去重数", "早期 348", f"当前正式 {wetlab['sequence_deduplicated_constructs']}", "以 master constructs 为准"],
        ["rescue 平均长度 range", "早期 1.5 aa", f"当前正式 {plate_balance['uniprot_rescue']['candidate_median_length_mean']['after_range']:.3f} aa", "Pfam 精确平衡修复后"],
        ["E2R Top-20", "旧生产 32.5%/34.0%", f"当前严格校准 {pct(top20_cal['base_hit_rate'])}; 独立确认 {pct(dual_confirm['fused_hit'])}", "双核 RRF 后"],
        ["测试数", "旧 39 passed", "当前 79 passed", "新增路由、资产和一致性测试"],
    ]
    parts.append(md_table(["项目", "历史记录", "当前正式值", "说明"], version_rows) + "\n")
    parts.append(
        """旧文件保留用于 provenance，不应删除或悄悄覆盖。本文与 `results/terpene_research_iteration_report.md`、`results/terpene_wetlab_execution_report.md` 和当前 JSON summaries 构成现行口径。
"""
    )

    parts.append(heading("二十四、指标完整解释", 2))
    metric_rows = [
        ["Rank", "候选在排序中的位置；1 最好", "越小越好"],
        ["Best positive rank", "一个查询有多个正例时，最靠前正例的名次", "越小越好"],
        ["Hit@K", "Top-K 中至少出现一个已知正例的查询比例", "回答‘给 K 个实验名额，是否至少抓到一个’"],
        ["Hits@K", "Top-K 中正例的数量", "多正例查询可大于 1"],
        ["Expected hits@K", "跨查询平均的 Hits@K", "不是概率"],
        ["Precision@K", "Hits@K / K", "候选列表中已知正例密度"],
        ["Positive recall@K", "Hits@K / 查询已知正例数", "多正例覆盖率"],
        ["MRR", "平均 1 / best-positive-rank", "对非常早的正例敏感，越大越好"],
        ["Median best rank", "best-positive-rank 的中位数", "比均值更抗长尾"],
        ["ROC-AUC", "可靠性分数区分‘命中/未命中查询’的排序能力", "0.5 约等于随机；不是活性率"],
        ["Average Precision", "沿可靠性阈值的 precision-recall 加权面积", "适合命中较稀少时"],
        ["Brier score", "预测概率与 0/1 结果的均方误差", "越低越好；受 base rate 影响"],
        ["Bootstrap 95% CI", "按 query-cell 有放回重采样得到差值区间", "反映抽样不确定性"],
        ["Percentage point", "两个百分比直接相减", "34%→43% 是 +9 pp，不是 +9%"],
        ["Relative improvement", "(新-旧)/旧", "同一变化可对应更大的相对百分比"],
        ["New hits", "新路线命中而旧路线未命中的查询数", "体现新增覆盖"],
        ["Lost hits", "旧路线命中而新路线未命中的查询数", "体现替换风险"],
        ["Jaccard", "两个候选集合交集/并集", "1 表示完全相同"],
        ["Top-1 candidate share", "最常见 Top-1 候选占全部查询比例", "过高可能是 hub"],
        ["Effective Top-1 candidates", "exp(Top-1 分布 Shannon entropy)", "等效均匀候选数，越大越分散"],
        ["Normalized slot entropy", "孔位分布 Shannon entropy / log2(可用槽数)", "0 集中，1 近均匀"],
        ["Candidate length range", "不同板的统计量最大值减最小值", "越小表示板间更平衡"],
        ["Reliability tier", "基于严格双冷校准的排序证据层", "不是生化结论"],
    ]
    parts.append(md_table(["指标", "定义", "如何理解"], metric_rows) + "\n")
    parts.append(
        r"""### 24.1 一个 Hit@20 的例子

假设有 100 个查询，每个输出 20 个候选，其中 39 个查询的 Top-20 至少包含一个已知正例，则 Hit@20=39%。这不表示 2,000 个候选中 39% 有活性，也不表示每个查询有 39% 成功概率。

### 24.2 MRR 的例子

若三个查询最早正例名次分别为 1、10、100，则 MRR=(1+0.1+0.01)/3=0.37。它会强烈奖励 rank 1，因此可能与 Hit@20 的最优路线不同。

### 24.3 置信区间为什么重要

原冻结 Top-20 提升 +5.23 pp，但区间跨 0，所以不能排除该切分上偶然波动；独立确认 +8.60 pp，区间下界 +5.02 pp，证据更强。开发集只报告高分而不报告确认，会系统性夸大效果。
"""
    )

    parts.append(heading("二十五、名词与缩写解释", 2))
    glossary_rows = [
        ["TPS", "Terpene synthase，萜类合酶/环化酶相关集合"],
        ["R2E", "Reaction to Enzyme，给定反应找酶"],
        ["E2R", "Enzyme to Reaction，给定酶找反应"],
        ["Rhea", "标准化生化反应知识库及其反应 ID"],
        ["MARTS", "本项目使用的外部萜类反应—酶关联数据源"],
        ["UniProt", "蛋白序列与注释数据库；本项目只受控使用扩展层"],
        ["ESM-C", "蛋白语言模型；这里使用 600M mean embedding"],
        ["DRFP", "Differential Reaction Fingerprint，反应差分指纹"],
        ["Dual tower", "蛋白与反应分别编码到共享向量空间的模型"],
        ["InfoNCE", "基于正对与对比负例的表示学习损失"],
        ["Multi-positive", "一个查询允许多个正例，而非强制一一配对"],
        ["PU learning", "Positive–Unlabeled；未标注不等于负例"],
        ["Cold-start", "泛称查询实体或实体簇在训练中未见；必须继续说明是 exact 还是 cluster 层级"],
        ["Exact-entity holdout", "测试蛋白或反应本身及其全部标签未见，但同簇同源物/相似反应可以保留"],
        ["Homolog-visible", "测试 exact 蛋白未见，但训练中实际存在同一 50% identity cluster 的其他蛋白"],
        ["Cluster-cold", "整个蛋白相似簇或反应化学簇在训练中未见"],
        ["Double-cold", "查询蛋白簇与反应簇同时未见"],
        ["Cluster", "相似实体的组；蛋白使用 50% identity，反应使用化学相似簇"],
        ["Fold", "交叉验证拆分的一部分"],
        ["Query-cell", "蛋白 fold × 反应 fold 的一个查询统计单元"],
        ["Zero-shot", "没有查询已知阳性 seed 的排序"],
        ["Few-shot", "给定少量已知阳性 seed 的扩展"],
        ["RRF", "Reciprocal Rank Fusion，倒数名次融合"],
        ["Kernel", "相似度函数；双核分别作用于反应和蛋白"],
        ["Collaborative support", "通过训练关联图把两侧相似性联合传播"],
        ["Degree normalization", "降低高关联度节点的天然优势"],
        ["Temperature", "softmax 集中程度；越小越偏向最相似邻居"],
        ["Hard negative", "与正例相近但应区分的负例"],
        ["Motif", "催化相关短序列模式，如 DDxxD"],
        ["Pfam", "蛋白家族/结构域 HMM 数据库"],
        ["Architecture", "一个蛋白包含哪些结构域及其组合"],
        ["CAGE", "旧方案使用的结构/配对辅助分数体系"],
        ["Reservoir", "高召回候选池，不等同最终 Top-K"],
        ["Canonical ranking", "受严格验证的主候选排名"],
        ["Rescue slot", "主排名末尾为外部证据源保留的少量位置"],
        ["MILP", "Mixed-Integer Linear Programming，混合整数线性规划"],
        ["Hungarian algorithm", "线性分配算法，用于候选—孔位匹配"],
        ["Hub", "在许多查询中反复占据高位的候选"],
        ["Calibration", "把诊断特征映射为可解释的经验可靠性分数"],
        ["Abstention", "证据不足时拒绝自动接受，而非强行给高置信结论"],
        ["Exact sequence deduplication", "氨基酸序列完全相同的构建只合成一次"],
        ["Codon optimization", "按表达宿主调整编码序列；本项目尚未执行"],
    ]
    parts.append(md_table(["名词", "解释"], glossary_rows) + "\n")

    parts.append(heading("二十六、统计与方法学上的威胁", 2))
    parts.append(
        """1. **标签不完整**：未知关联可能是真阳性，Hit@K 会低估真实发现能力，也会使 UniProt 自由合并看起来更差；
2. **反应簇定义依赖表示与阈值**：不同指纹或阈值会改变 double-cold 难度；
3. **蛋白 50% identity 不是功能边界**：同簇可功能分化，异簇也可同功能；
4. **查询单元相关性**：query-cell 并非完全独立，bootstrap 区间仍是近似；
5. **超参数探索多重比较**：大量开发实验会产生偶然高点，因此必须依赖冻结与独立确认；
6. **生产候选分布漂移**：未来加入新反应或新蛋白后，双核资产、校准器和集中度都需重建；
7. **ESM-C 与 DRFP 局限**：序列表示不等于结构与催化轨迹，反应指纹不完整表达立体化学和过渡态；
8. **可靠性不是活性概率**：高可靠查询仍可能因表达、底物、宿主、辅因子或检测失败而无阳性；
9. **湿实验板间平衡不等于无偏**：MILP/Hungarian 只能控制已建模因素；
10. **确认切分数量仍有限**：双核 Top-20 已有正区间确认，但未来新数据外部验证仍最重要。
"""
    )

    parts.append(heading("二十七、从计算结果到湿实验决策的正确方式", 2))
    parts.append(
        """### 27.1 对 R2E 候选板

- 把 Top-3 视为高优先精排；
- Top-10/20 用于扩大机制和序列多样性；
- canonical prefix 不应被大规模未标注 UniProt 自由挤占；
- 对 motif/Pfam/结构证据冲突的候选安排人工复核，而不是自动删除；
- 记录候选来源、序列簇、架构、证据层和表达长度，便于阳性率分层统计。

### 27.2 对 E2R 功能注释

- Top-20 双核路线适合生成验证面板或底物/产物候选集合；
- Top-3 仍应使用早期精度更强的生产路线；
- 有已知 reaction seed 时应进入 few-shot，而不是继续套 zero-shot 校准；
- 已知关联屏蔽后，可靠性标注应明确为不适用，避免把回归审计冒充开放验证。

### 27.3 第一轮实验返回后

需要按以下维度统计：

- route / candidate source；
- canonical vs UniProt；
- 证据层 A/B/C/D；
- Pfam architecture；
- 最近训练蛋白相似度；
- 可靠性 tier；
- 反应前体类别和骨架类别；
- 表达成功、可溶性、底物消耗、目标产物、旁产物。

只有把阴性拆分为“未表达、未检测、底物不兼容、产生其他产物、真正无活性”，才能有效反馈模型。
"""
    )

    parts.append(heading("二十八、下一阶段最值得做的研究", 2))
    next_rows = [
        ["真实湿实验反馈闭环", "最高", "这是解决标签不完整与校准偏差的唯一直接途径"],
        ["按候选来源分层阳性率", "最高", "验证 canonical、MARTS、UniProt 各层真实价值"],
        ["E2R Top-20 外部新批次验证", "高", "确认双核收益跨数据时间与来源迁移"],
        ["R2E Top-10/20 新机制特征", "高", "当前严格 R2E 仍仅 13.5/19.0%"],
        ["结构证据 tie-aware learned rescue", "中高", "需严格 train-only reservoir 与 nested selection"],
        ["反应级不确定性与 coverage-aware abstention", "中", "把低覆盖反应显式拒绝"],
        ["可表达性/构建风险模型", "中", "减少计算高分但实验不可执行的候选"],
        ["动态注册表资产重建", "中", "新增实体后自动重建双核、校准与审计"],
    ]
    parts.append(md_table(["方向", "优先级", "理由"], next_rows) + "\n")

    parts.append(heading("二十九、可复现性、验证与关键文件", 2))
    parts.append(
        """### 29.1 关键生产代码

- `projects/active/terpene_screening/rank_open_world.py`
- `projects/active/terpene_screening/rank_registry_batch.py`
- `projects/active/terpene_screening/dual_kernel_runtime.py`
- `projects/active/terpene_screening/prepare_production_dual_kernel_assets.py`
- `projects/active/terpene_screening/evaluate_open_world_uncertainty.py`
- `projects/active/terpene_screening/validate_dual_kernel_deployment.py`

### 29.2 关键结果

- `results/terpene_marts_dual_kernel_rescue_route_v1/summary.json`
- `results/terpene_marts_dual_kernel_confirmatory20260726/locked_confirmatory_summary.json`
- `results/terpene_production_models/marts_dual_kernel_e2r_top20/`
- `results/terpene_open_world_uncertainty_rrf_routing/`
- `results/terpene_registry_batch/`
- `results/terpene_deployment_validation_e2r_top20_dual_kernel.json`
- `results/terpene_research_iteration_report.md`
- `results/terpene_wetlab_execution_report.md`

### 29.3 测试与静态审计

当前 TPS 测试套件为 **79 passed**。10 条 warning 来自 DRFP 对未来 NumPy int32 越界转换行为的弃用提醒，不是当前测试失败。`git diff --check` 通过。

### 29.4 生产资产重建命令

```bash
.venv/bin/python projects/active/terpene_screening/prepare_production_dual_kernel_assets.py
.venv/bin/python projects/active/terpene_screening/evaluate_open_world_uncertainty.py \
  --output-dir results/terpene_open_world_uncertainty_dual_kernel_candidate_v1 \
  --device cuda
.venv/bin/python projects/active/terpene_screening/rank_registry_batch.py \
  --direction both --objectives 3,10,20 --device cuda \
  --output-dir results/terpene_registry_batch
.venv/bin/python projects/active/terpene_screening/validate_dual_kernel_deployment.py
.venv/bin/pytest -q projects/active/terpene_screening/tests
```

实际生产合并还包含“保留旧 E2R Top-3/10，只替换 Top-20”的最小变更审计，不能简单用全量重跑产物覆盖历史注册表。
"""
    )

    parts.append(heading("三十、最终决策表", 2))
    final_rows = [
        ["旧 gate + reaction similarity", "保留并用于 current-library / 同源可用场景", "数据库补全与高成功率 exploitation 有效；不能单独代表开放泛化"],
        ["旧 CAGE RF/HGB meta-ranker", "不部署", "缺少当前完整可复现严格双冷产物"],
        ["MARTS-adapted dual tower", "部署", "双向主表示和主召回"],
        ["E2R Top-10 neural RRF", "部署", "多个确认切分正向"],
        ["E2R Top-20 dual-kernel RRF", "部署", "独立确认 +8.60 pp，CI 下界 >0"],
        ["Motif-only / pocket-only", "不部署", "全库召回弱"],
        ["Structured skeleton losses", "保留代码与消融", "开发小增益，冻结失败"],
        ["Two-stage motif/carbon residual", "不部署", "冻结选择回到 scale 0"],
        ["Exact Pfam combination auxiliary", "小规模辅助证据", "冻结 Top-10 +0.42 pp"],
        ["Hierarchical single-domain Pfam", "淘汰", "冻结负向"],
        ["UniProt free merge", "禁止", "已知命中保留率大幅下降"],
        ["UniProt controlled tail quota", "部署", "3+0、9+1、18+2"],
        ["Reaction-specific architecture contract", "部署", "208 支持、32 canonical-only"],
        ["MILP + Hungarian wetlab layout", "锁定", "板间与孔位混杂显著降低"],
    ]
    parts.append(md_table(["组件", "当前决策", "核心依据"], final_rows) + "\n")

    parts.append(heading("三十一、常见问题", 2))
    parts.append(
        """### Q1：43.4% 是否表示每 20 个候选里有 43.4% 会有活性？

不是。它表示在独立严格双冷 query-cell 中，43.4% 的查询至少有一个已知正例进入 Top-20。

### Q2：为什么总体严格 Top-20 写 39.2%，独立确认又写 43.4%？

39.2% 来自用于生产校准的原严格双冷全体 query-cell；43.4% 来自重新分配簇的独立锁定确认。二者是不同切分上的同一路线。

### Q3：双核是不是标签泄漏？

不是。每个评测 fold 的双核只使用 train folds 的关联图和训练实体邻域；测试 query exact self-neighbor 被排除。生产 discovery 中已知关联也在 RRF 前屏蔽。

### Q4：为什么不用双核替换神经模型？

双核单独在 R2E 和早期 E2R 预算上不稳定；它最适合作为 Top-20 覆盖型互补源。RRF 保留了神经模型的早期排序结构。

### Q5：为什么不把 Pfam 设成硬过滤？

关联数据包含经典 TPS、P450、prenyltransferase 和其他相关功能；全局硬过滤会丢真阳性。Pfam 应反应条件化使用。

### Q6：现有六块板是否已经可以直接订购 DNA？

蛋白序列与去重构建清单已生成，但尚未进行宿主特异 codon optimization。订购前必须锁定表达宿主、载体、标签、起止密码子、信号肽/转运肽处理和合成供应商约束。

### Q7：模型低分的候选是否一定无活性？

不是。标签不完整、反应表示不足和未见机制都会造成漏检。低分只能降低优先级，不能作为生物学否定。

### Q8：下一次新增数据后哪些东西必须重做？

候选注册表、蛋白/反应特征、训练关联图、双核稀疏资产、可靠性校准、候选集中度审计和可能受影响的湿实验候选选择都应重建。
"""
    )

    parts.append(heading("三十二、结论", 2))
    parts.append(
        """旧方案证明了反应相似度和 seed expansion 在数据库补全与同源扩展中的价值；新版没有取消这两项能力，而是在其之外补齐了 reaction-cold、protein-cold、double-cold、外部实体注册、可靠性校准和实验执行链路。系统的正确描述不是“从旧任务升级为双冷任务”，而是从单一口径扩展成覆盖 exploitation 与 exploration 的多场景检索系统。

实际应用中，相似蛋白和相似反应应被主动利用：它们对应更高的阳性获取效率。cluster-cold 与 double-cold 则用于衡量当这些证据不可用时的远缘发现和完整开放外推能力。两类能力必须并列呈现，并在实验面板中分配不同配额。

最重要的新进展不是增加了一个更复杂的神经网络，而是找到了一种低参数、可解释、可稀疏部署的双核协同证据，并通过开发、冻结和独立确认三个阶段把它限制在最适合的 E2R Top-20 预算。生产接入遵循最小变更原则，未授权路线逐字节保持不变，注册表 0 泄漏，可靠性重新校准，六个部署包全部有效。

当前系统已经具备可执行性，但最终科学价值仍取决于真实湿实验。下一阶段应同时统计同源层、模型层和跨簇探索层的阳性率、表达失败和产物偏差，再据此优化不同目标下的候选配额，而不是追求一个脱离使用场景的单一总分。
"""
    )

    report = "\n".join(parts).strip() + "\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(report, encoding="utf-8")
    digest = hashlib.sha256(report.encode("utf-8")).hexdigest()
    summary = {
        "status": "complete",
        "report": str(OUTPUT),
        "report_sha256": digest,
        "characters": len(report),
        "bytes": len(report.encode("utf-8")),
        "sections": 32,
        "current_date": "2026-07-24",
        "protocol_reassessment": {
            "taxonomy_rows": int(len(protocol_taxonomy)),
            "capability_rows": int(len(protocol_capability)),
            "same_model_rows": int(len(protocol_same_model)),
            "fewshot_rows": int(len(protocol_fewshot)),
            "exact_entity_rows": int(len(exact_entity_protocols)),
            "exact_visibility_rows": int(len(exact_entity_visibility)),
            "principle": "homology-enabled practical retrieval and cold-start generalization are separate co-primary tracks",
            "practical_exact_protein_e2r_hit_at_10": 0.7241003271537623,
            "same_cluster_visible_exact_protein_e2r_hit_at_10": 0.8257790368271954,
            "no_same_cluster_exact_protein_e2r_hit_at_10": 0.38388625592417064,
            "protein_cluster_cold_e2r_hit_at_10": 0.36095965103598693,
        },
        "test_status": "79 passed",
        "deployment_status": {
            "neural_packages": 5,
            "dual_kernel_package": "valid",
        },
        "primary_new_result": {
            "direction": "enzyme_to_reaction",
            "budget": 20,
            "route": "rrf_primary0.7_dual_kernel0.3_c60",
            "calibration_hit": float(top20_cal["base_hit_rate"]),
            "independent_confirmation_baseline": dual_confirm["production_hit"],
            "independent_confirmation_fused": dual_confirm["fused_hit"],
            "independent_confirmation_delta": dual_confirm["difference"],
            "bootstrap_ci": [
                dual_confirm["bootstrap_ci_low"],
                dual_confirm["bootstrap_ci_high"],
            ],
        },
        "registry": {
            "e2r_queries": registry["enzyme_to_reaction"]["n_unique_queries"],
            "r2e_queries": registry["reaction_to_enzyme"]["n_unique_queries"],
            "ranking_rows": registry["discovery_audit"]["ranking_rows_checked"],
            "known_association_leaks": registry["discovery_audit"]["known_association_leaks"],
        },
        "wetlab": {
            "plates": wetlab["n_plates"],
            "wells": wetlab["n_wells"],
            "reactions": wetlab["n_reactions"],
            "sequence_deduplicated_constructs": wetlab[
                "sequence_deduplicated_constructs"
            ],
        },
        "source_artifacts": [
            "docs/terpene_retrieval_protocol_reassessment_zh.md",
            "results/terpene_protocol_reassessment/summary.json",
            "results/terpene_protocol_reassessment/protocol_taxonomy.csv",
            "results/terpene_protocol_reassessment/capability_spectrum.csv",
            "results/terpene_exact_entity_protocols/metrics.csv",
            "results/terpene_protocol_reassessment/exact_entity_visibility_matrix.csv",
            "docs/terpene_candidate_retrieval_scheme_comparison_metrics.json",
            "results/terpene_research_iteration_summary.json",
            "results/terpene_marts_dual_kernel_rescue_route_v1/summary.json",
            "results/terpene_marts_dual_kernel_confirmatory20260726/locked_confirmatory_summary.json",
            "results/terpene_registry_batch/dual_kernel_top20_change_audit.json",
            "results/terpene_uniprot_expansion_report_summary.json",
            "results/terpene_combined_wetlab_campaign/summary.json",
        ],
    }
    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
