from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results/terpene_protocol_reassessment"
REPORT = ROOT / "docs/terpene_retrieval_protocol_reassessment_zh.md"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * float(x):.{digits}f}%"


def table(headers: list[str], rows: list[list[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" if i == 0 else "---:" for i in range(len(headers))) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    exact = pd.read_csv(
        ROOT / "results/terpene_current_library_dual_fusion_restricted_v1/nested_metrics.csv"
    )
    cold = pd.read_csv(ROOT / "results/terpene_dual_tower_multiview/metrics.csv")
    exact_entity = pd.read_csv(ROOT / "results/terpene_exact_entity_protocols/metrics.csv")
    exact_visibility = pd.read_csv(
        ROOT / "results/terpene_protocol_reassessment/exact_entity_visibility_matrix.csv"
    )
    few = pd.read_csv(ROOT / "results/terpene_sequence_fewshot_strict/metrics.csv")
    marts = pd.read_csv(ROOT / "results/terpene_marts_fewshot_open_world/metrics.csv")

    # Same-model cold matrix: multiview dual tower, current-only, 1,391 proteins / 513 reactions.
    cold_rows: list[dict[str, object]] = []
    for row in cold.itertuples(index=False):
        cold_rows.append(
            {
                "protocol": str(row.scope),
                "direction": str(row.direction),
                "n_queries": int(row.n_queries),
                "hit_at_3": float(row.hit_probability_at_3),
                "hit_at_5": float(row.hit_probability_at_5),
                "hit_at_10": float(row.hit_probability_at_10),
                "hit_at_20": float(row.hit_probability_at_20),
                "mrr": float(row.mean_reciprocal_rank),
                "median_best_positive_rank": float(row.median_best_positive_rank),
            }
        )
    cold_matrix = pd.DataFrame(cold_rows)
    cold_matrix.to_csv(OUT_DIR / "same_model_cold_protocol_matrix.csv", index=False)
    exact_entity.to_csv(OUT_DIR / "exact_entity_protocol_matrix.csv", index=False)

    # Few-shot practical homolog expansion: report explicit methods rather than silently choosing one.
    few_rows: list[dict[str, object]] = []
    for scope in ["random_positive", "protein_cluster_cold"]:
        for m in [1, 2, 3, 5]:
            subset = few[(few.scope == scope) & (few.m == m) & (few.B == 10)]
            for method in ["esmc_max_cosine", "esmc_centroid_cosine", "kmer3_max_jaccard"]:
                hit = subset[subset.method == method]
                if hit.empty:
                    continue
                r = hit.iloc[0]
                few_rows.append(
                    {
                        "protocol": scope,
                        "n_seeds": m,
                        "method": method,
                        "budget": 10,
                        "n_trials": int(r.n_trials),
                        "hit_at_10": float(r.hit_probability),
                        "expected_hits_at_10": float(r.expected_hits),
                        "precision_at_10": float(r.precision),
                        "hidden_recall_at_10": float(r.hidden_recall),
                    }
                )
    few_matrix = pd.DataFrame(few_rows)
    few_matrix.to_csv(OUT_DIR / "fewshot_protocol_matrix.csv", index=False)

    external_rows: list[dict[str, object]] = []
    ext = marts[
        (marts.direction == "reaction_to_enzyme")
        & (marts.category == "external_reaction")
        & (marts.method == "seed_esmc_max")
    ]
    for r in ext.itertuples(index=False):
        external_rows.append(
            {
                "protocol": "marts_external_reaction_fewshot",
                "n_seeds": int(r.m),
                "n_trials": int(r.n_trials),
                "n_unique_queries": int(r.n_unique_queries),
                "hit_at_3": float(r.hit_probability_at_3),
                "hit_at_10": float(r.hit_probability_at_10),
                "hit_at_20": float(r.hit_probability_at_20),
                "mrr": float(r.mean_reciprocal_rank),
            }
        )
    external = pd.DataFrame(external_rows)
    external.to_csv(OUT_DIR / "external_fewshot_matrix.csv", index=False)

    exact_rows = [
        {
            "protocol": "legacy_exact_reaction_nested_current_library",
            "budget": int(r.budget),
            "n_reactions": int(r.n_reactions),
            "hit": float(r.hit_probability),
            "expected_hits": float(r.expected_hits),
            "mrr_within_budget": float(r.mean_reciprocal_rank_within_budget),
        }
        for r in exact.itertuples(index=False)
    ]
    exact_matrix = pd.DataFrame(exact_rows)
    exact_matrix.to_csv(OUT_DIR / "current_library_exact_matrix.csv", index=False)

    # Orthogonal task taxonomy.  Seed availability is a third axis, not part of the
    # reaction-novelty × protein-novelty zero-shot plane.
    taxonomy_rows = [
        {
            "track": "current_exact_completion",
            "seed_status": "zero_shot",
            "reaction_exact_seen_in_fold_training": False,
            "reaction_cluster_may_be_seen": True,
            "positive_protein_cluster_may_be_seen": True,
            "homology_allowed": True,
            "primary_question": "Can a held-out exact reaction ID be completed using related chemistry and known protein families?",
        },
        {
            "track": "protein_exact_holdout",
            "seed_status": "zero_shot",
            "reaction_exact_seen_in_fold_training": True,
            "reaction_cluster_may_be_seen": True,
            "positive_protein_cluster_may_be_seen": True,
            "homology_allowed": True,
            "primary_question": "Can a completely unseen exact protein be annotated or recovered while homologous proteins remain legal evidence?",
        },
        {
            "track": "reaction_exact_holdout",
            "seed_status": "zero_shot",
            "reaction_exact_seen_in_fold_training": False,
            "reaction_cluster_may_be_seen": True,
            "positive_protein_cluster_may_be_seen": True,
            "homology_allowed": True,
            "primary_question": "Can a completely unseen exact reaction be retrieved while chemically similar reactions and known protein families remain legal evidence?",
        },
        {
            "track": "seeded_homolog_expansion",
            "seed_status": "few_shot_1_to_5_positive_enzymes",
            "reaction_exact_seen_in_fold_training": True,
            "reaction_cluster_may_be_seen": True,
            "positive_protein_cluster_may_be_seen": True,
            "homology_allowed": True,
            "primary_question": "Can known positive enzymes be expanded to homologous or near-homologous alternatives?",
        },
        {
            "track": "reaction_cluster_cold",
            "seed_status": "zero_shot",
            "reaction_exact_seen_in_fold_training": False,
            "reaction_cluster_may_be_seen": False,
            "positive_protein_cluster_may_be_seen": True,
            "homology_allowed": True,
            "primary_question": "Can a new reaction family be mapped using the known protein-family space?",
        },
        {
            "track": "protein_cluster_cold",
            "seed_status": "zero_shot",
            "reaction_exact_seen_in_fold_training": True,
            "reaction_cluster_may_be_seen": True,
            "positive_protein_cluster_may_be_seen": False,
            "homology_allowed": False,
            "primary_question": "Can the system find a remote enzyme family or annotate a new protein family within known reaction space?",
        },
        {
            "track": "double_cold",
            "seed_status": "zero_shot",
            "reaction_exact_seen_in_fold_training": False,
            "reaction_cluster_may_be_seen": False,
            "positive_protein_cluster_may_be_seen": False,
            "homology_allowed": False,
            "primary_question": "Can both an unseen reaction family and an unseen protein family be extrapolated simultaneously?",
        },
        {
            "track": "seeded_cross_cluster_expansion",
            "seed_status": "few_shot_1_to_5_positive_enzymes",
            "reaction_exact_seen_in_fold_training": True,
            "reaction_cluster_may_be_seen": True,
            "positive_protein_cluster_may_be_seen": False,
            "homology_allowed": False,
            "primary_question": "Given positive seeds, can hidden positives be recovered outside the seed protein clusters?",
        },
    ]
    taxonomy = pd.DataFrame(taxonomy_rows)
    taxonomy.to_csv(OUT_DIR / "protocol_taxonomy.csv", index=False)

    capability_rows: list[dict[str, object]] = []
    exact_by_budget = {int(r.budget): r for r in exact.itertuples(index=False)}
    capability_rows.append({
        "track": "current_exact_completion", "direction": "reaction_to_enzyme",
        "route": "nested current-library expert + dual-tower fusion",
        "comparison_group": "best_practical_route", "n_queries_or_trials": 513,
        "hit_at_3": float(exact_by_budget[3].hit_probability),
        "hit_at_5": float(exact_by_budget[5].hit_probability),
        "hit_at_10": float(exact_by_budget[10].hit_probability),
        "hit_at_20": float(exact_by_budget[20].hit_probability),
        "interpretation": "Exact reaction ID held out; related reaction clusters and homologous proteins remain legal evidence.",
    })
    for protocol in ["protein_exact", "reaction_exact"]:
        for direction in ["reaction_to_enzyme", "enzyme_to_reaction"]:
            r = exact_entity[(exact_entity.protocol == protocol) & (exact_entity.direction == direction)].iloc[0]
            capability_rows.append({
                "track": protocol, "direction": direction,
                "route": "same current-only multiview dual tower",
                "comparison_group": "same_model_exact_entity_holdout",
                "n_queries_or_trials": int(r.n_query_cells),
                "hit_at_3": float(r.hit_probability_at_3),
                "hit_at_5": float(r.hit_probability_at_5),
                "hit_at_10": float(r.hit_probability_at_10),
                "hit_at_20": float(r.hit_probability_at_20),
                "interpretation": "The exact entity and all its labels are absent, but same-cluster homologs or similar reactions may remain.",
            })
    for scope in ["reaction_cold", "protein_cold", "double_cold"]:
        for direction in ["reaction_to_enzyme", "enzyme_to_reaction"]:
            r = cold_matrix[(cold_matrix.protocol == scope) & (cold_matrix.direction == direction)].iloc[0]
            capability_rows.append({
                "track": scope, "direction": direction, "route": "same current-only multiview dual tower",
                "comparison_group": "same_model_split_ablation", "n_queries_or_trials": int(r.n_queries),
                "hit_at_3": float(r.hit_at_3), "hit_at_5": float(r.hit_at_5),
                "hit_at_10": float(r.hit_at_10), "hit_at_20": float(r.hit_at_20),
                "interpretation": "Only the named novelty axis/axes are isolated; use this group to compare split difficulty.",
            })
    for seeds in [1, 5]:
        r = few_matrix[(few_matrix.protocol == "random_positive") & (few_matrix.n_seeds == seeds) & (few_matrix.method == "kmer3_max_jaccard")].iloc[0]
        capability_rows.append({
            "track": f"seeded_homolog_expansion_{seeds}_seed", "direction": "reaction_to_enzyme",
            "route": "3-mer maximum Jaccard to positive seeds", "comparison_group": "best_practical_route",
            "n_queries_or_trials": int(r.n_trials), "hit_at_3": None, "hit_at_5": None,
            "hit_at_10": float(r.hit_at_10), "hit_at_20": None,
            "interpretation": "Seed and hidden positives may share a protein cluster; this is the practical homolog-expansion track.",
        })
    pd.DataFrame(capability_rows).to_csv(OUT_DIR / "capability_spectrum.csv", index=False)

    # Human-facing tables.
    same_model_rows = []
    scope_name = {
        "reaction_cold": "只隔离反应簇；蛋白空间可复用",
        "protein_cold": "只隔离蛋白簇；反应空间可复用",
        "double_cold": "反应簇与蛋白簇同时隔离",
    }
    direction_name = {
        "reaction_to_enzyme": "R2E：反应找酶",
        "enzyme_to_reaction": "E2R：酶找反应",
    }
    for scope in ["reaction_cold", "protein_cold", "double_cold"]:
        for direction in ["reaction_to_enzyme", "enzyme_to_reaction"]:
            r = cold_matrix[(cold_matrix.protocol == scope) & (cold_matrix.direction == direction)].iloc[0]
            same_model_rows.append(
                [
                    scope_name[scope],
                    direction_name[direction],
                    int(r.n_queries),
                    pct(r.hit_at_3),
                    pct(r.hit_at_5),
                    pct(r.hit_at_10),
                    pct(r.hit_at_20),
                    f"{r.mrr:.3f}",
                ]
            )

    exact_entity_rows = []
    exact_protocol_name = {
        "protein_exact": "Exact 蛋白未见；同簇同源物允许可见",
        "reaction_exact": "Exact 反应未见；同簇相似反应允许可见",
    }
    for protocol in ["protein_exact", "reaction_exact"]:
        for direction in ["reaction_to_enzyme", "enzyme_to_reaction"]:
            r = exact_entity[
                (exact_entity.protocol == protocol) & (exact_entity.direction == direction)
            ].iloc[0]
            exact_entity_rows.append([
                exact_protocol_name[protocol], direction_name[direction], int(r.n_query_cells),
                pct(r.hit_probability_at_3), pct(r.hit_probability_at_5),
                pct(r.hit_probability_at_10), pct(r.hit_probability_at_20),
                f"{r.mean_reciprocal_rank:.3f}",
            ])

    visibility_rows = []
    for protocol in ["protein_exact", "reaction_exact"]:
        for direction in ["reaction_to_enzyme", "enzyme_to_reaction"]:
            for visibility in ["visible", "not_visible"]:
                r = exact_visibility[
                    (exact_visibility.protocol == protocol)
                    & (exact_visibility.direction == direction)
                    & (exact_visibility.same_cluster_evidence == visibility)
                ].iloc[0]
                visibility_rows.append([
                    exact_protocol_name[protocol], direction_name[direction],
                    "有同簇训练邻居" if visibility == "visible" else "无同簇训练邻居",
                    int(r.n_query_cells), pct(r.hit_probability_at_3),
                    pct(r.hit_probability_at_10), pct(r.hit_probability_at_20),
                ])

    exact_table_rows = []
    for r in exact.itertuples(index=False):
        exact_table_rows.append([f"Top-{int(r.budget)}", pct(r.hit_probability), f"{r.expected_hits:.2f}"])

    random_table_rows = []
    for m in [1, 2, 3, 5]:
        esmc = few_matrix[
            (few_matrix.protocol == "random_positive")
            & (few_matrix.n_seeds == m)
            & (few_matrix.method == "esmc_max_cosine")
        ].iloc[0]
        kmer = few_matrix[
            (few_matrix.protocol == "random_positive")
            & (few_matrix.n_seeds == m)
            & (few_matrix.method == "kmer3_max_jaccard")
        ].iloc[0]
        strict = few_matrix[
            (few_matrix.protocol == "protein_cluster_cold")
            & (few_matrix.n_seeds == m)
            & (few_matrix.method == "esmc_centroid_cosine")
        ].iloc[0]
        random_table_rows.append(
            [
                m,
                pct(esmc.hit_at_10),
                pct(kmer.hit_at_10),
                pct(strict.hit_at_10),
            ]
        )

    external_table_rows = [
        [int(r.n_seeds), int(r.n_unique_queries), pct(r.hit_at_3), pct(r.hit_at_10), pct(r.hit_at_20)]
        for r in external.itertuples(index=False)
    ]

    report = f"""# 萜类合酶检索任务与评测协议重新评估报告

## 一、修正后的总原则

**没有理由要求所有评测都同时隔离相似反应簇与 50% 蛋白序列簇。** 是否隔离哪一侧，取决于实际问题中哪些信息本来就是允许使用的。

在真实酶筛选里，找到已知阳性酶的相似蛋白不仅允许，而且通常是成本最低、成功率最高、最应该优先利用的证据。若任务是“已有一个催化该反应的酶，继续寻找可替代同源物”，把所有近同源蛋白强制排除，会改变任务本身，而不是让原任务变得更公平。

另一方面，若研究问题是“模型能否找到远缘酶家族”或“新反应家族与新蛋白家族同时出现时能否泛化”，就必须使用 protein-cluster-cold 或 double-cold。它们是**外推能力测试和压力测试**，不是所有生产任务的唯一主指标。

因此，本项目以后采用**多轨评测**：

1. 数据库内关联补全；
2. 允许同源的 seeded homolog expansion；
3. 新反应簇、但可利用已知蛋白空间；
4. 已知反应空间中的远缘蛋白发现；
5. 双侧同时未见的完整开放发现；
6. 外部库 few-shot 扩展。

这些任务必须分别命名、分别报告，不能把一个数字称为整个系统的“总能力”。

## 二、为什么之前的表述不够准确

此前报告过度强调 double-cold，并把 random-positive few-shot 主要描述为“上界”或“同源捷径”。这个说法只对“评估跨家族泛化”成立；对实际同源扩展任务并不成立。

更准确的说法应是：

- random-positive / homolog-visible 指标是**实际同源扩展能力**；
- protein-cluster-cold 是**远缘家族扩展能力**；
- reaction-cluster-cold 是**新化学空间映射到已知蛋白空间的能力**；
- double-cold 是**两侧同时外推的最难压力测试**。

“允许利用同源”与“检验不依赖同源时是否仍能工作”是两个并列问题，不是一个正确、另一个错误。

## 三、三个彼此独立的评测轴

评测不应被压成“普通指标”和“双冷指标”两档，而应先说明三个独立问题：

1. **Seed 轴**：查询时是否已经给出 1–5 个阳性酶；zero-shot 只表示没有 seed，不等于双冷。
2. **反应新颖性轴**：exact reaction ID、相似反应簇或整个反应簇是否在训练中可见。
3. **蛋白新颖性轴**：正确蛋白及其 50% identity cluster 是否可见；允许同源时，这通常是合法生产证据。

因此，reaction-cold × protein-cold 是无 seed 条件下的二维平面，double-cold 只是其中“两个轴都冷”的一个角。Few-shot 是第三个轴，不能被塞进这个二维表里。

### 3.1 无 seed 的二维新颖性矩阵

| 反应侧 | 蛋白同源空间可用 | 正确蛋白簇不可用 |
|---|---|---|
| exact ID 留出，但相似反应簇可用 | Current-library exact completion | Protein-cluster-cold：已知反应空间中的远缘酶发现 |
| 整个反应簇不可用 | Reaction-cluster-cold：新反应映射到已知蛋白家族 | Double-cold：新反应簇 × 新蛋白簇同时外推 |

### 3.2 有 seed 时是另一条轴

- seed 与 hidden positives 可同簇：实际同源扩展；
- seed 与 hidden positives 强制跨簇：远缘扩展；
- 外部反应 + 外部 seed：external few-shot。

## 四、场景定义：到底允许什么信息

| 场景 | 查询条件 | 允许复用相似蛋白 | 允许复用相似/相同反应空间 | 回答的问题 |
|---|---|---:|---:|---|
| Current-library exact-reaction holdout | 该折整条 exact reaction ID 不进入训练；相似反应簇仍可见 | 是 | 是（不含 exact ID） | 数据库内候选补全能做到多好 |
| Seeded homolog expansion | 已知反应且已有 1–5 个阳性酶 | **是，且这是核心证据** | 是 | 能否找到更多同源或近同源可替代酶 |
| Reaction-cluster-cold R2E | 目标反应簇未见，无阳性 seed | 是 | 否 | 新反应能否映射到已有蛋白家族 |
| Protein-cluster-cold R2E | 反应可见，正确蛋白簇未见 | 否 | 是 | 能否找到远缘或不同家族的同功能酶 |
| Protein-cluster-cold E2R | 查询蛋白簇未见，反应目录可见 | 否 | 是 | 新蛋白能否注释到已有反应类别 |
| Reaction-cluster-cold E2R | 查询蛋白可见，正确反应簇未见 | 是 | 否 | 已知/相似酶能否发现新反应或新产物 |
| Double-cold | 两侧簇均未见 | 否 | 否 | 完整开放世界外推能力 |
| MARTS external few-shot | 外部反应，有少量外部阳性 seed | 部分允许 | 外部查询 | 外部库中利用 seed 的实用扩展能力 |

## 五、同一模型、只改变隔离条件

为了说明指标下降究竟由哪一侧隔离造成，下表固定为同一个 current-only multiview dual tower、相同 1,391 条蛋白和 513 个反应，只改变 split。这张表用于比较协议难度。后面的 current exact 与 few-shot 表使用各场景最合适的实用路线，用于描述系统能力，不能拿来宣称同一个模型在不同 split 上提升了多少。

{table(["隔离协议", "方向", "查询数", "Hit@3", "Hit@5", "Hit@10", "Hit@20", "MRR"], same_model_rows)}

这张表说明：

- 对 R2E 而言，允许使用已有蛋白空间、只隔离反应簇时，Top-10 为 **28.7%**；
- 若反应已知，但强制正确蛋白来自未见的 50% 序列簇，Top-10 降为 **18.1%**；
- 两侧同时隔离后，Top-10 才降到 **6.4%**。

因此，把 6.4% 当作“系统找酶的能力”是不准确的。它只代表最难的双侧外推条件。

E2R 同理：

- 查询新蛋白、但候选反应目录与反应类别可见时，Top-10 为 **36.1%**；
- 若要为已知/相似蛋白发现训练中未见的反应簇，Top-10 为 **31.3%**；
- 两侧都未见时为 **14.1%**。

### 5.1 Exact 实体未见，但相似簇允许可见

Cluster-cold 并不是“测试实体未见”的唯一方式。更贴近很多实际任务的协议是：测试蛋白或反应本身及其全部关联完全不进入训练，但同一 50% 蛋白簇中的其他同源物、或同一化学簇中的相似反应仍然允许出现。

{table(["协议", "方向", "查询单元", "Hit@3", "Hit@5", "Hit@10", "Hit@20", "MRR"], exact_entity_rows)}

其中：

- exact 新蛋白做已有反应目录注释（E2R）时，Top-10 为 **72.4%**；
- 已知反应寻找从未进入训练的 exact 新蛋白（R2E）时，Top-10 为 **51.2%**；
- exact 新反应寻找酶（R2E）时，Top-10 为 **38.0%**；
- 为蛋白寻找训练中未出现过的 exact 反应（E2R）时，Top-10 为 **37.9%**。

这些数字较高不是因为测试实体本身泄漏，而是因为同源蛋白或相似反应邻域被合法保留。这正对应实际中“新序列，但数据库已有同源物”或“新 exact 反应记录，但已有相似化学”的场景。

### 5.2 训练中是否真的存在同簇证据

为避免把 singleton 实体也笼统称为“允许同源”，进一步按训练中是否实际存在同簇邻居分层：

{table(["Exact 协议", "方向", "训练中同簇证据", "查询单元", "Hit@3", "Hit@10", "Hit@20"], visibility_rows)}

最关键的是 exact-protein E2R：

- 有同一 50% identity cluster 的训练同源物时，Top-10 为 **82.6%**；
- 没有同簇训练同源物时，Top-10 为 **38.4%**；
- 整个蛋白簇强制未见的 protein-cluster-cold E2R 为 **36.1%**。

R2E 也呈现同样趋势：已知反应寻找 exact 新蛋白时，有同簇训练同源物的 query-cell Top-10 为 **62.6%**，没有时只有 **15.5%**。这证明同源证据本身贡献巨大，而且在实际筛选中应被主动利用。Cluster-cold 用来测量这条证据不可用时的后备能力，不能取代同源可见场景。

## 六、实际数据库补全能力

当前库 exact-reaction 五折嵌套融合的结果如下。每个目标折中的整条 exact reaction ID 不进入该折模型训练，但相似反应簇、候选蛋白及其在其他反应上的信息仍可合法使用。它测的是数据库补全，不是仅隐藏单条 pair，也不是 reaction-cluster-cold。

{table(["预算", "Hit@K", "平均已知正例数/查询进入预算"], exact_table_rows)}

因此，对“在现有 TPS 数据库和已知家族中给一个目标反应补候选”这个问题，正式 Top-10 是 **48.1%**，Top-20 是 **57.5%**，而不是双冷中的个位数或十几个百分点。

## 七、同源扩展与远缘扩展必须并列报告

下表固定 Top-10。random-positive 表示 seed 和隐藏阳性可以属于同一序列家族；protein-cluster-cold 强制隐藏阳性来自其他 50% identity cluster。

{table(["Seed 数", "Random-positive ESM-C max", "Random-positive 3-mer max", "跨50%蛋白簇 ESM-C centroid"], random_table_rows)}

结论：

- 已知 1 个阳性时，简单序列同源扩展 Top-10 约 **72%–74%**；
- 已知 5 个阳性时可达 **92%–93%**；
- 强制跨 50% 序列簇后才降到约 **20%–30%**。

这不是前一种评测“虚高”、后一种才“真实”。它们分别回答：

- **同源扩展**：能否找到 seed 周围可直接实验的替代酶；
- **远缘发现**：当近同源家族不可用时，能否找到结构和序列差异更大的同功能酶。

在湿实验资源有限时，通常应先利用前者，再把一部分预算留给后者增加多样性和知识增量。

## 八、外部 MARTS few-shot

外部反应给定少量阳性 seed、使用 ESM-C 最大相似度扩展：

{table(["Seed 数", "不同外部反应", "Hit@3", "Hit@10", "Hit@20"], external_table_rows)}

这一协议比 current random-positive 更开放，但没有强制 seed 与所有 hidden positives 跨 50% identity cluster，因此它应单独命名为 **external few-shot**，不能冒充严格跨簇，也不能被 double-cold 替代。

## 九、以后应该怎样选择主指标

### 9.1 已有目标反应和阳性 seed

主指标：random-positive / family-stratified few-shot、Top-3/10/20、hidden-positive recall。相似蛋白完全允许。

补充指标：protein-cluster-cold，用于判断是否能覆盖远缘家族。

### 9.2 目标反应已知，但没有阳性 seed

需要先判断“新”到哪一层：

- 若只是一个从未进入训练的 exact reaction ID，但数据库中存在同簇相似反应，主指标使用 **exact-reaction holdout R2E**；
- 若整个反应化学簇都未见，但允许利用已有蛋白家族，主指标使用 **reaction-cluster-cold R2E**；
- 若正确蛋白家族也必须未见，才进一步使用 **double-cold R2E**。

这三者不是同一指标的宽松版和严格版，而是三个不同的生产条件。

### 9.3 给一个新蛋白做功能注释

若候选反应目录已知，并且现实中允许利用数据库同源物，主指标应是 **exact-protein holdout E2R**：测试蛋白本身和全部标签未见，但同簇同源物可以提供合法证据。本次该指标 Top-10 为 72.4%，训练中实际存在同簇同源物时为 82.6%。

**Protein-cluster-cold E2R** 是补充的远缘泛化指标，用于回答“整个同源簇都没有训练证据时还能否注释”，本次 Top-10 为 36.1%。

若同时要求发现训练中未见的反应簇，才使用 **double-cold E2R**。

### 9.4 论文或模型方法学比较

至少同时报告：

- current-library association completion；
- exact-protein 与 exact-reaction holdout；
- reaction-cluster-cold；
- protein-cluster-cold；
- double-cold；
- seeded homolog expansion；
- seeded cross-cluster expansion。

任何方法都不应只挑最有利的一列作为总成绩。

## 十、生产策略应如何对应这些能力

生产排名不应因为双冷是更难的 benchmark，就主动禁止同源证据。正确的分层策略是：

1. **Exploitation / 高成功率层**：seed sequence similarity、已知家族、reaction similarity；
2. **Model-mediated 层**：双塔、方向专用模型、RRF；
3. **Exploration / 新颖性层**：protein-cluster-diverse、reaction-cluster-diverse、dual-kernel 或架构差异候选；
4. 在一个实验面板内显式分配配额，而不是让一种目标覆盖全部候选。

例如 Top-20 面板可以同时保留：

- 8–12 个同源/已知家族高置信候选；
- 4–6 个模型支持但序列较远的候选；
- 2–4 个跨架构或外部库探索候选。

具体比例应根据目标是“尽快拿到阳性”还是“发现新家族”调整。

## 十一、修正后的结论

1. 双冷隔离不是所有任务的必要条件；它只在评估完整开放外推时必要。
2. 相似蛋白是合法且极有价值的生产证据，不应因为方法学压力测试而被从实际筛选中删除。
3. 旧同源扩展 70%–90% 的结果应保留并作为对应使用场景的主指标。
4. protein-cold 和 double-cold 指标用于量化同源证据失效后的后备能力与探索能力。
5. 系统没有一个单一“总准确率”；它有数据库补全、同源扩展、新反应映射、远缘蛋白发现和双冷外推等不同能力。
6. 主报告和后续论文应使用任务矩阵，并明确每个数字允许复用了哪些信息。
"""

    REPORT.write_text(report, encoding="utf-8")
    summary = {
        "status": "complete",
        "principle": "homology-enabled practical retrieval and double-cold generalization are separate co-primary evaluation tracks",
        "report": str(REPORT),
        "report_sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(),
        "same_model_cold_matrix": str(OUT_DIR / "same_model_cold_protocol_matrix.csv"),
        "exact_entity_matrix": str(OUT_DIR / "exact_entity_protocol_matrix.csv"),
        "exact_entity_visibility_matrix": str(OUT_DIR / "exact_entity_visibility_matrix.csv"),
        "fewshot_matrix": str(OUT_DIR / "fewshot_protocol_matrix.csv"),
        "external_fewshot_matrix": str(OUT_DIR / "external_fewshot_matrix.csv"),
        "current_library_exact_matrix": str(OUT_DIR / "current_library_exact_matrix.csv"),
        "protocol_taxonomy": str(OUT_DIR / "protocol_taxonomy.csv"),
        "capability_spectrum": str(OUT_DIR / "capability_spectrum.csv"),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
