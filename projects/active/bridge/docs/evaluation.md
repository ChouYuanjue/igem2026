# BRIDGE 评测说明

## 1. 当前评测规则

当前主评测只控制一件事：**目标酶–反应关系在 `clean2023` 中从未出现过**。酶本身和反应本身可以已经见过，不再额外要求 protein-cold 或 reaction-cold。

七个 fair benchmark 的 train/test 关系合并后共有 242,298 条关系；移除 `clean2023` 已知关系后，得到 23,773 条 relation-unseen 目标关系，覆盖 1,923 个 R2E 查询和 17,486 个 E2R 查询。

训练侧已知关系不会再次作为发现目标。Seed / homology context 只读取训练侧已知阳性；测试目标永远不可作为 seed。

## 2. 历史连续结果

原固定七类综合表继续保留，用于和前期结果对齐：

| 系统 | MRR | Hit@10 | Hit@100 | Hit@1000 |
| --- | ---: | ---: | ---: | ---: |
| Broad Core | 0.2093 | 36.87% | 59.41% | 77.91% |
| BRIDGE 完整系统 | 0.2369 | 41.54% | 64.92% | 77.91% |

当前专家因果分析统一使用第 3 节的最大 relation-unseen 测试面。

## 3. 最大 relation-unseen 双向专家大类消融

### R2E：反应 → 酶

| 系统 | MRR | Hit@10 | Hit@100 | Hit@1000 |
| --- | ---: | ---: | ---: | ---: |
| **完整 BRIDGE** | **0.076723** | **13.365%** | **30.213%** | **47.322%** |
| 去除功能/同源 | 0.070580 | 12.272% | 27.613% | 47.322% |
| 去除结构/机制 | 0.069037 | 12.428% | 28.237% | 47.322% |
| 去除 Domain Specialists | 0.076723 | 13.365% | 30.213% | 47.322% |

功能/同源组贡献：MRR **+0.006143**，Hit@10 **+1.092 pp**。结构/机制组贡献：MRR **+0.007686**，Hit@10 **+0.936 pp**。Domain Specialists 在这一最大综合面上的全局消融差值为 **0**：共有 12 个查询激活专项专家（P450 10、phosphatase 1、TPS 1），但没有改变这些查询的最佳阳性最终排名。专项专家的价值继续由第 8 节的 family 局部测试刻画。

R2E 当前成员：

- 功能/同源：EnzGFM；Seed / homology 作为可选关系上下文单独评测。
- 结构/机制：CLIPZyme、bounded reaction-center、Pocket-Reaction Interaction Expert。
- Domain Specialists：P450 / phosphatase / terpene family CAGE 与 TPS；当前表按整组删除，不逐个删除。

### E2R：酶 → 反应

| 系统 | MRR | Hit@10 | Hit@100 | Hit@1000 |
| --- | ---: | ---: | ---: | ---: |
| **完整系统** | **0.103721** | **19.947%** | **47.221%** | **79.029%** |
| 去除功能/同源 | 0.057857 | 10.769% | 29.927% | 72.121% |
| 去除结构/机制 | 0.096381 | 17.465% | 46.980% | 76.141% |

功能/同源组贡献：MRR **+0.045863**，Hit@10 **+9.179 pp**。结构/机制组贡献：MRR **+0.007339**，Hit@10 **+2.482 pp**。

E2R 当前零样本成员为 EnzGFM 与 CLIPZyme。reaction-center 与 Pocket Interaction 目前只完成了 R2E 方向验证，因此不在 E2R 中强行加入。

## 4. 关系未见前提下的不同干净评测面

最大 relation-unseen 是最宽、最难的主测试面。为了把“广域端到端能力”和更贴近日常数据库扩展的使用情境分开，还可以报告两个不引入关系泄漏的辅助表面。三者都要求目标酶–反应关系从未出现在 `clean2023`。

### 4.1 Query-seen relation-unseen：数据库扩展

这里只额外要求查询端在训练知识中出现过；目标关系仍然完全未见。它模拟“已知一个反应/酶已有若干关联，现在继续发现新的关联”。

| 方向 | 查询数 | MRR | Hit@10 | Hit@100 | Hit@1000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| R2E | 890 | **0.08826** | **14.49%** | **31.24%** | 47.19% |
| E2R | 908 | **0.22742** | **57.38%** | **85.68%** | **96.70%** |

这是最推荐的第二张性能表。它比 maximal relation-unseen 更容易，但没有暴露任何测试关系，也没有重新挑选阳性。

### 4.2 Broad Top1000 conditional reranking：只测重排能力

BRIDGE 的专家层本来工作在 Broad 候选之上，因此还可以条件化在“Broad 已经把至少一个正确目标召回 Top1000”的查询上，单独测重排。这一表面仍然保持目标关系未见，但它是阶段性指标，不能替代端到端结果。

| 方向 | 查询数 | Broad MRR | BRIDGE MRR | Broad Hit@10 | BRIDGE Hit@10 | BRIDGE Hit@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R2E | 910 | 0.13302 | **0.16189** | 22.53% | **28.24%** | **63.85%** |
| E2R | 11,118 | 0.05989 | **0.15519** | 14.27% | **29.62%** | **67.62%** |

R2E 的候选池由 Broad 固定，因此该子面 Hit@1000 保持 100%。E2R 当前专家排序不是严格候选集保持，因此最终 Hit@1000 可低于 100%。

“酶和反应两端实体都见过、只有边未见”的经典 transductive link-prediction 面在当前数据里只有 198 条边（33 个 R2E 查询、88 个 E2R 查询），样本太小，不建议作为 headline。

建议最终报告层次：**最大 relation-unseen 作为主压力测试；query-seen relation-unseen 作为真实数据库扩展面；Broad-covered conditional reranking 作为架构阶段分析。**

## 5. Seed / homology relation-context

Seed 强度在独立开发面冻结：R2E `α=20`，E2R `α=0.75`。最大 relation-unseen 测试不参与强度选择。

| 方向 | 系统 | MRR | Hit@10 | Hit@100 | Hit@1000 |
| --- | --- | ---: | ---: | ---: | ---: |
| R2E | Broad | 0.06306 | 10.66% | 25.74% | 47.32% |
| R2E | Broad + train seed | **0.16899** | **23.09%** | **38.90%** | **55.90%** |
| E2R | Broad | 0.03824 | 9.08% | 24.07% | 63.58% |
| E2R | Broad + train seed | **0.04726** | **10.25%** | **24.96%** | **64.19%** |

有合法训练 seed 的查询中：R2E MRR 0.07198 → **0.30086**；E2R 0.07183 → **0.24559**。没有 seed 时精确回退 Broad。

## 6. Pocket-Reaction Interaction Expert

Pocket 现在作为真正的 pair-level 结构专家使用。输入来自 EnzymeCAGE 在全局 ESM / DRFP 融合之前的 256 维 pocket–substrate 双向 cross-attention 表示；训练使用 relation-level leave-one-positive-out 与 Broad hard negatives。

最终权限固定为：`α=0.35`，**Top20 完全保护**，Pocket 只允许在 rank 21–1000 的可评分候选之间做局部纠正。缺失结构时贡献为 0。

在完整 BRIDGE 中：

| 系统 | MRR | Hit@10 | Hit@100 | Hit@1000 |
| --- | ---: | ---: | ---: | ---: |
| 不含 Pocket Interaction | 0.07671852 | 13.365% | 30.213% | 47.322% |
| **加入 Pocket Interaction** | **0.07672257** | **13.365%** | **30.213%** | **47.322%** |

全局 MRR 增益为 **+0.00000405**，其余主指标不回退。1,396 个查询具备至少两个可评分 pocket 候选，最终只有 11 个查询的最佳阳性排名实际发生变化，其中 7 个改善、4 个变差；这些变化查询的中位排名由 66 提升到 49。

Pocket 的全局增益很小，这是当前 bounded permission 的直接结果。它的价值是提供与整体结构和反应中心不同尺度的局部催化环境证据，同时不破坏 Broad 已经可靠的头部顺序。

## 7. CAGE 分层比较

| 方法 | MRR | Hit@10 | Hit@100 |
| --- | ---: | ---: | ---: |
| 原始 CAGE | 0.100763 | 20.53% | 27.56% |
| Broad Top-1000 + 通用 CAGE | 0.025714 | 5.44% | 18.76% |
| BRIDGE（原固定七类） | **0.236857** | **41.54%** | **64.92%** |

通用 CAGE 不能直接接管 Broad 上层排序。CAGE 的有效使用方式是 family-specific specialist，以及 Pocket Interaction 中被拆出的局部结构交互表示。

## 8. Family 专项专家

family-tuned CAGE 相比 generic CAGE：

- P450：MRR 0.0370 → **0.0705**；
- phosphatase：MRR 0.2522 → **0.3169**；
- terpene：MRR 0.0189 → **0.0387**。

Family/TPS 按适用域激活，在 R2E 大类消融中整体保留。

## 9. 当前冻结入口

当前入口为 `reproducibility/bridge/canonical.json`。核心记录：

- `BRIDGE_BIDIRECTIONAL_CATEGORY_ABLATION_V3_RESULT.json`；
- `BRIDGE_RELATION_SEED_CONTEXT_V3_RESULT.json`；
- `BRIDGE_POCKET_INTERACTION_V1_RESULT.json`；
- `BRIDGE_RELATION_UNSEEN_SURFACES_V1_RESULT.json`。
