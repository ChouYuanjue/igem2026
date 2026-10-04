# BRIDGE 评测说明

## 1. 评测原则

BRIDGE 区分零样本排序、关系上下文与领域专项能力。训练关系与测试目标关系严格分开；外部测试只用于冻结确认，不参与回头调参。

当前关系规则：

- `clean2023` 中的酶–反应对应关系视为训练侧已知知识；
- relation-clean 目标必须是 `clean2023` 中从未出现过的精确对应关系；
- 对当前查询已经在训练库中已知的关系，不再作为待发现候选；
- seed/context 只能来自训练侧已知阳性，测试阳性永远不能作为 seed。

## 2. 原七类综合结果

为保持与前期结果连续，保留原固定七类综合表：

| 系统 | MRR | Hit@10 | Hit@100 | Hit@1000 |
| --- | ---: | ---: | ---: | ---: |
| Broad Core | 0.2093 | 36.87% | 59.41% | 77.91% |
| BRIDGE 完整系统 | 0.2369 | 41.54% | 64.92% | 77.91% |

这张表用于历史连续比较。由于其中部分关系相对当前 `clean2023` 训练源已经可见，新的专家因果分析以第 3 节 relation-clean 结果为准。

## 3. Relation-clean 双向专家大类消融

R2E 中 Family/TPS 专项层保持完整、不逐个消融；E2R 当前没有对应的 Family CAGE/TPS 排序层，不人为补入。seed/homology context 单独在第 4 节评价，不混入零样本表。

### R2E：反应 → 酶

1,165 个唯一查询，1,954 个 query-cell 实例。

| 系统 | MRR | Hit@10 | Hit@100 | Hit@1000 |
| --- | ---: | ---: | ---: | ---: |
| 完整零样本系统 | **0.05402** | **10.90%** | **27.33%** | **45.75%** |
| 去除功能/同源专家 | 0.05032 | 10.08% | 25.23% | 45.75% |
| 去除结构/机制专家 | 0.04842 | 10.34% | 25.95% | 45.75% |

贡献：功能/同源组带来 MRR +0.00370、Hit@10 +0.82 pp；结构/机制组带来 MRR +0.00560、Hit@10 +0.56 pp。Hit@1000 不变，说明两类专家主要改变头部顺序。

R2E 中：功能/同源的零样本成员是 EnzGFM；结构/机制成员是 CLIPZyme + bounded reaction-center。

### E2R：酶 → 反应

4,237 个唯一查询，5,323 个 query-cell 实例。

| 系统 | MRR | Hit@10 | Hit@100 | Hit@1000 |
| --- | ---: | ---: | ---: | ---: |
| 完整零样本系统 | **0.10485** | **24.99%** | 51.04% | **84.60%** |
| 去除功能/同源专家 | 0.05367 | 10.14% | 36.76% | 77.19% |
| 去除结构/机制专家 | 0.09099 | 18.67% | **52.58%** | 82.34% |

功能/同源组带来 MRR +0.05118、Hit@10 +14.84 pp；结构/机制组带来 MRR +0.01387、Hit@10 +6.31 pp。结构证据明显改善头部排序，但 Hit@100 有约 1.54 pp 的交换损失，因此其作用应表述为头部精排收益。

E2R 中：功能/同源的零样本成员是 EnzGFM；结构/机制当前由 CLIPZyme 提供，reaction-center 仅支持 R2E。

## 4. Seed / homology relation-context

Seed 不再从测试阳性中抽取。开发时只使用训练侧关系构建上下文，并在所有 frozen outer-test 关系被提前隔离的 relation-delta 开发面上选择强度；outer test 不参与强度选择。

冻结强度：R2E `α=20`，E2R `α=0.75`。

| 方向 | 系统 | MRR | Hit@10 | Hit@100 | Hit@1000 |
| --- | --- | ---: | ---: | ---: | ---: |
| R2E | Broad | 0.04573 | 8.75% | 23.69% | 45.75% |
| R2E | Broad + train seed context | **0.10906** | **17.45%** | **33.47%** | **52.25%** |
| E2R | Broad | 0.03757 | 8.10% | 26.58% | 73.21% |
| E2R | Broad + train seed context | **0.04786** | **9.51%** | **28.31%** | **74.21%** |

只看真正有训练侧 seed 的实例，增益更明显：R2E MRR 0.05825 → 0.20626；E2R 0.10076 → 0.26142。没有合法 seed 的 query 精确回退 Broad。

因此 seed/homology 作为**可选关系上下文路由**保留，不把它伪装成所有 query 都存在的零样本专家。

## 5. Pocket / structure-support

Pocket support 只作为 router 支持特征做强度扫描。内部 validation 上 `scale=0.2` 略优于 `scale=0`，但 frozen relation-clean outer 确认时：

- `scale=0.2`：MRR 0.05356，Hit@10 10.90%；
- `scale=0`：MRR **0.05402**，Hit@10 10.90%。

因此最终排序中将 pocket/structure-support 的权重设为 **0**。它仍可作为“结构是否存在”的元数据，但不获得排序权。

## 6. CAGE 分层比较

| 方法 | MRR | Hit@10 | Hit@100 |
| --- | ---: | ---: | ---: |
| 原始 CAGE | 0.100763 | 20.53% | 27.56% |
| Broad Top-1000 + 通用 CAGE | 0.025714 | 5.44% | 18.76% |
| BRIDGE | 0.236857 | 41.54% | 64.92% |

Broad Top-1000 阳性 query coverage 约 77.91%，通用 CAGE 可评分阳性 query coverage 约 25.08%。因此仅替换候选池、仍把统一上层排序交给 generic CAGE，并不能解决广域排序。

## 7. Family 专项专家

family-tuned CAGE 相比 generic CAGE：

- P450：MRR 0.0370 → **0.0705**；
- phosphatase：MRR 0.2522 → **0.3169**；
- terpene：MRR 0.0189 → **0.0387**。

Family/TPS 专项能力按匹配领域评价；R2E 大类消融中保持完整，不逐个移除。

## 8. 冻结记录

当前结果入口为 `reproducibility/bridge/canonical.json`。新的核心记录：

- `BRIDGE_BIDIRECTIONAL_CATEGORY_ABLATION_V2_RESULT.json`；
- `BRIDGE_RELATION_SEED_CONTEXT_V2_RESULT.json`；
- `BRIDGE_POCKET_SUPPORT_V2_RESULT.json`。
