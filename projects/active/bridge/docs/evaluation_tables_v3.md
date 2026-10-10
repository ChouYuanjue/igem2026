# BRIDGE 最终分层主实验 V3

母集始终为 23,773 条相对 clean2023 严格 relation-unseen 关系。只有涉及 EnzymeCAGE 的两条路径共享候选预算；Broad Retrieval、BRIDGE 与四组消融始终保留各自原生完整候选空间。

## R2E：召回层（全 1,923 query）

| 召回器 | Query hit | Macro positive recall | Edge recall |
|---|---:|---:|---:|
| EnzymeCAGE 原生门控 | 7.44% | 6.26% | 3.43% |
| Broad 同预算召回 | 27.61% | 20.99% | 14.72% |

## R2E：排序层（212 个 CAGE 可评分严格 query）

| 模型/路径 | 候选路径 | MRR | Hit@3 | Hit@10 | Hit@100/正例召回 |
|---|---|---:|---:|---:|---:|
| EnzymeCAGE | CAGE 同预算路径 | 0.1489 | 16.98% | 29.72% | 38.27% 正例召回 |
| Broad Retrieval + CAGE Reranking | CAGE 同预算路径 | 0.0882 | 9.43% | 20.28% | 29.20% 正例召回 |
| Broad Retrieval | 原生完整候选空间 | 0.1034 | 10.38% | 20.28% | 38.68% |
| BRIDGE | 原生生产路径 | 0.1896 | 21.23% | 29.72% | 42.92% |
| BRIDGE - Functional | 去功能证据 | 0.1790 | 19.81% | 27.83% | 43.40% |
| BRIDGE - Structure/Mechanism | 去结构/机制证据 | 0.1898 | 20.75% | 28.30% | 42.92% |
| BRIDGE - Relational Memory | 去长期关系记忆 | 0.1207 | 11.32% | 25.00% | 44.34% |
| BRIDGE - Family/Domain | 去家族/领域专家 | 0.1896 | 21.23% | 29.25% | 42.92% |

## E2R：召回层（全 17,486 query）

E2R 的低预算门控采用与 CAGE R2E 对称的规则：Top-10 ESM-C 相似 clean2023 酶 → 汇总其已知反应。候选池中位数 6，均值约 6.86。

| 召回器 | Query hit | Macro positive recall | Edge recall |
|---|---:|---:|---:|
| 相似酶门控 | 15.11% | 14.58% | 15.47% |

## E2R：排序层（284 个 CAGE 可评分严格 query）

| 模型/路径 | 候选路径 | MRR | Hit@3 | Hit@10 | Hit@100/正例召回 |
|---|---|---:|---:|---:|---:|
| EnzymeCAGE | CAGE 同预算路径 | 0.0364 | 4.58% | 5.99% | 4.46% 正例召回 |
| Broad Retrieval + CAGE Reranking | CAGE 同预算路径 | 0.0439 | 5.99% | 8.10% | 6.16% 正例召回 |
| Broad Retrieval | 原生完整候选空间 | 0.0718 | 5.99% | 16.90% | 41.90% |
| BRIDGE | 原生生产路径 | 0.2235 | 25.00% | 32.39% | 55.99% |
| BRIDGE - Functional | 去功能证据 | 0.2112 | 24.30% | 31.69% | 49.65% |
| BRIDGE - Structure/Mechanism | 去结构/机制证据 | 0.2277 | 25.00% | 32.39% | 54.93% |
| BRIDGE - Relational Memory | 去长期关系记忆 | 0.0857 | 8.80% | 17.96% | 48.24% |
| BRIDGE - Family/Domain | 去家族/领域专家 | 0.2237 | 25.00% | 32.39% | 55.99% |

## 口径

- EnzymeCAGE 与 Broad→CAGE：只为比较 CAGE 门控/重排而共享逐 query 候选预算。
- Broad Retrieval：始终使用完整候选空间的 Broad 原生排名。
- BRIDGE 与四组消融：始终使用完整候选空间的生产路径，不因 CAGE 的候选预算缩小。
- R2E 的 CAGE 神经重排目前受可物化结构特征支持限制，因此排序层报告 212 个严格 query；召回层仍覆盖全部 1,923 query。
- E2R 的 CAGE 神经重排报告 284 个严格 query；低预算门控本身覆盖全部 17,486 query。
