# BRIDGE 主实验与分层评测表

前两张表仅保留为训练重叠诊断：R2E/E2R controlled 正例与 clean2023 的精确关系重叠均约 85.5%，因此不能用于证明 BRIDGE 性能。后两张 23,773 条严格 relation-unseen 表才承担广域泛化结论。

## 诊断表一：反应到酶——同预算召回 × 重排

459 个 query；1,526 条正例中 1,305 条（85.52%）已在 clean2023 出现，仅 71 个 query 没有任何测试正例与训练图精确重叠。下表只用于审计召回/重排行为。

| 模型 | MRR | MAP | Hit@3 | Hit@10 | Hit@20 | Query 正例覆盖 | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| EnzymeCAGE | 0.1367 | — | 16.99% | 30.28% | 39.65% | 46.84% | 作者原生相似反应候选门 + generic EnzymeCAGE 排序 |
| Broad Retrieval | 0.3017 | 0.2592 | 35.51% | 48.80% | 56.43% | 62.96% | Broad 召回；每个 query 的 K 等于 CAGE 原生候选预算 |
| Broad Retrieval + CAGE Reranking | 0.1335 | 0.1163 | 13.51% | 33.99% | 48.15% | 62.96% | Broad 同预算召回后，仅用 EnzymeCAGE pred_logit 重排同一批候选 |
| BRIDGE | 0.5626 | 0.4650 | 58.61% | 60.13% | 61.22% | 62.96% | 与 Broad→CAGE 完全相同的 Broad 候选池，由当前 BRIDGE 重排 |
| BRIDGE - Functional | 0.5629 | 0.4653 | 58.82% | 60.13% | 61.22% | 62.96% | 去功能证据 |
| BRIDGE - Structure/Mechanism | 0.5632 | 0.4656 | 59.04% | 60.13% | 61.22% | 62.96% | 去结构/机制与 pocket 证据 |
| BRIDGE - Relational Memory | 0.3782 | 0.3270 | 40.96% | 55.56% | 59.26% | 62.96% | 去长期关系记忆 |
| BRIDGE - Family/Domain | 0.5611 | 0.4633 | 58.82% | 59.91% | 61.22% | 62.96% | 去 family-CAGE 与 TPS 专家 |

## 诊断表二：酶到反应——固定域同预算排序

852 个有效酶 query、1,529 条正例中 1,308 条（85.55%）已在 clean2023 出现，仅 88 个 query 没有任何测试正例与训练图精确重叠。固定 465 反应、K=155 的数值仅作为泄漏诊断保留。

| 模型 | MRR | MAP | Hit@3 | Hit@10 | Hit@20 | 宏正例召回 | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| EnzymeCAGE | 0.0312 | 0.0223 | 2.00% | 5.99% | 14.44% | 40.30% | 在固定 465 反应域内由 CAGE 选/排 Top155 |
| Broad Retrieval | 0.1799 | 0.1496 | 18.78% | 36.97% | 48.94% | 83.43% | 在同一 465 反应域内由 Broad 选/排 Top155 |
| Broad Retrieval + CAGE Reranking | 0.0603 | 0.0450 | 5.87% | 12.44% | 21.13% | 83.43% | Broad Top155 召回后用同一 CAGE pred_logit 重排 |
| BRIDGE | 0.7887 | 0.7364 | 86.97% | 93.08% | 94.72% | 96.93% | 当前 BRIDGE 在同一 465 反应域内选/排 Top155 |
| BRIDGE - Functional | 0.7898 | 0.7394 | 87.21% | 94.01% | 94.48% | 95.90% | 去功能证据 |
| BRIDGE - Structure/Mechanism | 0.8001 | 0.7536 | 87.21% | 93.78% | 95.19% | 97.13% | 去结构证据 |
| BRIDGE - Relational Memory | 0.1924 | 0.1651 | 21.01% | 34.98% | 47.42% | 80.89% | 去长期关系记忆 |
| BRIDGE - Family/Domain | 0.7887 | 0.7364 | 86.97% | 93.08% | 94.72% | 96.93% | 当前 E2R 无 family/TPS 成员，与完整 BRIDGE 相同 |

## 广域泛化：反应到酶（23,773 relation-unseen）

| 模型 | MRR | Hit@10 | Hit@100 | Hit@1000 | 说明 |
|---|---:|---:|---:|---:|---|
| Broad Retrieval | 0.0933 | 14.85% | 32.00% | 48.82% | 冻结 Broad 全候选排序 |
| BRIDGE | 0.1995 | 29.71% | 39.23% | 54.93% | 当前冻结 FinalBridgeRuntime |
| BRIDGE - Functional | 0.1989 | 29.79% | 38.92% | 54.98% | 去除 EnzGFM 功能证据 |
| BRIDGE - Structure/Mechanism | 0.1924 | 28.68% | 39.05% | 54.98% | 去除结构/机制证据；R2E 同时去除 pocket 重排 |
| BRIDGE - Relational Memory | 0.1148 | 21.27% | 34.50% | 48.82% | 去除 clean2023 长期关系记忆；不含运行时情景记忆 |
| BRIDGE - Family/Domain | 0.1995 | 29.65% | 39.25% | 54.98% | 去除 family-CAGE 与 TPS |

## 广域泛化：酶到反应（23,773 relation-unseen）

| 模型 | MRR | Hit@10 | Hit@100 | Hit@1000 | 说明 |
|---|---:|---:|---:|---:|---|
| Broad Retrieval | 0.0638 | 13.49% | 44.73% | 76.41% | 冻结 Broad 全候选排序 |
| BRIDGE | 0.2829 | 40.28% | 67.58% | 88.33% | 当前冻结 FinalBridgeRuntime |
| BRIDGE - Functional | 0.2678 | 37.32% | 61.90% | 84.74% | 去除 EnzGFM 功能证据 |
| BRIDGE - Structure/Mechanism | 0.2773 | 38.92% | 65.89% | 86.83% | 去除结构/机制证据；R2E 同时去除 pocket 重排 |
| BRIDGE - Relational Memory | 0.2138 | 36.54% | 60.99% | 86.79% | 去除 clean2023 长期关系记忆；不含运行时情景记忆 |
| BRIDGE - Family/Domain | 0.2829 | 40.28% | 67.58% | 88.33% | 当前 E2R 没有 family/TPS 成员，因此与完整 BRIDGE 相同 |

## EnzymeCAGE 作者原生基准

### Enzyme-405（226-query aligned comparison）

| 模型 | MRR | Hit@10 | Hit@20 |
|---|---:|---:|---:|
| EnzymeCAGE generic pretrain | 0.2517 | 51.33% | 70.09% |
| Broad Retrieval | 0.2671 | 52.21% | 68.14% |
| Layered mainline | 0.2864 | 54.42% | 70.35% |

### Orphan-335

| 模型 | MRR | Hit@10 | Hit@20 |
|---|---:|---:|---:|
| Selenzyme (author retrieval stage) | 0.2088 | 31.04% | 39.10% |
| Broad Retrieval | 0.2605 | 42.39% | 50.45% |
| Catalyst layered | 0.3007 | 48.36% | 57.61% |

> Orphan-335 当前仓库没有覆盖完整作者候选池的 EnzymeCAGE generic-pretrain 神经分数；Selenzyme 保留真实名称，不冒充 EnzymeCAGE。

## 四组证据定义

- 功能证据：EnzGFM 功能兼容性。
- 结构与机制证据：CLIPZyme、反应中心/机制，以及 R2E pocket 局部重排。
- 关系记忆证据：clean2023 长期训练图记忆；只有提供训练后/用户确认 support 时才额外启用情景记忆。
- 家族与领域专家：family CAGE 与 TPS。

controlled 表中长期关系记忆几乎直接回放训练图，因此完整 BRIDGE 的高分无泛化含义。专家的广域独立价值只读取 23,773 条严格 relation-unseen 泛化表。

