# BRIDGE 分层主评测表

主表使用全部 23,773 条相对 clean2023 严格未见关系。难度宏平均对实体新旧类别等权，并在类别内部对训练图度数分层等权；不删除难样本。

EnzymeCAGE 作者原始候选检索可处理 1,917/1,923 个 R2E query，因此不另设“CAGE-compatible”公平子集。其候选门只召回 3.43% 的 held-out relation edge，这是检索能力结果，不是本地特征缓存覆盖率。

## 主表：反应到酶

| 模型 | MRR | Hit@10 | Hit@100 | Hit@1000 | 说明 |
|---|---:|---:|---:|---:|---|
| EnzymeCAGE（候选门上界） | 0.3048 | 30.48% | 30.48% | 30.48% | 作者 Top-10 相似反应候选门之后的最乐观上界；1,917/1,923 个主测试反应可执行，神经重排真实结果只能低于或等于此上界 |
| Broad Retrieval | 0.0933 | 14.85% | 32.00% | 48.82% |  |
| Broad Retrieval + CAGE Reranking | 0.0659 | 8.86% | 23.42% | 48.82% | 由 Broad 负责候选召回，再用 EnzymeCAGE 神经分数重排已有完整 CAGE 证据的候选槽位；未物化证据保持 Broad 原位，不作负证据 |
| BRIDGE | 0.1995 | 29.71% | 39.23% | 54.93% | 当前冻结 FinalBridgeRuntime |
| BRIDGE - Functional | 0.1989 | 29.79% | 38.92% | 54.98% | 去除 EnzGFM 功能证据 |
| BRIDGE - Structure/Mechanism | 0.1924 | 28.68% | 39.05% | 54.98% | 去除结构与机制证据 |
| BRIDGE - Relational Memory | 0.1148 | 21.27% | 34.50% | 48.82% | 去除 clean2023 训练图长期关系记忆；本测试不提供运行时情景记忆 |
| BRIDGE - Family/Domain | 0.1995 | 29.65% | 39.25% | 54.98% | R2E 去除 family-CAGE 与 TPS；当前 E2R 没有对应专项成员 |

## 主表：酶到反应

| 模型 | MRR | Hit@10 | Hit@100 | Hit@1000 | 说明 |
|---|---:|---:|---:|---:|---|
| EnzymeCAGE | — | — | — | — | N/A：EnzymeCAGE 没有与当前 11,081 反应全空间同口径的原生酶到反应全局检索入口 |
| Broad Retrieval + CAGE Reranking | — | — | — | — | N/A：当前没有与 11,081 反应全空间同口径的 CAGE 神经重排；固定 pair reservoir 的反向统计不视为全局 E2R 检索 |
| Broad Retrieval | 0.0638 | 13.49% | 44.73% | 76.41% |  |
| BRIDGE | 0.2829 | 40.28% | 67.58% | 88.33% | 当前冻结 FinalBridgeRuntime |
| BRIDGE - Functional | 0.2678 | 37.32% | 61.90% | 84.74% | 去除 EnzGFM 功能证据 |
| BRIDGE - Structure/Mechanism | 0.2773 | 38.92% | 65.89% | 86.83% | 去除结构与机制证据 |
| BRIDGE - Relational Memory | 0.2138 | 36.54% | 60.99% | 86.79% | 去除 clean2023 训练图长期关系记忆；本测试不提供运行时情景记忆 |
| BRIDGE - Family/Domain | 0.2829 | 40.28% | 67.58% | 88.33% | R2E 去除 family-CAGE 与 TPS；当前 E2R 没有对应专项成员 |

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
| FIBRE Broad Core | 0.2605 | 42.39% | 50.45% |
| Catalyst layered | 0.3007 | 48.36% | 57.61% |

> Orphan-335 当前仓库未保存覆盖完整作者候选池的 EnzymeCAGE generic-pretrain 神经分数，因此保留作者检索阶段 Selenzyme 的真实名称，不作替换。

## 证据组定义

- 功能证据：EnzGFM 功能兼容性。
- 结构与机制证据：CLIPZyme、反应中心/机制以及 R2E pocket 局部重排。
- 关系记忆证据：clean2023 训练图长期记忆；本主实验不提供运行时情景记忆。
- 家族与领域专家：家族 CAGE 与 TPS；当前 E2R 生产路径没有对应专项成员。
