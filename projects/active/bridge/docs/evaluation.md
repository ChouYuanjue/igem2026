# BRIDGE：统一广域关系评测

## 1. 数据和任务

模型采用**训练集、统一验证集、测试集**三个数据层级。

| 数据部分 | 已确认酶–反应关系 | 实体规模与用途 |
| --- | ---: | --- |
| 训练集 | **218,537** | clean2023，161,749 个蛋白、9,750 个反应，训练基础表示并构造只读催化关系图 |
| 统一验证集 | **5,216** | 3,454 个蛋白查询、990 个反应查询；用于两方向的专家门控、关系强度和功能证据校准 |
| 测试集 | **21,505** | 15,751 个酶查询、1,857 个反应查询；R2E 与 E2R 共用同一批训练期未见的正关联 |

训练、验证和测试的精确关联互不重叠。验证与测试的蛋白查询无交集。R2E 允许**516 个反应查询实体在验证与测试中重复出现**，其对应的酶–反应关联始终不同。来源文件、冻结哈希与每一类冷启动定义均写入复现记录。

R2E 在完整 **185,918 个候选蛋白**中定位催化酶；E2R 在完整 **11,081 个候选反应**中定位对应反应。过滤排名在推理时排除 clean2023 已知正关联，并在计算逐关系目标排名时过滤同一查询的其他已标注正例。EnzymeCAGE 原生候选门控实验使用固定候选集合，未召回的目标按对应完整候选库尾端计分。

## 2. 统一验证与模型校准

所有当前关系专家门控只使用同一组 **5,216 条验证关联**。R2E 使用其中 990 个反应查询进行五折查询分组验证，基于训练图邻域、局部一致性、关系覆盖和候选排序信号估计关系证据权限，取查询相关的权限与条件强度进行排序修正。功能专家 EnzGFM 的输出强度通过相同验证集选择为 **1.25**；校准时其余专家、基础编码器及候选全集固定。

E2R 的联合查询路由在同一验证集内以蛋白查询分组完成拟合与验证，训练图归纳关系证据从 **16 个已注释近邻蛋白**汇集真实催化关联，并以 **8 个近邻反应原型**支持无历史关联反应。该方向的非负关系系数分别为 **0.3092** 与 **0.8209**。E2R 其余兼容调用路径的关系门控同样在统一验证集内交叉验证。

主实验采用四项证据消融：功能、结构与机制、长期关系、家族与领域。删除长期关系证据时，R2E 和 E2R 均取消训练图关系项；E2R 同时取消归纳传播与反应原型项。其余专家及校准参数保持冻结。

## 3. 指标和四类难度

**Direct** 为逐关系 MRR、Hit@3、Hit@10、Hit@100 的直接平均。**Balanced** 在同一批排序结果上，先将 both-seen、protein-cold、reaction-cold、double-cold 四种历史暴露类型等权，再对每类实际出现的训练图节点 degree 层等权，层内逐关系等权：

\[
\mathrm{Balanced}(M)=rac{1}{4}\sum_{c\in C}rac{1}{|S_c|}\sum_{s\in S_c}rac{1}{|D_{c,s}|}\sum_{i\in D_{c,s}}M_i.
\]

Hit@K 为目标排名不超过 K 的比例，MRR 为过滤排名倒数均值。Balanced 保留原始度量，**不扣除随机排序基线**。

| 历史暴露类型 | 测试正关联 |
| --- | ---: |
| both-seen | 180 |
| protein-cold | 7,496 |
| reaction-cold | 1,070 |
| double-cold | 12,759 |
| **合计** | **21,505** |

## 4. 双向正式测试与四组消融

### R2E：反应 → 酶 · Direct：自然分布
| 方法 | MRR | Hit@3 | Hit@10 | Hit@100 |
| --- | ---: | ---: | ---: | ---: |
| EnzymeCAGE | 0.00522 | 0.49% | 0.99% | 2.46% |
| Broad Retrieval | 0.01413 | 1.22% | 2.22% | 7.39% |
| Broad Retrieval + CAGE Reranking | 0.00537 | 0.46% | 1.20% | 3.30% |
| CAGE Gate + BRIDGE Reranking | 0.01218 | 1.24% | 1.92% | 3.17% |
| BRIDGE | 0.02590 | 2.68% | 4.05% | 8.04% |
| 去除Functional | 0.02481 | 2.58% | 3.78% | 7.77% |
| 去除Structure/Mechanism | 0.02637 | 2.72% | 4.10% | 7.75% |
| 去除Long-term Relation Context | 0.01762 | 1.51% | 2.99% | 8.73% |
| 去除Family/Domain | 0.02589 | 2.68% | 4.04% | 8.03% |

### R2E：反应 → 酶 · Balanced：难度宏平均
| 方法 | MRR | Hit@3 | Hit@10 | Hit@100 |
| --- | ---: | ---: | ---: | ---: |
| EnzymeCAGE | 0.06294 | 5.66% | 11.42% | 24.62% |
| Broad Retrieval | 0.09412 | 8.67% | 15.12% | 32.60% |
| Broad Retrieval + CAGE Reranking | 0.04056 | 4.56% | 8.96% | 19.75% |
| CAGE Gate + BRIDGE Reranking | 0.16423 | 17.47% | 23.60% | 29.87% |
| BRIDGE | 0.20441 | 22.94% | 31.13% | 40.56% |
| 去除Functional | 0.20374 | 22.95% | 31.07% | 40.44% |
| 去除Structure/Mechanism | 0.20146 | 22.10% | 30.88% | 40.01% |
| 去除Long-term Relation Context | 0.11772 | 10.18% | 21.84% | 35.06% |
| 去除Family/Domain | 0.20439 | 22.94% | 31.11% | 40.54% |

### E2R：酶 → 反应 · Direct：自然分布
| 方法 | MRR | Hit@3 | Hit@10 | Hit@100 |
| --- | ---: | ---: | ---: | ---: |
| EnzymeCAGE | 0.07090 | 8.45% | 9.74% | 10.22% |
| Broad Retrieval | 0.03357 | 2.87% | 7.62% | 24.10% |
| Broad Retrieval + CAGE Reranking | 0.01810 | 2.21% | 3.53% | 4.16% |
| CAGE Gate + BRIDGE Reranking | 0.11286 | 12.94% | 14.94% | 15.20% |
| BRIDGE | 0.14980 | 15.80% | 25.50% | 45.42% |
| 去除Functional | 0.11524 | 12.35% | 18.30% | 39.94% |
| 去除Structure/Mechanism | 0.13370 | 13.82% | 22.63% | 45.19% |
| 去除Long-term Relation Context | 0.09727 | 9.65% | 18.28% | 40.33% |
| 去除Family/Domain | 0.14980 | 15.80% | 25.50% | 45.42% |

### E2R：酶 → 反应 · Balanced：难度宏平均
| 方法 | MRR | Hit@3 | Hit@10 | Hit@100 |
| --- | ---: | ---: | ---: | ---: |
| EnzymeCAGE | 0.08566 | 8.92% | 11.51% | 12.44% |
| Broad Retrieval | 0.06317 | 5.50% | 13.43% | 44.55% |
| Broad Retrieval + CAGE Reranking | 0.03246 | 3.70% | 5.26% | 6.26% |
| CAGE Gate + BRIDGE Reranking | 0.11113 | 11.72% | 14.21% | 14.29% |
| BRIDGE | 0.28897 | 30.26% | 40.34% | 60.69% |
| 去除Functional | 0.27668 | 29.33% | 37.65% | 57.05% |
| 去除Structure/Mechanism | 0.19998 | 20.38% | 29.86% | 60.10% |
| 去除Long-term Relation Context | 0.20449 | 20.70% | 32.60% | 54.62% |
| 去除Family/Domain | 0.28897 | 30.26% | 40.34% | 60.69% |

## 5. 结果与专项适用域

R2E 完整系统的 Balanced MRR、Hit@10、Hit@100 分别为 **0.20441、31.13%、40.56%**。移除功能证据后对应为 **0.20374、31.07%、40.44%**；删除长期关系证据后为 **0.11772、21.84%、35.06%**。完整系统的 Direct MRR、Hit@10、Hit@100 分别为 **0.02590、4.05%、8.04%**。

E2R 完整系统的 Direct MRR、Hit@10、Hit@100 分别为 **0.14980、25.50%、45.42%**；Balanced 为 **0.28897、40.34%、60.69%**。移除长期关系证据后 Balanced Hit@10 为 **32.60%**。E2R 家族专项通道在这一任务中未激活，对应消融与完整结果相同。

P450 与 Phosphatase 的领域专项仍在各自官方独立家族测试中进行。TPS 适用性切片取自本次 21,505 条正式关系，包含 **10 个反应查询和 12 条正关联**；完整系统的查询级 MRR 和 Hit@10 为 **0.0221、10.00%**，移除家族与领域证据后为 **0.0086、0.00%**。其他真实时间增长及作者外部集合采用各自独立专项协议，单独报告其原生分母。

**统计边界。** 本次关系门控和功能强度的重新设计发生于先前已查看正式测试结果之后。因此本表构成探索性复评；推广性结论仍需在后续未参与方案选择的新增关联上独立确认。

## 6. 复现入口

统一样本和九行表：reproducibility/bime_rank/scripts/prepare_bridge_canonical_query_split_v1.py

R2E 关系门控：reproducibility/bime_rank/scripts/fit_bridge_r2e_single_validation_v1.py。R2E 功能证据验证：reproducibility/bime_rank/scripts/evaluate_bridge_r2e_functional_single_validation_v1.py。

R2E 正式排序与消融：reproducibility/bime_rank/scripts/evaluate_bridge_r2e_single_validation_ablation_v1.py；CAGE 原生门控组合：reproducibility/bime_rank/scripts/evaluate_bridge_r2e_single_validation_cage_hybrid_v1.py。

E2R 联合门控沿用冻结版本；兼容关系门控由 reproducibility/bime_rank/scripts/fit_bridge_e2r_fallback_single_validation_v1.py 生成。

全部正式汇总：reproducibility/bime_rank/records/BRIDGE_CANONICAL_21505_RESULT.json。
