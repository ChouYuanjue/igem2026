# Catalyst Retrieval Capability Scorecard

这份表只回答一个问题：**不同场景分别测什么能力，应该看什么指标。** 不再用一个总分覆盖所有任务。


> **证据身份：** TPS practical R2E 现在已有本地复现的 pure EnzymeCAGE 外部基线；`RF/HGB+CAGE` 仍只是 Catalyst 自己的历史系统。外部 ranking delta 只在 EnzymeCAGE 原生可直接比较的 common support 上计算，完整 513×1391 能力与 applicability 单独报告。详细身份见 [`RETRIEVAL_EVIDENCE_LEDGER.md`](RETRIEVAL_EVIDENCE_LEDGER.md)。

## A. TPS 专项：数据库补全 / 日用筛选

完整 Catalyst 任务仍是 **513 reactions × 1,391 proteins**：当前 locked nested route 的 Hit@10 **48.15%**、Hit@20 **57.50%**。这代表完整 TPS candidate universe 的实际筛选能力。

真正外部 baseline 使用 **pure EnzymeCAGE + EnzymeCAGE 官方检索算法复现**（自己的 reaction-similarity gate + generic-pretrain seed42 `epoch_19`），不掺 Catalyst 的 RF/HGB 或融合组件。在其原生可公平比较的 **459 reactions × 1,379 proteins** 上：

| 方法 | Hit@5 | Hit@10 | Hit@20 | Macro positive recall@10 | Macro positive recall@20 |
|---|---:|---:|---:|---:|---:|
| pure EnzymeCAGE + EnzymeCAGE-style retrieval | 22.22% | 30.28% | 39.65% | 22.80% | 30.94% |
| Catalyst locked practical route | **37.47%** | **47.49%** | **56.21%** | **36.80%** | **45.10%** |
| 差值 | +15.25 pp | **+17.21 pp** | **+16.56 pp** | +14.00 pp | +14.16 pp |

这张表的主叙事是：**同一 CAGE-native 适用域内，Catalyst 在实际 Top-10/20 筛选预算上明显更强。** 另外单列 applicability：EnzymeCAGE reaction feature 可处理 **465/513 (90.64%)**，raw scorer 可形成有正例的检索任务 **462/513 (90.06%)**；其中 **459** 个有预先归档的 EnzymeCAGE-style retrieval gate，作为外部 ranking 主对比，protein native pocket support **1379/1391 (99.14%)**；Catalyst 覆盖完整 **513/513 reactions + 1391/1391 candidate proteins**。44 个 EnzymeCAGE canonicalization failure 和另外 4 个 native reaction-feature failure没有替它修复，也没有算进 ranking delta。

旧 RF/HGB+CAGE hybrid → 当前 route 的 39.57→48.15% / 45.22→57.50% 只保留为内部迭代历史，不再占外部 baseline 列。

## B. TPS 专项：exact 新实体，但合法邻域证据可用

| 情境 | Query | Hit@10 | Hit@20 | MRR |
|---|---:|---:|---:|---:|
| E2R exact 新蛋白，全体 | 917 | 72.41% | 77.54% | 0.552 |
| E2R 且有 ≥50% 同簇训练同源物 | 706 | **82.58%** | **86.69%** | **0.654** |
| E2R 无同簇训练同源物 | 211 | 38.39% | 46.92% | 0.209 |
| R2E exact 新蛋白且同源可见 | 722 cells | **62.60%** | **68.14%** | **0.440** |
| R2E 同源不可见 | 232 cells | 15.52% | 21.98% | 0.072 |

这组不是 baseline→current，而是同一个能力在**邻域证据可用/不可用**时的差异，直接体现同源利用能力。

## C. TPS 专项：few-shot 同源扩展

- 内部 random-positive：1 seed 时 Top-10 约 **72–74%**；5 seeds 时约 **92–93%**。
- 外部 MARTS：1 seed 的 ESM-C max 为 Hit@3 **38.85%**、Hit@10 **50.94%**、Hit@20 **54.27%**。
- 强制跨 50% protein cluster 后，1-seed ESM-C centroid Top-10 **27.64%**、Top-20 **40.54%**；简单 3-mer 为 22.44% / 31.24%。

因此这里最能区分能力的不是 MRR，而是 **Hit@10 + hidden-positive recall**，并把 homolog expansion 与 remote-family expansion 分开。

## D. TPS 专项：严格双冷探索

当前对称主表统一使用 **1,421 proteins × 453 reactions 的 MARTS universe**，但两个方向讲不同能力：

| 方向 | 旧路线 | 已确认新路线 | 最合适的叙事 |
|---|---|---|---|
| R2E，155 frozen query-cells | MRR 0.03890；Hit@10 5.81%；Hit@20 15.48% | **MRR 0.04183**；**Hit@10 6.45%**；Hit@20 **15.48% 不退** | **precision-first stabilization**：MRR 相对约 +7.5%，前排质量提高且 Top20 recall 保持 |
| E2R，279 query-cells | MRR 0.0764；Hit@20 34.77% | **MRR 0.0874；Hit@20 43.37%** | **recall expansion**：Hit@20 **+8.60 pp**，95% CI [+5.02,+12.54]，同时 MRR 提高 |

这里没有 aligned external model baseline，因此**不讲外部 superiority**；讲的是同一个 current MARTS strict benchmark 上，两方向都完成了锁定路线确认。R2E 的贡献是稳定精排，E2R 的贡献是明显扩展候选召回。旧 513×1391 strict R2E 只保留历史参考，不再拼进当前双向主表。

## E. 通用能力：大候选宇宙时间外推

Rhea128→141，relative to clean2023 的 protein/reaction/pair 都 100% 未见。

| 方向 | Candidate universe | 最区分能力的归一化指标 | Baseline → Current | MRR |
|---|---:|---|---:|---:|
| R2E | 185,918 proteins | Success@0.1%（前186） | 20.19% → **31.73%** | 0.0128 → **0.0299** |
| E2R | 11,081 reactions | Success@0.2%（前23） | 6.40% → **23.93%** | 0.0153 → **0.0275** |

在如此大的候选池里，Success@候选池百分比比单独 Hit@10 更有解释力：它直接回答“把实验空间压缩到前千分之一/千分之二后，多少 query 仍保留真阳性”。

## F. 通用内部严格确认：旧式 Hit@K

这组负责和历史 50%+ 数字保持连续：R2E Hit@50 **42.50%→49.18%**；E2R Hit@50 **51.98%→55.36%**。它不替代 E 的外部时间快照。

## 最终指标映射

| 能力 | 主指标 |
|---|---|
| TPS 数据库补全 | Hit@10/20，预算内已知正例数 |
| TPS exact 新实体 | Hit@10/20 + MRR；同源 visible/not-visible 对照 |
| TPS few-shot | Hit@10 + hidden-positive recall |
| TPS 远缘探索 | paired Hit@10/20 delta + CI + MRR |
| 通用大库检索 | Success@0.1–0.2% + MRR，Hit@50 辅助 |
| Production | latency / memory / determinism / retention |

## G. 真正外部 baseline 证据

- **TPS practical pure EnzymeCAGE（当前主 TPS 外部对比）**：459×1379 common support；CAGE Hit@10/20 **30.28/39.65%**，Catalyst **47.49/56.21%**。双方本地实测；CAGE 使用官方 `retrieve.py` 语义的 reaction-similarity gate 复现 + generic-pretrain checkpoint；归档 TPS gate 不声称是作者原始文件。适用率另报，不把 CAGE unsupported inputs 算 ranking miss。
- **Enzyme-405 complete226**：official EnzymeCAGE seeds40–44 与 Catalyst 双方本地实测、同 support；属于独立通用/外部 benchmark，不再用它填 TPS baseline 的空。
- **Orphan-335 Selenzyme**：author score 与 Catalyst 同 author pool，本地同 support；reaction-novel secondary。
- **CLIPZyme common support**：official checkpoint 与 Catalyst 同 score matrix，本地实测；adapted common-support secondary alignment。
- **Paper-only**：Enzyme-405 full295 的 EnzymeCAGE paper 数字、ReactZyme native 的 EnzGFM-1.5B paper 数字，不得混成“本地基线”。

### TPS 当前对称性

当前 MARTS **1,421×453** strict universe 的 R2E/E2R 都已经完成 finalized locked route evidence：R2E 是“前排精度/稳定性增强”，E2R 是“Top20 recall 扩展”。二者同 universe、同 partition family，但不强行用同一个指标讲同一种故事。
