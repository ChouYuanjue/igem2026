# Catalyst Retrieval Capability Scorecard

这份表只回答一个问题：**不同场景分别测什么能力，应该看什么指标。** 不再用一个总分覆盖所有任务。


> **证据身份纠正：** `RF/CAGE rescue` 是 Catalyst 自己的旧 TPS hybrid，不是 EnzymeCAGE 外部 baseline。当前 TPS 513×1391 没有完整 same-support 的纯外部 baseline。纯 EnzymeCAGE 本地分数存在，但只覆盖部分 native scored support。真正双方本地实测、同 support 的纯 EnzymeCAGE 对比是独立的 Enzyme-405 complete226 benchmark。详细身份见 [`RETRIEVAL_EVIDENCE_LEDGER.md`](RETRIEVAL_EVIDENCE_LEDGER.md)。

## A. TPS 专项：数据库补全 / 日用筛选

候选宇宙：513 reactions × 1,391 proteins。这里 Top-10/20 就是实际实验预算，因此直接报告 Hit@K。

| 方法 | Hit@5 | Hit@10 | Hit@20 |
|---|---:|---:|---:|
| 历史 TPS RF/HGB+CAGE hybrid（内部旧系统） | 34.50% | 39.57% | 45.22% |
| 当前 best nested TPS route | **38.60%** | **48.15%** | **57.50%** |

相对内部旧系统，Top-10 **+8.58 pp**，Top-20 **+12.28 pp**。这只证明内部迭代，**不是外部 baseline delta**。TPS 513×1391 当前缺完整 same-support 外部 baseline；纯 EnzymeCAGE 已本地实测，但 support 不完整。

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

| 方向 | 基线 | 当前 | 主要指标 |
|---|---:|---:|---|
| R2E（旧 513×1391 universe） | 内部 RF/HGB+CAGE：Hit@10 0.56%, Hit@20 1.54% | dual tower **8.96% / 16.81%** | 仅内部历史 delta；不可作为当前对称 TPS 主表 |
| E2R（当前 1421×453 universe） | 内部旧 production：Hit@20 34.77%, MRR 0.0764 | dual-kernel RRF **43.37%**, MRR **0.0874** | 内部路线确认 +8.60pp, CI [+5.02,+12.54] |

绝对值低是任务定义造成的；这里重点看**旧方法是否还能工作，以及改进是否配对稳定**。

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

- **Enzyme-405 complete226**：official EnzymeCAGE seeds40–44 与 Catalyst 双方本地实测、同 support。这是最干净的 pure-EnzymeCAGE apples-to-apples：CAGE SR@10 51.33±1.66%，Catalyst 49.12%；Catalyst MRR 0.2631 vs CAGE 0.2517，MAP 0.2575 vs 0.2521。
- **Orphan-335 Selenzyme**：author score 与 Catalyst 同 author pool，本地同 support；但只作为 reaction-novel secondary。
- **CLIPZyme common support**：official checkpoint 与 Catalyst 同 score matrix，本地实测；但属于 adapted common-support secondary alignment。
- **Paper-only**：Enzyme-405 full295 的 EnzymeCAGE paper 数字、ReactZyme native 的 EnzGFM-1.5B paper 数字，不得混成“本地基线”。

### TPS 当前对称性

当前 MARTS universe 是 **1,421 proteins × 453 reactions**。同一 universe 上两方向都有测量，但最终证据并不对称：E2R 已有独立融合确认（Hit@20 34.77%→43.37%）；R2E 同 split 已测到 direct candidate Hit@20 16.74%，但还没有与 E2R 同等级的 finalized route confirmation。旧 513×1391 R2E 结果只保留历史能力展示。
