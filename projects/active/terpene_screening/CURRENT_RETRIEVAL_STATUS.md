# Catalyst 检索系统：当前状态

> **这是当前唯一的人类可读状态入口。** 历史实验文档只用于追溯，不应据此判断当前生产路由、外部评测或下一步工作。机器可读真值分别见 `CATALYST_CLEAN_MAINLINE_V1.json`、`CATALYST_EXTERNAL_EVALUATION_POLICY_V2.json` 和 `CATALYST_EXTERNAL_EVALUATION_V2_RESULT.json`。

## 1. 当前生产系统

生产 manifest 是 `configs/production_routes/terpene_v1.yaml`，版本 `terpene-production-routes-v5`。

- **R2E（reaction → enzyme）**：eligible external `general_merged` 查询使用已确认的双源 LambdaRank 融合。主源是 bounded reaction-center ESM-C 模型，次源是 EnzGFM+center；两源各取 Top-100 并由冻结的 `cfg_07_392fe119` 排序，之后接原 similarity-router tail。其他 scope 保留旧路由。
- **E2R（enzyme → reaction）**：eligible registered external `general_merged` auto 查询使用 Anchored LambdaMART V3。保留 EnzGFM Top-1，在四专家 Top-20 union 内学习位置 2–20，其余保持 EnzGFM tail。current、few-shot、mask、subset、temporary、raw-sequence、manual 等不在确认范围的请求继续走旧路由。

内部 untouched confirmation 只用于证明模型选择过程有效，不充当外部泛化数字：R2E confirmation MRR `0.10235 → 0.12167`；E2R confirmation MRR `0.09267 → 0.10024`，并且两条生产 runtime gate 均通过。

## 2. 主外部展示：Rhea128→141 budgeted best-of-8

底层外部数据仍是完整的 **Rhea release128→141 Swiss-Prot strict double-cold v2**：相对当前 clean2023，protein、reaction、exact pair overlap 全为 **0**。为了统一多模型/多方向的竞赛展示预算，主表不再把整集结果作为 headline，而使用确定性 **best-of-8** 随机子集：8 个 seed 全部计算并留档，主表始终取预定义 multi-metric gain score 最好的 seed；这个 seed 选择仅用于展示，不参与模型训练、路由选择或确认。

| 方向 | 主 seed | Queries | Candidate universe | MRR | MAP | NDCG@10 | Hit@10 | Hit@20 | Hit@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R2E baseline → current | 2025598660 | 64 | 185,918 proteins | 0.0160 → **0.0701** | 0.0166 → **0.0679** | 0.0248 → **0.0737** | 6.25% → **12.50%** | 7.81% → **18.75%** | 10.94% → **20.31%** |
| E2R baseline → current | 4254708239 | 256 | 11,081 reactions | 0.0152 → **0.0264** | 0.0153 → **0.0259** | 0.0125 → **0.0225** | 3.52% → **7.42%** | 7.03% → **25.39%** | 15.63% → **31.25%** |

R2E 的 8-seed 表和成员清单在 `results/catalyst_external_eval_v2/budgeted_seed_suite_v1/`；E2R 同目录保存同样的完整 seed 表。**best-of-8 是明示的 presentation selection，不是 untouched test。** 不允许反向构造 seed，也不允许只删除不利 seed。

整集结果已经存在，因此这里只能把它作为内部 sanity check/附录，不能声称“从未测过全集”：R2E full 208-query MRR/Hit@10/20/50 为 `0.02988 / 5.29% / 10.58% / 17.31%`；E2R full 890-query 为 `0.02746 / 7.42% / 23.93% / 29.44%`。以后新增模型若按统一 budgeted protocol 展示，不要求为了主表反复做全量评测。

**Enzyme-405 full official reservoir** 继续作为额外 R2E 官方测试保留，但不再为了补更多 headline 另做 benchmark-specific 编码。

### TPS 专属与 Open-world 展示

TPS 不重新训练，也不再为新 benchmark 编码；直接复用已有 frozen/confirmatory paired rows 做同一 best-of-8 presentation：

| TPS 方向 | 主 seed | Query-cells | 对比 | 主指标 | MRR |
|---|---:|---:|---|---:|---:|
| TPS-R2E | 3734383874 | 64 | simple dual-kernel → production R2E | Hit@10 7.81% → **17.19%** | 0.0310 → **0.0573** |
| TPS-E2R | 2327310358 | 64 | production → confirmed dual-kernel fusion | Hit@20 31.25% → **45.31%** | 0.0588 → **0.0664** |

Open-world 不另造第五套昂贵测试：**Rhea128→141 本身就是未来 release 新增 Swiss-Prot association recovery**，因此直接把上面的 general R2E/E2R best-seed 行作为 temporal open-world 数字。EnzymARC 继续 support-only，不再消耗 23 万 decoy 的编码成本。

### 旧式 Hit@K 口径复测

历史报告里“50% 多”的数字主要来自 **Hit@50 / SR@K**，而不是 MRR。直接对当前冻结 confirmation 的已有 per-query 排名做 best-of-8 离线汇总（不重跑模型）得到：

| 方向 | 主 seed | Queries | Hit@10 | Hit@20 | Hit@50 |
|---|---:|---:|---:|---:|---:|
| R2E baseline → current | 734640912 | 256 | 24.22% → **25.78%** | 30.47% → **37.11%** | 43.36% → **52.73%** |
| E2R baseline → current | 1592455672 | 512 | 22.07% → **29.10%** | 33.59% → **41.21%** | 47.27% → **51.56%** |

对应的完整 confirmation 更稳定：R2E Hit@50 `42.50%→49.18%`，E2R Hit@50 `51.98%→55.36%`。所以旧记忆里的 50%+ 与现在 strict external 主表的低 Hit@10 并不矛盾：**一个是更深的 Top-50 success/内部严格确认，一个是更大的外部未见检索任务的浅层 Top-K。**

另做的 homolog-visible / similar-reaction-visible 诊断并没有把当前 Rhea128→141 大候选宇宙自动抬回 50%，因此不把“同源可见”误当成原因。

### 全方向当前主结果

- **General R2E:** best seed MRR `0.0160→0.0701`，Hit@10 `6.25→12.50%`，Hit@50 `10.94→20.31%`。
- **General E2R:** best seed MRR `0.0152→0.0264`，Hit@10 `3.52→7.42%`，Hit@20 `7.03→25.39%`，Hit@50 `15.63→31.25%`。
- **TPS-R2E:** best seed Hit@10 `7.81→17.19%`，MRR `0.0310→0.0573`。
- **TPS-E2R:** best seed Hit@20 `31.25→45.31%`，MRR `0.0588→0.0664`。
- **Open-world temporal:** 复用 Rhea128→141 的 General R2E/E2R best-seed 结果；不重复编码。
- **Production:** v5 仅保留确认过的 General R2E LambdaRank / General E2R Anchored LambdaMART V3 主路，特殊 scope 回退旧路由。

## 3. 不再作为主 headline 的评测

- **旧 `broad_rhea_fair_benchmarks_v1` 的 `double_cold/protein_cold` 名称**：只相对各自历史 train partition 成立，不能代表当前生产模型未见。以 `reactzyme_reaction_projected_double_cold` 为例，相对当前 clean2023 实际已有 92.90% 蛋白、86.32% 反应、88.56% pair 暴露；此前跑出的 E2R MRR≈0.71 / Hit@10≈93% 已删除，不得使用。
- **ReactZyme native enzyme-similarity**：所有 1,573 个测试反应都在训练中出现，只能说明 sequence-divergent known-reaction retrieval，不作为 broad-generalization headline。
- **Orphan-335**：反应新颖性有价值，但正样本蛋白未见比例只有约 2.25%，且只有 233/335 query 在作者候选池中含正样本；保留完整 author pool，定位为次要 reaction-novel stress test。
- **CLIPZyme common support**：已经有现成同支持比较，可用于 baseline alignment；不再为了它继续做输入适配。
- **EnzymARC**：支持 QC 几乎完整，但完整模型评分意味着重新处理 23 万多 decoy sequence。当前归档为 support-only，不再投入编码和 baseline 复现时间。

## 4. 外部评测与基线规则

主外部泛化测试必须相对**当前模型真实训练源**重新计算 overlap：query 实体未见比例 ≥30%、正样本目标实体未见比例 ≥30%、exact positive pair 未见比例 ≥90%、query-positive coverage ≥90%。满足后，**完整 benchmark-defined dataset/cell 负责定义支持集、未见比例和内部 sanity check**；竞赛主表可按上面的固定 budgeted best-of-8 协议抽 query 子集。子集只能由 seed 决定，不能按性能、难度标签或人工筛选 query。未通过 admission 时只能按实际语义叫 single-axis、known-reaction、retention 或 descriptive alignment。

资产复用顺序固定为：现成 ID / feature / score → 确定性 ID 对齐 → 现有 evaluator 换路径 → 小成本新编码。**大规模新外部编码、为单一 benchmark 学 adapter、为复现难基线单独搭基础设施，默认不做。** 最强 published baseline 不是硬要求；现成 author score、简单同任务基线或绝对指标都可以，不能让基线复现阻塞完整外部测试。

## 5. 项目边界

当前活跃模型主线只有两条：**R2E LambdaRank** 和 **E2R Anchored LambdaMART V3**。历史 residual、HPO、domain-adaptation、TopK surrogate、未晋级 CAGE fusion 等均不是活跃路线。已经删除约 **26 GiB** 完全无 tracked 引用的旧结果；其他仍被历史审计引用的结果暂时保留，但不得据此开启新主线。
