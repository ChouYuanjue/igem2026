# FIBRE / BiME-Rank / TPS / Starase 原始研发历史档案

> **用途**：本文件不是比赛页面成稿，也不是“当前方法”说明书，而是一份供后续写作、答辩、溯源和团队交接使用的高密度原始历史。它已经做了研究史层面的组织，但刻意保留成功、失败、走弯路、复现修正、协议重定义和工程调优，不把历史压缩成一条漂亮但失真的线。
>
> **范围**：模型、评测、候选宇宙、结构/口袋尝试、TPS 专项、通用检索、BiME-Rank、FIBRE 几何重建、Starase/Catalyst 科研 Agent，以及这些方向的复现与发布治理。**本轮明确不整理数据库专项历史**：数据库并非本文作者负责的工作，且仓库状态未做同等级审计，因此只在确实影响模型/Agent 边界时提到“数据库事实/记录”这一抽象接口，不叙述数据库工程本身。
>
> **证据等级**：正文优先依据 Git 历史、冻结 JSON/Markdown、当前代码和结果目录。用户补充的 Human Practices（HP）事实——“HP 组组织外界广泛试用，真实用户先提出需求，随后我们修改 Agent”——作为团队事实保留；但因为本轮仓库中未发现同等级的原始访谈材料，本文不会虚构具体受访者、企业或逐字引语。

---

## 0. 如何阅读这份“Raw History”

“Raw”不等于“把 commit 按时间抄一遍”。本项目的 Git 历史有两个特点：第一，很多科学尝试以一串连续 commit 逐步冻结、确认、否决；第二，2026-09-18 的 `b1e12c7` 一次性 checkpoint 了此前工作区中近 2.9 万行几何研究，因此单看 commit 粒度会严重低估实际迭代次数。本文采用四层结构：

1. **阶段**：为什么研究问题发生变化；
2. **实验族**：同一个假设被怎样连续测试；
3. **决定**：promote / reject / supersede / keep-as-evidence；
4. **追溯索引**：文末保留逐 commit、逐 `results/` 目录、逐冻结 record/config/script 的全集索引，确保小调优不会因为正文没有单独升格为章节而消失。

全文中“当前”均指 2026-09-27、commit `6f29ccf` 之后的仓库状态。

---

# 第一编：问题形成——从结构/口袋实验到 TPS 全库检索失败

## 1. 2026-07-10：仓库被整理时，研究已经不是空白起点

最早的整理 commit `8c606bf`（`Reorganize project layout by research block`）并不是“开始写模型”的第一天，而是把此前已经存在的研究资产重新编排进统一仓库。此时至少有两条互相关联但尚未统一的工作线：

- **Pocket robustness / EnzymeCAGE 输入干预**：研究口袋来源与多口袋聚合是否改变酶-反应排序；
- **Terpene screening**：把 EnzymeCAGE/P2Rank/反应相似性等方法用于 TPS 候选筛选，并开始面向湿实验候选选择。

这一阶段的重要意义不是“得到最终模型”，而是最早暴露了后来反复出现的两个问题：**有结构信息不等于有好的大规模检索能力；信息源越多也不等于简单融合越好。**

### 1.1 Pocket robustness：先问“哪个口袋”而不是“哪个模型”

当时的口袋矩阵相当完整，至少覆盖：

- 官方预提取 pocket；
- P2Rank top-1；
- P2Rank top-k + `max / mean / rank-weighted / softmax-pool`；
- fpocket top-1 / top-k；
- P2Rank + fpocket union，分别做 max 与 source-weighted；
- 预留 catalytic-residue prior 与 ScanNet residue prior 方向。

Enzyme-405 的 50-reaction slice 中有 50 个有效反应、1,675 个候选 pair、86 个 positive、1,556 个 unique enzymes。代表性结果为：

- official pocket：Hit@5 = 0.54，Hit@10 = 0.68；
- P2Rank top-1：0.54 / 0.68；
- P2Rank top-k 的四种朴素聚合：全部仍为 0.54 / 0.68；
- 但约 52.46% 的样本最高 EnzymeCAGE pocket score 来自非第一口袋；
- fpocket top-1 / top-k：约 0.50 / 0.50；
- P2Rank + fpocket union：仍约 0.50 / 0.50，source-balanced 二阶段校准也没有把结果救回来。

这里得到的不是“P2Rank 永远最好”，而是一组更细的经验：P2Rank top-1 在这个切片上足以替代官方 pocket；多 pocket 明确暴露了**定位不确定性**，但简单扩大 pocket 候选并没有增加 retrieval 命中；fpocket 与 union 还可能引入更多与催化无关的几何凹陷，造成假阳性。这个结论后来会以另一种形式重现：**一个表征在生物学上合理，不代表它应该无条件进入全局排序。**

值得保留的失败边界还有：官方 full-eval 配置引用了缺失的预计算路径，因此当时没有伪造“完整论文复现”；`enzyme405_100` 只有 official-precomputed 方案完整运行，`all_feasible` 更大规模受资源限制。这套诚实的“能跑多少就说多少”的习惯后来成为 BiME/FIBRE 证据治理的一部分。

### 1.2 直接对 1,391 条 TPS 做 CAGE 全库 pair scoring：结构模型几乎不能直接承担 retrieval

`results/terpene_cage_screen/terpene_screen_report.md` 是一个非常重要的原点实验：

- 10 条 Rhea TPS 反应；
- 1,391 条 TPS 候选蛋白；
- 预计 13,910 个 reaction-enzyme pair；
- AlphaFold 结构成功 1,381/1,391；
- P2Rank 成功 1,379/1,381；
- EnzymeCAGE 成功打分 13,790/13,910；
- 但是 Hit@1 / Hit@5 / Hit@10 全部为 **0**；
- MRR = **0.0037**；
- median best-positive rank = **387**。

这是整个项目最值得反复保留的负结果之一。它说明当结构、口袋和 pair scorer 的“计算覆盖率”已经接近 99% 时，**retrieval quality 仍可能极差**。模型会不会给某个 pair 一个分数，与它能否在上千候选里把真正阳性稳定推到前列，是两件完全不同的事情。

### 1.3 相似反应 gate：排名改善，但出现硬召回上限

10-query 的 similarity gate 把 MRR 提到约 0.0622，median best-positive rank 降到 16.5，Hit@3/5/10 到 0.1；说明先找相似反应再传播候选是有效的工程捷径。但代价也直接暴露：

- 10-query positive-pair coverage 仅约 66.4%；
- 放大到 all-Rhea 513 reactions 后，1,640 个真实酶记录中只有 720 个被 gate 覆盖；
- positive-pair coverage 约 **43.98%**。

因此候选池一旦构造完成，池外正例无论后端 CAGE/RF 多强都不可能被“救回”。后来的 RF/HGB/CAGE rescue 实际是在这个 reservoir 内优化，而不是消除 reservoir ceiling。

### 1.4 reaction-only / few-shot / wet-lab intention / gate-matrix：早期已经在区分知识状态

`8c606bf` 同时保留了 `evaluate_fair_cage_reaction_only.py`、`evaluate_fair_cage_fewshot.py`、`evaluate_wetlab_intentions.py`、`evaluate_wetlab_workflows.py` 和 `gate_matrix.py`。这说明项目很早就没有把所有 query 当成一种任务：

- 有些场景完全没有 seed；
- 有些场景已有少量已验证酶；
- 有些场景目标是“推荐列表”，有些场景直接面向实验工作流；
- 相似性、序列、motif、CAGE 等证据会在不同 workflow 中承担不同角色。

当时这还没有统一成后来的“knowledge state / context-conditioned expert”语言，但问题已经出现。

---

# 第二编：V1——TPS 开放世界双向检索系统

## 2. 2026-07-24：不再把任务叫分类或候选池内 rerank

commit `fcffa06`（`feat(tps): add open-world retrieval system and technical report`）是第一次数学/任务定义层面的跃迁。技术报告明确把问题改写为两个方向的开放世界检索：

- R2E：给定反应，找可能催化它的蛋白；
- E2R：给定蛋白，找可能催化的反应；
- query 和 candidate 都允许是注册库中没出现过的新实体；
- 新实体进入候选库后不要求重新训练类别头；
- 输出不仅是分数，还需要可靠性、候选预算和可执行湿实验列表。

### 2.1 主体模型：multi-positive dual tower

当时的主表示为：

- protein：ESM-C 600M mean embedding，约 1,152 维；
- reaction：DRFP 2,048 维 + precursor / product categorical，总约 2,115 维；
- 双塔投影到 256 维单位球；
- 用 multi-positive contrastive objective 处理 many-to-many enzyme-reaction relation；
- 未标注 pair 不直接当真负例，用 50% sequence cluster 与 reaction cluster 做 PU mask。

这一步奠定了后来一直保留的原则：**检索学习的是可扩展的 relation score，而不是固定类别 softmax。**

### 2.2 旧 RF/CAGE rescue 与新 dual-tower 的真实比较：不能只挑漂亮数字

在旧的 exact-reaction holdout、1,391 TPS candidate universe 上：

| 方法 | Hit@5 | Hit@10 | Hit@20 |
|---|---:|---:|---:|
| 旧 reaction-similarity backbone | 31.19% | 36.65% | 41.72% |
| 旧 RF rescue | **34.50%** | 39.57% | 45.22% |
| 新 controlled dual tower | 32.36% | **39.96%** | **46.98%** |

新模型并没有在所有旧口径上全面胜出，尤其 Hit@5 还下降约 2.14 pp。真正的差异出现在共同严格 25-cell double-cold：

| 方法 | Hit@5 | Hit@10 | Hit@20 |
|---|---:|---:|---:|
| 每 fold 重建 reservoir 的旧 RF/CAGE | 0.00% | 0.56% | 1.54% |
| 新 controlled dual tower | **4.20%** | **8.96%** | **16.81%** |

因此 V1 的科学价值不能写成“神经模型把旧模型分数全面提高”，而应写成：**它重新定义了任务，使双向、cluster-cold、开放候选和新实体参与成为原生能力。**

### 2.3 MARTS domain adaptation：专门化没有被“通用化”抹掉

V1 后续不是简单追求一个 universal model，而是保留 TPS 专项：current 数据训练得到 broad/current model，再用 MARTS 进行 domain adaptation；还尝试过 freeze-reaction、PU、hard-negative、R2E loss 权重偏置、neighbor hybrid、mechanism rescue 等多个变体。它们的意义在 Raw History 中要保留：项目很早就发现**specialization 是真实需求，问题是如何在不牺牲开放范围的情况下使用它**。

### 2.4 方向与预算专用 route：Top-3、Top-10、Top-20 并不共享同一最优操作

早期生产系统针对方向和预算做过不同 route：

- E2R 外部 Top-10 使用 reciprocal-rank fusion（RRF）；
- current / external、zero-shot / few-shot 可能调用不同路径；
- Top-3 更重 precision，Top-20 更重 recall；
- seed similarity 与 direct model score 在应用中不能当成同一概率尺度。

这套 route fragmentation 后来会直接推动 Agent/semantic routing，以及 BiME-Rank 对“专家为什么存在”的重新解释。

## 3. 2026-07-29 至 08-13：从模型变成可部署、可校准、可做实验的系统

### 3.1 portable runtime 与 reproducibility

`5a6dc78` 加入 portable production runtime 与 reproducibility assets。之后多次 deployment commit 不是单纯“运维”：它们把 candidate universe、model bundle、route id、hash 和输出解释绑在一起，避免同名 route 背后模型资产漂移。

### 3.2 cycle consistency：试过，但没有因为概念漂亮就上线

08-05 的 cycle-consistency / competition evidence 以及第二轮 cycle grid，测试了 R2E→E2R / E2R→R2E 反查是否能提升稳定性。第二轮 grid 用 12 个 queries、3 个 Top-K objectives、5 个 weights、3 个 applicability gates；最终 `promotion_candidates = 0`，`production_ranking_modified = false`。这是早期“有理论吸引力但没有足够证据就不改生产”的例子。

### 3.3 conformal retrieval set：尝试把“不确定”变成集合而非一句 confidence

`cd8cfec` 建立 normalized-best-positive-rank split-conformal：

- 1,215 unique query calibration units；
- 6 个 calibrators；
- alpha = 0.20 / 0.10 / 0.05；
- global holdout coverage 全部通过；
- 一部分 strong/moderate Mondrian groups 可启用，样本不足或经验 coverage 不够的组自动退回 global。

它没有成为最终 FIBRE 的核心数学对象，但留下两条方法习惯：**calibration population 必须匹配；无法校准时不要把内部 score 包装成概率。**

### 3.4 taxonomy scope 与真实用户要求的候选约束

08-09 将候选蛋白按本地可支持 taxonomy 信息保守地分成 eukaryote / prokaryote / other / unknown。随后 `requested_r2e20_prokaryote_fewshot_20260809` 在 20 个 sheet rows / 17 个 reaction groups 上输出 200 个 prokaryotic candidates，并混合 zero-shot 与 seed-guided groups。

这不是一个新的模型，而是一次重要的使用层经验：研究者问的通常不是“给我全库排名”，而是“在可表达、可采购、可在目标宿主工作的候选约束下给我可测列表”。这类约束后来不应被塞进模型 score 本身，而应作为任务/候选域和 Agent orchestration 的一部分。

### 3.5 staged experimental touch 与 131,532 条本地候选宇宙

08-13 的实验工作流把本地 GBK-derived 候选扩到 131,532 条，针对 17 个 reaction groups / 20 sheet rows 生成 200 个候选：

- exact known-positive overlap = 0；
- 11 个 hybrid groups，6 个 direct groups；
- prokaryote scope 强制执行；
- direct-model score 与 seed ESM-C cosine 明确标注为“ranking score，不是概率”，不同原始尺度不能直接混当概率。

这一步证明 V1 已经从 benchmark 进入真正候选构建，但也暴露了 route、scope、seed、evidence 状态越来越难让普通用户手工选择。

---

# 第三编：真实用户反馈推动入口变化——从 route menu 到科学 Agent

## 4. 2026-08-20：正式 Agent 之前，真实用户试用已经迫使语义路由重写

团队的 HP 事实是：**HP 组组织外界广泛试用，真实用户先提出需求，团队随后修改系统。** Git 历史支持“需求变化先于正式 agent harness”的时间结构：08-20 已出现一整个 semantic-routing 改造批次，而 `bounded scientific agent harness` 到 08-28 才正式出现。

这批改动应当作为一个实验族理解，而不是九个同级产品 commit：

1. `94cfdfe` 一次性加入/重构 intent routing、ambiguous task handling、R2E/E2R routing graph、route design、pathway compatibility、thermodynamics/FBA/feasibility、feedback report；
2. `f85e4af` 为低置信 intent 加 confirmation；
3. `9609021` 修复 route/pathway 的 autonomous intent routing；
4. `5be0dd6` 允许对话中切换 intent；
5. `73d96e4` 将“只问潜在反应”的 E2R follow-up 送到 novel-discovery scope；
6. `1bc369a` 开始用 DeepSeek 做 semantic scope selection；
7. `1233c00` 统一语义权威；
8. `6c1faf9` / `35d453f` 把完整上下文带入 scope switching 与 continuation。

从研究史角度，这里真正发生的是：**模型 route 的复杂性超过了人应该承担的认知负担。** 这不是“为了做 AI Agent 而做 Agent”，而是用户反馈说明“科研问题是自然语言和连续上下文，内部 route menu 只是实现细节”。

08-26 的 evidence-first bilingual UI、中文科研任务 guidance 和 readability 改动也属于这条用户反馈线：不改变 retrieval 数学，但影响系统是否能被非开发者真正使用。

## 5. 2026-08-27 至 08-29：候选宇宙扩张与正式科学 Agent 同时发生

### 5.1 general candidate universe：TPS 不再等于搜索空间

`c76afb2` 将 Catalyst retrieval 重构到 general universe。后来稳定下来的通用规模是 185,918 proteins 与 11,081 reactions。TPS 从“整个候选库”逐渐变成一个 specialization / family state；这一转变为后面的 BiME-Rank 和最终 atlas 留出了空间。

### 5.2 08-28 bounded scientific agent harness：把此前 routing 经验收敛成可约束执行框架

`4f13385` 建立 contracts、session store、tool registry、evidence query、entity resolution 和 bounded harness。紧接着：

- `305add9`：从规则入口改成 model-led end-to-end agent；
- `e9bf41f`：structured protein recovery；
- `3db10d0`：raw reaction evidence resolution；
- `9d8999b`：扩充 scientific tools，并支持 Markdown 结构化输出；
- `eb088c1`：verified entity memory 与 UX；
- `25fddd4`：ChEBI 前的 compound-name normalization。

这里的“bounded”非常重要：LLM/semantic layer 可以做语言理解和有限决策，但实体 ID、候选 universe、科学不变量和可追溯 evidence 仍由 deterministic components 保证。后来的 Starase 继续沿用这条边界。

### 5.3 08-29 research workspace：Agent 第一次从“选 route”变成“持续科研状态”

`5c0be8f` 加入 research workspace、多轮 state 和 `scientific_research_service.py`；随后 composable workflows、generalized followups、evidence synthesis/pagination、local-first recorded associations、relation/protein evidence、session-target provenance、完整 source list 和 literature coverage 相继出现。

从此 Agent 有两个清晰阶段：

- **阶段 A：routing convenience**——让用户不用学习模型菜单；
- **阶段 B：science workspace**——保存对象、证据、上下文和后续操作，让“得到候选”不再是对话终点。

这条演化是用户需求驱动的产品/科研接口事实，但后续模型从 BiME 到 FIBRE 的数学重构仍需由实验与理论本身证明，不能把 HP 反馈当成数学因果替代品。

---
# 第四编：V2 的形成——通用检索研究矩阵与 BiME-Rank

## 6. 2026-08-30：不是“一次做出 MoE”，而是一次密集的研究爆发

08-30 至 09-03 是仓库里实验密度最高的一段。若只说“我们加入了多个 expert”，会遗漏真正重要的研究过程：团队几乎同时在问五类问题——能否继续训练一个统一模型、如何防止旧能力遗忘、哪些表征真的有增益、怎样组合不同检索器、怎样避免在 benchmark 上自我欺骗。很多后来成为 BiME-Rank 设计原则的东西，恰恰来自失败实验。

为了不把 200 多次提交写成流水账，下面按**科学假设族**整理；每个族内部保留关键版本、调参方向和否决原因。逐 commit 完整索引见附录 A。

## 7. 实验族 A：继续训练一个统一模型，还是保留多条能力路线？

### 7.1 general known-recovery 与 continuation：先尝试“在旧模型上继续学”

一开始并没有直接选择 MoE。`e6fdbd0` / `a60b7d5` 建立 general known-recovery evaluator 并做加速，随后 `a06fc7a` 引入 full-evidence directional continuation，`da90f30` 加 strict known-recovery retention guard。目的很直白：能否在吸收更广泛关系的同时不丢掉原有 TPS/已有检索能力？

后面连续尝试了多种 continual-learning / model-merging 思路：

- checkpoint blending；
- directional continuation towers；
- RecAdam；
- FusionBench Fisher consolidation；
- source-weighted RegMean；
- Mammoth LwF；
- bidirectional score-matrix distillation；
- Margin-MSE 保留相对排序 margin；
- Pareto 选择而不是单指标挑模型；
- bootstrap broad recovery gains；
- TIES merging；
- post-hoc domain expert routing；
- AdaMerging。

`3e0b863 Reject mismatched recovery ensembles` 很早就说明：简单把不同来源模型拼起来可能在某些恢复指标上好看，但不能维持同一 retrieval semantics。后续的 retention gate、Pareto guard、exact-positive-rank export 都是在阻止“新数据一来就把原系统冲坏”。

### 7.2 这条线留下的结论

没有一种统一 continuation 方法自然解决所有方向/场景；不同知识状态、候选支持和表示法的优势明显是条件性的。于是“保留多个专家”不是先验架构偏好，而是**统一持续训练反复遭遇 retention trade-off 后的工程/统计结论**。这也是后来 BiME-Rank 的专家池比传统 MoE 更强调 admission、fallback 与 provenance 的原因。

## 8. 实验族 B：cleanroom 与评测纪律——先防止高分来自暴露，再谈模型创新

这一阶段另一条几乎同等重要的工作不是模型，而是不断重写“什么数字允许相信”。

### 8.1 从 broad benchmark 到 nested / full-candidate / train-only negative

连续引入：

- leakage-aware fair benchmark evaluator；
- 更完整的 retrieval metrics；
- ReactZyme native cutoff；
- broad leakage-clean Rhea benchmark；
- leakage-controlled retraining；
- exact full-candidate metrics；
- full-candidate benchmark runner；
- cold benchmark negative 只允许 train-only 产生；
- leakage-safe nested dev splits；
- cleanroom + capability evaluation；
- mandatory baselines；
- multi-metric routing guard；
- joint reliability evidence；
- external baseline metric provenance；
- “audit gains without metric cherry-picking”；
- exact positive-rank export。

这里形成了后来非常典型的实验生命周期：**proposal → preregister/freeze → internal development → untouched confirmation → promote/reject → revealed set 禁止继续调参**。

### 8.2 seed ensemble / hard-negative curriculum / bootstrap 也必须先冻结

项目没有把“seed ensemble”当作可以无限刷的免费提升：先 preregister，再 freeze before holdout。hard-negative curriculum 也一样。`paired bootstrap ranking audit`、Enzyme-405 uncertainty、difficulty slices 等工作把“是否偶然”与“在哪类 query 有增益”变成标准报告内容。

### 8.3 一个后来被撤回的高分：旧 projected-cold 并不是真的 current-cold

历史上曾出现 ReactZyme projected double-cold 上极高的 E2R 数字（MRR 约 0.71、Hit@10 约 93%）。后来重新相对 current clean2023 training source 做 exposure audit，发现 protein/reaction/pair 大比例已经出现，旧 `cold` 只对旧 partition 成立。该 headline 被删除并降级为历史能力诊断。

这是 Raw History 必须保留的“自我推翻”：**评测名字不能代替相对当前训练源的真实 overlap。**

## 9. 实验族 C：Horizyn——先试 loss，再试 representation，不把论文名当魔法

官方 `FullBatchMLNCELoss` 被直接复用到现有 cleanroom features 上，协议在运行前冻结。结果：

- existing baseline mean joint percentile ≈ 0.9833；
- Horizyn MLNCE ≈ 0.5167；
- SR@1 有极小上升；
- SR@3/5/10、DCG@10、EF@1%、MRR、MAP、AUROC、NDCG@10 全面下降。

因此 `f464080` 明确记录 **negative Horizyn MLNCE result**，不允许花 outer benchmark 预算继续验证这个 candidate。之后才把 Horizyn 拆成 representation hypotheses：

- ProtT5 bridge；
- reaction-feature distillation；
- protein-space distillation；
- adapter；
- residual / fusion；
- canonical exact residual；
- distillation preprocessing 的可复现重建。

这条线的教训是：论文中一个整体系统有效，不意味着其中某个 loss 单独移植就有效；必须隔离变量。

## 10. 实验族 D：RDKit+——更丰富 reaction 表示确实有价值，但 novelty routing 不稳定

RDKit+ 作为反应表示经历 preregistration、outer-freeze、paired bootstrap 和后续多模型复用，最终成为通用模型重要的 reaction-side base。但“RDKit+ 对低相似度 query 更有优势”这个直觉被单独做成 novelty hard-slice route 时没有通过。

`CLEANROOM_R2E_RDKITPLUS_NOVELTY_HARD_SLICE_V1` 在 reaction similarity < 0.3 的 204 queries 上：

- MRR 0.02867 → 0.03721；
- MAP 0.02170 → 0.02953；
- Hit@50 14.71% → 18.63%；
- 但 Hit@10 9.80% → 8.33%；
- macro-AUROC 略降；
- fold 间方向不稳定；
- 全 R2E 上 Hit@10 / Hit@50 也有退化。

因此阈值 0.7 / repeat=1 的 novelty expert 被直接拒绝，不允许看到结果后继续扫阈值。后来如果再做 low-sim mechanism，必须另起 preregistered protocol。

## 11. 实验族 E：reaction center——“科学上有信息”与“能进入全局模型”被明确区分

这是后续几何重构最重要的实验前史之一。

### 11.1 V1 直接拼接 reaction-center features：总体提升，hard slice 早期排序失败

显式 mapped reaction-center channel（约 4,419 维组合）在 3,226 个 R2E queries 上有明显 aggregate 增益：MRR 0.09861 → 0.10987，Hit@20 / Hit@50 也提升；低相似度 slice 的深层指标亦变好。但 preregistered hard-slice gate 关注 NDCG@10 / Hit@10，二者轻微退化，因此 `02372dc` 仍然拒绝 V1。

这个决定非常关键：项目没有把“整体平均高一点”当成无限正当化。结论是**reaction-center information scientifically useful，但这个 concatenation 不是可晋升形式**。

### 11.2 identity-preserving residual V2 与 bounded V3

随后不是丢弃 reaction center，而是把它做成更保守的 residual：初始化为 identity、限制扰动、冻结 base。V2 在 fresh confirmation 上通过，V3 再做 bounded selection / confirmation，成为一度被确认的 R2E center capability。

### 11.3 fresh Rhea128→141 外部 transfer 又把它打回去

真正新鲜的 Rhea release128→141 Swiss-Prot strict double-cold v2 上：

- base MRR 0.01282；
- reaction-center residual MRR 0.00976；
- MAP / NDCG@10 / Hit@10 / Hit@20 / median rank 均变差；
- 只有 macro-AUROC 与 Hit@50 有改善。

因此 external promotion 被拒绝，并明确规定不能基于这次 reveal 再调 center dimension、mapping、learning rate、loss weight、route threshold 或 snapshot definition。

这段历史后来几何阶段再次出现：**局部化学信息不能因为“看起来更机制化”就被强塞进一个全局 metric。**

### 11.4 comprehensive directional center：几乎全面改善，仍因预先定义的 material-gain gate 不够而不晋级

09-01 的 comprehensive directional center 把 EnzGFM / equal-block protein features 与 bounded center 组合。R2E、E2R 三个 fold 都没有明显回退，hard low-sim R2E 甚至有较大改善；但 pooled gain 没达到 preregistered “material” threshold，因此没有跑 confirmation，决定仍是保留现有 routed mainline。

### 11.5 center + Top-2000 residual：局部 Hit@K 增益不能抵消 MRR 回退

另一次组合中，Top-2000 residual 在 high-sim queries 上使 Hit@10/20/50 上升，但 pooled MRR 轻微下降，material-gain gate 仍失败。更细的 fallback audit 还发现 scale=0 时因 Top-2000 re-sort 导致一个 tie-order mismatch；修复 generic fallback 后也没有重开 promotion，因为核心 gate 仍未通过。

## 12. 实验族 F：functional prototype residual——九个候选都没过，不靠“机制解释”救模型

`CLEANROOM_R2E_FUNCTIONAL_PROTOTYPE_RESIDUAL_V1` 对 residual scale 与 confidence margin 组合形成 9 个 candidate。结果每个都至少违反一组预声明约束，例如 primary macro-AUROC、Hit@10、NDCG@10、median rank 或全局 Hit@50/MRR。最终 `b960cb8` 记录 `reject_v1_no_confirmation_no_retuning`。

这里的历史意义是：family/prototype 层面的生物学直觉没有被自动升格成生产专家；只有“解释得通 + 在预定协议中通过”才允许晋级。

## 13. 实验族 G：EnzGFM——从表示替换走向 direction-aware block fusion

EnzGFM 的测试比“换一个 embedding 看分数”复杂得多：

- 先 preregister protein representation test；
- balanced bidirectional cleanroom selection；
- equal-block feature fusion，避免高维 block 仅凭维数占优；
- Gate A 比较 ESM-C / EnzGFM / equal-block；
- Gate B 与 RDKit+ 联动；
- full-candidate completion scope；
- partial-feature guard；
- common Gate-A universe；
- nested outer selection；
- temporal protein-cold directional router；
- native same-support benchmark 与 official asset audit。

结果并不是“EnzGFM 全面替换 ESM-C”，而是不同方向和 candidate support 下，EnzGFM / ESM-C / equal-block 的最优使用方式不同。这进一步强化了“representation is conditional evidence”的观点。

## 14. 实验族 H：CLIPZyme——结构表示的价值与 support boundary 同时被审计

CLIPZyme 研究同时做了：

- train-overlap audit；
- reaction similarity audit；
- stereo-robust canonicalization；
- graph-input compatibility；
- support-aware reaction extractor；
- native protein support；
- directed reaction fallback；
- ReactZyme native support；
- 明确 **revert encoder substitution**，不把自己替换 encoder 的版本冒充 official CLIPZyme。

这条线最后成为 BiME-Rank promoted structural expert 的重要基础，但它从一开始就带着“availability-aware”的条件：结构 expert 没有 support 时必须 exact fallback，不能把 missing structure 解释成 low score。

## 15. 实验族 I：R2E learning-to-rank——从 similarity router 到确认过的 LambdaRank

R2E 先有 fast similarity router，冻结后 `7f00476` promoted。随后做 automated LambdaRank fusion：

1. preregister；
2. cache 与 full evaluator 数值对齐；
3. 固定 rank-based AUROC semantics；
4. 固定 hard-negative sampling；
5. materialize caches once；
6. 选定 config；
7. fresh confirmation evaluator；
8. confirmation passed；
9. candidate runtime；
10. production route promotion。

这一条是 V2 最典型的“多个专家不是直接平均，而是 learning-to-rank 学习在 frozen candidate evidence 上组合”的成熟路径。

## 16. 实验族 J：E2R learning-to-rank——先失败，再用 anchoring 保护强基线

E2R 的路径更能说明为什么最终方法不是简单 MoE。

### 16.1 unconstrained four-expert LambdaRank 失败

V1 stack：

- baseline MRR 0.09841 → stack 0.05254；
- Hit@10 27.33% → 11.92%；
- Hit@20 35.95% → 23.37%；
- AUROC 反而略升。

这表明 unconstrained reranker 可以优化全局区分度却毁掉研究者真正关心的 early ranking。于是 V1 被明确拒绝，并禁止在同一 dev 上改 shortlist size、depth、rounds、learning rate、hard-negative count、feature set。

### 16.2 baseline-anchored rescue 与 Anchored LambdaMART V3

后续设计把强 EnzGFM order 变成结构约束：保留 Top-1，只在有限 expert union / positions 2–20 中学习重排，其余 tail 保持 baseline。Anchored LambdaMART V3 再经过独立搜索、tie-break、confirmation、hash lock、runtime gate，最终 promoted。

这条线留下的设计原则是：**强 base rank 不是另一个待融合 expert，而是需要保护的几何/顺序先验。**

## 17. 实验族 K：TPS successor——基础模型、active-site XAttn、MARTS symmetry 与纯 CAGE baseline

### 17.1 TPS foundation model

09-02 单独构建 TPS foundation features、R2E development runner 和 temporal support，不把 broad model 的结果自动当成 family-specialist 结果。

### 17.2 active-site cross-attention：一次非常重要的“作废实验”

TPS active-site XAttn 尝试了 token materialization、fresh transfer split、final ranking semantics、hard pools、residual training、guarded HPO、runtime freeze 等完整链条。第一次 confirmation 后才发现 candidate ID alignment bug：token asset 的 1,421 蛋白顺序与 transfer score 文件不一致，导致 evaluator 用错 row index。

`99df860` 把**全部 V1 performance metrics 明确标为 forbidden**，并规定已经 reveal 的 folds 3/4 永远不能再做 confirmation；只允许在纯 evaluator 修复后重做 development，并把 replacement confirmation 移到 fresh MARTS temporal protein-cold。

这是整个仓库科学完整性最值得保留的例子之一：一个复杂模型即使已经“跑完 confirmation”，只要发现 identity/provenance 错误，就必须把漂亮数字全部作废。

### 17.3 MARTS 双向 symmetry 补齐

为了避免早期只讲 E2R 或混用两个 TPS universe，09-03 冻结 MARTS R2E symmetry confirmation。在 1,421 proteins × 453 reactions 的同一 universe 上，155 frozen R2E query-cells：

- baseline MRR 0.03890 → 0.04183；
- Hit@10 5.81% → 6.45%；
- Hit@20 15.48% 保持。

定位是 precision stabilization，不夸大为 SOTA。

### 17.4 pure EnzymeCAGE baseline 重建：纠正“RF/CAGE 就是 EnzymeCAGE”的旧叙述

后期把 external EnzymeCAGE 官方式 retrieval pipeline 在 TPS common support 上重新跑通，并明确把历史 RF/HGB+CAGE hybrid 改称**内部旧系统**。459 reactions × 1,379 proteins common native support 上：

- pure EnzymeCAGE Hit@10 ≈ 30.28%，Hit@20 ≈ 39.65%；
- Catalyst locked route Hit@10 ≈ 47.49%，Hit@20 ≈ 56.21%；
- delta +17.21 / +16.56 pp。

同时 raw neural full-matrix diagnostic 只有约 1.95% / 2.38% Hit@10/20，进一步说明 EnzymeCAGE 自身也高度依赖 candidate retrieval gate。

## 18. 实验族 L：外部 baseline 与“缺什么就写 N/A”

### 18.1 Enzyme-405

最终可执行的 strict local same-support 是 226 reactions / 11,665 canonical pairs。5 个 official EnzymeCAGE checkpoints（seed 40–44）本地重跑：SR@10 51.33±1.66%；Catalyst 49.12%。但 Catalyst MRR/MAP/DCG@10 略高。因此严格结论是**same-order performance with mixed metric wins**，不是强行宣布谁全面胜出。

### 18.2 Orphan-335

保留作者 top-10-similar-Rhea retrieval pool：335 queries，90,804 canonical pairs，44,889 candidate UIDs。作者 Selenzyme vs Catalyst：

- MRR 0.2088 → 0.3007；
- MAP 0.2113 → 0.2942；
- NDCG@10 0.2092 → 0.3124；
- Hit@10 31.04% → 48.36%。

但只有 233/335 query 的 author pool 含至少一个 positive，而且 positive-protein novelty 很有限，所以后来只作为 secondary reaction-novel stress test。

### 18.3 CLIPZyme common support

official checkpoint 与 Catalyst 在共同 native support 上做过 local descriptive comparison；但仓库后来重新审计 current-training exposure 后，旧 projected-cold headline 被降级。真正 current-strict 的 temporal evidence改用 Rhea release128→141 v2。

### 18.4 EnzymARC

支持审计几乎完整，但完整 scoring 会要求重新处理大量 decoys。项目选择做 support/QC 和 source evaluator，而不是为了“baseline 列表更长”投入新的大规模编码。难复现 baseline 不允许阻塞自身外部测试。

---

# 第五编：BiME-Rank 收敛——从实验矩阵到受约束的多专家系统

## 19. 2026-09-05：结构专家被正式 admission，而不是“看起来合理就加进去”

`3d8f24c Promote BiME-Rank structural expert routing` 标志多专家体系从实验矩阵进入正式 admission。`BIME_RANK_EXPERT_ADMISSION_V1.json` 固化了几条规则：

- expert 必须先在 internal clean development 选择；
- frozen external / temporal double-cold 只允许 veto，不能用于 retune；
- missing expert = unavailable，不是低分；
- expert unavailable 时必须 exact fallback 到 incumbent route；
- candidate manifest 与 production manifest 分开。

### 19.1 R2E CLIPZyme structure expert

内部：MRR 0.11645 → 0.13485，Hit@10 26.85% → 31.06%，Hit@20 35.16% → 40.73%。严格 external temporal evidence 后来的 frozen metrics约为 MRR 0.08733、MAP 0.08018、NDCG@10 0.08780、Hit@10 13.19%、Hit@20 15.97%、Hit@50 22.22%。

### 19.2 E2R CLIPZyme structure expert

内部 V3 → V4：MRR 0.13234 → 0.18540，Hit@10 35.16% → 47.60%，Hit@20 46.64% → 59.17%。strict 248-query / 10,131-reaction support 上，5-expert route MRR 0.02777、MAP 0.02438、Hit@20 15.32%；与 official CLIPZyme 比较时并非每个浅层指标都赢，因此仍按 same-support mixed evidence 描述。

## 20. Context / seed expert：有已验证 pair 时才改变第二阶段，zero-shot 顺序先形成

`e813d52`、`d69e700` 把 known-positive context、multi-seed context 和 evidence contract 统一起来。核心 invariant 是：

1. 先形成 frozen zero-shot BiME order；
2. 只有存在合法 verified seed 时 context expert 才激活；
3. seed IDs 要 mask，避免 trivially return input；
4. tail 保留 zero-shot BiME order。

这不是把 few-shot 和 zero-shot 混成一个分数，而是明确“观测条件变化后允许一个条件性 second-stage expert”。

## 21. Cost-aware hierarchical expert execution：专家越多，不能每次全跑

`420eb90` 将 expert admission 进一步变成成本分层：先用廉价、覆盖广的表示形成粗候选，再只对 shortlist 或 support-covered query 运行昂贵结构/上下文 expert。这里第一次把**计算成本**当成方法约束，而不只是部署优化：真正的 open-world system 不可能为 185,918 × 11,081 的潜在 pair 全部运行最重结构模型。

## 22. 09-07 至 09-08：发布、归档、证据边界本身成为研究成果的一部分

这两天大量 commit 看似“chore/release”，但 Raw History 必须保留它们的科学意义：

- canonicalize assets；
- archive superseded evidence；
- judge release 与 canonical evidence 对齐；
- portable / full-asset release checks 分离；
- source role classification；
- lineage-only tests 降级；
- canonical claim provenance；
- model asset hash / rebuild contract；
- pytest release boundary；
- current route evidence 与 legacy runtime weights 分离；
- scientific data/result surface lock；
- **correct current wetlab reproduction boundary**；
- **demote legacy TPS wetlab campaign bundle**；
- ReactZyme/Horizyn preprocessing/candidate universe exact replay；
- 记录每个 production model lineage。

这段工作修正了几个后来叙事中很容易犯的错误：旧 TPS wet-lab campaign 只能支撑 TPS lineage/application boundary，不能自动变成 BiME/FIBRE 的湿实验验证；historical runtime result 也不能因为文件还在就继续做 headline。

---

# 第六编：从“多专家怎么组合”到“催化关系究竟是什么”——几何重建

## 23. 2026-09-09 至 09-18：大量几何实验在一个 checkpoint 中集中出现

`b1e12c7 checkpoint: preserve pre-refactor retrieval and geometry state` 一次新增约 149 个文件、近 2.9 万行。它不是“一天突然发明一个 product manifold”，而是把一段此前未逐 commit 固化的研究工作整体 checkpoint。因此这段必须按结果资产/文档族还原，而不能用 commit 数量衡量迭代密度。

核心问题发生了变化：BiME-Rank 已经证明多源信息与专家 routing 能提高检索，但从解释上仍然像“很多 score 的受控组合”。新的问题是：

> pair labels 稀疏，但 reaction 与 enzyme 两侧的 molecular observations 丰富且异质。能否先给两侧建立几何，再把已知 pair 看成两侧之间的 sparse correspondence，而不是把每种信息都训练成一个额外 scorer？

## 24. Product-manifold / correspondence-defect 主假设

定义 reaction space \(M_R\)、enzyme catalytic-state space \(M_E\)，用 product metric

\[
d_M((r,e),(r',e'))^2=d_R(r,r')^2+d_E(e,e')^2.
\]

已知正关系 \(\Omega\) 形成 product-space sparse support：

\[
J_\Omega(r,e)=\min_{(r_i,e_i)\in\Omega}
[d_R(r,r_i)^2+d_E(e,e_i)^2].
\]

再定义 marginal novelty

\[
m_R(r)=d_R(r,\Omega_R)^2,\qquad m_E(e)=d_E(e,\Omega_E)^2,
\]

以及 correspondence defect

\[
\Delta_\Omega(r,e)=J_\Omega(r,e)-m_R(r)-m_E(e)\ge0.
\]

这套理论第一次把“pair compatibility”解释为：两个对象各自可能都很 familiar，但它们最近的 reaction precedent 与 enzyme precedent 能否由**同一个已知 pair**共同解释。也首次明确完整生物对象本应包含 context：\(\Gamma\subset M_R\times M_E\times M_C\)。

## 25. 几何实验族全集：不是一个公式，而是对“怎样延拓 sparse relation”的系统搜索

这段研究至少包含以下互相竞争的 operator / geometry families；它们的 `results/` 目录都保留在附录 B：

### 25.1 Product geodesic 系列

`terpene_product_geodesic_dev_v1/v2/v3/v4`、normalized product geodesic、atlas-refined geodesic、geodesic potential。主要探索 product-space shortest-path / intrinsic normalization 是否足以让 sparse positives 形成稳定 compatibility。

### 25.2 Coupling defect / correspondence

`product_coupling_defect_v4b`、`product_correspondence_dev_v1`、`strict_inductive_v1`。这里逐渐从“绝对离已知 pair 多远”转向“joint precedent 相对两个 marginal precedent 多出的 coherence cost”。

### 25.3 Energy / PDE / heat 类延拓

包括：

- symmetric correspondence energy；
- free-energy correspondence；
- heat-density ratio；
- anisotropic product measure；
- screened Poisson correspondence；
- stable level-set correspondence；
- multiplex product heat；
- multiresolution product heat v1/v2/v3；
- geometric product flow clean-dev v1 到 v11。

这些尝试的共同目标是让 sparse positive measure 在 product geometry 中传播，但每种方法对平滑、尺度、各向异性、边界和 support density 的假设不同。最终没有把这些全部塞进最终 FIBRE，而是保留为“为什么要寻找更局部、更弱假设结构”的研究证据。

### 25.4 Lexicographic / Pareto / partial relation

`lexicographic_correspondence_v2/v3`、partial biological relation 等尝试处理“多个 defect coordinate 无法合理压成一个标量”的问题。最终更偏向 Pareto：一个 candidate 只有在所有声明坐标都不差且至少一项更好时才 dominate；否则标记 trade-off / incomparable。

严格 audit 发现很多原先 global score 能强行排序的 pair，在 pocket/global multi-coordinate 下会变成 trade-off；front-0 并没有膨胀到毫无区分度。但 strict-inductive 仍有小回退，因此 partial relation 最终作为非排序的解释层，而不是替换 total rank。

### 25.5 Multiresolution factor geometry

reaction 与 protein 两侧都做过多尺度：

- reaction multiresolution v1/v2；
- protein multiresolution v1-v4；
- hierarchical factor geometry v1/v2；
- global ESM-C aligned / sequence fill；
- reaction atlas refined；
- tangent / local information geometry。

目的不是多造模型，而是问：global sequence/whole reaction 与 catalytic-local observations 是否应在同一 factor metric 的不同尺度表达。

### 25.6 Structure / pocket / motif / active-site coordinates

尝试过：

- P2Rank current；
- pocket local aligned v1/v2；
- pocket sequence；
- 3Di / pocket-3Di / pocket-OT builders；
- family-aware motif coordinates；
- TPS active-site / structural observations；
- mechanistic chart geometry；
- correspondence structural basis / deployment atlas。

这条线最终产生一个非常明确的 coverage objection：当前 canonical 185,918 proteins 中，materialized P2Rank pocket 只有约 1,287 条（约 0.69%）。许多 432-node local protein charts 的 median pocket count 甚至为 0。若把 pocket 直接放进 global \(g_E\)，大多数地方实际上没有形成“局部结构几何”，只有孤立 observed points；若为了让它起作用再加 availability gate，又破坏“一个 global metric”的原始主张。

## 26. Exact seed update：train-free 更新第一次有了严格代数性质，但也暴露副作用

在 min-plus correspondence 中加入 verified positive，可以把 joint/marginal transforms 用 pointwise min 精确更新，结果与 full reconstruction 完全一致。`terpene_correspondence_seed_update_audit_v1` 验证 incremental/full field difference = 0。

但“代数精确”不等于“所有 unrelated query 都变好”：130 次 leave-one-seed interventions 中，平均 unrelated-query RR worsen fraction 约为 E2R 8.1%、R2E 4.4%。影响 footprint 与平均收益有一定相关，但与 worsen fraction 几乎无关。于是项目拒绝凭 anchor density 设计手调 penalty，而改为记录 influence footprint / novelty provenance。

这条经验后来转化为最终 atlas 的**bounded finite-rank update**：允许实验影响模型，但必须给影响幅度一个数学预算。

## 27. Level-set uncertainty：确定性 score 中也可能存在真实的“不可分辨”

min-plus defect 会出现 exact / machine-scale ties。level-set audit 显示：

- E2R 约 29.6% queries 的 best-positive rank 有非平凡区间；
- R2E 约 18.1%；
- dense reference 与 exact section 的 joint min-plus cost 可完全一致，但浮点减法顺序造成约 1.78e-15 的差异，就足以在近似平 level 内交换 candidate 顺序。

因此历史 geometry 引入 optimistic/pessimistic/neutral expected rank 与 expected reciprocal rank，而不是把稳定 ID tie-break 冒充科学分辨率。最终 interaction atlas 没继续以 level-set 为核心，但“输出分数不等于已校准置信度”的纪律被保留。

## 28. Reaction-center tangent information：一次直接推动理论转向的负结果

为了尽量不破坏 global reaction geometry，后期最保守的 reaction-center 尝试只在已有 base edges 上增加 tangent energy：

- 不新增长程边；
- 不删除 base edge；
- missing local observation 完全中性；
- duplicate refinement 幂等；
- refinement 不能缩短 base edge。

即便如此，`fibre_reaction_tangent_information_dev_v1` 中 E2R MRR 仍从 0.07230 降到 0.04923，paired bootstrap RR delta 的 95% CI 全部低于 0；R2E 也没有可靠增益。结论不是“reaction center 没价值”，而是：**它不适合被强制解释成 global reaction geodesic 的一个统一 metric correction。**

## 29. Atlas consistency crisis：理论声称 one field，但 direction-centered sections 尚未证明相同

旧 geometry 最强的哲学主张是 R2E/E2R 应当是同一个 scalar field \(F(r,e)\) 的 fibres。但实现为了计算效率分别做 reaction-centered 与 enzyme-centered local sections。`legacy_geometry/atlas.md` 很诚实地指出：如果 overlap 上数值不一致，就不能说它们是同一个 global field 的 restrictions。

当时提出的修复候选是 partition-of-unity variational assembly，甚至设计了 query-independent tensor-product atlas / Galerkin-Nyström discretization。但文档也明确说：**数学更漂亮不是 promotion 理由**，必须实际减少 section defect、保持 no-leakage、且 retrieval 不退化。

这段“自我质疑”直接预示了最终 FIBRE：partition of unity 被保留，但“所有 molecular views 必须先压进两个 global manifolds”被放弃。

## 30. 09-19 至 09-20：从 correspondence field 走向 stratified / mechanistic / partial relation

`1c84205` 引入 biological correspondence stratification；`c54d33a` integrate mechanistic strata；`c334b38` 暴露 exact seed stability provenance；`5078f70` 加 partial biological relation。此时团队已经在主动松动“一个 total scalar order 包办所有生物信息”的要求：

- global correspondence 保持检索主线；
- pocket / mechanism / motif 可形成 local strata；
- local strata 若 strict-inductive 不稳，就只解释、不排序；
- missing coordinate 不补 0、不当 negative；
- context-specific inactivity 不升级成永久 global negative。

这正是 product-manifold 走向 interaction-atlas 的理论中间态。

---

# 第七编：Starase 第二阶段——从模型入口变成可持续科研工作区

## 31. 2026-09-21 至 09-22：Agent 不再只解决“调用哪个模型”

这两天的改动应按科研工作流理解：

- FIBRE biological evidence 与 Starase agent 合并；
- controller action recovery / public runtime 恢复；
- 优先展示 informative enzymology evidence；
- conversation isolation 与 layered context；
- compound stereochemistry normalization；
- scientific cards 与 evidence bundle；
- workspace objects 可直接复用；
- route 结果数量语义保留；
- AI-native local route patching；
- derived route lineage；
- planning budget-aware；
- evidence count provenance；
- contextual recommendation 保持 user-facing；
- route design 绑定 verified compound identities；
- pathway ambiguity 用 verified refs 消解。

因此 Agent 的第二阶段不是“聊天界面更好看”，而是把候选、证据、路线、结构和后续操作变成**可复用科研对象**。它与模型数学并行演化：Agent 吸收 workflow complexity，FIBRE 解决 interaction modeling；两者互相服务，但不能把其中一条伪装成另一条的唯一原因。

---

# 第八编：V3——FIBRE interaction atlas 最终重定义

## 32. 2026-09-26：先把 context domain 说清，再放弃过强的 global geometry

`49fde84` formalize context-restricted domain，明确真正物理 activity 应与 assay/process context 有关；`ea202c5` 让 verified agent observations 可复用；`65a1b5a` 尝试 tensor-product atlas field。这些仍保留 product-field 的“统一对象”冲动，但已经开始把 local coordinate 与 observation context 分开。

随后 `f358a8a` 增加 interaction-atlas primitives，`f8a51f9` 正式 **redefine FIBRE as interaction atlas**。这是 ontology change，不只是换名。

## 33. 最终对象：不是多个 expert 的投票，而是多个局部 interaction charts

当前定义先区分：

- 物理层：\(A(r,e;c)\)，context-conditioned catalytic activity；
- 任务层：\(K^\star(r,e)\)，在声明观测语义下的 working catalytic compatibility。

对每个 biochemical / information regime \(U_\alpha\)，有不同维度甚至不同语义的坐标：

\[
\phi_\alpha(r)\in H^R_\alpha,
\qquad
\psi_\alpha(e)\in H^E_\alpha,
\]

以及局部 interaction form

\[
K_\alpha(r,e)=\phi_\alpha(r)^\top G_\alpha\psi_\alpha(e).
\]

这解决了旧 global-manifold 的核心困难：sequence、whole reaction、reaction center、structure、pocket、family、mechanism 不必拥有同一个维度、不必 metrically aligned，也不必在每个 pair 上都可观测。

## 34. Partition of unity：把旧 global/expert mixture 精确重解释为 atlas gluing

理想 pair-aware atlas 使用

\[
\rho_\alpha(r,e)\ge0,
\qquad \sum_\alpha \rho_\alpha(r,e)=1,
\]

并 glue

\[
\widehat K(r,e)=\sum_\alpha \rho_\alpha K_\alpha.
\]

历史 multi-expert 的 convex mixture 并不需要丢掉：若 expert mass 为 \(m\)，softmax gate 为 \(g_k\)，则

\[
\rho_0=1-m,\qquad \rho_k=mg_k,
\]

权重严格和为 1。这说明 final atlas 不是“把旧模型推倒重做”，而是把旧融合函数放进一个更一般、能处理 missing modalities 和 heterogeneous coordinates 的数学对象里。

## 35. 方向性被诚实保留：共享 local interaction，不虚构 R2E=E2R

冻结实现仍然 directional：

\[
S_d(r,e)=\sum_\alpha \rho_\alpha^{(d)}(r,e)K_\alpha(r,e),
\qquad d\in\{R2E,E2R\}.
\]

R2E 用 reaction-side gates / mix，E2R 用 protein-side gates / 自己的 mix。当前文档明确**不声称**同一 pair 的最终两个方向 score 点对点相等。

为了让“共享 interaction、方向化 readout”不只是口头解释，commit `6f29ccf` 加入了 deterministic bound：

\[
|S_p-S_q|
\le \frac12\|p-q\|_1
(K_{\max}-K_{\min}).
\]

它把方向差异分解为两件事：partition difference 与 active-chart score disagreement。若 charts 在 overlap 上越来越一致，即使两个 retrieval direction 使用不同 partition，最终差异也受到限制。

## 36. Gluing consistency：专家同时声称 applicable 时，必须约束同一 task-level compatibility

最终 atlas 加入

\[
\mathcal L_{glue}
=\sum_{\alpha<\beta}
\rho_\alpha\rho_\beta(K_\alpha-K_\beta)^2.
\]

`934272e` 把它接进 multi-expert training；冻结 atlas recipe 使用 weak glue weight 0.02，并把 universal chart 与 8 个 local charts 放在同一个 partition 中。

这里的核心变化是：expert diversity 仍允许存在，但 diversity 发生在**坐标和适用 regime**，不是“同一个 pair 可以随便投出互相矛盾的 scalar votes”。

## 37. Missing modality 的最终语义：zero mass，不是 zero biology

一个 broad raw-input universal chart 对所有有效 reaction/protein 都存在，因此 optional structure / pocket / mechanism / family chart 缺失时：

- unavailable chart partition mass = 0；
- 剩余 available charts renormalize；
- candidate 不会因缺结构而从 universe 消失；
- missing view 不产生负分或惩罚。

这条规则正是早期 pocket coverage、CLIPZyme support、EnzGFM partial feature guard 等一系列历史问题的统一答案。

## 38. Train-free experiment update：从 min-plus exact seed update 改成 bounded finite-rank operator

当前 chart-local update 写为

\[
\Delta_\alpha
=\sum_i w_i\rho_\alpha(r_i,e_i)
\phi_\alpha(r_i)\psi_\alpha(e_i)^\top.
\]

若有 \(n\) 个 accepted observations，则 rank 不超过 \(n\)。`FiniteRankInteractionUpdate` 支持 reaction/enzyme chart 维度不同，并用 Frobenius budget \(\epsilon\) 限制扰动；单位范数坐标下单 pair score change 绝对值不超过 \(\epsilon\)。历史 `PositivePairConditioner` 是 shared square latent space 的已使用特例。

边界同样写清：**数学 primitive 已实现，但“一个外部 wet-lab observation 自动解析、判定 applicable charts、投影并更新全部 chart”的 application orchestration 不能仅凭 kernel 存在就宣称已经完成。**

## 39. Final atlas 的冻结证据

当前 atlas native TPS 结果：

- legacy-exact R2E：MRR 0.23695，Hit@10 35.87%，Hit@20 45.81%，median rank 29；
- strict 25-cell double-cold R2E：MRR 0.04489，Hit@10 10.78%，Hit@20 19.05%，median 136；
- strict E2R：MRR 0.07229，Hit@10 19.02%，Hit@20 28.71%，median 64。

相对 historical MARTS-only 8-expert：

- R2E MRR +0.00490，Hit@20 +1.75 pp，但 Hit@10 **-1.45 pp**；
- E2R MRR +0.01841，Hit@10/20 均约 +7.07 pp。

这个结果被明确当成 whole-method trade-off，而不是要求每个 metric 都增加。

把同一 TPS strict R2E query 放进全部 185,918 proteins：

- full atlas：MRR 0.01245，Hit@10 3.22%，Hit@20 5.60%，median positive rank 5,488；
- 同一 trained model 的 universal-chart-only：MRR 0.00984，Hit@10 2.52%，Hit@20 4.62%，median 8,998。

因此 local TPS/multiview specialization 在扩大 candidate universe 后仍有可测贡献，而不是只在 TPS 小池里有效。

## 40. 几何/interaction-atlas 收尾时主动修正的几个理论过度声明

09-27 的 `6f29ccf` 专门做了“去过度声明”：

1. \(A(r,e;c)\) 才是更接近物理 activity 的概念对象，\(K^\star(r,e)\) 只是 task-level working compatibility；
2. frozen R2E/E2R readout 不宣称逐点相等；
3. 增加 directional discrepancy bound；
4. 增加 convex-hull / single-chart guarantees；
5. chart-local finite-rank update 从只存在公式推进到 kernel；
6. 明确 application-level 自动跨 chart update 尚未完成；
7. product-manifold 作为 derivational stage 保存，不再冒充 current ontology。

测试层面：FIBRE + atlas/multi-expert 相关 247 tests passed；materialized release validation 0 import-boundary violations，28/28 project-owned assets materialized。

---

# 第九编：把失败实验转成设计约束——全历史的交叉结论

## 41. 哪些尝试被明确拒绝或作废，以及它们后来贡献了什么

| 尝试 | 失败/边界 | 留下的设计约束 |
|---|---|---|
| direct all-pair CAGE | 近完整结构/口袋 coverage 仍 Hit@10=0 | pair scorer ≠ large-universe retriever |
| reaction-similarity gate | all-Rhea positive coverage ~44% | hard reservoir 有不可恢复 recall ceiling |
| P2Rank multi-pocket naive aggregation | 暴露 uncertainty 但无 Hit gain | 多信息 ≠ 简单多候选/聚合 |
| fpocket / pocket union | 排名退化 | local geometry 要有 catalytic relevance |
| cycle rerank grid | 0 promotion candidates | reciprocal consistency 不能只凭概念上线 |
| Horizyn MLNCE | 多数主指标退化 | 拆 loss 与 representation，不整包迷信论文 |
| RDKit+ novelty router | hard-slice MRR 好、Hit10/稳定性坏 | conditional gain 必须跨 fold + multi-metric 过门槛 |
| reaction-center concat V1 | aggregate 好，hard early-rank gate 失败 | local chemistry 有信息，但进入方式要受约束 |
| reaction-center residual fresh Rhea | internal confirm 后 external early-rank 退化 | internal success 不等于 transfer；revealed external 不可回头调参 |
| functional prototype residual | 9 candidates 全失败 | biological story 不替代预注册指标 |
| unconstrained E2R LambdaRank | AUROC 微升但 early ranking 大幅崩 | 强 baseline 要 anchoring |
| comprehensive center | 三 fold 都增益但不够 material | 不因“方向一致”降低预定 promotion bar |
| center + Top2000 fusion | Hit 提升但 MRR 退 | component individually useful ≠ composition automatically useful |
| TPS XAttn V1 | ID alignment bug，所有性能作废 | identity/provenance correctness 高于模型复杂度 |
| old projected-cold headline | 与 current training 大量 overlap | “cold”必须相对当前训练源重新计算 |
| global reaction tangent geometry | E2R MRR 0.0723→0.0492 | local mechanism 不应被强迫进入 global metric |
| pocket as global protein geometry | support 仅约 0.69% | missing modality 需要 local chart semantics |
| one global field on query sections | overlap consistency 未天然保证 | 理论对象与数值 realization 必须区分 |

## 42. 三次真正的模型范式变化

从全历史看，可以把对外模型版本压成三代，但 Raw History 必须记住每一代内部都包含大量支线：

1. **V1 TPS open-world retrieval**：从 CAGE/gate/rescue 走向双向 dual-tower + PU + domain adaptation + uncertainty + wet-lab candidate workflow；
2. **V2 BiME-Rank generalized retrieval**：从 continuation/representation/cleanroom 实验矩阵中形成 availability-aware experts、learning-to-rank、baseline anchoring、context experts 与 cost-aware execution；
3. **V3 FIBRE interaction atlas**：从 product-manifold/correspondence 的理论尝试进一步放松为 heterogeneous local interaction charts + partition/gluing + bounded evidence update。

Product-manifold 不是第四代正式产品模型，而是 V3 的关键中间理论阶段。

## 43. Agent 的两阶段与模型三阶段不是同一个时间轴

Agent 不能硬塞成“模型第几代”的附件。更准确的是：

- 真实用户试用先迫使 semantic routing、intent confirmation、scope switching 改造；
- formal bounded agent 把这些需求收敛成自然语言科研入口；
- research workspace 把证据、对象、路线和多轮状态变成持续科研资产；
- Starase 第二阶段再加入可复用对象、budget-aware planning、route patching、verified refs。

Agent 的外部动因来自 HP/真实用户，模型的架构动因主要来自实验与数学；二者在使用场景上互相施压，但不能用单向因果替代彼此。

## 44. 湿实验边界

TPS 是整个项目最直接的 wet-lab grounding：早期任务、candidate list、plate/campaign 等都围绕 TPS。09-08 后仓库主动修正 wet-lab reproduction boundary 并 demote legacy TPS campaign bundle。因此当前正确说法是：

- wet-lab history 支撑 TPS problem relevance 和历史 candidate-generation workflow；
- BiME/FIBRE 从这些问题中抽象出 broader modeling principles；
- **没有新的 FIBRE interaction-atlas-specific wet-lab validation 时，不能把旧 TPS 实验自动写成 V3 的独立实验验证。**

---

# 第十编：证据与命名冲突清理

## 45. 本文已经去掉/修正的历史叙述冲突

1. **“RF/CAGE = EnzymeCAGE baseline”**：错误。RF/HGB+CAGE 是项目内部历史 hybrid；pure EnzymeCAGE 后来单独复现。
2. **“TPS 一直只有一套 candidate universe”**：错误。旧 current-library 约 513×1391；MARTS strict 约 453×1421；后面还进入 general 185,918-protein universe。
3. **“Agent 做出来以后用户才提需求”**：与团队 HP 事实和 Aug20 Git 顺序不符。真实用户反馈先推动 routing，formal harness 在 Aug28。
4. **“Agent 导致 general model 诞生”**：过度单向。用户需求、general-universe refactor、模型扩域和 agent formalization 是相互推动的并行线。
5. **“多专家就是最终理论”**：错误。BiME-Rank 是有效工程/统计阶段；后续专门重建 interaction ontology。
6. **“Product manifold 是 current FIBRE”**：错误。它是 superseded derivational stage。
7. **“FIBRE 当前有一个方向完全相同的 score matrix”**：错误。共享 local chart scores，但 frozen partitions directional。
8. **“K(r,e) 是 context-free physical activity truth”**：错误。更完整对象应含 context；当前是 working compatibility。
9. **“所有新实验已经自动跨 chart train-free update”**：过度声明。kernel primitive 已有，application orchestration 未完整声称。
10. **“旧 TPS wetlab 验证了最终 FIBRE atlas”**：错误。只能支撑历史 TPS lineage/application grounding。

---

## 附录 A：逐 commit 可追溯索引（模型、Agent、评测与复现；数据库专项排除）

本附录不是正文的叙事层级，而是防遗漏索引。正文中不同尝试按科学问题成组组织；这里保留每个相关 commit 的日期、状态标签和主题，便于回到 Git 逐项核查。纯数据库专项 commit 按本次文档范围明确排除。

### P0 早期结构/口袋与 CAGE 全库筛选：问题暴露

| 日期 | commit | 类型 | 原始主题 |
|---|---|---|---|
| 2026-07-10 | `8c606bf` | 实验/实现/迭代 | Reorganize project layout by research block |

### P1 TPS 开放世界双向检索、可靠性与实验执行化

| 日期 | commit | 类型 | 原始主题 |
|---|---|---|---|
| 2026-07-24 | `fcffa06` | 实验/实现/迭代 | feat(tps): add open-world retrieval system and technical report |
| 2026-07-29 | `5a6dc78` | 实验/实现/迭代 | feat(terpene): add portable production runtime and reproducibility assets |
| 2026-08-05 | `83d3117` | 重构/重定义 | Refactor terpene production core and validation workflows |
| 2026-08-05 | `f0ddbc7` | 固化/发布/复现 | Record validated server06 terpene deployment |
| 2026-08-05 | `300cd19` | 实验/实现/迭代 | Add competition evidence and cycle-consistency layer |
| 2026-08-05 | `2764310` | 固化/发布/复现 | Record evidence-aware server06 deployment |
| 2026-08-05 | `cd8cfec` | 实验/实现/迭代 | Add conformal retrieval sets and second-round cycle grid |
| 2026-08-05 | `b5a20fa` | 固化/发布/复现 | Record conformal-aware server06 deployment |
| 2026-08-05 | `7608eff` | 实验/实现/迭代 | Build animated model portal and read-only data hub |
| 2026-08-09 | `4b9e7cd` | 实验/实现/迭代 | Refine route atlas and add R2E taxonomy scopes |
| 2026-08-09 | `6aa7f7b` | 固化/发布/复现 | Record R2E taxonomy-aware server06 deployment |

### P2a 用户试用驱动的语义路由与科研入口扩张

| 日期 | commit | 类型 | 原始主题 |
|---|---|---|---|
| 2026-08-20 | `94cfdfe` | 实验/实现/迭代 | Refine intent routing and isolate ambiguous task handling |
| 2026-08-20 | `f85e4af` | 确认/晋升/确认实验 | Add low-confidence intent confirmation flow |
| 2026-08-20 | `9609021` | 修正/边界澄清 | Fix autonomous intent routing for route and pathway tasks |
| 2026-08-20 | `5be0dd6` | 实验/实现/迭代 | Allow conversational intent switching in catalyst routing |
| 2026-08-20 | `73d96e4` | 实验/实现/迭代 | Route potential-only E2R followups to novel discovery scope |
| 2026-08-20 | `1bc369a` | 实验/实现/迭代 | Use DeepSeek semantic scope selection for E2R followups |
| 2026-08-20 | `1233c00` | 实验/实现/迭代 | Unify DeepSeek semantic authority for catalyst routing |
| 2026-08-20 | `6c1faf9` | 实验/实现/迭代 | Provide full semantic context for routing scope switches |
| 2026-08-20 | `35d453f` | 实验/实现/迭代 | Strengthen semantic routing continuation context |
| 2026-08-20 | `e232bba` | 实验/实现/迭代 | Merge pull request #1 from ChouYuanjue/feature/terpene-model-frontend-20260805 |
| 2026-08-26 | `c98587a` | 实验/实现/迭代 | Refine Catalyst evidence-first bilingual discovery UI |
| 2026-08-26 | `c869999` | 实验/实现/迭代 | Expand Catalyst Chinese research task guidance |
| 2026-08-26 | `920ace4` | 实验/实现/迭代 | Refine Catalyst bilingual copy and readability |

### P2b 通用候选宇宙、正式科学 Agent 与科研工作区

| 日期 | commit | 类型 | 原始主题 |
|---|---|---|---|
| 2026-08-27 | `c76afb2` | 重构/重定义 | Refactor Catalyst open-world retrieval and general universe |
| 2026-08-27 | `5bc24ea` | 实验/实现/迭代 | Add semantic TPS candidate-universe routing |
| 2026-08-27 | `dd302cd` | 重构/重定义 | Merge Catalyst open-world general-universe refactor |
| 2026-08-27 | `7da5e52` | 修正/边界澄清 | Harden Catalyst production service lifecycle |
| 2026-08-27 | `82a062d` | 实验/实现/迭代 | Make Catalyst service manager executable |
| 2026-08-28 | `278778b` | 实验/实现/迭代 | Support protein-family reaction evidence queries |
| 2026-08-28 | `4f13385` | 实验/实现/迭代 | Add bounded scientific agent harness |
| 2026-08-28 | `305add9` | 实验/实现/迭代 | Make Catalyst agent model-led end to end |
| 2026-08-28 | `e9bf41f` | 实验/实现/迭代 | Improve structured protein agent recovery |
| 2026-08-28 | `3db10d0` | 修正/边界澄清 | Harden raw reaction evidence resolution |
| 2026-08-28 | `9d8999b` | 实验/实现/迭代 | Expand model-led scientific tools and render Markdown |
| 2026-08-28 | `eb088c1` | 实验/实现/迭代 | Refine Catalyst agent UX and verified entity memory |
| 2026-08-28 | `25fddd4` | 实验/实现/迭代 | Normalize compound names before ChEBI lookup |
| 2026-08-29 | `5c0be8f` | 实验/实现/迭代 | Integrate research workspace and multi-turn state |
| 2026-08-29 | `2ceb7a6` | 实验/实现/迭代 | Make Catalyst research workflows composable |
| 2026-08-29 | `175be4c` | 实验/实现/迭代 | Unify Catalyst cards and responsive layout |
| 2026-08-29 | `1842080` | 实验/实现/迭代 | Generalize Catalyst follow-ups and scientific wrapping |
| 2026-08-29 | `e879a1e` | 实验/实现/迭代 | Generalize Catalyst evidence synthesis and pagination |
| 2026-08-29 | `6a092c7` | 固化/发布/复现 | Make recorded association lookup local-first |
| 2026-08-29 | `1c3fb28` | 固化/发布/复现 | Honor natural recorded-only evidence requests |
| 2026-08-29 | `064dacc` | 实验/实现/迭代 | Load ESM-C from local cache first |
| 2026-08-29 | `272cc81` | 实验/实现/迭代 | Strengthen Catalyst relation and protein evidence flows |
| 2026-08-29 | `e9ae566` | 实验/实现/迭代 | Enforce Catalyst session target provenance |
| 2026-08-29 | `ea8741a` | 修正/边界澄清 | Align research frontier with production ranking |
| 2026-08-29 | `8922ba7` | 实验/实现/迭代 | Paginate complete research source lists |
| 2026-08-29 | `ede7b03` | 实验/实现/迭代 | Broaden research literature coverage |
| 2026-08-29 | `9b69b16` | 实验/实现/迭代 | Keep retrieval evidence complete and current |
| 2026-08-29 | `da66de6` | 修正/边界澄清 | Clarify ranking modes and model scores |
| 2026-08-29 | `37fb9b2` | 实验/实现/迭代 | Simplify public Catalyst route catalog |
| 2026-08-29 | `f10cceb` | 修正/边界澄清 | Harden Catalyst evidence workflows and bilingual UI |
| 2026-08-29 | `f9b3317` | 修正/边界澄清 | Align bidirectional few-shot retrieval semantics |
| 2026-08-29 | `009646b` | 固化/发布/复现 | Clean retired code and untrack runtime artifacts |
| 2026-08-29 | `5bb1f92` | 实验/实现/迭代 | Bump Catalyst capability and frontend cache version |
| 2026-08-29 | `7800cb1` | 确认/晋升/确认实验 | Bind confirmed positives to verified cards |
| 2026-08-29 | `a313b62` | 实验/实现/迭代 | Recover grounded synthesis from qualifier drift |
| 2026-08-29 | `425436c` | 实验/实现/迭代 | Redact sensitive HTTP query logs |
| 2026-08-29 | `1cf3de8` | 实验/实现/迭代 | Cache successful remote research evidence |
| 2026-08-29 | `ef3e39a` | 实验/实现/迭代 | Remove Catalyst dependency on retired portal |
| 2026-08-29 | `c59c3b7` | 实验/实现/迭代 | Document current Catalyst architecture |

### P3 通用检索研究爆发：续训、专家、学习排序、严格评测与多表示

| 日期 | commit | 类型 | 原始主题 |
|---|---|---|---|
| 2026-08-30 | `e6fdbd0` | 实验/实现/迭代 | Add general known-recovery model evaluator |
| 2026-08-30 | `a60b7d5` | 实验/实现/迭代 | Speed up broad known-recovery evaluation |
| 2026-08-30 | `a06fc7a` | 实验/实现/迭代 | Add full-evidence directional continuation trainer |
| 2026-08-30 | `da90f30` | 实验/实现/迭代 | Add strict known-recovery retention guard |
| 2026-08-30 | `3e0b863` | 否决/失败/失效 | Reject mismatched recovery ensembles |
| 2026-08-30 | `313420b` | 实验/实现/迭代 | Add continuation checkpoint blending |
| 2026-08-30 | `421fe93` | 实验/实现/迭代 | Compose directional continuation towers |
| 2026-08-30 | `7c8c5ff` | 实验/实现/迭代 | Reuse RecAdam for low-forgetting continuation |
| 2026-08-30 | `fc539bf` | 实验/实现/迭代 | Document low-forgetting continuation methods |
| 2026-08-30 | `2b624f4` | 实验/实现/迭代 | Gate continuation on legacy retention |
| 2026-08-30 | `050a4a6` | 实验/实现/迭代 | Reuse FusionBench Fisher consolidation |
| 2026-08-30 | `bdcfc0c` | 实验/实现/迭代 | Add low-cost source-weighted RegMean consolidation |
| 2026-08-30 | `cf590a5` | 实验/实现/迭代 | Reuse Mammoth LwF for retrieval score retention |
| 2026-08-30 | `015b2ff` | 实验/实现/迭代 | Distill bidirectional retrieval score matrix |
| 2026-08-30 | `605e448` | 实验/实现/迭代 | Preserve retrieval margins with Margin-MSE |
| 2026-08-30 | `620409f` | 实验/实现/迭代 | Select continuation models on Pareto gains |
| 2026-08-30 | `9facd34` | 实验/实现/迭代 | Bootstrap broad recovery gains |
| 2026-08-30 | `3ff506c` | 实验/实现/迭代 | Reuse TIES for broad expert merging |
| 2026-08-30 | `b835681` | 实验/实现/迭代 | Evaluate post-hoc domain expert routing |
| 2026-08-30 | `92a785a` | 实验/实现/迭代 | Adapt AdaMerging to bidirectional retrieval |
| 2026-08-30 | `5aafa07` | 实验/实现/迭代 | Benchmark CAGE and neural experts fairly |
| 2026-08-30 | `fa3828c` | 实验/实现/迭代 | Route broad queries across retrieval experts |
| 2026-08-30 | `18bf609` | 实验/实现/迭代 | Benchmark auxiliary retrieval specialists |
| 2026-08-30 | `177ae1f` | 实验/实现/迭代 | Validate retrieval expert portfolio selection |
| 2026-08-30 | `e6541ab` | 实验/实现/迭代 | Stack retrieval experts with LambdaRank |
| 2026-08-30 | `3d1e86f` | 修正/边界澄清 | Fix standalone LambdaRank adapter imports |
| 2026-08-30 | `ffe0cb3` | 实验/实现/迭代 | Test LambdaRank adapter as standalone CLI |
| 2026-08-30 | `c7f7c71` | 实验/实现/迭代 | Evaluate pure CAGE on native support |
| 2026-08-30 | `74c1314` | 实验/实现/迭代 | Keep general queries on general experts |
| 2026-08-30 | `34f5f46` | 实验/实现/迭代 | Use seed-guided production few-shot retrieval |
| 2026-08-30 | `b73b47c` | 实验/实现/迭代 | Distinguish internal expert routing provenance |
| 2026-08-30 | `4e74ddc` | 修正/边界澄清 | Align few-shot route views with hybrid guidance |
| 2026-08-30 | `996d4c7` | 实验/实现/迭代 | Bind provenance to candidate universe assets |
| 2026-08-30 | `81ab49f` | 实验/实现/迭代 | Upgrade current TPS R2E specialist |
| 2026-08-30 | `8b9806b` | 实验/实现/迭代 | Add leakage-aware fair benchmark evaluator |
| 2026-08-30 | `4b0a0cb` | 实验/实现/迭代 | Broaden fair retrieval metrics |
| 2026-08-30 | `5dabf15` | 实验/实现/迭代 | Cover ReactZyme native retrieval cutoffs |
| 2026-08-30 | `c0e7f42` | 实验/实现/迭代 | Build broad leakage-clean Rhea benchmarks |
| 2026-08-30 | `e98ddb5` | 实验/实现/迭代 | Support leakage-controlled broad retraining |
| 2026-08-30 | `ee9fe82` | 实验/实现/迭代 | Separate ReactZyme reciprocal-rank metric |
| 2026-08-30 | `c59176a` | 实验/实现/迭代 | Add exact full-candidate retrieval metrics |
| 2026-08-30 | `2a3a089` | 实验/实现/迭代 | Test full-candidate metric parity |
| 2026-08-30 | `6236bb8` | 实验/实现/迭代 | Add broad full-candidate benchmark runner |
| 2026-08-30 | `b4db02b` | 否决/失败/失效 | Keep cold benchmark negatives train-only |
| 2026-08-30 | `2483291` | 实验/实现/迭代 | Summarize broad Rhea benchmark matrix |
| 2026-08-30 | `d5f304f` | 实验/实现/迭代 | Broaden pure EnzymeCAGE metrics |
| 2026-08-30 | `922155d` | 实验/实现/迭代 | Add leakage-safe nested retrieval dev splits |
| 2026-08-30 | `83fe985` | 实验/实现/迭代 | Broaden cleanroom and capability evaluation |
| 2026-08-30 | `212a796` | 修正/边界澄清 | Enforce mandatory baselines and align Enzyme-405 protocol |
| 2026-08-30 | `59b75b1` | 实验/实现/迭代 | Guard routed experts across retrieval metrics |
| 2026-08-30 | `d13d133` | 实验/实现/迭代 | Gate routed experts with joint reliability evidence |
| 2026-08-30 | `efd8200` | 固化/发布/复现 | Track external baseline metric provenance |
| 2026-08-30 | `5c23dca` | 审计 | Audit broad retrieval gains without metric cherry-picking |
| 2026-08-30 | `96c060b` | 实验/实现/迭代 | Select cleanroom retrievers across internal folds |
| 2026-08-30 | `f7a7492` | 实验/实现/迭代 | Match EnzymeCAGE valid-pocket evaluation reservoir |
| 2026-08-30 | `ed2905f` | 实验/实现/迭代 | Reconstruct EnzymeCAGE ESM-C features audibly |
| 2026-08-30 | `3e7c8b7` | 审计 | Audit Enzyme-405 sequence provenance without correction |
| 2026-08-30 | `3c05dee` | 实验/实现/迭代 | Require ESM validity for EnzymeCAGE author reservoir |
| 2026-08-30 | `de23188` | 实验/实现/迭代 | Export exact positive ranks for broad benchmarks |
| 2026-08-30 | `1f1ad78` | 实验/实现/迭代 | Snapshot strongest system versus EnzymeCAGE |
| 2026-08-30 | `2dc5aed` | 实验/实现/迭代 | Report only leakage-clean capability evidence |
| 2026-08-30 | `7767a66` | 实验/实现/迭代 | Replay train-only novel reactions in R2E continuation |
| 2026-08-30 | `7079db5` | 固化/发布/复现 | Record continuation top-k margin |
| 2026-08-30 | `fdd62e8` | 实验/实现/迭代 | Route low-similarity reactions to novelty expert |
| 2026-08-30 | `439019b` | 实验/实现/迭代 | Replay train-only novel reactions in cleanroom training |
| 2026-08-30 | `00ad433` | 预注册/冻结 | Freeze clean R2E novelty expert before outer tests |
| 2026-08-30 | `783f6c7` | 预注册/冻结 | Pre-register cleanroom novelty selection |
| 2026-08-30 | `91ccfd3` | 预注册/冻结 | Match cleanroom selector to preregistered metrics |
| 2026-08-30 | `f85a2a9` | 确认/晋升/确认实验 | Summarize frozen clean novelty confirmatory results |
| 2026-08-30 | `d5e9ce6` | 预注册/冻结 | Freeze cleanroom model before Enzyme-405 reveal |
| 2026-08-30 | `9122463` | 否决/失败/失效 | Pre-register cleanroom hard-negative curriculum |
| 2026-08-30 | `ae0a399` | 实验/实现/迭代 | Update clean-only benchmark snapshot |
| 2026-08-30 | `86fe570` | 预注册/冻结 | Pre-register cleanroom seed ensemble |
| 2026-08-30 | `23c86ea` | 实验/实现/迭代 | Quantify clean Enzyme-405 uncertainty |
| 2026-08-30 | `dd79d62` | 预注册/冻结 | Freeze cleanroom seed ensemble before holdout folds |
| 2026-08-30 | `e742edf` | 预注册/冻结 | Pre-register official Horizyn MLNCE cleanroom test |
| 2026-08-30 | `f464080` | 否决/失败/失效 | Record negative Horizyn MLNCE result |
| 2026-08-30 | `383ac94` | 预注册/冻结 | Pre-register raw RDKitPlus reaction feature test |
| 2026-08-30 | `94fd35b` | 实验/实现/迭代 | Support pluggable cleanroom protein features |
| 2026-08-30 | `d44d2b9` | 预注册/冻结 | Freeze RDKitPlus representation before outer tests |
| 2026-08-30 | `cfc9a9a` | 实验/实现/迭代 | Add identity-preserving RDKitPlus outer continuation |
| 2026-08-31 | `5c9daff` | 审计 | Add paired bootstrap ranking audit |
| 2026-08-31 | `057e762` | 实验/实现/迭代 | Summarize clean RDKitPlus outer gains |
| 2026-08-31 | `170af22` | 预注册/冻结 | Pre-register EnzGFM protein representation test |
| 2026-08-31 | `31545c4` | 实验/实现/迭代 | Add balanced bidirectional cleanroom selection |
| 2026-08-31 | `77a5d98` | 预注册/冻结 | Add equal-block protein feature fusion |
| 2026-08-31 | `fd7e7f3` | 预注册/冻结 | Freeze bidirectional EnzGFM development recipe |
| 2026-08-31 | `1cfdb35` | 预注册/冻结 | Freeze EnzGFM Gate A runner |
| 2026-08-31 | `819ab49` | 预注册/冻结 | Freeze EnzGFM RDKitPlus Gate B runner |
| 2026-08-31 | `ce99c18` | 审计 | Add CLIPZyme train overlap audit |
| 2026-08-31 | `94ac9c5` | 审计 | Add CLIPZyme reaction similarity audit |
| 2026-08-31 | `6d0ee17` | 审计 | Fix CLIPZyme similarity audit entrypoint |
| 2026-08-31 | `302fc9e` | 审计 | Make CLIPZyme similarity audit stereo robust |
| 2026-08-31 | `f2131f0` | 实验/实现/迭代 | Add internal full-candidate cleanroom cells |
| 2026-08-31 | `9093d5f` | 审计 | Add internal shortlist headroom audit |
| 2026-08-31 | `3192686` | 审计 | Add auditable RXNMapper reaction registry |
| 2026-08-31 | `2bb9a85` | 修正/边界澄清 | Fix RXNMapper progress failure count |
| 2026-08-31 | `ab91b7e` | 实验/实现/迭代 | Support exact metrics from reranked positive ranks |
| 2026-08-31 | `b5713a0` | 实验/实现/迭代 | Add support-aware CLIPZyme reaction extractor |
| 2026-08-31 | `1b35b2c` | 预注册/冻结 | Freeze internal Top-2000 reranker gate |
| 2026-08-31 | `ae69c08` | 实验/实现/迭代 | Respect CLIPZyme graph input support |
| 2026-08-31 | `d09f7b0` | 审计 | Audit CLIPZyme graph input compatibility |
| 2026-08-31 | `a76419d` | 预注册/冻结 | Freeze full-candidate EnzGFM completion scope |
| 2026-08-31 | `f2a9b92` | 实验/实现/迭代 | Guard EnzGFM gates against partial features |
| 2026-08-31 | `6b348de` | 实验/实现/迭代 | Enforce common Gate A protein universe |
| 2026-08-31 | `dd45c95` | 预注册/冻结 | Freeze EnzGFM RDKit+ outer protocol |
| 2026-08-31 | `504bac1` | 实验/实现/迭代 | Use nested selection for clean outer evaluation |
| 2026-08-31 | `b8916ab` | 实验/实现/迭代 | Preserve merged feature provenance |
| 2026-08-31 | `6276086` | 实验/实现/迭代 | Support scoped EnzGFM feature completion |
| 2026-08-31 | `f173e9b` | 预注册/冻结 | Freeze Top-2000 pair residual recipe |
| 2026-08-31 | `4078898` | 实验/实现/迭代 | Implement clean Top-2000 pair reranker |
| 2026-08-31 | `4f334df` | 修正/边界澄清 | Scope reaction feature contract correctly |
| 2026-08-31 | `70880ea` | 预注册/冻结 | Freeze EnzGFM RDKit protein-cold outer protocol |
| 2026-08-31 | `960d79a` | 预注册/冻结 | Correct post-reveal protein-cold freeze audit |
| 2026-08-31 | `17f339f` | 固化/发布/复现 | Record EnzymeCAGE baseline for every scenario |
| 2026-08-31 | `80fb3bd` | 实验/实现/迭代 | Validate expert eligibility per benchmark scenario |
| 2026-08-31 | `8a686ac` | 预注册/冻结 | Freeze directional expert routing before temporal outer |
| 2026-08-31 | `bca49f1` | 实验/实现/迭代 | Rebuild reranker coarse metrics from ranks |
| 2026-08-31 | `1684907` | 预注册/冻结 | Freeze bounded Top-2000 residual sweep |
| 2026-08-31 | `469b9a4` | 实验/实现/迭代 | Make residual scale zero exact coarse fallback |
| 2026-08-31 | `dc776f2` | 确认/晋升/确认实验 | Freeze bounded residual fold3 confirmation |
| 2026-08-31 | `c200121` | 审计 | Audit epoch gains by clean difficulty |
| 2026-08-31 | `fee4a31` | 审计 | Audit Enzyme-405 evidence chain |
| 2026-08-31 | `c5449ba` | 实验/实现/迭代 | Bind difficulty slices to active reaction schema |
| 2026-08-31 | `727167d` | 预注册/冻结 | Freeze difficulty-aware Top-2000 router |
| 2026-08-31 | `ae6e95b` | 实验/实现/迭代 | Parameterize internal reranker evaluation roots |
| 2026-08-31 | `82e4f34` | 实验/实现/迭代 | Prefer reproduced EnzymeCAGE evidence |
| 2026-08-31 | `2268dc0` | 确认/晋升/确认实验 | Record Top-2000 router confirmation |
| 2026-08-31 | `eaadd94` | 预注册/冻结 | Freeze low-similarity novelty screening |
| 2026-08-31 | `1e1a6cf` | 否决/失败/失效 | Reject unstable low-sim novelty replay |
| 2026-08-31 | `c5194c3` | 预注册/冻结 | Freeze reaction-center hard-slice feature screen |
| 2026-08-31 | `02372dc` | 否决/失败/失效 | Reject reaction-center v1 hard-slice promotion |
| 2026-08-31 | `49889f1` | 预注册/冻结 | Freeze identity residual center integration |
| 2026-08-31 | `370a9f9` | 审计 | Fix residual identity audit dropout mode |
| 2026-08-31 | `0ac8535` | 确认/晋升/确认实验 | Freeze fresh center residual confirmation |
| 2026-08-31 | `357750a` | 确认/晋升/确认实验 | Freeze center residual confirmation evaluator |
| 2026-08-31 | `0c30a1b` | 确认/晋升/确认实验 | Record passed center residual confirmation |
| 2026-08-31 | `8e54257` | 预注册/冻结 | Freeze Rhea 128 to 141 external benchmark |
| 2026-08-31 | `2698a3a` | 预注册/冻结 | Freeze Rhea snapshot external evaluator |
| 2026-08-31 | `8fc12fe` | 确认/晋升/确认实验 | Document confirmed reaction center capability |
| 2026-08-31 | `7049f94` | 修正/边界澄清 | Correct Rhea snapshot identifier provenance |
| 2026-08-31 | `c2e15f2` | 修正/边界澄清 | Align Rhea V2 manifest with evaluator |
| 2026-08-31 | `37d4bc0` | 预注册/冻结 | Freeze one-way Rhea V2 reveal runner |
| 2026-08-31 | `6a838b8` | 审计 | Audit Rhea 128 training source alignment |
| 2026-08-31 | `33bc30e` | 预注册/冻结 | Freeze Rhea V2 model asset hashes |
| 2026-08-31 | `df543c0` | 实验/实现/迭代 | Smoke frozen Rhea V2 model loaders |
| 2026-08-31 | `da56ff6` | 修正/边界澄清 | Fix Rhea V2 evaluator import path |
| 2026-08-31 | `4279d32` | 否决/失败/失效 | Record failed fresh Rhea snapshot transfer |
| 2026-08-31 | `c95dda7` | 审计 | Audit fresh Rhea transfer failure modes |
| 2026-08-31 | `72fc416` | 预注册/冻结 | Freeze bounded reaction center V3 development |
| 2026-08-31 | `6c75ce5` | 预注册/冻结 | Freeze bounded V3 selection evaluator |
| 2026-08-31 | `ced9484` | 确认/晋升/确认实验 | Freeze bounded V3 confirmation |
| 2026-08-31 | `9ac4f8e` | 确认/晋升/确认实验 | Confirm bounded reaction center V3 |
| 2026-08-31 | `eae6ee1` | 预注册/冻结 | Freeze clean mainline production package |
| 2026-08-31 | `c1c6f6b` | 实验/实现/迭代 | Validate directional expert routing |
| 2026-08-31 | `c27b62a` | 实验/实现/迭代 | Consolidate clean retrieval mainline |
| 2026-08-31 | `5c3905a` | 实验/实现/迭代 | Center docs on clean retrieval mainline |
| 2026-08-31 | `aea6fc8` | 固化/发布/复现 | Package clean mainline runtime registry |
| 2026-08-31 | `f86d7d0` | 实验/实现/迭代 | Reproduce Orphan-335 author retrieval |
| 2026-08-31 | `43c72b4` | 预注册/冻结 | Freeze functional prototype residual screening |
| 2026-08-31 | `8f46e7f` | 实验/实现/迭代 | Implement train-only functional prototype residual |
| 2026-08-31 | `b634043` | 预注册/冻结 | Freeze functional prototype selector |
| 2026-08-31 | `b960cb8` | 否决/失败/失效 | Record functional prototype rejection |
| 2026-08-31 | `865862d` | 修正/边界澄清 | Support fixed cleanroom partitions |
| 2026-09-01 | `4e1d90f` | 预注册/冻结 | Freeze ReactZyme retention policy |
| 2026-09-01 | `30c46a1` | 修正/边界澄清 | Fix ReactZyme selector serialization |
| 2026-09-01 | `797de4b` | 修正/边界澄清 | Complete ReactZyme selector boolean fix |
| 2026-09-01 | `b309ed8` | 否决/失败/失效 | Record ReactZyme retention rejection |
| 2026-09-01 | `6879eac` | 预注册/冻结 | Freeze native molecule-bag adapter v1 |
| 2026-09-01 | `5be9029` | 预注册/冻结 | Freeze native bag adapter selector |
| 2026-09-01 | `4178843` | 修正/边界澄清 | Fix native bag evaluator protein normalization |
| 2026-09-01 | `a787e85` | 修正/边界澄清 | Fix native bag teacher feature alignment |
| 2026-09-01 | `15b3378` | 确认/晋升/确认实验 | Freeze native bag adapter confirmation |
| 2026-09-01 | `6770223` | 确认/晋升/确认实验 | Implement train-only native bag confirmation |
| 2026-09-01 | `06b0724` | 确认/晋升/确认实验 | Record native bag adapter confirmation |
| 2026-09-01 | `2904307` | 预注册/冻结 | Freeze unified baseline-safe system objective |
| 2026-09-01 | `835ebaf` | 预注册/冻结 | Freeze full-universe E2R baseline contract |
| 2026-09-01 | `4f02195` | 审计 | Implement full-universe E2R baseline audit |
| 2026-09-01 | `bb0dc90` | 预注册/冻结 | Freeze baseline-safe E2R expert router |
| 2026-09-01 | `89f3247` | 预注册/冻结 | Freeze four-expert E2R portfolio audit |
| 2026-09-01 | `4545582` | 预注册/冻结 | Freeze four-expert E2R LambdaRank stack |
| 2026-09-01 | `185f02c` | 实验/实现/迭代 | Bound E2R LambdaRank to expert shortlist |
| 2026-09-01 | `2924233` | 实验/实现/迭代 | Implement four-expert E2R LambdaRank stack |
| 2026-09-01 | `89b6423` | 审计 | Record descriptive Rhea transfer failure audit |
| 2026-09-01 | `64e5a8e` | 审计 | Align Rhea posthoc audit to frozen direction |
| 2026-09-01 | `5d90dd1` | 否决/失败/失效 | Reject unconstrained E2R LambdaRank stack |
| 2026-09-01 | `cc3d3fb` | 预注册/冻结 | Freeze baseline-anchored E2R rescue V2 |
| 2026-09-01 | `2c76be6` | 实验/实现/迭代 | Implement baseline-anchored E2R rescue V2 |
| 2026-09-01 | `a43861c` | 修正/边界澄清 | Fix E2R rescue V2 transform test |
| 2026-09-01 | `d3a246b` | 实验/实现/迭代 | Enforce unique authoritative baselines |
| 2026-09-01 | `2f9c5eb` | 实验/实现/迭代 | Use executable EnzGFM baseline |
| 2026-09-01 | `b14931b` | 审计 | Audit official EnzGFM 1.5B assets |
| 2026-09-01 | `880d427` | 预注册/冻结 | Freeze EnzGFM native same-support benchmark |
| 2026-09-01 | `3c4e385` | 实验/实现/迭代 | Match author ReactZyme support identity |
| 2026-09-01 | `79404be` | 预注册/冻结 | Freeze EnzGFM native candidate selection |
| 2026-09-01 | `5e27a0a` | 审计 | Record audited EnzGFM native baseline result |
| 2026-09-01 | `858bb1a` | 预注册/冻结 | Freeze TIGER reaction-novel baseline contract |
| 2026-09-01 | `fd80a53` | 实验/实现/迭代 | Use reproducible CLIPZyme reaction-novel baseline |
| 2026-09-01 | `0592301` | 修正/边界澄清 | Align CLIPZyme baseline to ReactZyme molecule bags |
| 2026-09-01 | `cb5680f` | 实验/实现/迭代 | Revert CLIPZyme encoder substitution |
| 2026-09-01 | `e025286` | 修正/边界澄清 | Harden native CLIPZyme baseline boundary |
| 2026-09-01 | `2e45564` | 实验/实现/迭代 | Guard native CLIPZyme baseline semantics |
| 2026-09-01 | `c80d4a3` | 审计 | Add CLIPZyme ReactZyme native support audit |
| 2026-09-01 | `b98ab46` | 审计 | Test CLIPZyme direction support audit primitives |
| 2026-09-01 | `fbdb1a0` | 预注册/冻结 | Freeze directed CLIPZyme fallback order |
| 2026-09-01 | `b952bf5` | 实验/实现/迭代 | Test directed CLIPZyme fallback contract |
| 2026-09-01 | `6988171` | 实验/实现/迭代 | Skip optional Horizyn test in portable CI |
| 2026-09-01 | `b39bc86` | 实验/实现/迭代 | Skip optional XGBoost route in portable CI |
| 2026-09-01 | `6d46793` | 审计 | Add directed CLIPZyme fallback support audit |
| 2026-09-01 | `124eab7` | 审计 | Add native CLIPZyme protein support audit |
| 2026-09-01 | `f2cb803` | 实验/实现/迭代 | Test CLIPZyme protein identity boundary |
| 2026-09-01 | `a216baf` | 实验/实现/迭代 | Sync CLIPZyme native-boundary contract tests |
| 2026-09-01 | `742b090` | 否决/失败/失效 | Lazy-load rejected Rescue V2 ranking dependency |
| 2026-09-01 | `c7fa435` | 实验/实现/迭代 | Decouple universe parsing from strict asset validation |
| 2026-09-01 | `fc65108` | 实验/实现/迭代 | Validate candidate universes only at execution boundary |
| 2026-09-01 | `aace478` | 实验/实现/迭代 | Update candidate universe tests for explicit versions |
| 2026-09-01 | `10baf89` | 实验/实现/迭代 | Add fair external baseline evaluators |
| 2026-09-01 | `0eb86c7` | 实验/实现/迭代 | Consolidate authoritative baseline comparisons |
| 2026-09-01 | `d29e587` | 预注册/冻结 | Preregister comprehensive directional center model |
| 2026-09-01 | `f99826f` | 预注册/冻结 | Freeze directional base weights before evaluation |
| 2026-09-01 | `d68a4c8` | 实验/实现/迭代 | Add comprehensive directional development gate |
| 2026-09-01 | `4f40385` | 修正/边界澄清 | Fix directional center runtime schema metadata |
| 2026-09-01 | `4373a1e` | 否决/失败/失效 | Record rejected directional center integration |
| 2026-09-01 | `5de920d` | 预注册/冻结 | Preregister center Top-2000 comprehensive route |
| 2026-09-01 | `f63f7bc` | 审计 | Add center Top-2000 gate and rank audit |
| 2026-09-01 | `3b0e785` | 实验/实现/迭代 | Support direct coarse eval layout in joint gate |
| 2026-09-01 | `d4394d0` | 实验/实现/迭代 | Preserve exact coarse ranks on reranker fallback |
| 2026-09-01 | `201cad6` | 否决/失败/失效 | Record rejected center Top-2000 integration |
| 2026-09-01 | `072cc3d` | 审计 | Tolerate CSV noise in exact fallback audit |
| 2026-09-01 | `f778cee` | 预注册/冻结 | Preregister protected EnzGFM center route |
| 2026-09-01 | `0132287` | 实验/实现/迭代 | Add protected EnzGFM center route gate |
| 2026-09-01 | `12820a4` | 预注册/冻结 | Freeze fast R2E similarity router |
| 2026-09-01 | `7f00476` | 确认/晋升/确认实验 | Promote confirmed R2E similarity router |
| 2026-09-02 | `fc40180` | 预注册/冻结 | Preregister automated R2E LambdaRank fusion |
| 2026-09-02 | `bbf274c` | 实验/实现/迭代 | Match LambdaRank cache to full evaluator numerics |
| 2026-09-02 | `4b07e98` | 预注册/冻结 | Freeze LambdaRank rank-based AUROC semantics |
| 2026-09-02 | `c97d5ef` | 否决/失败/失效 | Freeze LambdaRank hard-negative sampling |
| 2026-09-02 | `09428e9` | 实验/实现/迭代 | Materialize LambdaRank caches once |
| 2026-09-02 | `a27edb0` | 预注册/冻结 | Freeze selected R2E LambdaRank configuration |
| 2026-09-02 | `e5dcddb` | 确认/晋升/确认实验 | Add frozen R2E LambdaRank confirmation evaluator |
| 2026-09-02 | `f76334f` | 确认/晋升/确认实验 | Record passed R2E LambdaRank confirmation |
| 2026-09-02 | `4086263` | 确认/晋升/确认实验 | Add confirmed R2E LambdaRank candidate runtime |
| 2026-09-02 | `e12753f` | 确认/晋升/确认实验 | Promote confirmed R2E LambdaRank production route |
| 2026-09-02 | `6904a24` | 预注册/冻结 | Freeze anchored E2R LambdaMART V3 search |
| 2026-09-02 | `d054d0d` | 修正/边界澄清 | Align E2R V3 automatic tie break |
| 2026-09-02 | `e6926e3` | 确认/晋升/确认实验 | Freeze E2R V3 confirmation protocol |
| 2026-09-02 | `0540cd1` | 预注册/冻结 | Lock E2R V3 final ranker hash |
| 2026-09-02 | `6d44186` | 确认/晋升/确认实验 | Record confirmed E2R V3 ranker |
| 2026-09-02 | `590bb9a` | 预注册/冻结 | Freeze E2R V3 production gates |
| 2026-09-02 | `e0e7d31` | 确认/晋升/确认实验 | Add confirmed E2R V3 candidate runtime |
| 2026-09-02 | `9810a16` | 预注册/冻结 | Freeze E2R V3 runtime gate queries |
| 2026-09-02 | `7855c8b` | 修正/边界澄清 | Fix E2R runtime gate conformal mode |
| 2026-09-02 | `e52563d` | 预注册/冻结 | Freeze open-world and TPS temporal benchmarks |
| 2026-09-02 | `09c96f9` | 实验/实现/迭代 | Add frozen TPS temporal support builder |
| 2026-09-02 | `a482941` | 实验/实现/迭代 | Add TPS foundation feature builder |
| 2026-09-02 | `bfc7794` | 预注册/冻结 | Freeze TPS foundation R2E search |
| 2026-09-02 | `c62d12d` | 实验/实现/迭代 | Add frozen TPS R2E development runner |
| 2026-09-02 | `4db7ee7` | 预注册/冻结 | Freeze EnzymARC open-world evaluation |
| 2026-09-02 | `4eb0676` | 预注册/冻结 | Freeze TPS active-site cross-attention successor |
| 2026-09-02 | `7a880fb` | 预注册/冻结 | Freeze TPS XAttn transfer semantics |
| 2026-09-02 | `b51f089` | 实验/实现/迭代 | Add TPS XAttn token materializers |
| 2026-09-02 | `beaf16b` | 实验/实现/迭代 | Add fresh TPS XAttn transfer split |
| 2026-09-02 | `a5246f5` | 预注册/冻结 | Freeze TPS XAttn final ranking semantics |
| 2026-09-02 | `15757f1` | 预注册/冻结 | Freeze TPS XAttn hard pools |
| 2026-09-02 | `10d6a3e` | 预注册/冻结 | Freeze TPS XAttn residual training |
| 2026-09-02 | `34ea648` | 实验/实现/迭代 | Add guarded TPS active-site XAttn HPO |
| 2026-09-02 | `e6de6eb` | 预注册/冻结 | Freeze TPS XAttn HPO runtime |
| 2026-09-02 | `6cb56ec` | 实验/实现/迭代 | Add EnzymARC open-world support adapter |
| 2026-09-02 | `6a98fbc` | 预注册/冻结 | Freeze TPS XAttn development winner |
| 2026-09-02 | `99df860` | 否决/失败/失效 | Invalidate misaligned TPS XAttn evaluation |
| 2026-09-02 | `6b8ea9f` | 预注册/冻结 | Freeze full EnzymARC support and scorer |
| 2026-09-02 | `b115f4a` | 预注册/冻结 | Freeze EnzymARC source evaluator |
| 2026-09-02 | `3e25f56` | 预注册/冻结 | Freeze EnzymARC sequence-form gate |
| 2026-09-02 | `de64d0d` | 实验/实现/迭代 | Require EnzymARC sequence-form gate |
| 2026-09-02 | `672ff50` | 预注册/冻结 | Close corrected TPS XAttn and freeze EnzymARC gate |
| 2026-09-02 | `988b705` | 预注册/冻结 | Freeze strong baseline absorption policy |
| 2026-09-03 | `6007575` | 确认/晋升/确认实验 | Promote E2R v3 and streamline retrieval evaluation |
| 2026-09-03 | `946169c` | 实验/实现/迭代 | Add legacy Hit@K retrieval presentation |
| 2026-09-03 | `8e436d4` | 实验/实现/迭代 | Organize retrieval metrics by capability scenario |
| 2026-09-03 | `0865d7b` | 修正/边界澄清 | Correct retrieval baseline provenance and TPS symmetry |
| 2026-09-03 | `5a55738` | 确认/晋升/确认实验 | Freeze TPS MARTS R2E symmetry confirmation |
| 2026-09-03 | `ed45a16` | 确认/晋升/确认实验 | Confirm symmetric TPS MARTS R2E route |
| 2026-09-03 | `5241228` | 预注册/冻结 | Freeze pure CAGE TPS baseline and record R2E symmetry result |
| 2026-09-03 | `da457ca` | 实验/实现/迭代 | Add complete-support pure CAGE evaluator |
| 2026-09-03 | `f2767cd` | 实验/实现/迭代 | Finalize bidirectional MARTS TPS evidence |
| 2026-09-03 | `3385227` | 预注册/冻结 | Freeze recovered-source pure CAGE TPS support |
| 2026-09-03 | `b87ccce` | 实验/实现/迭代 | Complete pure EnzymeCAGE TPS baseline evidence |
| 2026-09-03 | `9288479` | 实验/实现/迭代 | Close retrieval evidence gaps in current narrative |

### P4 BiME-Rank 收敛、上下文专家、成本分层与可复现发布

| 日期 | commit | 类型 | 原始主题 |
|---|---|---|---|
| 2026-09-05 | `3d8f24c` | 确认/晋升/确认实验 | Promote BiME-Rank structural expert routing |
| 2026-09-05 | `e813d52` | 实验/实现/迭代 | Unify BiME-Rank context routing and evidence |
| 2026-09-05 | `d69e700` | 实验/实现/迭代 | Document and validate BiME-Rank multi-seed context |
| 2026-09-06 | `420eb90` | 实验/实现/迭代 | Add cost-aware hierarchical expert execution |
| 2026-09-07 | `8ef9f88` | 固化/发布/复现 | Canonicalize BiME-Rank assets and archive superseded evidence |
| 2026-09-07 | `c871c38` | 实验/实现/迭代 | Archive superseded BiME-Rank experiment outputs |
| 2026-09-07 | `aacfc97` | 固化/发布/复现 | Normalize BiME-Rank judge release against canonical evidence |
| 2026-09-07 | `ca5d6fb` | 固化/发布/复现 | Merge BiME-Rank release mainline |
| 2026-09-07 | `7ed2c39` | 固化/发布/复现 | Package reproducible BiME-Rank research release |
| 2026-09-07 | `b4da3f0` | 固化/发布/复现 | Separate portable and full-asset release checks |
| 2026-09-07 | `8e6b7a0` | 修正/边界澄清 | Clarify BiME-Rank release narrative and asset roles |
| 2026-09-07 | `9a0ae80` | 固化/发布/复现 | Classify BiME-Rank release source roles |
| 2026-09-07 | `87ae7cb` | 固化/发布/复现 | Demote lineage-only tests from BiME-Rank release |
| 2026-09-07 | `ff64cfe` | 固化/发布/复现 | Record canonical claim source provenance |
| 2026-09-07 | `f3ceead` | 固化/发布/复现 | Demote isolated research auxiliaries from release |
| 2026-09-07 | `a99b9b2` | 预注册/冻结 | Lock current model assets and rebuild contracts |
| 2026-09-07 | `1715d52` | 实验/实现/迭代 | Recover canonical evaluators and demote unused TPS R2E evidence |
| 2026-09-07 | `7d8cdcb` | 固化/发布/复现 | Tighten BiME-Rank release evidence and replay assets |
| 2026-09-07 | `74c2458` | 实验/实现/迭代 | Close BiME-Rank canonical reproduction gaps |
| 2026-09-07 | `cb15337` | 固化/发布/复现 | Polish BiME-Rank research release metadata |
| 2026-09-07 | `d345339` | 实验/实现/迭代 | Demote isolated historical BiME-Rank artifacts |
| 2026-09-07 | `268e452` | 实验/实现/迭代 | Demote self-contained historical artifact chains |
| 2026-09-07 | `f6eb2f0` | 实验/实现/迭代 | Prune closed historical evaluation auxiliaries |
| 2026-09-07 | `b7baa62` | 固化/发布/复现 | Enrich research release validation artifact |
| 2026-09-07 | `96509df` | 修正/边界澄清 | Align Python package identity with BiME-Rank release |
| 2026-09-07 | `4e804fd` | 固化/发布/复现 | Set NJU-China as release author |
| 2026-09-07 | `a54c6cc` | 固化/发布/复现 | Make pytest obey the release test boundary |
| 2026-09-07 | `af7be5c` | 固化/发布/复现 | Refresh public release documentation map |
| 2026-09-07 | `3438042` | 实验/实现/迭代 | Separate current route evidence from legacy runtime weights |
| 2026-09-08 | `fed5bf1` | 实验/实现/迭代 | Demote isolated legacy runtime results |
| 2026-09-08 | `e9b23c8` | 预注册/冻结 | Lock scientific data result surface |
| 2026-09-08 | `758f96a` | 实验/实现/迭代 | Demote isolated historical evaluation auxiliaries |
| 2026-09-08 | `80d974d` | 实验/实现/迭代 | Demote isolated historical project artifacts |
| 2026-09-08 | `9b86c87` | 修正/边界澄清 | Correct current wetlab reproduction boundary |
| 2026-09-08 | `1f92806` | 实验/实现/迭代 | Demote legacy TPS wetlab campaign bundle |
| 2026-09-08 | `9d84498` | 实验/实现/迭代 | Rebuild ReactZyme transfer assets exactly |
| 2026-09-08 | `17de56f` | 实验/实现/迭代 | Recover Horizyn distillation preprocessing |
| 2026-09-08 | `0fc390f` | 实验/实现/迭代 | Make candidate universe tables exactly replayable |
| 2026-09-08 | `888bb34` | 实验/实现/迭代 | Close exact candidate universe source coverage |
| 2026-09-08 | `03fd328` | 固化/发布/复现 | Track current CAGE rescue runtime scores |
| 2026-09-08 | `b589f22` | 固化/发布/复现 | Track MARTS training split authorities |
| 2026-09-08 | `12328b7` | 固化/发布/复现 | Track DRFP production ancestor bundle |
| 2026-09-08 | `7a79243` | 实验/实现/迭代 | Pin independent Horizyn training checkpoint |
| 2026-09-08 | `81301d8` | 实验/实现/迭代 | Close Horizyn distillation input contract |
| 2026-09-08 | `e411f31` | 固化/发布/复现 | Track Horizyn reaction distiller artifact set |
| 2026-09-08 | `28381cb` | 固化/发布/复现 | Record deployed TPS fallback lineages |
| 2026-09-08 | `607995a` | 固化/发布/复现 | Track R2E center model ancestors |
| 2026-09-08 | `55a6267` | 固化/发布/复现 | Track E2R V3 runtime ranker |
| 2026-09-08 | `fa64101` | 预注册/冻结 | Register E2R equal-block features as rebuildable |
| 2026-09-08 | `f5aebf0` | 固化/发布/复现 | Record all production model lineages |
| 2026-09-08 | `f8dcfad` | 实验/实现/迭代 | Require reproduction lineage for every production model |
| 2026-09-08 | `8e0f6c3` | 固化/发布/复现 | Refresh canonical candidate universe identity |
| 2026-09-08 | `05762d2` | 实验/实现/迭代 | Make production lineage generation portable |
| 2026-09-08 | `77897fb` | 实验/实现/迭代 | Update external model asset regression |

### P5 几何重建：product manifold / correspondence / partial relation

| 日期 | commit | 类型 | 原始主题 |
|---|---|---|---|
| 2026-09-18 | `b1e12c7` | 重构/重定义 | checkpoint: preserve pre-refactor retrieval and geometry state |
| 2026-09-18 | `6a6563b` | 重构/重定义 | refactor: separate current retrieval from reproduction lineage |
| 2026-09-18 | `9964590` | 重构/重定义 | refactor: establish FIBRE and Starase Navigator |
| 2026-09-19 | `1c84205` | 实验/实现/迭代 | feat: stratify FIBRE biological correspondence |
| 2026-09-19 | `7eb8034` | 审计 | research: audit exact broad FIBRE sections |
| 2026-09-19 | `c54d33a` | 实验/实现/迭代 | feat: integrate mechanistic FIBRE strata |
| 2026-09-19 | `c334b38` | 实验/实现/迭代 | feat: expose exact seed stability provenance |
| 2026-09-20 | `5078f70` | 实验/实现/迭代 | feat: add partial biological relation to FIBRE |
| 2026-09-20 | `48dbc90` | 固化/发布/复现 | feat: package portable FIBRE and full-information Starase release |

### P6 Starase 科研工作区第二阶段：证据、对象复用与路线操作

| 日期 | commit | 类型 | 原始主题 |
|---|---|---|---|
| 2026-09-21 | `94b107d` | 实验/实现/迭代 | feat: unify Starase agent and biological FIBRE evidence |
| 2026-09-21 | `9b8d933` | 修正/边界澄清 | fix: harden Starase controller action recovery |
| 2026-09-21 | `2046bfb` | 修正/边界澄清 | fix: restore Starase public frontend runtime |
| 2026-09-21 | `7c6ee37` | 修正/边界澄清 | fix: prioritize informative enzymology evidence |
| 2026-09-21 | `4fce907` | 修正/边界澄清 | fix: isolate conversations and layer agent context |
| 2026-09-21 | `f5f12a3` | 实验/实现/迭代 | feat: fold scientific cards and bundle agent evidence |
| 2026-09-21 | `466ae62` | 实验/实现/迭代 | feat: make agent workspace objects directly reusable |
| 2026-09-21 | `2f1e70e` | 修正/边界澄清 | fix: preserve explicit route result counts |
| 2026-09-21 | `9833050` | 实验/实现/迭代 | feat: support AI-native local route patching |
| 2026-09-21 | `f08aa42` | 实验/实现/迭代 | feat: preserve derived route lineage in workspace |
| 2026-09-22 | `46204d6` | 实验/实现/迭代 | feat: make agent planning budget-aware |
| 2026-09-22 | `14eaae0` | 修正/边界澄清 | fix: preserve evidence count provenance |
| 2026-09-22 | `62a60d8` | 修正/边界澄清 | fix: bind route design to verified compound identities |
| 2026-09-22 | `4e455f5` | 实验/实现/迭代 | feat: let agent resolve pathway ambiguity with verified refs |

### P7 FIBRE 最终 interaction atlas：context domain、tensor atlas、gluing 与收尾

| 日期 | commit | 类型 | 原始主题 |
|---|---|---|---|
| 2026-09-26 | `49fde84` | 重构/重定义 | refactor: formalize context-restricted FIBRE domain |
| 2026-09-26 | `9043ae2` | 固化/发布/复现 | chore: refresh FIBRE release manifests |
| 2026-09-26 | `ea202c5` | 实验/实现/迭代 | feat: reuse verified agent observations |
| 2026-09-26 | `65a1b5a` | 实验/实现/迭代 | feat: add tensor-product atlas field |
| 2026-09-26 | `d03aed4` | 实验/实现/迭代 | chore: add agent evaluation suite runner |
| 2026-09-26 | `7e8f6a6` | 固化/发布/复现 | chore: refresh release manifests after closeout |
| 2026-09-26 | `ad18eb2` | 修正/边界澄清 | fix: make agent metamorphic evaluation fail safely |
| 2026-09-26 | `ef1a99f` | 修正/边界澄清 | chore: refresh release manifests for agent eval fix |
| 2026-09-26 | `f358a8a` | 实验/实现/迭代 | feat: add FIBRE interaction atlas |
| 2026-09-26 | `f8a51f9` | 重构/重定义 | refactor: redefine FIBRE as interaction atlas |
| 2026-09-26 | `e7cf582` | 修正/边界澄清 | refactor: align FIBRE release profiles with atlas model |
| 2026-09-26 | `c3067ee` | 固化/发布/复现 | chore: refresh FIBRE atlas release manifests |
| 2026-09-26 | `934272e` | 实验/实现/迭代 | feat: add atlas gluing loss to multi-expert training |
| 2026-09-26 | `1b897e8` | 固化/发布/复现 | chore: refresh atlas gluing release manifests |
| 2026-09-27 | `8d23b9b` | 实验/实现/迭代 | feat: close out FIBRE interaction atlas |
| 2026-09-27 | `a4517fa` | 固化/发布/复现 | chore: refresh FIBRE atlas closeout manifests |
| 2026-09-27 | `6f29ccf` | 实验/实现/迭代 | refine FIBRE interaction atlas semantics |

## 附录 B：`results/` 顶层实验产物全集索引

下表列出当前工作区 `results/` 的全部顶层结果目录。它们并不都具有同等证据等级：有的是正式冻结评测，有的是开发筛选、smoke、negative result、资产审计或运行时产物。正文只提升关键转折；本附录保留名称以避免小尝试在历史整理中消失。

### BiME-Rank / 多专家生产

- `bime_rank_unified_v1`
- `unified_safe_system_v1`

### 通用检索、表示与外部基线

- `broad_rhea_clean_models_paired_1epoch_v1`
- `broad_rhea_difficulty_performance_nested_selected_v1`
- `broad_rhea_difficulty_slices_v1`
- `broad_rhea_fair_benchmarks_v1`
- `broad_rhea_full_candidate_base_v1`
- `broad_rhea_full_candidate_clean_v1`
- `broad_rhea_full_candidate_nested_selected_v1`
- `broad_rhea_full_candidate_novelty_expert_frozen_v1`
- `broad_rhea_full_candidate_paired_1epoch_v1`
- `broad_rhea_nested_crossfold_difficulty_v1`
- `broad_rhea_nested_dev_v1`
- `broad_rhea_nested_dev_v2`
- `broad_rhea_nested_difficulty_v2`
- `broad_rhea_novelty_confirmatory_v1`
- `broad_rhea_novelty_expert_frozen_v1`
- `broad_rhea_novelty_route_distance_v1`
- `broad_rhea_novelty_route_frozen_v1`
- `broad_rhea_rdkitplus_outer_bootstrap_v1`
- `broad_rhea_rdkitplus_outer_eval_v1`
- `broad_rhea_rdkitplus_outer_matrix_v1`
- `cleanroom_internal_full_candidate_benchmarks_v1`
- `cleanroom_internal_full_candidate_difficulty_v1`
- `cleanroom_internal_full_candidate_rdkitplus_v1`
- `cleanroom_internal_functional_prototype_residual_v1`
- `cleanroom_internal_rdkitplus_novelty_hard_slice_v1`
- `cleanroom_internal_reaction_center_bounded_v3`
- `cleanroom_internal_reaction_center_feature_v1`
- `cleanroom_internal_reaction_center_residual_v2`
- `cleanroom_internal_reaction_center_residual_v2_confirmation_v1`
- `cleanroom_internal_shortlist_headroom_v1`
- `cleanroom_internal_top2000_bounded_residual_v2`
- `cleanroom_internal_top2000_bounded_residual_v2_fold3_confirmatory`
- `cleanroom_internal_top2000_difficulty_router_v1_salted_confirm`
- `cleanroom_internal_top2000_pair_reranker_v1`
- `clipzyme_atommap_compatibility_audit_v1`
- `clipzyme_catalyst_common_support_v1`
- `clipzyme_directed_fallback_support_v1`
- `clipzyme_native_extension_v1`
- `clipzyme_outer_overlap_audit_v1`
- `clipzyme_protein_support_v1`
- `clipzyme_reaction_similarity_audit_v1`
- `clipzyme_reactzyme_direction_support_v1`
- `enzgfm_directional_router_temporal_protein_cold_v1_eval`
- `enzgfm_directional_router_temporal_protein_cold_v1_models`
- `enzgfm_directional_router_temporal_protein_cold_v1_selection`
- `enzgfm_gate_a_bidirectional_v1`
- `enzgfm_gate_b_rdkitplus_v1`
- `enzgfm_native_same_support_catalyst_v1`
- `enzgfm_nested_outer_v2_eval`
- `enzgfm_nested_outer_v2_models`
- `enzgfm_nested_outer_v2_selection`
- `enzgfm_rdkitplus_outer_meta_exposure_audit_v1`
- `enzyme405_cleanroom_selected_confirmatory_v1`
- `enzyme405_current_mainline_v1`
- `enzyme405_evidence_chain_audit_v1`
- `enzymecage_405_sequence_consistency_v1`
- `enzymecage_cleanroom_rdkitplus_r2e_gate_confirm_v1`
- `enzymecage_cleanroom_rdkitplus_v1`
- `enzymecage_cleanroom_selected_full_v1`
- `enzymecage_fairness_audit_v1`
- `enzymecage_local_maxsupport_v1`
- `orphan335_author_retrieval_v1`
- `orphan335_fixed_pool_v1`
- `reactzyme_native_bag_adapter_v1`
- `reactzyme_native_bag_adapter_v1_confirmation`
- `reactzyme_native_support_adaptation_v1`
- `tiger_reactzyme_reaction_similarity_native_v1`

### 其他相关实验资产

- `capability_epoch_difficulty_audit_v1`
- `catalyst_clean_mainline_v1`
- `catalyst_external_eval_v2`
- `comprehensive_center_top2000_v1`
- `comprehensive_directional_center_v1`
- `comprehensive_enzgfm_center_top1_v1`
- `enzymarc_open_world_v1`
- `fast_r2e_similarity_router_v1`
- `r2e_lambdarank_fusion_v1`
- `requested_r2e20_bime_v2_20260906`
- `requested_r2e20_nju_lab_final_v2_20260813`
- `requested_r2e20_prokaryote_fewshot_20260809`
- `rhea128_to141_external_v1`
- `rhea128_to141_external_v2`

### 工程/部署/审计

- `deployment`
- `repo_refactor_analysis`
- `reports`

### FIBRE 几何/观测/interaction-atlas

- `fibre_application`
- `fibre_assay_context_v1`
- `fibre_broad_zero_temp_dev_v1`
- `fibre_catalytic_interaction_residual_dev_v1`
- `fibre_catalytic_interaction_residual_smoke_fold0`
- `fibre_catalytic_interaction_residual_smoke_fold0_v2`
- `fibre_catalytic_state_v1`
- `fibre_consensus_stratified_dev_v1`
- `fibre_consensus_stratified_strict_inductive_v1`
- `fibre_cross_source_catalytic_evidence_v1`
- `fibre_enrichment_martsrow_smoke_v1`
- `fibre_enrichment_query_smoke_v1`
- `fibre_enrichment_smoke_v1`
- `fibre_enrichment_smoke_v2`
- `fibre_enzymology_stress_v1`
- `fibre_geometric_foundation_synthetic_v1`
- `fibre_interaction_atlas_full_v1`
- `fibre_interaction_atlas_tps_broad_universe_v1`
- `fibre_levelset_uncertainty_dev_v1`
- `fibre_local_fiber_geometry_v1`
- `fibre_mechanistic_stratified_dev_v1`
- `fibre_mechanistic_stratified_strict_inductive_v1`
- `fibre_observation_index_v1`
- `fibre_partial_relation_dev_v1`
- `fibre_partial_relation_strict_inductive_v1`
- `fibre_promiscuity_holdout_v1`
- `fibre_publication_context_deepseek_broad_pilot_v1`
- `fibre_publication_context_deepseek_v1`
- `fibre_publication_context_pilot_v1`
- `fibre_reaction_tangent_information_dev_v1`
- `fibre_rhea_mapping_v1`
- `fibre_standards_cache_v1`
- `fibre_stratified_correspondence_dev_v1`
- `fibre_stratified_correspondence_strict_inductive_v1`
- `fibre_stratified_correspondence_structure_qualified_dev_v1`
- `fibre_stratified_correspondence_unqualified_dev_v1`
- `fibre_stratified_pocket_dev_v1`
- `fibre_stratified_pocket_strict_inductive_v1`
- `fibre_structure_materialize_smoke_v1`
- `fibre_uniprot_state_v1`
- `fibre_workflow_tooling`
- `fibre_workflow_tooling_venv`

### product-manifold / geometric field

- `geometric_compatibility_clean_dev_v1`
- `geometric_product_field_bidirectional_audit_v1`
- `geometric_product_field_bidirectional_audit_v2`
- `geometric_product_flow_clean_dev_v1`
- `geometric_product_flow_clean_dev_v10`
- `geometric_product_flow_clean_dev_v11`
- `geometric_product_flow_clean_dev_v2`
- `geometric_product_flow_clean_dev_v3`
- `geometric_product_flow_clean_dev_v4`
- `geometric_product_flow_clean_dev_v5`
- `geometric_product_flow_clean_dev_v6`
- `geometric_product_flow_clean_dev_v7`
- `geometric_product_flow_clean_dev_v8`
- `geometric_product_flow_clean_dev_v9`
- `geometric_product_flow_explain_probe_v1`
- `geometric_product_flow_external_retention_v1`
- `geometric_product_flow_protein_conformal_v1`
- `geometric_product_flow_protein_diffusion_v1`
- `product_field_e2r_section_probe_v1`
- `product_field_r2e_section_probe_v1`
- `product_field_section_consistency_v1`
- `product_manifold_applicability_audit_v1`
- `product_manifold_biological_witness_v1`
- `product_manifold_biological_witness_v2`

### 早期 pocket / structure robustness

- `pocket`

### Starase Agent / runtime / evaluation

- `starase_agent_live_replay_v6`
- `starase_navigator_eval_20260922`
- `starase_navigator_runtime`

### TPS / MARTS / 几何前史

- `terpene_anisotropic_product_measure_dev_v1`
- `terpene_architecture_auxiliary_reranking_double_cold`
- `terpene_cage_neural_common_reservoir_specialists_v1`
- `terpene_cage_residual_rescue`
- `terpene_cage_screen`
- `terpene_candidate_hub_normalization_double_cold`
- `terpene_clean_prior_field_dev_v1`
- `terpene_combined_wetlab_campaign`
- `terpene_conformal_retrieval_sets`
- `terpene_correspondence_seed_update_audit_v1`
- `terpene_cross_model_rank_fusion`
- `terpene_current_e2r_general_universe_audit_v1`
- `terpene_current_library_dual_fusion_restricted_v1`
- `terpene_current_library_dual_fusion_v1`
- `terpene_current_library_expert_v1`
- `terpene_current_me8_fusion_rankings_v1`
- `terpene_current_me8_top10_locked_base_v1`
- `terpene_current_pfam_fixed_rankings_v1`
- `terpene_current_pfam_hierarchical_full_v1`
- `terpene_current_pfam_on_locked_me8_v1`
- `terpene_current_r2e_general_universe_broad_experts_v1`
- `terpene_cycle_rerank_grid_v2`
- `terpene_dual_kernel_development_v1`
- `terpene_dual_kernel_frozen_v1`
- `terpene_dual_tower_cold`
- `terpene_dual_tower_multiview`
- `terpene_e2r_route_interleaving_confirmatory20260725`
- `terpene_esmc_fewshot`
- `terpene_exact_entity_protocols`
- `terpene_exact_residual_uncertainty`
- `terpene_free_energy_correspondence_dev_v1`
- `terpene_fusion_sources_100e_v1`
- `terpene_gate_matrix_smoke`
- `terpene_general_evidence_full_adapted_three_seed_eval`
- `terpene_general_evidence_probe_e2r_balanced_seed20260723`
- `terpene_general_evidence_probe_e2r_balanced_seeds20260724_20260725`
- `terpene_general_evidence_probe_e2r_balanced_three_seed`
- `terpene_general_evidence_probe_r2e_balanced_three_seed_repro`
- `terpene_graph_diffusion_double_cold`
- `terpene_heat_density_ratio_dev_v1`
- `terpene_hierarchical_factor_biological_coherence_v1`
- `terpene_horizyn_adapter_full`
- `terpene_horizyn_fusion_double_cold`
- `terpene_horizyn_protein_space_distillation`
- `terpene_horizyn_prott5_bridge_double_cold`
- `terpene_horizyn_reaction_feature_distillation`
- `terpene_horizyn_residual_canonical_exact`
- `terpene_hybrid_bidirectional_cold`
- `terpene_lexicographic_correspondence_dev_v2`
- `terpene_lexicographic_correspondence_dev_v3`
- `terpene_marts_adapted_neighbor_hybrid`
- `terpene_marts_domain_adaptation_cartesian_drfp`
- `terpene_marts_domain_adaptation_cartesian_pu`
- `terpene_marts_domain_adaptation_freeze_reaction`
- `terpene_marts_domain_adaptation_hardneg128_e50`
- `terpene_marts_domain_adaptation_r2e075`
- `terpene_marts_dual_kernel_development_v1`
- `terpene_marts_dual_kernel_frozen_v1`
- `terpene_marts_dual_kernel_rescue_route_v1`
- `terpene_marts_dual_rankings_v1`
- `terpene_marts_fewshot_open_world`
- `terpene_marts_freeze_reaction_neighbor_hybrid`
- `terpene_marts_mechanism_features_v1`
- `terpene_marts_mechanism_rescue_pu`
- `terpene_marts_open_world`
- `terpene_marts_r2e075_neighbor_hybrid`
- `terpene_marts_r2e_symmetry_confirm_v1`
- `terpene_mechanism_sheet_v1`
- `terpene_mechanistic_tie_refinement_dev_v1`
- `terpene_model_rank_fusion_double_cold`
- `terpene_multi_expert_marts_rankings_v1`
- `terpene_multiplex_product_heat_dev_v1`
- `terpene_multiresolution_product_heat_dev_v1`
- `terpene_multiresolution_product_heat_dev_v2`
- `terpene_multiresolution_product_heat_dev_v3`
- `terpene_neighbor_route_rankings_v1`
- `terpene_normalized_product_geodesic_dev_v3`
- `terpene_old_new_comparison`
- `terpene_open_world`
- `terpene_open_world_uncertainty`
- `terpene_open_world_uncertainty_dual_kernel_candidate_v1`
- `terpene_open_world_uncertainty_exact_routing`
- `terpene_open_world_uncertainty_final`
- `terpene_open_world_uncertainty_rrf_e2r`
- `terpene_open_world_uncertainty_rrf_routing`
- `terpene_open_world_uncertainty_rrf_routing_pre_dual_kernel_top20`
- `terpene_p2rank_current_v1`
- `terpene_p2rank_current_v1_smoke`
- `terpene_portal_runtime`
- `terpene_product_correspondence_dev_v1`
- `terpene_product_correspondence_strict_inductive_v1`
- `terpene_product_coupling_defect_dev_v4b`
- `terpene_product_geodesic_atlas_refined_dev_v2`
- `terpene_product_geodesic_dev_v1`
- `terpene_product_geodesic_dev_v2`
- `terpene_product_geodesic_dev_v3`
- `terpene_product_geodesic_dev_v4`
- `terpene_product_geodesic_potential_dev_v1`
- `terpene_production_models`
- `terpene_production_retention`
- `terpene_production_retention_e2r`
- `terpene_protocol_reassessment`
- `terpene_pure_cage_baseline_v1`
- `terpene_pure_cage_baseline_v3`
- `terpene_pure_cage_full_support_v1`
- `terpene_reactzyme_transfer_audit_v1`
- `terpene_registry_batch`
- `terpene_registry_batch_pre_dual_kernel_top20`
- `terpene_registry_batch_raw_rerun_dual_kernel_top20`
- `terpene_retrieval_adamerging_three_seed`
- `terpene_retrieval_adamerging_three_seed_eval_full`
- `terpene_screened_poisson_correspondence_dev_v1`
- `terpene_sequence_fewshot_strict`
- `terpene_smoke`
- `terpene_stable_levelset_correspondence_dev_v1`
- `terpene_symmetric_correspondence_energy_dev_v1`
- `terpene_temporal_holdout_readiness`
- `terpene_uniprot_controlled_rescue_batch`
- `terpene_uniprot_expanded_double_cold`
- `terpene_uniprot_expanded_r2e`
- `terpene_uniprot_expansion_quality`
- `terpene_uniprot_expansion_quality_contract`
- `terpene_uniprot_rescue_campaign`
- `terpene_uniprot_rescue_sequence_integrity`
- `terpene_uniprot_tiered_double_cold`
- `terpene_wetlab_discovery_panels`
- `terpene_wetlab_plate_balanced`
- `terpene_wetlab_plate_manifest`
- `terpene_wetlab_randomized_layout`
- `terpene_zero_shot_cold`
- `tps_active_site_xattn_v1`
- `tps_active_site_xattn_v1r1`
- `tps_foundation_r2e_v1`
- `tps_temporal_r2e_v1`

## 附录 C：BiME-Rank/FIBRE 冻结记录与配置全集

### records

- `reproducibility/bime_rank/records/AUTHORITATIVE_BASELINE_COMPARISONS_V2.md`
- `reproducibility/bime_rank/records/BIME_RANK_COST_AWARE_HIERARCHY_V1_RESULT.json`
- `reproducibility/bime_rank/records/BIME_RANK_EXPERT_ADMISSION_V1.json`
- `reproducibility/bime_rank/records/BIME_RANK_PRODUCTION_V1_RESULT.json`
- `reproducibility/bime_rank/records/BIME_RANK_PRODUCTION_V2_RESULT.json`
- `reproducibility/bime_rank/records/BIME_RANK_RETRIEVAL_CAPABILITY_SCORECARD_V2.json`
- `reproducibility/bime_rank/records/CATALYST_BASELINE_PROVENANCE_V1.json`
- `reproducibility/bime_rank/records/CATALYST_CAPABILITY_BASELINE_CONTRACT_V1.json`
- `reproducibility/bime_rank/records/CATALYST_COMPREHENSIVE_DIRECTIONAL_CENTER_V1.json`
- `reproducibility/bime_rank/records/CATALYST_EXTERNAL_EVALUATION_POLICY_V2.json`
- `reproducibility/bime_rank/records/CATALYST_EXTERNAL_EVALUATION_V2_RESULT.json`
- `reproducibility/bime_rank/records/CATALYST_LEGACY_BROAD_RHEA_CURRENT_TRAIN_AUDIT_V1.json`
- `reproducibility/bime_rank/records/CATALYST_R2E_LAMBDARANK_FUSION_V1.json`
- `reproducibility/bime_rank/records/CATALYST_R2E_LAMBDARANK_FUSION_V1_DEVELOPMENT_RESULT.json`
- `reproducibility/bime_rank/records/CATALYST_TPS_MARTS_R2E_SYMMETRY_CONFIRM_V1.json`
- `reproducibility/bime_rank/records/CATALYST_TPS_MARTS_R2E_SYMMETRY_CONFIRM_V1_RESULT.json`
- `reproducibility/bime_rank/records/CATALYST_TPS_PURE_CAGE_APPLICABILITY_BASELINE_V1_RESULT.json`
- `reproducibility/bime_rank/records/CLEANROOM_R2E_RHEA128_TO141_EXTERNAL_V1.json`
- `reproducibility/bime_rank/records/CLEANROOM_R2E_RHEA128_TO141_EXTERNAL_V1_INVALIDATION.json`
- `reproducibility/bime_rank/records/CLEANROOM_R2E_RHEA128_TO141_EXTERNAL_V2.json`
- `reproducibility/bime_rank/records/CLEANROOM_R2E_RHEA128_TO141_EXTERNAL_V2_RESULT.json`
- `reproducibility/bime_rank/records/CLIPZYME_DIRECTED_REACTION_NOVEL_FALLBACK_CONTRACT_V1.json`
- `reproducibility/bime_rank/records/CLIPZYME_REACTZYME_REACTION_SIMILARITY_BASELINE_CONTRACT_V1.json`
- `reproducibility/bime_rank/records/CURRENT_RETRIEVAL_STATUS.md`
- `reproducibility/bime_rank/records/CURRENT_STRONGEST_VS_ENZYMECAGE.md`
- `reproducibility/bime_rank/records/ENZGFM_NATIVE_SAME_SUPPORT_CATALYST_V1_RESULT.json`
- `reproducibility/bime_rank/records/ENZGFM_OFFICIAL_ASSET_AUDIT_V1.json`
- `reproducibility/bime_rank/records/FIBRE_CATALYTIC_INTERACTION_RESIDUAL_DEV_V1.json`
- `reproducibility/bime_rank/records/FIBRE_INTERACTION_ATLAS_SCORECARD_V1.json`
- `reproducibility/bime_rank/records/R2E_REACTION_CENTER_RESIDUAL_V2_TRAIN_GEOMETRY.json`
- `reproducibility/bime_rank/records/REACTZYME_NATIVE_BAG_ADAPTER_V1_CONFIRMATION_RESULT.json`
- `reproducibility/bime_rank/records/RETRIEVAL_CAPABILITY_SCORECARD.md`
- `reproducibility/bime_rank/records/RETRIEVAL_EVIDENCE_LEDGER.md`
- `reproducibility/bime_rank/records/RHEA128_CLEAN2023_SOURCE_ALIGNMENT_V2.json`
- `reproducibility/bime_rank/records/RHEA128_TO141_V2_FIXED_MODEL_ASSETS.json`
- `reproducibility/bime_rank/records/UNIFIED_SAFE_SYSTEM_E2R_ANCHORED_LAMBDAMART_V3_CONFIRMATION.json`

### configs

- `reproducibility/bime_rank/configs/bime_rank_candidate_v1.yaml`
- `reproducibility/bime_rank/configs/fibre_interaction_atlas_v1.yaml`
- `reproducibility/bime_rank/configs/terpene_lambdarank_candidate_v1.yaml`
- `reproducibility/bime_rank/configs/terpene_similarity_router_v3.yaml`

## 附录 D：当前可追溯的 BiME-Rank/FIBRE 实验脚本全集

- `reproducibility/bime_rank/scripts/__init__.py`
- `reproducibility/bime_rank/scripts/assemble_bime_cost_aware_hierarchy_v1.py`
- `reproducibility/bime_rank/scripts/assemble_bime_expert_admission_v1.py`
- `reproducibility/bime_rank/scripts/assemble_fibre_atlas_scorecard_v1.py`
- `reproducibility/bime_rank/scripts/audit_bime_r2e_homology_admission_v1.py`
- `reproducibility/bime_rank/scripts/build_cold_splits.py`
- `reproducibility/bime_rank/scripts/build_enzgfm_protein_features.py`
- `reproducibility/bime_rank/scripts/build_general_candidate_universe.py`
- `reproducibility/bime_rank/scripts/build_general_reaction_features.py`
- `reproducibility/bime_rank/scripts/build_rdkitplus_augmented_reaction_features.py`
- `reproducibility/bime_rank/scripts/combine_protein_feature_blocks.py`
- `reproducibility/bime_rank/scripts/evaluate_bime_cost_aware_shortlist_retention_v1.py`
- `reproducibility/bime_rank/scripts/evaluate_bime_e2r_seed_context_retention_v1.py`
- `reproducibility/bime_rank/scripts/evaluate_bime_multiseed_scaling_v1.py`
- `reproducibility/bime_rank/scripts/evaluate_bime_r2e_homology_context_retention_v1.py`
- `reproducibility/bime_rank/scripts/evaluate_bime_r2e_seed_context_retention_v1.py`
- `reproducibility/bime_rank/scripts/evaluate_broad_rhea_benchmark.py`
- `reproducibility/bime_rank/scripts/evaluate_catalytic_interaction_residual.py`
- `reproducibility/bime_rank/scripts/evaluate_enzyme405_bime_augmented_v1.py`
- `reproducibility/bime_rank/scripts/evaluate_enzyme405_fixed_support.py`
- `reproducibility/bime_rank/scripts/evaluate_enzymecage_405_cleanroom.py`
- `reproducibility/bime_rank/scripts/evaluate_enzymecage_official_aligned.py`
- `reproducibility/bime_rank/scripts/evaluate_fibre_atlas_tps_broad_universe_v1.py`
- `reproducibility/bime_rank/scripts/evaluate_marts_open_world.py`
- `reproducibility/bime_rank/scripts/extract_esmc_motif_context_embeddings.py`
- `reproducibility/bime_rank/scripts/merge_protein_feature_libraries.py`
- `reproducibility/bime_rank/scripts/prepare_production_dual_kernel_assets.py`
- `reproducibility/bime_rank/scripts/prepare_rhea_snapshot_delta_external_benchmark_v2.py`
- `reproducibility/bime_rank/scripts/pretrain_horizyn_reaction_feature_distillation.py`
- `reproducibility/bime_rank/scripts/rebuild_horizyn_distillation_preprocessing.py`
- `reproducibility/bime_rank/scripts/rebuild_reactzyme_transfer_assets.py`
- `reproducibility/bime_rank/scripts/rebuild_rhea128_to141_strict_support_v2.py`
- `reproducibility/bime_rank/scripts/run_bime_e2r_seed_context_v1.py`
- `reproducibility/bime_rank/scripts/run_bime_r2e_clipzyme_expert_v1.py`
- `reproducibility/bime_rank/scripts/run_bime_r2e_homology_context_v1.py`
- `reproducibility/bime_rank/scripts/run_bime_r2e_reciprocal_consistency_v1.py`
- `reproducibility/bime_rank/scripts/run_bime_r2e_seed_context_v1.py`
- `reproducibility/bime_rank/scripts/run_bime_tps_cage_topk_expert_v1.py`
- `reproducibility/bime_rank/scripts/run_e2r_anchored_lambdamart_v3_production_experts.py`
- `reproducibility/bime_rank/scripts/run_r2e_lambdarank_fusion_v1.py`
- `reproducibility/bime_rank/scripts/run_unified_safe_system_e2r_anchored_v3_experts.py`
- `reproducibility/bime_rank/scripts/train_cleanroom_directional_identity_aux_residual.py`
- `reproducibility/bime_rank/scripts/train_cleanroom_rhea_retriever.py`
- `reproducibility/bime_rank/scripts/train_general_evidence_retriever.py`
- `reproducibility/bime_rank/scripts/train_marts_adapted_production.py`
- `reproducibility/bime_rank/scripts/train_marts_domain_adaptation.py`
- `reproducibility/bime_rank/scripts/train_marts_horizyn_exact_residual_production.py`
- `reproducibility/bime_rank/scripts/train_marts_horizyn_reaction_residual.py`
- `reproducibility/bime_rank/scripts/train_marts_horizyn_residual_production.py`
- `reproducibility/bime_rank/support/__init__.py`
- `reproducibility/bime_rank/support/analyze_uniprot_expansion_quality.py`
- `reproducibility/bime_rank/support/balance_wetlab_reactions_across_plates.py`
- `reproducibility/bime_rank/support/benchmark_e2r_anchored_lambdamart_v3_runtime.py`
- `reproducibility/bime_rank/support/build_combined_wetlab_campaign.py`
- `reproducibility/bime_rank/support/build_wetlab_discovery_panels.py`
- `reproducibility/bime_rank/support/build_wetlab_plate_manifest.py`
- `reproducibility/bime_rank/support/evaluate_architecture_auxiliary_reranking_double_cold.py`
- `reproducibility/bime_rank/support/evaluate_fast_r2e_similarity_router_v1.py`
- `reproducibility/bime_rank/support/evaluate_legacy_cage_double_cold.py`
- `reproducibility/bime_rank/support/evaluate_locked_dual_kernel_route.py`
- `reproducibility/bime_rank/support/evaluate_marts_multi_expert_rank_fusion.py`
- `reproducibility/bime_rank/support/evaluate_model_rank_fusion_double_cold.py`
- `reproducibility/bime_rank/support/evaluate_motif_channel_reranker.py`
- `reproducibility/bime_rank/support/evaluate_multi_expert_protocol_comparison.py`
- `reproducibility/bime_rank/support/evaluate_pocket_local_reranker.py`
- `reproducibility/bime_rank/support/evaluate_r2e_lambdarank_fusion_v1_confirmation.py`
- `reproducibility/bime_rank/support/evaluate_two_stage_motif_residual.py`
- `reproducibility/bime_rank/support/manage_wetlab_feedback.py`
- `reproducibility/bime_rank/support/randomize_wetlab_candidate_positions.py`
- `reproducibility/bime_rank/support/rank_current_library.py`
- `reproducibility/bime_rank/support/rank_registry_batch.py`
- `reproducibility/bime_rank/support/run_unified_safe_system_e2r_anchored_v3_confirmation.py`
- `reproducibility/bime_rank/support/train_cleanroom_identity_aux_residual.py`
- `reproducibility/bime_rank/support/validate_dual_kernel_deployment.py`
