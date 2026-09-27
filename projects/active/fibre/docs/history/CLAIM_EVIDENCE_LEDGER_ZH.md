# FIBRE / BiME-Rank / TPS 叙事 Claim–Evidence Ledger

> 目的：给比赛页面、答辩、论文式写作和图注提供“哪些话可以说到什么程度”的统一边界。本文不替代 Raw History 或 Scientific Narrative；它只管理 claim 类型、证据入口和禁止的过度表述。

## 1. Claim 类型

- **数学命题**：在已声明假设下可以证明；应指向当前理论/代码 invariant。
- **已实现机制**：当前仓库代码确实具备；不自动意味着已做完整 application validation。
- **冻结实证**：在明确 protocol/candidate universe 上有冻结或可追溯结果。
- **历史实证**：真实做过，但协议/模型已 superseded；只用于解释迭代，不作为 current headline。
- **产品能力**：当前 Agent/runtime 能实际执行；应与模型性能 claim 分开。
- **HP / 用户反馈事实**：由团队或原始访谈记录支持；没有 primary record 时不得虚构受访者、逐字引语或企业背书。
- **未来扩展**：数学上自然或工程上计划做，但当前不能写成已实现。

## 2. 核心 claim ledger

| ID | 可使用的 claim | 类型 / 当前状态 | 主要证据入口 | 写作边界 |
|---|---|---|---|---|
| C01 | 直接对 1,391 条 TPS 候选做 EnzymeCAGE all-pair scoring 时，10 个 query 的 Hit@1/5/10 均为 0，MRR 约 0.0037。 | 历史实证 | `results/terpene_cage_screen/terpene_screen_report.md` | 可用于说明 pair scorer 不自动成为 large-universe retriever；不可推出 EnzymeCAGE 在所有任务上无效。 |
| C02 | reaction-similarity gate 能改善池内排序，但 all-Rhea known-positive coverage 约 43.98%，形成 hard recall ceiling。 | 历史实证 | `results/terpene_cage_screen/all_rhea_gate_summary.json`、`similarity_gate_metrics.json` | “池外 positive 无法被后端救回”是由候选构造逻辑决定；不要把 43.98% 写成真实生物活性 prevalence。 |
| C03 | TPS V1 将任务正式改成 bidirectional open-world retrieval，并使用 raw molecular inputs、multi-positive contrastive learning 与 PU-aware masking。 | 已实现的历史阶段 | `docs/terpene_candidate_retrieval_new_scheme_technical_report.md`、对应 July-24 lineage | V1 的价值不是“所有旧指标都更高”；exact-reaction Hit@5 并非全面占优。 |
| C04 | 在共同 strict 25-cell double-cold protocol 上，新 TPS dual tower 显著优于旧 RF/CAGE reservoir-rebuild route。 | 历史冻结实证 | `results/terpene_old_new_comparison/` 与技术报告 | 必须和 exact-reaction protocol 分开；不同 candidate/split 不能混成一个总提升。 |
| C05 | 真实用户/HP 反馈发生在 formal bounded Agent harness 之前，并推动 intent routing、ambiguity handling 与 scope switching。 | HP 团队事实 + Git chronology | Aug-20 routing commits；Aug-28 `4f13385` formal harness | 当前仓库未审计到同等级 primary interview archive；对外若要写“某企业/某用户说了什么”，需补原始记录。 |
| C06 | Agent V1 的主要作用是把 route/scope complexity 从用户界面中隐藏；随后发展为持久 scientific workspace。 | 产品演化事实 | Aug-20–29 Catalyst commits；Sep-21–22 Starase workspace commits；`frontend/starase_navigator/README.md` | 不写成“Agent 自己发现新催化规律”；LLM orchestration 与 deterministic scientific invariants 要分开。 |
| C07 | general retrieval candidate universe 扩展到 185,918 proteins / 11,081 reactions。 | 当前/冻结数据事实 | FIBRE/BiME frozen records 与 scorecard | 潜在 pair 密度计算只能说明记录稀疏，不能当成真实 negative rate。 |
| C08 | BiME-Rank 并不是一次性设计出的普通 MoE，而是 continuation、representation、cleanroom、expert admission、LTR、anchoring 和 cost-aware execution 的收敛结果。 | 历史综合结论 | `docs/archive/bime_rank/20260907/RETRIEVAL_EVIDENCE_LEDGER.md`、`CURRENT_RETRIEVAL_STATUS.md`、Git Aug30–Sep6 | 可以总结实验族；不要把每个中间 candidate 都写成正式“模型版本”。 |
| C09 | unconstrained E2R LambdaRank 曾显著破坏 MRR/Hit@10，即使 AUROC 略升；后续用 baseline anchoring 限制 reranking。 | 历史负结果 / 设计依据 | historical `UNIFIED_SAFE_SYSTEM_E2R_LAMBDARANK_STACK_V1_RESULT.json` 与 anchored LambdaMART records | 用于解释“强 base order 需要保护”；不能泛化成所有 LTR 都伤 E2R。 |
| C10 | reaction-center information 在部分 internal/aggregate protocol 有价值，但直接拼接、fresh temporal transfer 或 global tangent geometry 并不稳定。 | 多阶段历史实证 | reaction-center V1/V2/V3 records；Rhea128→141 external v2；`fibre_reaction_tangent_information_dev_v1` | 正确结论是“信息有价值但进入方式/适用域条件性强”，不是“reaction center 无用”。 |
| C11 | TPS active-site XAttn V1 因 candidate-ID alignment bug 被整体 invalidated，已 reveal folds 不再作为 confirmation。 | 历史作废记录 | historical `CATALYST_TPS_ACTIVE_SITE_XATTN_V1_INVALIDATION.json`、commit `99df860` | 禁止引用作废性能数字作为模型效果。可把它作为 identity/provenance 科研纪律案例。 |
| C12 | Product-manifold / correspondence geometry 是 FIBRE 的重要中间理论阶段，但不是 current ontology。 | 历史理论 | `projects/active/fibre/docs/legacy_geometry/`、Sep-18–20 geometry lineage | 可讲 sparse correspondence、defect、exact seed update、partial relation；不要把旧公式写成 current inference。 |
| C13 | Current FIBRE 的核心是 local interaction charts、chart-specific bilinear forms、availability-aware partition、overlap gluing。 | 当前数学 + 已实现机制 | `docs/method.md`、`docs/catalytic_kernel_foundation.md`、`kernel/atlas.py` | “同一个 scientific target”不等于 frozen R2E/E2R 最终 score 逐点相同。 |
| C14 | 旧 global/expert convex mixture 可以严格重写为包含 universal chart 的 partition of unity。 | 数学命题 + 测试 | `docs/theory/FIBRE_INTERACTION_ATLAS_THEORY_ZH.tex`、atlas tests、multi-expert evaluator | 这是代数重解释，因此 ontology change 本身不应被宣传成数值提升来源。 |
| C15 | Frozen implementation 当前仍是 directional：R2E 与 E2R 使用不同 query-side partitions，但共享 local pair scores。 | 当前实现事实 | `reproducibility/bime_rank/configs/fibre_interaction_atlas_v1.yaml`、multi-expert evaluator、`docs/method.md` | 禁止写“当前已经严格实现一个方向无关的全局 score matrix”。可写共享 task-level compatibility target。 |
| C16 | 两个 directional partitions 对同一 local score vector 的 readout discrepancy 有 total-variation/range 上界。 | 数学命题 + 已实现 invariant | `kernel/atlas.py`、`test_interaction_atlas.py`、理论专著 | 上界不代表两个方向实际很接近；实际 gap 仍需 empirical measurement。 |
| C17 | Missing optional modality 的语义是 chart unavailable / zero partition mass，而不是 low score 或 biological negative；universal raw-input chart 保证 valid pair 可打分。 | 当前数学 + 已实现 invariant | `kernel/atlas.py`、tests、`docs/method.md` | “可打分”不等于“准确”；strict double-cold/temporal evidence另行支撑 generalization。 |
| C18 | 新 accepted positive observations 可形成 rank≤n 的 chart-local finite-rank update，并受 Frobenius budget 约束。 | 数学命题 + kernel 实现 | `kernel/interaction.py`、interaction tests、理论专著 | 可以说 primitive 已实现；不能说任意外部湿实验已经自动端到端投影并更新所有 applicable charts。 |
| C19 | Final atlas 在 TPS strict broad-universe 185,918-protein retrieval 上优于同一 trained model 的 universal-chart-only ablation。 | 当前冻结实证 | `FIBRE_INTERACTION_ATLAS_SCORECARD_V1.json` 与 broad-universe TPS evaluation | 只能解释为 local specialization 在扩大 universe 后仍有增益；不能外推成所有 general enzyme discovery 都解决。 |
| C20 | Final atlas 对 historical MARTS-only 8-expert route 是 mixed trade-off：R2E MRR/Hit@20 提升但 Hit@10 下降；E2R 多项改善。 | 当前冻结实证 | scorecard / status | 必须保留 R2E Hit@10 -1.45 pp；禁止只报赢的指标。 |
| C21 | TPS wet-lab history 为最初任务和候选工作流提供真实 grounding。 | 历史 application evidence | TPS campaign/workflow lineage、Sep-8 reproduction-boundary corrections | 禁止把旧 TPS wet-lab campaign 写成 final FIBRE interaction-atlas-specific prospective validation。 |
| C22 | 不同 candidate universe/protocol 的指标构成 capability surface，不应压成一个 pooled scalar。 | 当前 research policy | `docs/research_policy.md`、scorecard | TPS legacy-exact、strict double-cold、temporal general、Enzyme-405、Orphan-335、broad universe 必须分开。 |

## 3. Human Practices 专用边界

目前可以安全写：

> HP 组组织外部真实用户试用；用户对自然语言任务表达、歧义澄清、连续追问和更多科研场景的需求推动了 semantic routing 与后续 Agent/workspace 的迭代。Git chronology 与这一先后关系一致。

目前不能仅凭已审计仓库写：

- “某企业明确指出 BiME-Rank 在生物学上不合理”；
- “某企业要求我们改成 interaction atlas”；
- 任何未找到 primary record 的逐字引语、企业名称或人数统计。

如果后续拿到 HP 原始访谈记录，应把它们作为独立 primary evidence 接入本 ledger，而不是反过来修改技术实验已经支持的因果链。

## 4. 当前模型最容易被误写的五句话

### 错误 1
“FIBRE 学到了真实的 context-free catalytic activity function。”

**应改为**：FIBRE 当前估计声明 retrieval 语义下的 task-level working catalytic compatibility；物理 activity 更完整地依赖 context `A(r,e;c)`。

### 错误 2
“R2E 和 E2R 只是同一 score matrix 的行和列。”

**应改为**：这是历史 product-field 的理想；当前 frozen atlas 共享 local pair interactions，但两个方向可使用不同 partitions。

### 错误 3
“结构/口袋缺失说明这个 candidate 不太可能催化。”

**应改为**：missing optional view 使对应 chart unavailable；其 mass 为 0，剩余 charts renormalize。

### 错误 4
“加入一个新湿实验后 FIBRE 已经能自动无训练更新整个系统。”

**应改为**：bounded finite-rank chart-local update primitive 已实现；完整实验解析与跨-chart orchestration 仍是独立 application step。

### 错误 5
“最终 FIBRE 已经被之前 TPS 湿实验直接验证。”

**应改为**：旧湿实验验证/grounding 属于 TPS lineage；final atlas 目前主要由冻结计算评测与同模型 ablation 支撑。

## 5. 使用顺序

写一条对外结论前，按顺序检查：

1. 这句话属于数学、实现、实证、产品、HP 还是未来扩展？
2. 它对应哪个明确 protocol / candidate universe / time snapshot？
3. 是否存在相反指标或 negative result 必须同时披露？
4. primary evidence 是当前 frozen record，还是只存在于历史 commit？
5. 是否把“可打分”“有增益”“已校准”“物理真实”“湿实验验证”这五种完全不同的强度混在了一起？

只有这五步都能回答清楚，才把 claim 升格到比赛主页面或答辩 headline。
