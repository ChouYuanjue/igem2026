# FIBRE / BiME-Rank / TPS 叙事 Claim–Evidence Ledger

> 目的：给比赛页面、答辩、论文式写作和图注提供“哪些话可以说到什么程度”的统一边界。本文不替代 Raw History 或 Scientific Narrative；它只管理 claim 类型、证据入口和禁止的过度表述。

## 1. Claim 类型

- **数学命题**：在已声明假设下可以证明；应指向当前理论/代码 invariant。
- **已实现机制**：当前仓库代码确实具备；不自动意味着已做完整 application validation。
- **冻结实证**：在明确 protocol/candidate universe 上有冻结或可追溯结果。
- **历史实证**：真实做过，但协议/模型已 superseded；只用于解释迭代，不作为 current headline。
- **产品能力**：当前 Agent/runtime 能实际执行；应与模型性能 claim 分开。
- **HP / 用户反馈事实**：由团队或原始访谈记录支持；没有 primary record 时不得虚构受访者、逐字引语或企业背书。
- **未实现内容**：只用于禁止过度表述，不进入主线、不作为模型设计内容展开。

## 2. 方法身份与证据归属

| 方法身份 | 在项目中的位置 | 主要证据归属 |
|---|---|---|
| EnzymeCAGE | 项目起点使用的外部结构配对工具。 | 早期口袋稳健性、TPS 全库结构配对与后续官方式同支持比较。 |
| 自研开放双塔 / MARTS 专项适配 | Catalyst 形成前的开放检索前身，建立从原始反应与蛋白输入直接进入双向排序的能力。MARTS 在这里是数据与专项适配来源，不单独作为最终方法名。 | 旧协议、严格双未见、专项适配、难负例与方向化生产路线。 |
| Catalyst | 通用候选宇宙、干净室评测、持续学习、反应中心与多表示研究收敛出的检索体系。 | Catalyst cleanroom、representation、continual-learning 与综合方向记录。 |
| BiME-Rank | 完成专家准入、候选级学习排序、缺失退回、上下文条件重排和成本分层的双向多专家方法。 | 通用严格时间、Enzyme-405、孤儿反应、结构专家、seed-context 与 production-v2。 |
| FIBRE-Atlas | FIBRE 第一版统一交互架构，把广域通用分量与局部专家组织成条件交互图册。 | 第一版 TPS 冻结协议、185,918 蛋白迁移和交互图册数学。 |
| FIBRE-Modes | FIBRE 第二版冻结科研模型；保留一个通用分量与八个查询条件局部模式，删除跨专家一致性惩罚。 | 第二版 TPS 熟悉协议、完整 25 格严格双冷启动、185,918 蛋白广域迁移、模式数学与反应中心模式更新实验。 |

写作时优先使用上表中的方法名作为数字主语。BiME-Rank 的通用广域结果可以出现在 FIBRE 的研发叙事中，因为它们构成前代已经建立的能力基线；FIBRE 自身的数字则用于证明统一交互架构在对应冻结协议上的表现。不同方法的证据可以形成连续研发链，但不能用“当前模型”把它们合并成同一次模型评测。

## 3. 核心 claim ledger

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
| C13 | 当前 FIBRE 的计算读出是两层条件结构：128 维通用交互块承担广域基线，八个 32 维局部专家承担潜在催化模式；专家总质量决定是否进入专项分支，查询侧门控决定专项分支内部的模式分布。 | 当前数学 + 已实现机制 | `FIBRE_CONDITIONAL_MODES_THEORY_ZH.md`、`kernel/atlas.py`、multi-expert evaluator | 通用分量没有被赋予第九种微观催化机制含义；八个局部模式也没有被预先命名成八种已知物理机制。 |
| C14 | 第一版 global/expert convex mixture 可以严格重写为包含 universal chart 的 partition of unity；第二版保留相同数值读出，数学解释改为 query-conditioned direct sum。 | 历史数学命题 + 当前重解释 | 第一版理论、atlas tests、第二版理论与 `directional_mode_readout` 测试 | 这是对真实计算图的重解释，不能把术语变化宣传成性能来源。 |
| C15 | 当前冻结实现仍然 directional：反应找酶与酶找反应共享九个 pair-score 分量，分别使用反应侧与蛋白侧条件权重。 | 当前实现事实 | `reproducibility/bime_rank/configs/fibre_conditional_modes_v2.yaml`、multi-expert evaluator、第二版理论 | 可以说两个方向共享底层配对证据；不能写成最终条件排序逐点相同。 |
| C16 | 条件读出可以由块对角算子精确表示，也可以写成通用基线/专项分支的两层潜变量期望；完整分歧满足全方差公式：局部模式内部方差 + 通用基线与专项均值之间的方差。 | 数学命题 + 已实现 invariant | `kernel/atlas.py`、`test_interaction_atlas.py`、第二版理论 | 这些分歧量当前不改动排序，也没有校准成实验成功概率。 |
| C17 | 当前冻结八模式均由统一的反应多视图和蛋白序列输入产生；结构、口袋、反应中心和种子上下文没有进入当前冻结八模式的显式输入。 | 当前实现事实 | 第二版配置、第二版理论、历史局部信息实验 | 不得把这些历史信息源写成当前冻结八模式的组成。 |
| C18 | 新 accepted positive observations 可形成 rank≤n 的 chart-local finite-rank update，并受 Frobenius budget 约束。 | 数学命题 + kernel 实现 | `kernel/interaction.py`、interaction tests、理论专著 | 可以说 primitive 已实现；不能说任意外部湿实验已经自动端到端投影并更新所有 applicable charts。 |
| C19 | 第二版条件模式在 TPS strict broad-universe 185,918-protein retrieval 上优于同一第二版训练模型的通用分量消融。 | 当前冻结实证 | `FIBRE_CONDITIONAL_MODES_SCORECARD_V2.json` 与第二版 broad-universe evaluation | 只能解释为局部模式在扩大 universe 后仍有增益；不能外推成所有 general enzyme discovery 都解决。 |
| C20 | 第二版相对第一版删除 0.02 跨专家一致性惩罚；完整协议中 R2E 多项改善，E2R 保留很小的指标交换。第一版对 historical MARTS-only 的 mixed trade-off 继续作为历史证据。 | 当前冻结实证 + 历史比较 | 第二版 scorecard、第一版 scorecard、status | 必须同时报告 E2R Hit@10/20 的小幅交换；不能把第二版写成所有指标逐项提高。 |
| C21 | TPS wet-lab history 为最初任务和候选工作流提供真实 grounding。 | 历史 application evidence | TPS campaign/workflow lineage、Sep-8 reproduction-boundary corrections | 禁止把旧 TPS wet-lab campaign 写成 final FIBRE interaction-atlas-specific prospective validation。 |
| C22 | 不同 candidate universe/protocol 的指标构成 capability surface，不应压成一个 pooled scalar。 | 当前 research policy | `docs/research_policy.md`、scorecard | TPS legacy-exact、strict double-cold、temporal general、Enzyme-405、Orphan-335、broad universe 必须分开。 |
| C23 | 当前 FIBRE 没有使用真实动力学常数训练，也不输出激活自由能、催化常数、米氏常数或能量单位。 | 当前实现边界 | `method.md`、`status.md`、第二版理论 | 禁止把当前排序分数包装成动力学量。 |
| C24 | 对现有无量纲分数做过加权对数指数和与二阶非线性聚合探针；前者冻结回退，后者开发阶段否决，均未晋级。 | 已完成负结果 | 选择历史、`FIBRE_SECOND_ORDER_SCORE_AGGREGATION_V1_RESULT.json`、对应结果目录 | 只能写成已完成的聚合负实验；不能把它们描述成已实现动力学模型。 |
| C25 | 模式概率单纯形上的相对熵最小改动更新已经实现：\(q_k^+\propto q_k^-e^{\ell_k}\)。零证据严格回退，多份对数证据可组合。 | 数学命题 + 已实现原语 | `kernel/mode_geometry.py`、`test_mode_geometry.py`、第二版理论 | 反应中心实验已经实际使用该结构；当前生产评分仍未启用该证据头。 |
| C26 | 排名置信度当前定义为预声明扰动下的 top-K 稳定进入概率、两候选顺序概率与名次分位区间。 | 数学命题 + 已实现原语 | `kernel/ranking_confidence.py`、`test_ranking_confidence.py`、第二版理论 | 这些量是排序稳定概率，不是催化成功概率；近似并列不能再由机器精度阈值决定。 |
| C27 | 训练中的门控熵只是一项离散模式分布的无量纲统计正则，用于抑制门控塌缩。 | 当前实现事实 | 第二版配置、`gate_regularization` 实现 | 不给门控熵附加任何未实现的物理含义。 |
| C28 | FIBRE 已实现冻结核心排序上的科学证据扩展层：新证据按查询内有效候选校准，以非负强度叠加；缺失证据贡献严格为零，也不会重分配其他证据的权重。 | 当前实现 + 已完成验证 | `kernel/evidence_fusion.py`、`runtime/scientific_evidence.py`、`FIBRE_SCIENTIFIC_EVIDENCE_V1_RESULT.json` | 该层位于冻结核心排序之上，不改写 FIBRE-Modes 的八个潜在催化模式。结构证据已完成一次严格时间确认；已知阳性上下文与反应中心机制证据已完成内部交叉拟合。 |
| C29 | 反应中心已经以统一“机制证据”接口完成一次独立准入：三个 clean2023 留出折的成对逻辑损失均下降，最终非负基础强度为 0.21333，RXNMapper 质量斜率为 0.11159。 | 当前实现 + 内部交叉拟合 | `export_reaction_center_evidence_v1.py`、`FIBRE_REACTION_CENTER_SCIENTIFIC_EVIDENCE_V1_RESULT.json` | 这里验证的是内部 held-out 候选级增量价值；没有新的外部严格时间确认，因此不把这组数写成外部泛化结果。 |
| C30 | 自部署用户可以用表格接口接入自己的科学证据：先通过 `fit_scientific_evidence.py` 在显式阳性/阴性历史数据上交叉拟合准入，再通过 `apply_scientific_evidence.py` 应用到冻结核心候选分数，并保存每条证据的实际贡献。 | 当前实现事实 | `runtime/scientific_evidence.py`、两个 CLI、对应测试 | 宿主、库存、表达性和实验成本属于候选约束/实验决策，不作为催化科学证据；没有历史标签的本地模块不会自动获得排序权重。 |

## 4. Human Practices 专用边界

目前可以安全写：

> HP 组组织外部真实用户试用；用户对自然语言任务表达、歧义澄清、连续追问和更多科研场景的需求推动了 semantic routing 与后续 Agent/workspace 的迭代。Git chronology 与这一先后关系一致。

目前不能仅凭已审计仓库写：

- “某企业明确指出 BiME-Rank 在生物学上不合理”；
- “某企业要求我们改成 interaction atlas”；
- 任何未找到 primary record 的逐字引语、企业名称或人数统计。

当前没有同等级原始访谈档案，因此 ledger 不写逐字引语、企业名称或人数统计。

## 5. 当前模型最容易被误写的五句话

### 错误 1
“FIBRE 学到了真实的 context-free catalytic activity function。”

**应改为**：FIBRE 当前估计声明 retrieval 语义下的 task-level working catalytic compatibility；物理 activity 更完整地依赖 context `A(r,e;c)`。

### 错误 2
“R2E 和 E2R 只是同一 score matrix 的行和列。”

**应改为**：当前第二版共享同一组通用分量和八个潜在催化模式分数；R2E 由反应侧条件权重读出，E2R 由蛋白侧条件权重读出。两个方向共享底层配对证据，最终条件排序允许不同。

### 错误 3
“结构/口袋缺失说明这个 candidate 不太可能催化。”

**应改为**：当前冻结八模式只依赖统一基础输入。结构、口袋和反应中心没有进入当前冻结八模式的显式输入。

### 错误 4
“加入一个新湿实验后 FIBRE 已经能自动无训练更新整个系统。”

**应改为**：模式后验更新原语已经实现，反应中心证据头也完成了独立开发—冻结实验；当前生产评分没有自动湿实验写回链路。

### 错误 5
“最终 FIBRE 已经被之前 TPS 湿实验直接验证。”

**应改为**：旧湿实验验证与问题来源属于 TPS 历史；当前第二版主要由开发选择、单次冻结确认、完整协议、广域同模型消融和独立外部证据支撑。

## 6. 使用顺序

写一条对外结论前，按顺序检查：

1. 这句话属于数学、实现、实证、产品、HP 还是未实现内容？
2. 它对应哪个明确 protocol / candidate universe / time snapshot？
3. 是否存在相反指标或 negative result 必须同时披露？
4. primary evidence 是当前 frozen record，还是只存在于历史 commit？
5. 是否把“可打分”“有增益”“已校准”“物理真实”“湿实验验证”这五种完全不同的强度混在了一起？

只有这五步都能回答清楚，才把 claim 升格到比赛主页面或答辩 headline。
