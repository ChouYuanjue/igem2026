# 从“该测哪几个酶”到 FIBRE Interaction Atlas：项目科学叙事主线

> **用途**：本文件为后续比赛网页、答辩、论文式 Methods/Results、视频脚本和团队统一口径服务。它不是 Raw History 的删减版，而是从完整研发史中重新组织一条逻辑稳定、证据边界清楚的科学故事。
>
> **范围**：TPS → 通用酶-反应检索 → BiME-Rank → FIBRE interaction atlas，以及与之并行、由真实用户反馈推动的 Catalyst/Starase Agent。数据库专项暂不纳入本叙事。

---

## 一、最好的开头不是“我们做了一个模型”，而是“实验位点太少”

合成生物学里，计算最终必须落到实验。一个反应可能对应成百上千个候选酶，但实验组真正能在一轮中构建、表达和测试的候选只有少数。于是最初的问题并不是：

> “这个 enzyme-reaction pair 是 0 还是 1？”

而是：

> **“在一个真实候选库里，我们最应该先测哪几个？”**

这一区别决定了整个项目从一开始就是 **retrieval / ranking**，而不是普通二分类。评价也自然围绕 Hit@K、MRR、positive rank、候选预算和 wet-lab list 展开。

最早的 TPS 场景尤其适合暴露这个问题：研究对象相对明确、实验目标真实、候选库规模已经足以让“一个一个看结构”不可行，同时又能直接检验推荐是否具有实验意义。

---

# 第一幕：结构信息很多，但“会给 pair 打分”并不等于“会从全库里找酶”

## 二、第一批实验先从当时最自然的结构路线开始

早期团队大量使用 EnzymeCAGE、AlphaFold、P2Rank，并专门做了 pocket robustness：官方 pocket、P2Rank、fpocket、多 pocket、不同聚合方式、不同 detector 的 union 都实际跑过。

这一阶段有两个很有价值的发现。

### 2.1 P2Rank 可以作为现实的 pocket fallback，但“多几个 pocket”没有自动产生更多检索能力

在 Enzyme-405 50-reaction slice 上，official pocket 与 P2Rank top-1 的 Hit@5/10 都是 0.54 / 0.68；P2Rank top-k 的 max、mean、rank-weighted、softmax 几种聚合仍然没有提升。虽然约一半样本的最高模型分来自非第一 pocket，说明 pocket localization uncertainty 确实存在，但**不确定性存在 ≠ 朴素扩大候选 pocket 就能利用它**。

fpocket 和 P2Rank+fpocket union 反而更差。这第一次给出后来反复出现的设计警告：

> **更多生物信息不是越多越好；信息只有在适合的模型语境中才应该影响排序。**

### 2.2 直接把 CAGE 用到 1,391 条 TPS 候选上，几乎完全失败

10 个 TPS reactions × 1,391 proteins 的 all-pair CAGE：结构/P2Rank/打分覆盖都接近完整，但 Hit@1/5/10 全部为 0，MRR 只有 0.0037，真实酶的 median best-positive rank 为 387。

这个结果比任何“某个模型提升几点”都更决定项目方向：

> **pair scorer 与 retrieval model 不是同一种能力。**

结构网络可以判断一对输入，但大候选空间中大量“看起来也合理”的 decoys 会淹没真正正例。

## 三、相似反应 gate 能救排名，却把未知正例永久挡在门外

用 reaction similarity 先召回候选，再让 CAGE/RF 排序，MRR 明显改善。但 all-Rhea audit 发现已知 positive pair coverage 只有约 44%。

这意味着一个不可绕开的上限：只要真实酶没有进入 gate reservoir，后面的任何 scorer 都不可能救回来。

于是团队第一次真正改变任务定义：

> **不能先决定“谁有资格成为候选”，再只在一个熟悉的小池里做分类式打分；模型必须原生支持 open-world retrieval。**

---

# 第二幕：V1 TPS Open-World Retrieval——先把“找酶”这个任务定义正确

## 四、双向检索替代固定类别：reaction 和 enzyme 都应该可以是新实体

2026-07-24 的 TPS 新方案把任务正式写成两个方向：

- reaction → enzyme（R2E）；
- enzyme → reaction（E2R）。

模型以 raw molecular information 为输入：protein 使用 ESM-C；reaction 使用 DRFP 与 precursor/product 描述；两侧进入 dual tower 的共同检索空间。训练使用 multi-positive contrastive objective，因为一个 reaction 可以对应多个 enzymes，一个 enzyme 也可以有 promiscuity。

更关键的是，训练不把所有未知 pair 当成负例。sequence cluster / reaction cluster 被用于 positive-unlabeled masking，防止“数据库没有记录”被误解释为“生物学不可能”。

因此 V1 的价值不是“把 Hit@K 再调高一点”，而是第一次给出了完整的 open-world contract：

> **只要一个新 reaction 或 protein 能从分子输入得到表示，它就可以进入检索；不需要先在训练表中拥有 ID。**

## 五、严格 double-cold 说明为什么这个重定义值得做

在旧 exact-reaction protocol 上，新 dual tower 与旧 RF rescue 各有胜负：Hit@5 甚至略差。但在共同严格 double-cold 上：旧 RF/CAGE Hit@10/20 只有约 0.56% / 1.54%，新 dual tower 达到约 8.96% / 16.81%。

所以故事不能写成“deep learning 全面击败传统方法”，而应该写成：

> **旧系统擅长熟悉 reservoir 内的 reranking；新系统真正得到的是 unseen reaction + unseen enzyme 下的可扩展检索能力。**

## 六、TPS specialization 没有因此被丢掉

V1 很快发现另一个事实：generalizable 不等于 generic。MARTS domain adaptation、family mechanism、few-shot seeds 仍然能对 TPS 有帮助。因此项目从一开始就没有把“通用模型”和“专用模型”理解成二选一。

这个矛盾后来贯穿整个项目：

- 我们需要 universal coverage；
- 也希望 family/mechanism/structure information 在它真正有效时发挥作用；
- 但不能让 specialist 把 candidate universe 重新缩成一个 closed world。

最终 FIBRE atlas 对“universal chart + local family chart”的处理，其实是在解决这里早已出现的问题。

## 七、V1 同时被做成真正能交给实验组的 workflow

随后加入了：

- direction / Top-K 目标专用 route；
- few-shot seed retrieval；
- candidate taxonomy scope；
- controlled external expansion；
- uncertainty / applicability；
- conformal retrieval set；
- cycle-consistency audit；
- wet-lab panel / plate / dedup / feedback；
- 本地 GBK-derived 131,532-protein candidate universe 的实际候选生成。

这里最值得讲的不是每个模块名字，而是研究目标发生了第二次变化：

> **模型输出不只是一个 benchmark score，而要变成“下一轮实验真正测什么”的决策材料。**

同时，一套越来越复杂的 route / scope / seed / candidate-universe 菜单也由此产生。

---

# 第三幕：真实用户把问题从“模型怎么调用”推向“科研问题怎么表达”

## 八、Human Practices 的作用：先有真实试用反馈，再有 Agent 迭代

这里必须尊重项目真实发生的顺序。HP 组邀请外界广泛试用，真实用户先提出问题，团队随后修改系统。Git 中 2026-08-20 已经出现成组的 intent routing、ambiguity handling、follow-up scope switching 和 semantic context 改造；正式的 bounded scientific agent harness 到 08-28 才出现。

因此最合适的叙事不是：

> “我们觉得 Agent 很酷，所以给模型套了聊天框。”

而是：

> **当非开发者真正开始提科研问题时，我们发现不能要求研究者先学会内部 route menu 才能使用模型。**

真实问题会有歧义，会连续追问，会从“这个反应找酶”切到“只看原核”“我已经有一个已知酶”“那这个蛋白还能做什么”“帮我看证据”“能不能接到路径设计”。这些都是自然科研语言，而不是 API 参数。

HP 在这里提供的是**外部需求与可用性验证**。但 HP 不是后续数学模型重构的替代证据：BiME/FIBRE 为什么采用某种技术结构，仍然必须由数据、负结果和理论推导支持。

## 九、Agent 第一阶段：隐藏 route complexity，而不是替模型做科学判断

08-20 到 08-29，系统经历：

- ambiguous intent confirmation；
- conversational intent switching；
- DeepSeek semantic scope selection；
- full-context continuation；
- bounded tool harness；
- model-led planning；
- verified entity memory；
- raw reaction / protein recovery；
- evidence synthesis；
- multi-turn research workspace。

最终形成一条边界：

> **LLM 负责理解用户想做什么、组织工具；deterministic scientific code 负责实体身份、候选域、模型调用、科学不变量和证据 provenance。**

这使 Agent 从“route selector”逐渐变成 research workspace，但不会替模型凭空创造催化关系。

---

# 第四幕：用户场景扩大之后，一个模型已经不够——但“多专家”也不是终点

## 十、General universe 把问题真正放大到广域 enzyme discovery

候选空间被扩展到约 185,918 proteins 与 11,081 reactions。此时 TPS 只是众多 biochemical regimes 中的一个；同时 query 的 knowledge state 也变多：

- zero-shot；
- known positive / few-shot；
- structure available / unavailable；
- reaction-center information available / unavailable；
- broad/general；
- family-specialized；
- novel reaction；
- novel protein；
- temporal transfer。

此时继续用“一个固定表示 + 一个固定 scoring function”覆盖全部场景，开始反复遇到 trade-off。

## 十一、BiME-Rank 是从大量失败/局部成功中长出来的，不是直接拍脑袋做 MoE

### 11.1 第一条尝试：继续训练一个统一模型

团队实际尝试了 RecAdam、Fisher consolidation、RegMean、LwF、score distillation、Margin-MSE、TIES、AdaMerging 等 continual-learning / merging 方法。它们都在试图解决同一件事：吸收 broad evidence，又不忘掉旧能力。

结果是不同方向、场景、表示经常出现 retention trade-off。于是 expert pool 开始比“一个越来越复杂的 universal checkpoint”更合理。

### 11.2 第二条尝试：更多 representation

RDKit+、EnzGFM、CLIPZyme、reaction center、TPS active site、functional prototype 都被独立测试。

但这些实验给出的不是“加信息一定更高”，而是：

- Horizyn MLNCE 单独移植失败；
- RDKit+ 是有价值的 reaction base，但低相似度 novelty router 不稳定；
- reaction-center concat 总体有增益，却没过 hard-slice early-rank gate；
- reaction-center residual 内部确认后，在 fresh Rhea temporal transfer 上 early ranking 退化；
- functional prototype residual 9 个候选全部失败；
- CLIPZyme structure 在有 support 时成为有价值 expert；
- EnzGFM 在不同方向的最佳组合不同；
- TPS active-site XAttn 甚至因 candidate ID alignment bug 把整套 V1 结果全部作废。

这些结果逐渐把一个核心事实变得无法忽视：

> **不同 molecular views 是条件性的。一个 view 有生物学意义，不代表它应该在所有 pair 上以同一种方式改变 score。**

### 11.3 第三条尝试：Learning-to-rank 组合 experts

R2E 最终通过 frozen LambdaRank fusion；E2R 则经历一次很有教育意义的失败：unconstrained LambdaRank 让 AUROC 略升，却把 MRR 和 Hit@10 大幅打崩。后来 Anchored LambdaMART V3 明确保留强 EnzGFM base 的 Top-1，只在受控 shortlist 中调整 2–20 位，再保持 tail。

因此 BiME-Rank 的核心不是“MoE 有很多塔”，而是：

> **专家必须有 admission；missing expert 必须 neutral；强基线必须保护；不同 direction 可以有不同 retrieval risk；昂贵专家只能在值得的 shortlist 上运行。**

## 十二、严格评测纪律与 BiME-Rank 一起成熟

这条主线必须和模型一起讲，否则高分会失去可信度。项目建立了：

- leakage-clean / nested / full-candidate protocol；
- preregister → development → freeze → confirmation；
- revealed external set 不再用于调参；
- baseline support 对齐；
- exact positive ranks；
- candidate-universe identity hash；
- rejected / invalidated results 继续留档；
- old “cold” 如相对 current training 不再 cold，就撤回 headline。

尤其 TPS XAttn 的例子非常适合说明研究纪律：已经完成 HPO 与 confirmation 后发现 ID alignment bug，团队直接宣布全部 V1 performance forbidden，并永久封存已 reveal folds，而不是偷偷修 evaluator 后继续用原 test。

---

# 第五幕：BiME-Rank 已经有效，但它暴露了一个更根本的问题

## 十三、真正稀缺的不是每个对象的信息，而是“谁和谁对应”的可靠监督

到了 broad universe，reaction 与 protein 本身的信息其实非常丰富：

- reaction：substrate/product、DRFP、RDKit descriptors、atom mapping、changed bonds、reaction center；
- protein：sequence、protein LM、family、structure、pocket、motif、cofactor 等。

但可靠 enzyme-reaction correspondence 相对于潜在 pair 空间极稀疏，而且是**选择性观察、positive-unlabeled、context incomplete** 的。

以 185,918 proteins × 11,081 reactions 为例，潜在 pair 超过 20 亿；已记录 association 只有约 24.7 万条，机械密度约 0.012%。这个数字不是“真实活性 prevalence”，而是在提醒我们：**大多数 pair 是 unknown，不是经过实验确认的 negative。**

于是一个更基础的问题出现：

> **怎样利用 rich but heterogeneous per-object information 去推断 sparse pair relation，而不是继续训练更多互相独立的 scorers？**

这才是从 BiME-Rank 走向 FIBRE 的核心科学动因。

## 十四、为什么“企业/用户觉得 MoE 不优雅”只能做外部共鸣，不能做数学因果

真实用户/HP 反馈非常适合支持这些实践问题：

- 为什么这次预测可信？
- 哪些信息真正贡献了？
- 完全没见过的新 reaction/protein 怎么办？
- 结构缺失时系统还工作吗？
- 我自己的新实验怎样进入系统？

这些反馈能验证“可解释、多场景、可吸收新证据”确实重要。但即便有用户觉得多专家堆叠不自然，**FIBRE 也不能仅因为审美反馈而成立**。真正技术依据来自前面那一整批实验：global fusion、local structure、reaction-center、missingness、directional routing 的行为确实显示现有 score ensemble 缺少统一 interaction semantics。

---

# 第六幕：先尝试 Product-Manifold——这是重要的中间理论，而不是最终答案

## 十五、第一次数学重构：把正样本看成两个 molecular spaces 之间的 sparse correspondence

团队先做了一个很自然也很激进的尝试：

- reaction 构成 \(M_R\)；
- enzyme catalytic state 构成 \(M_E\)；
- candidate pair 是 \(M_R\times M_E\) 上的点；
- known positive pairs \(\Omega\) 是 product space 上的 sparse support；
- compatibility 不再来自多个 experts 投票，而来自 candidate 到已知 correspondence 的 intrinsic relation。

核心 correspondence defect：

\[
J_\Omega(r,e)=\min_i[d_R(r,r_i)^2+d_E(e,e_i)^2],
\]

\[
\Delta_\Omega(r,e)=J_\Omega(r,e)-m_R(r)-m_E(e).
\]

它回答一个很漂亮的问题：如果 reaction 和 enzyme 各自都接近已知 support，它们是否能被**同一个已知 biochemical precedent**同时解释？

## 十六、这一阶段并不是一个公式，而是完整的 geometric operator search

团队实际试过 product geodesic、normalized geodesic、potential、coupling defect、symmetric/free-energy correspondence、heat-density ratio、anisotropic measure、screened Poisson、stable level set、lexicographic/Pareto、multiplex/multiresolution heat、product flow v1–v11、multiresolution protein/reaction geometry、pocket/3Di/OT/motif/tangent 等大量版本。

这些实验很重要，因为它们最终证明了**“一个 global molecular metric”仍然是太强的假设。**

## 十七、三个证据迫使我们放弃统一 global factor geometry

### 17.1 Reaction-center 放进 global reaction geometry 会伤害另一方向

最保守的 tangent refinement 已经不新增长程边、不删除 base edge、missing neutral、duplicate idempotent，但 E2R MRR 仍从 0.0723 降到 0.0492。说明 local reaction-center chemistry 不是一个可以无条件写进全局 reaction distance 的修正。

### 17.2 Pocket 很重要，但 coverage 极低

当时 canonical 185,918 proteins 中 materialized P2Rank pocket 约 1,287，只有约 0.69%。多数 local charts 根本没有足够 pocket nodes 构成“结构流形邻域”。如果为了让它起作用再加 availability branch，就已经承认它不是一个 universal global metric。

### 17.3 理论上的“one field”与数值 directional sections 之间存在 consistency gap

Product-field 理论希望 R2E/E2R 是同一个 \(F(r,e)\) 的 fibres，但 query-centered solver 分别在两个方向构造 section，不能仅凭愿望证明 overlap 一致。团队甚至写了 atlas-consistency 文档明确指出：没有 overlap agreement 就不能称同一个 global field。

这三点共同推动下一步：

> **不要继续逼所有信息先成为 global reaction/enzyme manifold 的一部分；允许每种 biochemical/information regime 有自己的局部坐标，只要求它们在同一个 enzyme-reaction interaction 上能够兼容。**

---

# 第七幕：V3 FIBRE Interaction Atlas——从“融合模型”转向“组织 interaction 的局部坐标”

## 十八、先区分物理 activity 与当前任务中的 working compatibility

真实催化依赖 assay、pH、温度、cofactor、浓度、宿主和其它 context。因此更完整对象应写成

\[
A(r,e;c).
\]

当前数据库通常不提供足够密集的 context-resolved observations，所以 FIBRE 不声称自己学习了一个 context-free physical law。它定义任务层 working compatibility：

\[
K^\star(r,e).
\]

这是一个非常重要的谦逊边界：模型用于 retrieval，不等于宣称某 pair 在所有实验条件下都会活跃。

## 十九、每个信息 regime 是一个 local interaction chart

对 chart \(\alpha\)：

\[
\phi_\alpha(r)\in H^R_\alpha,
\qquad
\psi_\alpha(e)\in H^E_\alpha,
\]

\[
K_\alpha(r,e)=
\phi_\alpha(r)^\top G_\alpha\psi_\alpha(e).
\]

这带来几个直接好处：

- sequence 与 structure 不需要同维度；
- reaction-center 与 whole-reaction 不需要共用一个 metric；
- TPS 可以是 family chart，而不是 TPS-only candidate universe；
- pocket 缺失时 chart 直接 unavailable，而不是负证据；
- 一个 broad raw-input chart 始终存在，所以 valid input 永远可打分。

## 二十、Partition of unity 把旧多专家变成一个更明确的数学对象

局部 interaction 用 non-negative weights glue：

\[
\rho_\alpha\ge0,
\qquad\sum_\alpha\rho_\alpha=1.
\]

旧 BiME 的 `global + expert convex mixture` 可以**精确**写成：

\[
\rho_0=1-m,
\qquad
\rho_k=mg_k.
\]

所以 interaction atlas 不是为了数学故事重写所有有效工程，而是解释了：旧 global channel 其实可以视作 universal chart，local experts 是其他 charts，routing/gates 是 partition 的工程近似。

## 二十一、Overlap gluing：多个 chart 可以不同，但不能在同一 pair 上毫无约束地互相矛盾

FIBRE 加入：

\[
\mathcal L_{glue}
=\sum_{\alpha<\beta}
\rho_\alpha\rho_\beta(K_\alpha-K_\beta)^2.
\]

只有两个 chart 都认为自己 applicable 时才惩罚 disagreement；它不强迫 latent coordinates 同构或 isometric。

这恰好把 BiME-Rank 的“expert diversity”重新解释为：**坐标和生物 regime 可以多样，但被估计的 task-level compatibility 是同一个问题。**

## 二十二、方向性：生物 interaction 不因查询方向改变，但当前 retrieval estimator 可以方向化

当前 frozen implementation 并没有强行伪装成严格相同 score：

\[
S_d(r,e)=\sum_\alpha
\rho_\alpha^{(d)}(r,e)K_\alpha(r,e),
\quad d\in\{R2E,E2R\}.
\]

R2E/E2R 使用不同 query-side partitions。我们只把 \(K^\star\) 当成共同 latent scientific target，不声称 \(S_{R2E}=S_{E2R}\) 点对点成立。

而且这种 discrepancy 有确定性上界：

\[
|S_p-S_q|
\le
\frac12\|p-q\|_1(K_{max}-K_{min}).
\]

它说明方向差异只有两种来源：partition 不同、charts 本身意见不同。Overlap consistency 越强，directional partition 的差异越不容易放大成 score disagreement。

## 二十三、新实验如何进入：bounded finite-rank interaction update

接受的新 positive observations 可以形成：

\[
\Delta_\alpha
=\sum_i w_i\rho_\alpha(r_i,e_i)
\phi_\alpha(r_i)\psi_\alpha(e_i)^\top.
\]

它的 rank 不超过 observation 数；再给 Frobenius norm 一个预算，就能严格限制单 pair score perturbation。这样“train-free feedback”不再是一条随手规则，而是 interaction operator 的有限秩更新。

当前 kernel 已实现这个数学 primitive；自动把任意外部 assay 解析并投影到所有适用 charts 的完整 application pipeline 仍应作为独立工程 claim，不能混为一谈。

---

# 第八幕：Starase 的最终角色——不是替代 FIBRE，而是让科研过程连续起来

## 二十四、Agent 第二阶段：Research workspace

随着真实用户问题越来越像连续科研过程，Starase 加入：

- verified entity resolution；
- scientific cards；
- literature / structure / relation evidence；
- reusable workspace objects；
- route design / local route patching；
- budget-aware planning；
- pathway ambiguity resolution；
- provenance-preserving derived routes；
- experimental feedback semantics。

这里可以用一句很清楚的话区分两部分：

> **FIBRE 负责“在 sparse correspondence 下怎样推断 enzyme-reaction compatibility”；Starase 负责“怎样让研究者在模型、证据和后续科学操作之间连续工作”。**

HP 提供真实使用需求与反馈闭环；模型/工具各自仍保持可审计边界。

---

# 第九幕：最终对外应该保留的三代模型和两代 Agent

## 二十五、三代模型

### V1：TPS Open-World Retrieval

**问题**：结构 pair scorer 与 similarity gate 无法可靠支持大候选库、double-unseen 检索。

**回答**：multi-positive dual tower + PU masking + bidirectional retrieval + MARTS specialization + uncertainty/application workflow。

**最值得讲的证据**：直接 CAGE Hit@10=0；gate coverage ceiling；严格 double-cold 中 dual tower 相对旧 rescue 明显提升；TPS 真实候选生成/wet-lab workflow。

### V2：BiME-Rank Generalized Retrieval

**问题**：候选 universe 和 user scenarios 扩大后，单一 representation / checkpoint 无法同时吸收 sequence、structure、reaction center、family、few-shot context。

**回答**：expert admission + direction-aware learning-to-rank + baseline anchoring + conditional context experts + cost-aware execution + strict evaluation lifecycle。

**最值得讲的证据**：大量“更多信息但并不自动更好”的 negative results；R2E LambdaRank、E2R anchored LambdaMART、structure/context admission；external/same-support audits。

### V3：FIBRE Interaction Atlas

**问题**：BiME 能组合 experts，却还没有解释这些 heterogeneous representations 为什么应该属于一个 coherent catalytic model；global product-manifold 尝试又对 partial/local information 假设过强。

**回答**：local interaction charts + bilinear pairings + partition of unity + overlap gluing + universal coverage + bounded finite-rank observation update。

**最值得讲的证据**：global tangent reaction-center 负结果、pocket coverage 极低、directional section consistency gap，以及 final atlas 在 broad candidate universe 中相对 universal-only 的增益。

## 二十六、两代 Agent

### Agent V1：自然语言 routing / model access

真实试用反馈促使 intent confirmation、scope switching、context continuation 和 bounded tool harness 出现。目标是让用户不必先学模型菜单。

### Agent V2：Scientific workspace

把 evidence、entity、route、model result 和后续操作变成可复用科研对象，并保持 deterministic scientific invariants 和 provenance。

---

# 第十幕：叙事时必须主动说明的边界

## 二十七、不要把不同 candidate universe 的数字混成一个“总分”

TPS practical、TPS MARTS strict、Rhea temporal、Enzyme-405、Orphan-335、CLIPZyme common support、185,918-protein broad TPS 都是不同 protocol。它们应展示 capability surface，而不是加权成一个神秘 scalar。

## 二十八、不要把旧湿实验写成最终 FIBRE atlas 的直接验证

TPS wet-lab history 是真实应用 grounding；FIBRE 是从这条历史继续抽象的最终数学模型。除非未来真的运行 FIBRE-specific prospective experiment，否则两者之间必须保留 validation boundary。

## 二十九、不要把 Human Practices 写成“企业一句话让我们换模型”

更可信的结构是：

1. 技术实验已经显示 heterogeneous information / missingness / sparse correspondence 是真实困难；
2. HP 真实用户反馈独立验证研究者确实需要更自然的入口、更可解释 evidence、更广场景和实验反馈；
3. 两条证据共同施压系统演化，但模型数学仍由技术证据负责。

## 三十、不要把失败实验藏掉

这个项目最有说服力的地方之一恰恰是：

- cycle consistency 没过就没上线；
- Horizyn loss 失败就分离变量；
- novelty routing 不稳就拒绝；
- reaction-center 有总体增益但没过 hard gate 仍拒绝；
- external transfer 失败后不回头调参；
- TPS XAttn identity bug 发现后整套 performance 作废；
- global geometry 发现假设过强后主动 supersede。

这不是“做错了很多”，而是一条可审计的 research iteration chain。

---

# 结语：一句话与一段话版本

## 一句话

**我们从 TPS 中“有限实验预算下先测谁”的真实检索问题出发，在真实用户不断扩大的科研需求中把系统从开放世界双向检索推进到广域多专家，再发现真正的统计瓶颈不是缺少单个分子的描述，而是 enzyme-reaction correspondence 稀疏、选择性且 context 不完整；最终 FIBRE 用 interaction atlas 让异质 molecular views 成为同一催化相容性问题的局部坐标，而 Starase 将模型、证据与科研操作连接为可持续工作流。**

## 一段话

项目最初面对的是一个很具体的 TPS 实验决策：候选酶很多，而真正能测的只有少数。直接结构/CAGE 全库打分和 reaction-similarity gate 暴露了 pair scorer、candidate reservoir 与 open-world retrieval 之间的鸿沟，于是 V1 建立双向、PU-aware 的 TPS open-world retrieval。HP 组随后组织真实用户试用；自然科研语言中的歧义、连续追问和更多场景推动 semantic routing 逐步发展成 Catalyst/Starase Agent。与此同时，候选宇宙扩展到广域 enzyme discovery，团队系统测试 continuation、RDKit+、EnzGFM、CLIPZyme、reaction center、few-shot context 和 learning-to-rank，形成 V2 BiME-Rank。大量正负实验又揭示：更丰富的 molecular view 往往只在局部 regime 有效，不能无条件塞入一个 global score。我们先尝试 product-manifold / sparse correspondence geometry，但 reaction-center transfer、pocket coverage 和 directional-section consistency 证明全局 factor geometry 的假设仍过强。最终 V3 FIBRE Interaction Atlas 允许不同 biochemical/information regimes 使用不同 reaction/enzyme coordinate spaces，通过 partition of unity 和 overlap consistency 共同估计 task-level catalytic compatibility，并用 bounded finite-rank operator 吸收新实验。Starase 则把这一模型与可追溯 evidence、workspace objects 和后续科研操作连接起来；Human Practices 验证真实需求，技术证据负责支撑模型选择。
