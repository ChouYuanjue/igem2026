# 萜类合酶双向开放世界检索系统技术报告

> **报告状态**：生产方案与评测实现说明<br>
> **比较对象**：仅比较原阶段文档中的 `recall_union_core + reaction similarity + learned RF rescue / CAGE` 旧方案与当前新方案<br>
> **日期**：2026-07-24
> **代码入口**：`projects/active/terpene_screening/`

---

## 0. 一页结论

本项目要解决的不是普通的“给一个反应分类到某个已知酶类别”，而是两个方向的开放世界检索：

1. 给定一条萜类反应，寻找可能催化它的酶；
2. 给定一条酶序列，寻找它可能催化的反应；
3. 查询或候选可以是数据库中从未出现过的新实体；
4. 新实体加入候选库后，不重新训练模型也能参加 Top-3、Top-10、Top-20 排名；
5. 输出不仅要有排序，还要说明该排序在严格冷启动条件下是否值得信任；
6. 排名最终需要落到可采购、可建板、可回填的湿实验流程。

原文旧方案的核心思想很直观：先用相似反应和若干规则构造一个高召回候选池，再让随机森林等模型从候选池中“救回”少量可能被主排序漏掉的酶。它对**当前数据库内部的反应补全**是有价值的，但本质上依赖已知的反应—酶关联图，且只实现了反应到酶的单向流程。

新方案把问题重写为共享向量空间中的双向检索：

- 蛋白由 ESM-C 600M 编码；
- 反应由 DRFP、前体类型和产物骨架编码；
- 双塔模型把蛋白与反应投影到同一个 256 维单位球面；
- 已知反应—酶对在空间中被拉近；
- 同一反应的多个正酶和同一酶的多个正反应同时参与多正例对比学习；
- 用 50% 序列簇和反应簇处理潜在假负例；
- 用 MARTS 外部数据做域适配；
- 不同方向、不同 Top-K 预算采用经过验证的专用路由；
- 外部酶 Top-10 使用两条互补路线的 Reciprocal Rank Fusion（RRF）；
- 三 seed 分歧和近邻度经过严格双冷校准，形成可拒答的经验可靠性层；
- UniProt 扩展不自由混入主排序，而以 0/1/2 个受控尾部插槽进入；
- 排名进一步生成六块 96 孔板、去重构建清单和结果反馈闭环。

### 0.1 最重要的比较结论

在原文的旧指标——**只按 exact reaction ID 留出、候选仍是当前 1,391 条 TPS**——下，新方案并非在所有预算上都压倒旧方案：

| 方法 | Hit@5 | Hit@10 | Hit@20 |
|---|---:|---:|---:|
| 原文旧方案最终 RF rescue | **34.50%** | 39.57% | 45.22% |
| 新方案受控 current-only 双塔 | 32.36% | **39.96%** | **46.98%** |
| 新方案减旧方案 | -2.14 pp | +0.39 pp | +1.76 pp |

因此，不能把新方案表述成“在旧任务上全面提分”。更准确的说法是：

- 旧方案在极短列表 Top-5 上仍有优势；
- Top-10 基本持平；
- 新方案在 Top-20 上略好；
- 新方案的决定性优势不在这张旧表，而在更严格的冷启动泛化、双向能力、开放世界扩展、可靠性校准和实验执行闭环。

在新的共同严格指标——**50% 蛋白序列簇 × 反应簇的 5×5 双冷**——下，旧方案必须在每个格子中重新生成 gate，不能复用全量数据库预计算的候选池和特征。最终结果为：

| 方法 | Hit@5 | Hit@10 | Hit@20 |
|---|---:|---:|---:|
| 原文旧方案，按格重建 reservoir 与 RF rescue | 0.00% | 0.56% | 1.54% |
| 新方案受控 current-only 双塔 | 4.20% | 8.96% | 16.81% |
| 新方案减旧方案 | +4.20 pp | +8.40 pp | +15.27 pp |

这张严格表才回答“目标反应簇和目标蛋白家族都没有在训练中出现时，系统还能否发现正确配对”。

---

## 1. 任务定义

### 1.1 反应到酶（Reaction → Enzyme，R2E）

输入可以是：

- 一个已有 Rhea ID；
- 一条从未注册过的 reaction SMILES；
- 可选的已知催化酶列表。

输出是候选酶排序。已知酶会被屏蔽，不会作为“新发现”返回。

典型使用场景：

- 已知某反应有一两个催化酶，希望找到更多远缘同功能酶；
- 只有目标反应，没有任何已知催化酶；
- 把新的蛋白序列临时或永久加入候选库后立即参与排名。

### 1.2 酶到反应（Enzyme → Reaction，E2R）

输入可以是：

- 一个已有或已注册的酶 ID；
- 一条全新的蛋白序列；
- 可选的已知反应列表。

输出是候选反应排序。已知反应会被屏蔽。

典型使用场景：

- 对新测序的 TPS 做功能探索；
- 已知一个酶催化某反应，继续寻找其潜在旁路或多功能反应；
- 将数据库外的新反应加入候选库后直接参与排名。

### 1.3 为什么这是检索，而不是分类

分类器通常假定标签集合固定，例如“该蛋白属于 513 个 Rhea 反应中的哪一个”。本项目不采用这个假设：

- 候选反应可以新增；
- 候选酶可以新增；
- 一个酶可以有多个反应；
- 一个反应可以有多个酶；
- 同一个模型必须支持两个方向。

因此，系统学习的是蛋白与反应之间的**可比较表示和相容性分数**，而不是固定类别的 softmax 概率。

---

## 2. 原文旧方案：最简但准确的说明

原文旧方案可以压缩成四步。

### 2.1 第一步：用相似反应构造候选酶

对目标反应 \(r_q\)，在已有反应中寻找化学结构相似的 seed reactions。反应相似度综合：

- 底物指纹相似度；
- 产物指纹相似度；
- 前体碳数类别是否一致；
- 产物骨架类别是否一致。

再把 seed reactions 已知的酶传播给目标反应。

### 2.2 第二步：构造 `recall_union_core` 高召回池

旧方案不是只依靠一种 gate，而是合并多个证据通道：

- balanced reaction similarity；
- product-oriented reaction similarity；
- precursor-compatible enzymes；
- 与 seed enzymes 的 sequence k-mer 相似度；
- motif/mechanism 规则。

合并后每条反应最多保留 300 个候选。这样避免对完整 1,391×513 矩阵做昂贵结构推理，但也意味着：**不在 reservoir 中的正例永远不可能被后续模型救回。**

### 2.3 第三步：加入 CAGE 与 learned meta-ranker

候选 pair 的特征包括：

- `reaction_similarity`；
- `sequence_kmer`；
- `motif_score`；
- precursor/product/mechanism evidence；
- evidence channel 数量；
- EnzymeCAGE pair score 及其组内排名；
- 多种交互项，如 reaction similarity × CAGE。

随机森林、HGB、ExtraTrees、逻辑回归在按 reaction ID 分组的 OOF 交叉验证中训练。

### 2.4 第四步：主排序 + 少量 rescue 插槽

旧方案最终没有让 learned model 完全替代 reaction similarity，而采用更保守的混合面板：

- Top-5：4 个 reaction-similarity 主候选 + 1 个 RF rescue；
- Top-10：5 个主候选 + 5 个 RF rescue；
- Top-20：10 个主候选 + 10 个 RF rescue。

这个设计的优点是保留化学近邻主排序，同时给 learned model 一些纠错空间。

### 2.5 旧方案做对了什么

旧方案并不是简单的答案泄漏：

- 目标 reaction ID 不作为 seed；
- canonical reaction SMILES 完全相同的反应也被排除；
- meta-ranker 使用 reaction-grouped OOF，而不是在同一目标反应的行上训练再预测。

所以，旧指标确实衡量了某种泛化能力：**给定一个未直接参与 RF 训练的 reaction ID，能否利用数据库中其他反应及其已知酶进行补全。**

### 2.6 旧方案真正的边界

它没有解决以下问题：

1. 没有隔离 50% 序列同源簇；
2. 没有隔离相似反应簇；
3. gate 特征依赖完整已知关联图；
4. reservoir 固定，未进入池的候选不可恢复；
5. 新蛋白通常要先准备结构、口袋和 CAGE pair score；
6. 没有酶到反应方向；
7. 没有经过严格双冷校准的可靠性分数；
8. 没有持久化开放世界注册、批处理和湿实验反馈系统。

---

## 3. 为什么必须同时报告“旧指标”和“新指标”

### 3.1 旧指标：exact reaction ID held-out

旧文档使用的核心评测口径是：

- 513 条当前反应；
- 1,391 条当前 TPS 候选；
- 按 reaction ID 做 5 折 GroupKFold；
- 测试 reaction ID 不进入 meta-ranker 训练；
- 但相似反应、近同源酶和完整数据库关联图仍可见。

它适合回答：

> 在当前数据库内部，遇到一个未直接参与监督训练的 reaction ID，能否借助相似反应和同源家族补全其已知酶？

它不适合回答：

> 一个全新的反应簇和全新的蛋白家族同时出现时，系统能否发现正确关联？

### 3.2 新共同指标：25-cell double-cold

新共同指标把两个轴同时隔离：

- 蛋白按 MMseqs2 50% identity 聚类；
- 反应按前体分层的 Butina 化学簇聚类；
- 各自分为 5 folds；
- 评测遍历 5×5=25 个 protein-fold × reaction-fold 单元；
- 每个已知 pair 只在它所属的格子中测试；
- 训练同时排除测试 protein fold 与 test reaction fold；
- 同一反应中属于其他蛋白 folds 的已知正例从候选排名中屏蔽，避免把已知正例当作新发现。

它回答的是更接近真实发现的问题：

> 目标蛋白家族未见、目标反应簇也未见时，模型能否把至少一个真实正例放入 Top-K？

### 3.3 旧方案在严格指标下为何必须“重新生成 gate”

这是本次对比中最重要的审计点。

若直接拿旧方案全量数据库预计算的 `gate_candidate_pools_with_evidence.csv`，只重新训练 RF，看起来严格双冷结果会非常高。但这是不合法的，因为这些行中的：

- 候选 reservoir；
- reaction-similarity 传播；
- sequence-kmer seed；
- precursor/product/mechanism evidence；
- evidence channel 数量；

都已经使用了全量 reaction→enzyme `true_map`。

因此，本报告中的旧方案严格结果采用**fold-local regeneration**：每个25格中只用训练关联重新生成 seed reactions、reservoir 和全部关联传播特征。测试反应只提供反应化学，测试蛋白只提供序列/结构。Pairwise CAGE 原始分数可保留，因为它是给定反应与蛋白结构后的独立模型输出，不直接使用本任务标签；但历史 `cage_rank_score` 依赖当时的候选集合，因此不能复用，必须在每个 fold-local reservoir 内由原始 CAGE score 重新计算排名百分位。

这一步将“模型重训”与“特征无泄漏”同时保证。

---

## 4. 新方案总体架构

新系统由六层组成：

1. 数据与实体注册层；
2. 蛋白/反应表示层；
3. 双塔对比学习层；
4. 方向和预算专用路由层；
5. 可靠性与拒答层；
6. UniProt rescue 与湿实验执行层。

### 4.1 当前生产数据规模

| 内容 | 数量 |
|---|---:|
| 当前 TPS 蛋白 | 1,391 |
| MARTS 注册外部蛋白 | 694 |
| 主候选蛋白总数 | 2,085 |
| 当前 Rhea 反应 | 513 |
| MARTS 注册外部反应 | 240 |
| 主候选反应总数 | 753 |
| current+MARTS 去重训练关联 | 3,439 |
| UniProt 受控 rescue 主层 | 5,672 |

MARTS 实体不是一次性测试文件，而是持久化开放世界 registry 的初始内容。用户新加入的实体使用同一注册格式。

---

## 5. 表示层

### 5.1 蛋白表示：ESM-C 600M

每条蛋白序列由 ESM-C 600M 编码为 1,152 维向量。对超过模型单段长度的序列采用重叠切片后聚合，最终做 L2 归一化。

直观理解：

- 传统 k-mer 只知道局部字符是否相同；
- ESM-C 表示包含更广泛的进化、结构和功能上下文；
- 它仍然不是“功能答案”，而是一个适合进一步监督对齐的蛋白表示。

### 5.2 反应表示：DRFP + 类别特征

反应表示总计 2,115 维：

- 2,048 维 DRFP；
- 10 维前体类别 one-hot；
- 57 维产物骨架类别 one-hot。

DRFP 编码反应前后结构差异；类别特征显式提供 GPP/FPP/GGPP 等前体尺度和产物碳数、环数、含氧状态等粗粒度先验。

对于无法稳定解析的极少数反应，DRFP 块设为零，但类别和注册元数据仍保留；该情况会被审计记录。

### 5.3 Horizyn exact-residual 反应表示

R2E Top-10/20 还使用一个外部预训练反应表示作为残差分支。形式为：

\[
z_r = \operatorname{norm}\left(z_{\text{DRFP}} + \sigma(g)\,z_{\text{Horizyn}}\right).
\]

其中：

- `base_reaction_tower` 编码 2,115 维主特征；
- `aux_reaction_tower` 编码 512 维 Horizyn 反应 embedding；
- 可学习 gate 初始值较小，防止外部表示一开始覆盖主表示；
- 对官方编码失败的少量反应使用打包的 distiller 回退，不把零向量经过网络偏置伪装成有效外部表示。

该分支只用于通过严格开发/冻结协议的 R2E Top-10/20，不用于 E2R。

---

## 6. 双塔模型

### 6.1 网络结构

蛋白塔与反应塔结构相同但参数不共享：

\[
z_p = \operatorname{norm}\bigl(W_{p2}\,\operatorname{GELU}(W_{p1}\,\operatorname{LN}(x_p))\bigr),
\]

\[
z_r = \operatorname{norm}\bigl(W_{r2}\,\operatorname{GELU}(W_{r1}\,\operatorname{LN}(x_r))\bigr).
\]

生产配置：

- hidden dimension：512；
- shared embedding dimension：256；
- dropout：0.1；
- 输出 L2 normalization。

pair score 是余弦相似度：

\[
s(r,p)=z_r^\top z_p.
\]

由于两个输出都归一化，点积就是 cosine similarity。

### 6.2 多正例双向对比学习

一个反应可以对应多个酶，一个酶也可以对应多个反应，因此不能把 batch 中除单个配对外的所有项都当作负例。

对每个 reaction query，损失比较：

- 所有已知正酶的 log-sum-exp；
- 所有允许候选的 log-sum-exp。

蛋白作为 query 时对称计算。总损失为：

\[
L=\lambda L_{R\rightarrow E}+(1-\lambda)L_{E\rightarrow R}.
\]

不同目标可以设置不同 \(\lambda\)：

- 共享模型采用0.5；
- R2E短名单模型采用0.75；
- R2E exact-residual也采用面向reaction-query的专用权重。

### 6.3 PU-style 假负例屏蔽

数据库中“没有标注关联”不代表真正不催化。尤其同一50%序列簇内的未标注酶，很可能只是尚未实验验证。

训练时：

- 所有已知正例永远保留；
- 与正例属于相同蛋白簇或反应簇的未标注项从 denominator 中排除；
- 其他未标注项作为弱负例参与对比学习。

这不是完整的正未标记学习理论估计器，但能显著减少明显的同源假负例冲突。

### 6.4 Hard-negative 专用模型

E2R Top-10 的副路线不再让大量显然无关的远负例主导训练，而只保留每个 query 分数最高的128个允许负例：

- 所有正例保留；
- 其余只取最难 K=128；
- 训练50 epochs，避免后期过拟合。

它单独并不稳定优于主模型，但会命中一批主模型漏掉的查询，因此作为 RRF 副路线保留。

---

## 7. MARTS 域适配与方向专用模型

### 7.1 为什么需要域适配

只在当前数据库训练的模型，对 MARTS 外部蛋白和外部反应存在明显分布偏移。当前生产基线在严格 external double-cold 中：

- E2R Hit@10：9.7%；
- R2E Hit@10：4.2%。

加入 MARTS 关联做 25-cell 域适配后，模型学到更广泛的蛋白和反应分布。

### 7.2 冻结反应塔的 E2R 专用模型

E2R 的输入是新蛋白。实验发现，只适配蛋白塔、冻结已经学习好的反应空间，更能保持反应坐标系稳定：

- reaction tower 不动；
- protein tower 适配 MARTS 外部序列；
- 用于 E2R Top-3、Top-20 和 RRF 主路线。

### 7.3 R2E 专用模型

R2E 外部反应采用不同目标：

- Top-3：reaction-loss weight 0.75 的直接双塔；
- Top-10/20：Horizyn exact-residual 直接双塔。

当前库 reaction queries 仍使用共享 PU 模型，避免外部专用模型破坏 current-database retention。

---

## 8. 近邻迁移与自动路由

双塔直接分数回答“这个蛋白和这个反应在学习空间中是否相容”。近邻迁移则加入可解释的数据库证据。

### 8.1 新蛋白的反应迁移

对外部蛋白 \(p_q\)：

1. 在当前已注释蛋白中找 ESM-C 最近邻；
2. 取前 K 个正相似邻居；
3. 收集这些邻居的已知反应；
4. 在反应 embedding 空间中，把已知反应的证据扩散到相近反应；
5. 邻居蛋白相似度作为权重。

直接分数与迁移分数先分别转为 tied-rank percentile，再加权：

\[
s_{hybrid}=w\,rank(s_{direct})+(1-w)\,rank(s_{neighbor}).
\]

使用排名百分位而非原始分数，可以避免不同模型、不同证据源量纲不一致。

### 8.2 外部酶 Top-10 的两路线 RRF

主路线：

- freeze-reaction 模型；
- 5 个蛋白邻居；
- direct weight 0.5。

副路线：

- hard-negative K=128 模型；
- 3 个蛋白邻居；
- direct weight 0.9。

最终：

\[
S_{RRF}(c)=\frac{0.35}{60+r_{primary}(c)}+
           \frac{0.65}{60+r_{secondary}(c)}.
\]

RRF 不要求两个模型的分数可校准，只依赖各自排名。参数在开发切分确定后，在两个新的 fold assignments 上锁定验证：

| 切分 | RRF Hit@10 | 原主路线 Hit@10 |
|---|---:|---:|
| 历史严格切分 | 25.37% | 19.40% |
| confirmatory 20260724 | 22.66% | 19.42% |
| locked confirmatory 20260725 | 21.91% | 17.67% |

生产中只有同时满足以下条件才自动启用 RRF：

- enzyme→reaction；
- 外部酶；
- zero-shot；
- Top-10；
- 自动路由；
- 没有手动覆盖模型。

当前库酶、few-shot、Top-3、Top-20 和手动 direct 都不会误触发。

### 8.3 最终自动路由表

| 查询 | Top-3 | Top-10 | Top-20 |
|---|---|---|---|
| 当前 reaction → enzyme | shared PU direct | shared PU direct | shared PU direct |
| 外部 reaction → enzyme | R2E 0.75 direct | exact-residual direct | exact-residual direct |
| 当前 enzyme → reaction | direct | direct | direct |
| 外部 enzyme → reaction | freeze + 5NN，w=0.75 | 双路线 RRF | freeze + 5NN，w=0.75 |
| 任意 few-shot query | supplied-seed route，并屏蔽已知关联 | 同左 | 同左 |

每行结果记录：

- `score_source`；
- `ranking_objective`；
- 主模型目录；
- RRF启用时的副模型目录；
- candidate-level ensemble disagreement；
- query-level reliability 字段。

---

## 9. 可靠性校准与拒答

### 9.1 为什么不能把 cosine score 当概率

双塔分数只表示排序相容性，不能直接解释为“有80%概率催化”。不同查询的分数分布也不同。

因此系统单独计算：

- 三 seed 对 Top-1 的投票一致率；
- Top-1 rank standard deviation；
- score standard deviation；
- Top-K 集合 Jaccard；
- Top-K candidate vote fraction；
- 排名边界 margin；
- 查询与当前训练库最近相似度。

### 9.2 校准方式

在严格 external 5×5 double-cold 查询上：

1. 为每个 query 记录上述诊断；
2. 标签为该 query 是否在对应 Top-K 命中；
3. 以 query ID 分组做 GroupKFold；
4. median imputation + standardization + class-balanced logistic regression；
5. 对 cross-validated probabilities 做 bootstrap AUC；
6. 只有95% AUC下界大于0.5才部署。

这仍然不是生化概率，而是“在历史严格冷启动查询中，这类排序特征有多像成功查询”的经验分数。

| 路由 | CV AUC | 95%下界 | 状态 |
|---|---:|---:|---|
| E2R Top-3 | 0.874 | 0.802 | 部署 |
| E2R Top-10 RRF | 0.711 | 0.632 | 部署 |
| E2R Top-20 | 0.735 | 0.666 | 部署 |
| R2E Top-3 | 0.435 | 0.270 | 不部署 |
| R2E Top-10 exact residual | 0.626 | 0.519 | 部署 |
| R2E Top-20 exact residual | 0.610 | 0.514 | 部署 |

E2R Top-10中，可靠性最高的25%查询 Hit@10为52.2%，总体为25.4%。

### 9.3 拒答策略

默认只标注风险，不阻止输出。自动化流程可选择：

- `require_calibrated`：必须存在通过双冷验证的校准器；
- `require_intermediate`：至少中等证据；
- `require_higher`：只允许高证据层。

不支持的查询不会得到伪造概率，而会明确标记“未校准/需人工复核”。

---

## 10. 开放世界注册

### 10.1 新酶

用户提供：

- `enzyme_id`；
- amino-acid sequence。

系统生成 ESM-C embedding，追加到 registry，并立即参与 R2E 候选排名或作为 E2R query。

### 10.2 新反应

用户提供：

- `reaction_id`；
- reaction SMILES。

系统按当前 feature schema 生成反应特征，并在需要时生成 exact-residual auxiliary feature，随后参与检索。

### 10.3 不重训的含义

“不重训”不是说新实体自动获得真标签，而是：

- 表示编码器能处理它；
- 双塔可计算它和已有候选的相似度；
- registry 维护稳定 row mapping；
- 排名和mask逻辑无需重建分类头。

持久化集成测试曾将一个已知正例以全新蛋白ID和反应ID注册：新酶对新反应排第3，新反应对新酶排第5；删除后 registry 恢复为694蛋白、240反应。

---

## 11. UniProt 扩展为何采用受控 rescue

### 11.1 自由合并失败

从五个 TPS 相关 Pfam 域获取并聚类后，得到5,672条具名 UniProt 代表序列。若全部自由加入主候选池，严格 external R2E：

| 候选宇宙 | Hit@3 | Hit@10 | Hit@20 |
|---|---:|---:|---:|
| canonical current+MARTS | 4.6% | 12.7% | 18.1% |
| 自由加入5,672条 UniProt | 2.5% | 6.3% | 10.5% |

原因包括：

- 大量 C/D evidence sequences 的功能标签弱；
- embedding hub 抢占多个反应前列；
- 未标注扩展序列与MARTS正例不在同一监督分布；
- 仅靠均值中心化、z-score或局部密度不能修复。

### 11.2 受控尾部插槽

因此主排序前缀不动，只给扩展层固定尾部位：

| 目标 | canonical prefix | UniProt tail | 原命中保留率 |
|---|---:|---:|---:|
| Top-3 | 3 | 0 | 100.0% |
| Top-10 | 9 | 1 | 93.3% |
| Top-20 | 18 | 2 | 97.7% |

### 11.3 reaction-specific Pfam architecture contract

仅按产物碳数判断 TPS 家族会误把：

- PF13243-only class-II；
- PF13249-only OSC fragment；
- 完整 PF13243+PF13249 OSC；
- PF00348 prenyltransferase；
- PF00494 squalene/phytoene synthase；

混在一起。

新方案从已知阳性酶的 accession、精确序列和高覆盖 MMseqs nearest match 建立 reaction→allowed architecture contract：

- 208/240注册反应得到五-Pfam支持；
- 32个不支持或无法可靠解析的反应保持 canonical-only；
- 完整OSC必须同时有PF13243+PF13249；
- 单域片段不得占用rescue位；
- raw external reaction默认不给UniProt位，除非调用方显式提供允许架构。

---

## 12. 从排名到湿实验

### 12.1 Canonical discovery panel

每条注册反应选择12个新候选：

- 6 exploitation；
- 3 uncertainty；
- 3 ESM-C sequence diversity。

已知阳性单独作为control，不占12个新发现位置。序列小于200 aa、大于1000 aa或含非标准残基的候选会自动替换。

### 12.2 UniProt rescue panel

24个受支持目标，每个4个 UniProt rescue：

- evidence anchor；
- homology-named；
- predicted-named；
- sequence-diversity。

最终候选要求完整允许架构，并排除高置信序列风险。

### 12.3 两级板位平衡

仅随机打乱孔位并不够，因为反应类型可能集中在某一块板。

第一层：MILP分板

- canonical每板6个reaction blocks；
- rescue每板12个reaction blocks；
- 平衡terpene type、TPS class、底物、阳性对照、长度、来源、证据层和Pfam architecture。

第二层：Hungarian assignment

- 在每个reaction block内平衡candidate role与位置；
- control/blank固定不动；
- role-slot normalized entropy由0.201升到0.974；
- 单一role在某孔位的最大占比由100%降到33.3%。

### 12.4 六板采购主清单

最终：

- 6×96=576 wells；
- 480 protein assay wells；
- 29个不同反应；
- 352个跨campaign精确序列去重构建体；
- 184,501 aa；
- 553,503 coding nt，不含stop codon。

采购可合并，但 canonical discovery 和 UniProt rescue 保持独立 QC scope。

### 12.5 反馈闭环

只有同时满足：

- reaction controls通过；
- candidate表达合格；
- assay未检出；

才将pair记为实验负例。

以下均保持 inconclusive：

- expression failure；
- control failure；
- untested pair。

反馈系统输出：

- confirmed positives；
- expression-qualified negatives；
- inconclusive rows；
- control reruns；
- 下一轮8候选panel。

---

## 13. 新旧方案在旧指标下的直接对比

### 13.1 公平设置

为了不把候选空间和外部数据优势混入方法比较，新方案旧指标结果采用：

- 与旧方案相同的513条current reactions；
- 相同的1,391条current TPS candidates；
- 相同的旧GroupKFold reaction→fold mapping；
- target reaction全部正例从训练关联中移除；
- current-only双塔；
- 固定单seed 20260723；
- 100 epochs；
- PU同簇假负例屏蔽。

这不是完整生产ensemble，而是新方案核心在旧任务上的受控版本。

### 13.2 结果

| 方法 | Hit@5 | Hit@10 | Hit@20 |
|---|---:|---:|---:|
| 旧方案 reaction-similarity backbone | 31.19% | 36.65% | 41.72% |
| 旧方案最终 RF rescue | **34.50%** | 39.57% | 45.22% |
| 新方案受控双塔 | 32.36% | **39.96%** | **46.98%** |

解释：

- Top-5：旧RF rescue高2.14 pp，说明手工reservoir+短名单纠错在数据库内部仍很强；
- Top-10：差0.39 pp，基本可视为同一水平；
- Top-20：新双塔高1.76 pp；
- 新双塔不依赖300候选reservoir，因此没有“未进入池就永远无法召回”的硬上限；
- 但旧指标允许近同源和相似反应跨折，因此不能用于证明开放世界冷启动。

旧方案另有50次 tune/test selection validation：Top-10平均39.81%，10th percentile 38.60%，说明其旧指标结果不是单次偶然。但这个验证仍属于相同的exact-reaction任务定义。

---

## 14. 新旧方案在严格新指标下的对比

### 14.1 公平设置

双方使用：

- current-only 1,391 proteins；
- current-only 513 reactions；
- 相同的50% protein clusters；
- 相同的reaction clusters；
- 相同的25个Cartesian cells；
- 相同的已知正例mask；
- 714个reaction query-cells。

旧方案每格重新构造所有关联传播特征；新方案每格重新训练双塔。两者都不使用MARTS外部训练关联，以隔离方法本身。

### 14.2 结果

| 方法 | Hit@5 | Hit@10 | Hit@20 |
|---|---:|---:|---:|
| 原文旧方案，fold-local gate + RF rescue | 0.00% | 0.56% | 1.54% |
| 新方案受控双塔 | 4.20% | 8.96% | 16.81% |
| 绝对变化 | +4.20 pp | +8.40 pp | +15.27 pp |

因为双方使用完全相同的714个查询单元，可以做逐查询配对：

| 预算 | 新版独占命中 | 旧版独占命中 | 双方都命中 | 增益的25-cell bootstrap 95%区间 |
|---|---:|---:|---:|---:|
| Top-5 | 30 | 0 | 0 | +2.83 至 +5.69 pp |
| Top-10 | 62 | 2 | 2 | +6.46 至 +10.33 pp |
| Top-20 | 115 | 6 | 5 | +12.78 至 +18.07 pp |

这说明严格提升不是单一大fold造成的：三个预算的配对区间都完全高于0。Top-10中有2个查询仅旧方案命中、2个双方都命中，说明旧特征仍保留少量互补性；但新版独占命中62个，净收益明显。

辅助指标：

| 方法 | MRR/可用替代 | median best positive rank | mean pool size |
|---|---:|---:|---:|
| 旧方案 | 不适用：reservoir panel无全库连续排名 | 不适用 | 205.6 |
| 新方案 | 0.0403 | 128.5 | 1,391全库 |

旧方案使用有限reservoir，因而更像“候选生成+panel选择”；新方案对全库给连续分数。两者的Hit@K可直接比较，但MRR不宜强行比较。

### 14.3 对严格结果的正确理解

严格双冷分数显著低于旧指标并不代表实现退化，而是问题变了：

- target protein family未见；
- target reaction cluster未见；
- 近同源seed被隔离；
- 相似反应关联不能通过全量图提前传播；
- 正确答案必须在完整候选空间中重新发现。

这正是实际发现任务中最困难、也最值得优化的部分。

---

## 15. 最终生产外部基准

受控current-only比较用于方法审计；实际生产还加入MARTS域适配、三seed ensemble、方向专用路由、exact residual和RRF。

严格 external double-cold生产结果：

| 方向 | Hit@3 | Hit@10 | Hit@20 |
|---|---:|---:|---:|
| 外部 enzyme → reaction | 7.84% | **25.37%** | 32.46% |
| 外部 reaction → enzyme | 4.64% | **13.50%** | 18.99% |

旧方案没有可直接对应的完整外部生产指标，因为：

- 它只实现R2E；
- 新候选蛋白通常需要结构下载、口袋检测和逐pair CAGE推理；
- reservoir依赖已知关联图；
- 没有原生的新反应/新蛋白registry；
- 没有E2R路线。

因此，本报告不把“旧方案缺失”写成0，也不拿旧exact-reaction数字与MARTS external数字做直接减法。

---

## 16. 新方案相对原文旧方案的具体改进

| 维度 | 原文旧方案 | 新方案 |
|---|---|---|
| 基本范式 | gate候选生成 + RF/CAGE rescue | 双塔共享空间 + 目标专用路由 |
| 方向 | R2E | R2E与E2R |
| 蛋白输入 | k-mer、motif、结构/CAGE | ESM-C；结构外部分支仅在验证有效时使用 |
| 反应输入 | 手工反应相似度 | DRFP+类别；R2E可加exact-residual |
| 候选范围 | 最多300个reservoir | canonical全库连续排名 |
| 多正例 | 特征表分类 | 原生multi-positive contrastive loss |
| 假负例 | 未系统处理 | 同簇PU denominator mask |
| 冷启动 | exact reaction ID | protein-cluster × reaction-cluster double-cold |
| 新酶 | 需要重新准备结构/pocket/CAGE | 序列编码后直接检索 |
| 新反应 | 可做相似反应gate，但非统一registry | SMILES编码、注册后直接检索 |
| Top-K策略 | 固定主+rescue配额 | objective-specific模型与路由 |
| 模型互补 | RF rescue slots | 两路RRF，经两次确认切分 |
| 可靠性 | 无严格校准 | grouped calibration + bootstrap gate + abstention |
| 大规模扩展 | 无独立压力测试 | UniProt自由合并压力测试后采用0/1/2受控尾部位 |
| 家族约束 | 粗粒度规则 | reaction-specific Pfam architecture contract |
| 实验执行 | 候选表 | 六板、去重FASTA、板间/板内平衡、反馈闭环 |

---

## 17. 没有进入生产的激进路线

为了避免“试过就算改进”，所有方法都必须在严格双冷和开发/冻结协议下胜出。以下路线未进入生产：

- 原样Horizyn global MLNCE；
- 冻结Horizyn反应编码器的ESM-C适配器；
- ESM-C→ProtT5 bridge；
- DRFP+Horizyn简单拼接；
- motif-context 5,774维蛋白表示；
- Pfam架构分类软重排；
- 多跳图传播；
- LambdaRank candidate stacking；
- 单独hard-negative模型替换；
- hard-negative curriculum；
- 多种candidate hub normalization。

保留一个方法不取决于它在某一张全体表上偶然高，而取决于：

1. 开发格先选中；
2. 冻结格不退化；
3. 必要时新fold assignment确认；
4. 能嵌入生产路由且有明确适用边界。

RRF是少数跨历史切分和两个confirmatory切分都保持正向的方法。

---

## 18. 代码结构与复现

### 18.1 关键实现

| 文件 | 作用 |
|---|---|
| `train_dual_tower_cold.py` | 双塔、multi-positive loss、PU mask、hard negatives、top-K surrogate |
| `train_marts_domain_adaptation.py` | 25-cell MARTS域适配与方向消融 |
| `train_marts_adapted_production.py` | current+MARTS全量生产训练 |
| `rank_open_world.py` | 单查询、自动路由、RRF、可靠性、开放世界编码 |
| `rank_registry_batch.py` | 694蛋白×240反应批处理与泄漏审计 |
| `evaluate_open_world_uncertainty.py` | grouped reliability calibration |
| `manage_open_world_registry.py` | 新蛋白/反应持久注册 |
| `prepare_uniprot_tps_expansion.py` | UniProt扩展、去重与50%聚类 |
| `rank_uniprot_rescue.py` | canonical prefix + controlled tail |
| `build_reaction_architecture_contracts.py` | reaction-specific Pfam contract |
| `build_wetlab_discovery_panels.py` | 12候选panel |
| `balance_wetlab_reactions_across_plates.py` | MILP板间平衡 |
| `randomize_wetlab_candidate_positions.py` | Hungarian板内角色平衡 |
| `manage_wetlab_feedback.py` | QC、标签分流和下一轮panel |

### 18.2 本报告新增的可比性评测

| 文件 | 作用 |
|---|---|
| `evaluate_dual_tower_protocol_comparison.py` | 新方案在旧exact-reaction和共同25-cell上的受控重训 |
| `comparison_assets/legacy_exact_reaction_folds.csv` | 从原文GroupKFold固化的513反应fold映射 |
| `evaluate_legacy_cage_double_cold.py` | 旧方案在每格用train-only关联重建gate与RF rescue |

旧指标新方案复现：

```bash
.venv/bin/python projects/active/terpene_screening/evaluate_dual_tower_protocol_comparison.py \
  --protocols legacy_exact \
  --seeds 20260723 --epochs 100
```

双方共同严格复现：

```bash
# 新方案
.venv/bin/python projects/active/terpene_screening/evaluate_dual_tower_protocol_comparison.py \
  --protocols double_cold_25cell \
  --seeds 20260723 --epochs 100

# 旧方案；要求原始pairwise CAGE分数可用
.venv/bin/python projects/active/terpene_screening/evaluate_legacy_cage_double_cold.py \
  --n-estimators 250 --cells all
```

严格旧方案脚本只复用历史 raw pairwise CAGE score；候选 reservoir、所有关联传播特征以及 CAGE rank percentile 均在每个双冷单元内重新生成。

---

## 19. 限制与下一步

1. **严格双冷绝对召回仍不高。** 特别是R2E Top-3尚未得到可靠校准器。
2. **MARTS标签仍不完整。** 未标注候选可能是真正正例，指标对false negative敏感。
3. **RRF主要改善E2R Top-10。** 它不是统一提升所有预算的万能融合。
4. **Horizyn exact-residual有许可证边界。** 外部预训练资产必须按其学术/非商业条款使用，不能默认作为无约束商业部署资产。
5. **UniProt扩展尚未有本项目湿实验标签。** 受控插槽是风险控制，不是证明扩展候选一定有效。
6. **可靠性分数不是催化概率。** 它只能用于排序风险分层和自动拒答。
7. **表达宿主与载体尚未冻结。** 因此主构建清单未做宿主特异密码子优化。
8. **旧方案严格重建依赖pairwise CAGE历史分数。** 若更换结构模型或口袋方法，应重新计算，而不能把历史CAGE分数视为永久真值。

推荐的下一步不是继续无边界搜索模型，而是：

- 完成第一轮canonical和UniProt rescue湿实验；
- 将严格QC通过的正例/负例纳入PU-aware再训练；
- 分析RRF新增命中是否集中于特定TPS class；
- 针对R2E Top-3构建独立的反应化学预训练或listwise目标；
- 用新增实验标签重新校准可靠性，而不是继续只依赖数据库注释。

---

## 20. 最终判断

原文旧方案在它擅长的任务——当前数据库内部、相似反应和近同源家族可见的R2E补全——依然是一个合理且有效的工程方案。新方案不应通过贬低旧方案或混用不同指标来证明价值。

新方案的真正进步是：

- 把单向候选生成改造成双向统一检索；
- 把固定数据库改造成无需重训的开放世界registry；
- 把exact reaction held-out升级为protein/reaction双冷；
- 把全量关联图预计算改为严格可审计的fold-local评测；
- 把单一分数改成目标专用路由和经确认的RRF；
- 把“模型自信”改成可验证、可拒答的经验可靠性；
- 把自由扩库改成压力测试后的受控rescue；
- 把候选CSV推进到构建去重、六板平衡和反馈再训练闭环。

因此，对新版最准确的描述不是“旧指标全面更高”，而是：

> **在保留旧数据库补全能力的同时，新方案把问题扩展成了可严格评测、可双向查询、可开放扩展、可风险控制并可直接进入实验执行的检索系统。**
