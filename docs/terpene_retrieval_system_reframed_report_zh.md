# 萜类合酶双向检索与候选推荐系统技术报告

**副标题：从数据库补全、同源扩展到远缘发现与开放世界检索的统一实现**

- 文档性质：全新撰写的独立技术报告
- 项目位置：`/home/s241850073/igem2026`
- 核心代码：`projects/active/terpene_screening/`
- 数据与结果：`data/`、`results/`
- 文档日期：2026-07-24

> 本报告最重要的结论不是“哪一个模型分数最高”，而是：**萜类合酶筛选并不是一个单一任务。** 已知反应的数据库补全、已有阳性酶后的同源扩展、新反应映射、远缘酶发现、新蛋白注释和双侧完全未见的开放外推，允许使用的信息不同，也必须使用不同的评测协议。相似蛋白和相似反应在实际筛选中通常是合法且有价值的证据；double-cold 只是最困难的压力测试，不能替代其他生产场景。

---

## 一、执行摘要

旧版方案以候选门控、反应相似度、CAGE 结构分数和 RF/HGB 元排序为主。它能够较好地完成当前数据库内部的反应补全，也能在已经给出阳性酶时利用序列相似性找到大量近同源候选。旧方案真正的问题不是“使用了同源信息”，而是没有把不同任务分开报告，容易把数据库内补全成绩解释成跨反应簇、跨蛋白簇的开放发现能力。

新版系统没有把旧能力全部推翻，而是把项目改造成一个**场景感知、双向、可扩展、可校准、可进入湿实验闭环的检索系统**。它现在支持：

1. **Reaction → Enzyme（R2E）**：给定目标反应，排序候选酶；
2. **Enzyme → Reaction（E2R）**：给定酶序列，排序候选反应；
3. **Zero-shot**：查询时没有已知阳性 seed；
4. **Few-shot**：已有 1–5 个阳性酶后继续扩展；
5. **外部实体临时注册**：新酶和新反应无需重新训练即可进入索引和排序；
6. **不同新颖性层级的独立评测**：exact、reaction-cold、protein-cold、double-cold；
7. **受控候选库扩展**：current、MARTS 和 UniProt rescue 分层使用；
8. **目标分预算路由**：Top-3、Top-10、Top-20 使用不同的融合策略；
9. **可靠性分层**：估计一次排序在对应评测协议下是否更可能命中，而不是伪装成生化活性概率；
10. **湿实验执行**：候选配额、阳性/阴性对照、板间平衡、孔位随机化和反馈闭环。

当前 canonical 系统包含：

| 项目 | 数量 |
|---|---:|
| 当前库蛋白 | 1,391 |
| 当前库反应 | 513 |
| 当前库正关联 | 1,640 |
| 加入 MARTS 后的 canonical 蛋白 | 2,085 |
| 加入 MARTS 后的 canonical 反应 | 753 |
| canonical 训练关联 | 3,439 |
| 外部注册酶 | 694 |
| 外部注册反应 | 240 |
| UniProt 受控 rescue 候选 | 5,672 |

系统没有一个可以覆盖所有用途的“总准确率”。当前最有代表性的能力谱为：

- 当前数据库 exact-reaction 补全：Top-10 **48.1%**，Top-20 **57.5%**；
- exact 新蛋白、同簇同源物允许可见的 E2R：Top-10 **72.4%**；实际有同簇训练同源物时为 **82.6%**；
- 已知反应寻找 exact 新蛋白的 R2E：Top-10 **51.2%**；
- 已知 1 个阳性 seed 的同源扩展：Top-10 **73.7%**；
- 已知 5 个阳性 seed 的同源扩展：Top-10 **92.8%**；
- 只隔离反应簇、允许复用蛋白空间的 R2E：Top-10 **28.7%**；
- 只隔离蛋白簇、反应空间可见的 E2R：Top-10 **36.1%**；
- 双侧同时未见时，新版仍显著优于旧 RF/CAGE 方案，但绝对指标明显更低；
- 外部 MARTS few-shot：1–3 seeds 的 Top-10 为 **50.9%–60.4%**；
- 当前外部严格路由中，E2R Top-10 为 **25.4%**，E2R Top-20 为 **39.2%**；独立锁定切分中，Top-20 双核融合达到 **43.4%**，相对原生产路线提升 **8.60 个百分点**。

这些数字回答的是不同问题，不能直接排成一条“从差到好”的模型排行榜。

---

## 二、项目到底要解决什么问题

### 2.1 反应找酶：R2E

输入一个目标反应，输出最值得验证的候选酶。实际又分为两类：

- **已有阳性酶**：可以合法利用同源、近同源和家族信息，目标通常是快速获得更多可用酶；
- **没有阳性酶**：需要依靠反应表示、相似化学空间、蛋白表示和训练关联进行 zero-shot 排序。

### 2.2 酶找反应：E2R

输入一个酶序列，输出其最可能催化的反应。它可用于：

- 给外部新酶做功能注释；
- 发现已知酶的潜在多功能性；
- 为湿实验选择底物或产物检测集合；
- 将新酶加入注册表后自动参与后续检索。

### 2.3 “Zero-shot”不等于“Double-cold”

这两个概念必须严格区分：

- zero-shot 只表示查询时没有 seed；
- reaction-cold 表示目标反应簇未见；
- protein-cold 表示正确蛋白簇未见；
- double-cold 表示两侧簇同时未见。

一个查询完全可以是 zero-shot，但仍允许使用相似反应或已知蛋白家族。把所有 zero-shot 都当作 double-cold，会错误地删除实际生产中允许使用的信息。

### 2.4 新酶和新反应必须能够临时加入

最终系统不是固定的 513 类 Rhea 分类器，也不是固定的 1,391 类酶分类器。新实体通过独立编码器加入索引：

```text
encode_enzyme(sequence) -> enzyme_vector
encode_reaction(reaction_smiles, metadata) -> reaction_vector
add_enzyme(temp_id, enzyme_vector)
add_reaction(temp_id, reaction_vector)
rank_enzymes(reaction_vector, known_enzyme_ids=[], top_k=20)
rank_reactions(enzyme_vector, known_reaction_ids=[], top_k=20)
```

因此，库外酶和库外反应可以在不重新训练的情况下参与 Top-3、Top-10 和 Top-20 排序。

---

## 三、旧方案：它做对了什么，又缺少什么

旧方案可概括为四层：

1. **Gate matrix**：利用底物、产物、反应类别和已知关系缩小候选池；
2. **Reaction similarity backbone**：从相似反应迁移已知候选酶；
3. **CAGE/结构证据**：给蛋白—反应 pair 提供通用模型分数；
4. **RF/HGB meta-ranker + rescue slots**：融合规则分数、相似度和 CAGE 信号。

### 3.1 旧方案的合理部分

旧方案并不是整体错误：

- 相似反应对数据库内部补全非常有用；
- 已知阳性附近的序列同源扩展是实际实验中成功率最高、成本最低的路线之一；
- gate 能快速缩小候选集合，便于早期工程验证；
- rescue slots 能在规则主列表外保留少量探索候选；
- 旧方案对当前库 exact-reaction holdout 的 Top-K 表现确实较好。

因此，新版没有禁止 reaction similarity 或 homolog similarity，而是把它们放回正确场景：**exploitation，高成功率补全与同源扩展。**

### 3.2 旧方案的主要问题

旧报告最重要的问题是**任务边界不清**：

- exact reaction ID 被留出，但相似反应簇仍可见；
- 候选蛋白及其在其他反应中的关系可能仍出现在训练中；
- 因而旧指标适合解释为数据库补全，不能解释为新反应簇 × 新蛋白簇的开放发现；
- 未同时报告 reaction-cold、protein-cold 和 double-cold；
- gate、融合权重、rescue 配额和模型可能在同一批 OOF 结果上多次选择；
- 旧文档中的部分 tune/test 过程缺少当前服务器上完整固化的脚本和结果；
- CAGE sigmoid 概率严重饱和，大量候选完全同分或近似同分；
- 排名中的并列有时由 ID 顺序打破，而不是模型证据；
- 部分 few-shot 随机种子曾使用 Python 内置 `hash()`，跨进程不可稳定复现。

### 3.3 CAGE 为什么被降级

在当前 TPS 数据上，通用 CAGE 模型出现显著退化：大量分数接近 0，反应内部差异极小，部分反应所有候选完全同分。其问题不是结构信息无价值，而是通用模型的概率刻度无法直接作为 TPS 精确产物选择的主排序真值。

新版保留了：

- 原始 logit；
- 反应内 percentile；
- rank z-score；
- tie group size；

但 CAGE 不再进入生产主排序，仅在少量冲突候选或结构救援中作为辅助证据。

---

## 四、新版总体架构

新版不是一个单一神经网络，而是一个分层检索系统：

```text
输入查询
  ├─ 场景识别：R2E / E2R，zero-shot / few-shot
  ├─ 新颖性与信息边界：exact / reaction-cold / protein-cold / double-cold
  ├─ 实体编码：ESM-C 蛋白表示 + DRFP/多视图反应表示
  ├─ 主检索：多正例双塔
  ├─ 场景专用证据：seed similarity / reaction similarity / neighbor transfer
  ├─ 预算专用融合：Top-3 / Top-10 / Top-20
  ├─ 可靠性标注
  ├─ 候选库策略：canonical prefix + controlled rescue
  └─ 湿实验面板：exploitation + uncertainty + diversity
```

### 4.1 蛋白表示

主蛋白表示为 ESM-C 600M 全序列 mean embedding，维度为 1,152。它承担：

- 当前库和 MARTS 蛋白的统一编码；
- 外部新酶的零重训练注册；
- seed 相似度与近邻迁移；
- 双塔蛋白输入；
- 三 seed ensemble 的一致性和不确定性特征。

新版同时提取过 motif-context、局部 pocket 和结构域相关表示，但这些路线在冻结评测中没有稳定超过全序列 ESM-C，因此没有进入主生产路由。

### 4.2 反应表示

反应输入不是单一类别 ID，而是组合表示：

- DRFP 反应差分指纹；
- 前体类别；
- 产物或骨架类别；
- 方向和基础反应元信息；
- multiview 模式下的多个反应视图。

当前生产模型中：

- `drfp_categorical` 反应输入维度为 2,115；
- `multiview` 反应输入维度为 8,270；
- 蛋白塔和反应塔最终映射到 256 维共享空间。

### 4.3 多正例双塔

一个反应可以对应多个酶，一个酶也可能对应多个反应，因此不能把任务简化成单标签分类。双塔学习：

\[
s(e,r)=\frac{f_e(e)^\top f_r(r)}{\tau}
\]

训练时保留全部已知正关联，并使用多正例对比目标，使同一反应的多个阳性酶或同一酶的多个阳性反应可以同时被拉近。

### 4.4 PU cluster mask

TPS 数据高度不完整，未标注 pair 不能等同于真负例。系统采用 positive–unlabeled 思路：

- 同一 50% 蛋白簇中的未标注 pair 不轻易作为负例；
- 同一反应簇中的未标注 pair 也从部分对比学习分母中屏蔽；
- 减少“潜在同功能同源物被当成负样本”的伤害。

### 4.5 MARTS 域适配与 ensemble

当前库模型进一步在 current + MARTS 数据上适配。三个随机种子共同训练并组成 ensemble：

- seeds：20260723、20260724、20260725；
- canonical 蛋白：2,085；
- canonical 反应：753；
- 训练关联：3,439。

ensemble 不仅提高稳定性，也提供：

- 三模型候选排名标准差；
- Top-K 投票比例；
- Top-K Jaccard；
- 分数 margin；
- 查询到训练库的最近邻相似度。

这些特征用于可靠性分层。

---

## 五、评测框架：不是“宽松 vs 严格”，而是三条独立轴

### 5.1 Seed 轴

- zero-shot：没有阳性 seed；
- few-shot：给出 1–5 个阳性酶；
- seed 与 hidden positives 是否同簇，是另一个独立条件。

### 5.2 反应新颖性轴

- exact reaction ID 留出，但相似反应簇可见；
- 整个反应簇未见；
- 外部反应临时注册。

### 5.3 蛋白新颖性轴

- 正确蛋白簇在训练中可见，允许同源；
- 正确 50% identity cluster 完全未见；
- 外部酶临时注册。

### 5.4 无 seed 的二维任务矩阵

| 反应侧 | 蛋白同源空间可用 | 正确蛋白簇不可用 |
|---|---|---|
| exact ID 留出、相似反应可见 | Current-library exact completion | Protein-cluster-cold R2E |
| 整个反应簇未见 | Reaction-cluster-cold R2E | Double-cold R2E |

E2R 可对称理解：

- 新蛋白注释到已知反应目录：protein-cold E2R；
- 已知/相似蛋白寻找新反应簇：reaction-cold E2R；
- 两侧均未见：double-cold E2R。

### 5.5 为什么不能只报告 double-cold

double-cold 会主动删除两类最强证据：相似反应和相似蛋白。它适合回答：

> 当新反应家族和新蛋白家族同时出现，系统能否进行完整外推？

但它不适合回答：

- 已知阳性酶后能否快速找到替代同源酶；
- 当前数据库中一个反应能否补全候选；
- 新反应能否映射到已有蛋白家族；
- 新蛋白能否注释到已有反应目录。

生产和论文都应采用多轨报告，而不是把最难的一格冒充整个系统。

---

## 六、同一模型下，不同隔离条件造成的真实差异

为避免把模型变化和协议变化混在一起，下面固定为同一个 current-only multiview dual tower、同一批 1,391 条蛋白和 513 个反应，只改变 split。

| 协议 | 方向 | 查询数 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| Reaction-cluster-cold | R2E | 513 | 15.4% | 20.1% | **28.7%** | **36.3%** | 0.154 |
| Reaction-cluster-cold | E2R | 1,320 | 14.1% | 20.5% | **31.3%** | **40.1%** | 0.128 |
| Protein-cluster-cold | R2E | 714 | 9.0% | 12.0% | **18.1%** | **24.5%** | 0.095 |
| Protein-cluster-cold | E2R | 917 | 24.0% | 29.1% | **36.1%** | **48.0%** | 0.225 |
| Double-cold | R2E | 141 | 2.1% | 3.5% | **6.4%** | **12.8%** | 0.029 |
| Double-cold | E2R | 269 | 2.2% | 5.2% | **14.1%** | **30.1%** | 0.050 |

这张表给出三个重要结论：

1. R2E 在只隔离反应簇、允许复用蛋白空间时，Top-10 是 28.7%，而不是 double-cold 的 6.4%；
2. 给新蛋白注释到已知反应目录时，protein-cold E2R Top-10 是 36.1%；
3. double-cold 的低分主要来自两侧强证据同时被移除，不能被解释成系统日常筛选能力。

---

## 七、旧方案与新版的公平比较

### 7.1 在旧方案原本擅长的 exact-reaction 协议下

| 方法 | Hit@5 | Hit@10 | Hit@20 |
|---|---:|---:|---:|
| 旧 reaction similarity | 31.2% | 36.6% | 41.7% |
| 旧 RF/CAGE rescue | **34.5%** | 39.6% | 45.2% |
| 新 controlled dual tower | 32.4% | **40.0%** | **47.0%** |

结论应当冷静表述：

- 新版没有牺牲旧数据库补全能力；
- Top-10 基本持平，Top-20 小幅提高；
- Top-5 仍略低于旧 RF/CAGE rescue；
- 因此新版的主要价值不是在旧指标上“碾压”，而是补齐双向检索、开放注册、冷启动评测和生产闭环。

当前最佳的嵌套 current-library expert + dual-tower 融合，在 513 个反应上的正式数据库补全指标为：

| 预算 | Hit@K | 平均进入预算的已知正例数 |
|---|---:|---:|
| Top-3 | 31.8% | 0.38 |
| Top-5 | 38.6% | 0.51 |
| Top-10 | **48.1%** | 0.77 |
| Top-20 | **57.5%** | 1.07 |

这条路线允许相似反应簇和同源蛋白作为合法证据，测的是数据库补全，不是开放外推。

### 7.2 在共同 current-only double-cold 协议下

| 方法 | Hit@5 | Hit@10 | Hit@20 |
|---|---:|---:|---:|
| 旧 fold-local RF rescue | 0.0% | 0.6% | 1.5% |
| 新 controlled dual tower | **4.2%** | **9.0%** | **16.8%** |
| 绝对提升 | +4.20 pp | +8.40 pp | +15.27 pp |

配对 bootstrap：

- Top-10 提升 95% CI：+6.46 至 +10.33 pp；
- Top-20 提升 95% CI：+12.24 至 +18.40 pp。

这说明旧 RF/CAGE 路线主要依赖数据库内迁移，在两侧邻域都被切断时基本失效；新双塔虽然绝对分数仍不高，但具备显著更强的外推能力。

### 7.3 为什么旧指标和新指标都要保留

两组对比不能互相替代：

- exact-reaction 说明系统能否利用现有知识高效补全；
- double-cold 说明同源和相似化学证据都不可用时，模型是否仍有后备能力。

真正的生产系统应同时追求：

- **高成功率 exploitation**；
- **远缘与新机制 exploration**。

---

## 八、Few-shot：70%–90% 的同源扩展为什么仍然成立

### 8.1 允许同簇的实际同源扩展

| Seed 数 | ESM-C max Top-10 | 3-mer max Top-10 |
|---:|---:|---:|
| 1 | 71.9% | **73.7%** |
| 2 | 80.2% | **82.8%** |
| 3 | 84.3% | **87.1%** |
| 5 | 91.6% | **92.8%** |

这类评测中，seed 和 hidden positives 可以属于同一 50% identity cluster。它回答：

> 已经知道一个催化该反应的酶，能否找到它周围更多可替代、可表达或可测试的同源酶？

这是实际筛选中的正式能力，不是“虚高分数”。

### 8.2 强制跨 50% 蛋白簇的远缘扩展

| Seed 数 | ESM-C centroid Top-10 |
|---:|---:|
| 1 | 27.6% |
| 2 | 25.5% |
| 3 | 29.6% |
| 5 | 27.0% |

它回答另一个问题：

> 当 seed 所在近同源家族被排除后，能否发现序列差异明显但催化相同反应的远缘酶？

数值降低并不代表模型退化，而是任务从“找亲戚”变成了“找远缘同功能者”。

### 8.3 外部 MARTS few-shot

| Seed 数 | 不同外部反应 | Hit@3 | Hit@10 | Hit@20 |
|---:|---:|---:|---:|---:|
| 1 | 48 | 38.9% | **50.9%** | 54.3% |
| 2 | 26 | 46.0% | **60.4%** | 66.9% |
| 3 | 21 | 41.2% | **59.3%** | 70.7% |

它比 current random-positive 更开放，但没有强制所有 hidden positives 与 seed 跨簇，因此应独立命名为 external few-shot。

### 8.4 生产建议

当目标是尽快获得阳性：

- 优先选择同源和近同源候选；
- 再加入模型支持但序列更远的候选；
- 保留少量跨架构探索位。

当目标是发现新家族：

- 提高 protein-cluster-diverse 配额；
- 不用同源扩展指标证明远缘发现；
- 单独统计跨簇阳性率。

---

## 九、当前生产路由

不同预算的目标不同，因此系统不使用一个固定权重覆盖 Top-3、Top-10 和 Top-20。

| 方向 | 预算 | 生产路线 | 对应严格外部指标 |
|---|---:|---|---:|
| R2E | Top-3 | reaction-loss 0.75 direct | 4.6% |
| R2E | Top-10 | Horizyn exact-residual direct | 13.5% |
| R2E | Top-20 | Horizyn exact-residual direct | 19.0% |
| E2R | Top-3 | freeze-reaction + 5-neighbor，direct 0.75 | 7.8% |
| E2R | Top-10 | 双神经路线 RRF | 25.4% |
| E2R | Top-20 | 神经主路线 + 双核协同 RRF | 39.2% |

这些数字来自严格外部双冷评测，用于描述整个同源蛋白簇和反应簇均不可用时的后备能力；它们不应覆盖 exact-new entity、exact completion 和 few-shot 的更高实用指标。

### 9.1 E2R Top-10 双神经路线 RRF

两条路线分别是：

- primary：freeze-reaction，5 个蛋白邻居，direct weight 0.5；
- secondary：hard-negative K=128，3 个蛋白邻居，direct weight 0.9。

使用 Reciprocal Rank Fusion：

\[
S(x)=\frac{0.35}{60+rank_{primary}(x)}+
\frac{0.65}{60+rank_{secondary}(x)}.
\]

RRF 只依赖排序名次，不直接混合尺度不同的 cosine 或概率，因此比固定分数相加更稳定。

### 9.2 E2R Top-20 双核协同

双核分数同时利用：

- 查询蛋白与训练蛋白的 ESM-C 相似性；
- 候选反应与训练反应的化学相似性；
- 训练反应—蛋白关联图。

设训练关联矩阵为 \(A\)，反应核为 \(K_R\)，蛋白核为 \(K_P\)：

\[
S_{DK}(r,p_q)=\sum_{(r_i,p_j)\in A_{train}}
K_R(r,r_i)\,\tilde A_{ij}\,K_P(p_j,p_q).
\]

直观解释是：只有当候选反应接近某个训练反应，同时查询蛋白也接近该训练反应的已知酶时，该训练关联才提供强支持。

生产中再与神经主路线做 RRF：

\[
S_{final}(x)=\frac{0.70}{60+rank_{neural}(x)}+
\frac{0.30}{60+rank_{dual-kernel}(x)}.
\]

### 9.3 为什么双核只用于 Top-20

双核是覆盖型证据，擅长把神经模型遗漏但同时具有化学邻域和序列邻域支持的候选拉进较深列表。它并不稳定提高 Top-3，因此不能替代早期精排路线。

### 9.4 独立确认结果

| 阶段 | Query-cells | 融合 Hit@20 | 原生产 Hit@20 | 差值 |
|---|---:|---:|---:|---:|
| 原冻结 16 cells | 153 | 34.0% | 28.8% | +5.23 pp |
| 独立锁定 fold seed 20260726 | 279 | **43.4%** | 34.8% | **+8.60 pp** |

独立确认中：

- 新增命中 27；
- 丢失命中 3；
- 净增加 24 个 query-cell；
- 配对 bootstrap 95% CI：+5.02 至 +12.54 pp；
- MRR：0.0764 → 0.0874。

这条路线经过开发、冻结和独立锁定确认后才进入生产，而不是根据单次高分直接部署。

---

## 十、可靠性校准

系统的可靠性分数不是酶活概率。它估计的是：

> 当前查询的排序特征是否类似于对应冷启动验证中更容易命中的查询？

当前主要部署校准器：

| 校准器 | 部署 | ROC-AUC | 95% CI | AP | Base hit | Brier |
|---|---:|---:|---:|---:|---:|---:|
| E2R Top-3 | 是 | 0.874 | [0.802, 0.936] | 0.344 | 7.8% | 0.144 |
| E2R Top-10 | 是 | 0.711 | [0.632, 0.787] | 0.545 | 25.4% | 0.203 |
| E2R Top-20 | 是 | 0.695 | [0.627, 0.758] | 0.612 | 39.2% | 0.219 |
| R2E Top-10 | 是 | 0.626 | [0.519, 0.731] | 0.185 | 13.5% | 0.234 |
| R2E Top-20 | 是 | 0.610 | [0.514, 0.704] | 0.338 | 19.0% | 0.243 |
| R2E Top-3 | 否 | 0.435 | [0.270, 0.622] | 0.049 | 4.6% | 0.242 |

只有 bootstrap ROC-AUC 下界超过 0.5 的校准器才允许部署。R2E Top-3 因未达到这一标准，明确标记为不可校准。

可靠性特征包括：

- 三 seed 候选分歧；
- Top-K Jaccard；
- 排名标准差；
- Top-1 与后续候选 margin；
- 查询到训练库的最近蛋白或反应相似度；
- 路由间一致性。

校准器必须与具体协议绑定。double-cold 校准器不能被解释为同源扩展场景的实际阳性概率。

---

## 十一、候选库扩展与 UniProt rescue

### 11.1 UniProt 扩展规模

基于五个 TPS 相关 Pfam 域构建了候选宇宙：

| 阶段 | 数量 |
|---|---:|
| UniProt 原始命中 | 46,064 |
| 有效非片段标准序列 | 45,780 |
| 精确去重 | 44,961 |
| 排除现有 ID/序列后 | 43,812 |
| 50% identity 聚类代表 | 6,494 |
| 具名主扩展层 | 5,672 |
| domain-only rescue | 822 |

### 11.2 为什么不能自由并入主排名

自由加入 5,672 条未标注 UniProt 候选后，严格双冷真实正例会被大量挤出：

- Top-3 原命中损失约 45.5%；
- Top-10 原命中损失约 50.0%；
- Top-20 原命中损失约 41.9%；
- 真实正例中位排名从约 149–158 恶化到 761–815；
- 扩展候选中出现明显 universal hub。

因此，当前策略是：

- canonical 主候选池保持 current + MARTS；
- UniProt 作为单独 rescue 层；
- 使用反应条件化 Pfam architecture contract；
- 对高频 hub 施加约束；
- 为每个反应只开放少量受控尾部候选；
- 在获得湿实验标签前不把 UniProt 分数与 canonical 分数视为同等置信。

### 11.3 证据分层

UniProt rescue 使用 A–D 证据分层：

- A：reviewed；
- B：实验或转录证据且具名；
- C：同源推断且具名；
- D：预测但具名。

最终 rescue 两块板中的 96 个候选包含：A=2、B=28、C=40、D=26，高置信序列风险为 0。

---

## 十二、湿实验候选与六块板执行方案

### 12.1 候选分配原则

每个反应的 discovery panel 不是简单取模型 Top-12，而是：

| 类型 | 数量 | 目的 |
|---|---:|---|
| exploitation | 6 | 最大化首轮阳性概率 |
| uncertainty | 3 | 选择模型分歧较大、最能改进边界的候选 |
| diversity | 3 | 覆盖不同序列簇、架构和候选来源 |

这种设计同时服务于“尽快获得阳性”和“获得下一轮训练信息”。

### 12.2 当前正式规模

| 项目 | 数量 |
|---|---:|
| 板数 | 6 |
| 总孔数 | 576 |
| 蛋白 assay wells | 480 |
| 不同反应 | 39 |
| canonical discovery wells | 384 |
| UniProt rescue wells | 192 |
| 候选 ID 构建 | 353 |
| exact-sequence 去重构建 | 352 |
| 总氨基酸 | 184,501 |
| coding nt（不含 stop） | 553,503 |

canonical 四块板包含：

- 288 个发现候选孔；
- 48 个阳性对照孔；
- 24 个空载体阴性孔；
- 24 个底物/流程空白孔。

### 12.3 板间 MILP 平衡

完整反应块使用容量约束 MILP 分配到板上，平衡：

- 萜类类型；
- TPS class；
- 底物；
- 阳性对照；
- 候选长度；
- 外部候选比例；
- A/B/C/D 证据层；
- Pfam architecture。

主要改善：

- canonical 候选 median-length mean 板间差：146.750 → 6.833 aa；
- canonical q90-length mean 板间差：221.650 → 30.533 aa；
- rescue median-length mean 板间差：100.125 → 4.875 aa；
- rescue q90-length mean 板间差：107.483 → 0.750 aa；
- rescue 的 A/B/C/D 和三类主要 Pfam 架构板间差均降到 0。

### 12.4 孔位 Hungarian 随机化

在不移动阳性对照、空载体和流程空白的条件下，对候选角色和孔位做 Hungarian assignment：

- 平均归一化孔位熵：0.201 → 0.974；
- 最大单孔位角色占比：100.0% → 33.3%；
- 最大角色孔位计数差：24 → 1；
- 候选 ID 全部保留；
- 对照和空白移动数：0。

### 12.5 负例反馈规则

以下情况不能直接作为模型负例：

- 表达失败；
- 蛋白不溶；
- 阳性对照失败；
- 底物或流程 QC 失败；
- 未测试的 pair；
- 检测限不足。

只有在表达、对照和检测流程均合格时，未检出目标产物的 pair 才可以作为较可信负反馈。旁产物和替代产物必须单独记录，因为它们可能表示酶活存在但反应标签不完整。

---

## 十三、哪些激进路线被尝试但没有进入生产

项目保留失败实验，避免只展示成功结果。

### 13.1 Motif-context

对 DDxxD、NSE/DTE、DxDD、QW 等局部区域提取 5,774 维表示。它能解释部分催化机制，但冻结评测中不能承担全库召回；局部 motif 相似并不足以决定精确产物骨架。

### 13.2 P2Rank / pocket-local

局部口袋模型在冻结集上的 Top-10/20 很低。可能原因包括结构覆盖不完整、口袋预测误差和缺少配体姿态监督。

### 13.3 结构化难负例

同前体、不同骨架的难负例在开发集有小幅改善，但冻结集不稳定。说明负例定义方向合理，现有反应骨架标签和数据量仍不足。

### 13.4 粗骨架、Morgan 簇、碳连接图和两阶段重排

这些方法提供了机制更合理的监督，但多数出现开发改善、冻结失败。它们没有被删除，而是明确标记为未通过确认。

### 13.5 Pfam 硬重排和三源融合

Pfam 对架构审计非常有价值，但关联数据中还包含 P450、prenyltransferase 和其他相关酶。全局硬过滤会丢失真实关联，因此仅用于反应条件化 contract、注释和 rescue 约束。

### 13.6 Horizyn 直接迁移

公开反应编码器和全局 MLNCE 路线在当前严格任务中没有稳定超过现有生产模型。其大规模预训练思想值得保留，但不能因为方法新颖就直接部署。

失败路线共同说明：

> 在当前数据规模下，增加局部特征或复杂监督不等于稳定提升。真正进入生产的方法必须在开发、冻结和独立确认中保持方向一致，并且只部署到它确实改善的预算和方向。

---

## 十四、代码与产物结构

### 14.1 主要训练与评测代码

```text
projects/active/terpene_screening/
├── train_dual_tower_cold.py
├── train_marts_domain_adaptation.py
├── train_marts_adapted_production.py
├── evaluate_dual_tower_protocol_comparison.py
├── evaluate_exact_entity_protocols.py
├── evaluate_sequence_fewshot_strict.py
├── evaluate_marts_fewshot_open_world.py
├── evaluate_e2r_route_interleaving.py
├── evaluate_locked_dual_kernel_route.py
├── evaluate_locked_marts_dual_kernel_confirmatory.py
└── evaluate_legacy_cage_double_cold.py
```

### 14.2 生产入口

```text
rank_open_world.py
rank_registry_batch.py
manage_open_world_registry.py
dual_kernel_runtime.py
validate_open_world_deployment.py
validate_dual_kernel_deployment.py
```

### 14.3 湿实验与候选库代码

```text
prepare_uniprot_tps_expansion.py
build_reaction_architecture_contracts.py
build_uniprot_rescue_campaign.py
build_wetlab_discovery_panels.py
build_wetlab_plate_manifest.py
balance_wetlab_reactions_across_plates.py
randomize_wetlab_candidate_positions.py
build_combined_wetlab_campaign.py
manage_wetlab_feedback.py
```

### 14.4 关键结果

```text
results/terpene_protocol_reassessment/
results/terpene_production_models/
results/terpene_open_world_uncertainty_rrf_routing/
results/terpene_registry_batch/
results/terpene_uniprot_rescue_campaign/
results/terpene_wetlab_discovery_panels/
results/terpene_wetlab_randomized_layout/
results/terpene_combined_wetlab_campaign/
```

---

## 十五、验证状态与已知限制

### 15.1 当前验证状态

- TPS 测试套件：74 passed；
- `git diff --check`：通过；
- 五个神经部署目录：全部 `valid`；
- 双核稀疏资产包：`valid`；
- 注册表排名审计：30,822 行；
- 已知关联泄漏：0；
- 单查询与批处理：候选、顺序和 RRF 分数一致；
- 当前非阻塞警告主要为 DRFP/NumPy 弃用提示。

### 15.2 数据限制

- 许多反应只有 1–2 个已知阳性；
- 未标注 pair 中可能包含大量真阳性；
- 反应标签可能只记录主产物，忽略副产物；
- 同一酶可催化多个反应；
- MARTS 和 Rhea 的粒度、命名和机制表示并不完全一致；
- UniProt 预测序列证据等级差异很大。

### 15.3 指标限制

- Hit@K 只说明 Top-K 中是否至少出现一个已知正例；
- 它不等于 Top-K 中有多少候选会在湿实验中有活性；
- 已知正例不完整会低估真实召回；
- query-cell 与 unique query 的统计单位不同；
- 不同候选宇宙、不同 split 和不同训练信息边界的数字不能直接比较；
- 可靠性校准只对对应协议有效。

### 15.4 实验限制

现有构建清单尚未进行宿主特异 codon optimization。正式订购 DNA 前仍需锁定：

- 表达宿主；
- 载体；
- 启动子和标签；
- 起止密码子；
- 信号肽或转运肽处理；
- 合成供应商长度、GC 和重复序列约束。

---

## 十六、实际使用决策表

| 用户目标 | 是否允许同源 | 推荐主路线 | 必须报告的主指标 | 补充指标 |
|---|---:|---|---|---|
| 已知反应和 1–5 个阳性酶，快速找替代物 | 是 | seed 3-mer/ESM-C + family-visible expansion | homolog-visible few-shot | cross-cluster few-shot |
| 已知反应，无阳性 seed，当前库内补全 | 是 | current exact nested fusion | exact-reaction holdout | reaction-cold |
| 新反应映射到已有蛋白家族 | 是 | R2E dual tower / reaction transfer | reaction-cluster-cold R2E | double-cold R2E |
| 已知反应寻找远缘酶家族 | 否 | protein-diverse R2E + model route | protein-cluster-cold R2E | architecture diversity |
| 新蛋白注释到已知反应目录 | 反应可见 | E2R production route | protein-cluster-cold E2R | double-cold E2R |
| 已知酶寻找新反应簇 | 蛋白可见 | E2R reaction-diverse route | reaction-cluster-cold E2R | double-cold E2R |
| 新反应簇和新蛋白簇同时外推 | 否 | 双塔 + 预算专用 RRF | double-cold | selective reliability |
| 扩展到大量 UniProt 候选 | 受控 | canonical prefix + rescue slots | retention / displacement | wet-lab rescue hit rate |

---

## 十七、下一阶段优先级

### P0：完成真实湿实验闭环

最重要的下一步不是继续堆模型，而是获得分层实验结果。必须同时记录：

- candidate source；
- route；
- homolog / model-mediated / cross-cluster；
- canonical / UniProt；
- Pfam architecture；
- evidence tier；
- 最近训练蛋白相似度；
- 表达和可溶性；
- 目标产物、旁产物和底物消耗。

### P1：建立场景专用校准器

当前可靠性主要来自严格外部协议。后续需要分别建立：

- homolog-visible few-shot 成功率校准；
- reaction-cold R2E 校准；
- protein-cold E2R 校准；
- UniProt rescue 专用校准。

### P2：机制感知反应表示

最有潜力的方法学方向仍然是 MARTS 逐步碳正离子机制图，而不是继续增加粗粒度反应指纹。目标是把“相似反应”从 SMILES 差分升级为：

- 前体启动方式；
- 首次环化；
- 碳正离子重排；
- 终止方式；
- 主/副产物机制路径。

### P3：更严格的时间切分与外部实验集

需要建立按文献时间或数据库发布日期划分的 temporal holdout，避免历史同源和重复条目造成过度乐观。

### P4：学习候选配额，而不是只学习排序

最终湿实验问题不是“谁排第一”，而是有限孔位如何在 exploitation、uncertainty 和 diversity 之间分配。后续可根据真实阳性率和信息增益学习 panel allocation policy。

---

## 十八、最终结论

旧方案证明了两件仍然成立的事实：

1. 相似反应对数据库补全有价值；
2. 已知阳性附近的同源扩展可以获得很高的 Top-10 命中。

新版没有否定这两项能力。它真正完成的升级是：

- 把 R2E 与 E2R 统一为双向检索；
- 把 exact、reaction-cold、protein-cold、double-cold 和 few-shot 拆成独立场景；
- 允许新酶和新反应无需重训练即可注册；
- 使用 ESM-C、DRFP、多正例双塔和 PU mask 建立可泛化的共享表示；
- 通过 MARTS 适配扩大数据与候选宇宙；
- 为 Top-3、Top-10 和 Top-20 设计不同路由；
- 用 RRF 和双核协同提升外部 E2R 覆盖；
- 对可靠性、候选库扩展和已知关联泄漏进行显式审计；
- 把计算结果转化为可执行的六块湿实验板和反馈闭环。

因此，项目最准确的描述不是：

> “从旧的高分方案换成了一个更严格但分数更低的开放世界模型。”

而是：

> **从一个主要面向数据库内补全的单向候选排序流程，扩展为同时覆盖同源利用、数据库补全、远缘发现、开放实体注册、双向检索和湿实验决策的多场景系统。**

实际使用时，应主动利用相似蛋白和相似反应来提高首轮阳性率，同时为远缘和跨架构候选保留明确探索配额。double-cold 用于回答“所有邻域证据都不可用时还能否工作”，而不是取代其他场景。最终科学结论必须来自不同候选层的真实湿实验阳性率、表达失败率和产物分布，而不是来自一个脱离任务边界的单一总分。
