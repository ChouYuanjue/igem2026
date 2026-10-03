# BRIDGE Engineering：完整决策树

本文件按技术因果重组历史。顺序服务于工程逻辑，允许与 commit 时间轻微交错。每个分支保留动机和结果。

## 0. 根问题

**目标：** 从 TPS 场景出发，在有限实验预算下，把真正相关的酶尽量排到前部，并逐步扩展到任意酶—反应查询。

```text
TPS 筛选
├── 候选从哪里来
├── 候选怎样排序
├── 怎样扩大候选空间
├── 怎样吸收不同证据
└── 怎样保证专家只在适用时发言
```

## 1. 候选池 + CAGE

### 相似反应迁移
动机：相似化学反应常共享可迁移酶家族。结果：形成最早候选生成主干，后续演化为反应邻域证据。

### Gate matrix / candidate gate
动机：缩小搜索范围。结果：数据库内有效；开放场景中，真实阳性一旦被 gate 排除便无法恢复。

### CAGE 主排序
动机：用结构和口袋信息区分候选。结果：早期有效；后续暴露分数饱和、结构覆盖和候选可评分范围问题。

### RF / HGB 元排序
动机：融合反应相似度、CAGE 和规则特征。结果：旧任务有效，依赖固定候选池和旧特征边界。

### rescue slots
动机：给规则或结构强证据保留少量补救位置。结果：局部保留，“主序 + 有限补救”成为后续锚定修正的早期原型。

## 2. 扩大候选池

### few-shot / seed 同源扩展
动机：已有阳性时同源证据极强。结果：保留为独立场景；同簇扩展和跨簇远缘扩展分开报告。

### MARTS 外部候选
动机：突破内部 TPS 数据。结果：保留，并推动外部反应、外部酶和严格切分评测。

### UniProt 自由扩池
动机：显著扩大 TPS 候选范围。结果：大量未标注候选争夺 Top-K，自由合并淘汰。

### canonical prefix + tail quota
动机：在扩展候选时保护已验证主排名。结果：保留为受控 rescue。

### Pfam 单域门控
动机：利用 TPS 家族架构知识。结果：共享单域会混入功能不同蛋白，淘汰。

### exact Pfam architecture / reaction-specific contract
动机：使用完整域组合并绑定具体反应。结果：保留用于 rescue、审计和解释，不承担全局硬过滤。

### taxonomy scope
动机：支持用户只搜索真核或原核。结果：使用候选宇宙约束，不训练额外总模型。

## 3. Broad Retrieval

### ESM-C 蛋白表示
动机：让数据库外蛋白可直接编码。结果：保留。

### DRFP + 反应类别
动机：让反应通过化学变化而非 ID 表示。结果：保留为广域反应表示。

### multi-positive dual tower
动机：一个反应可对应多个酶，一个酶也可对应多个反应。结果：形成 Broad 基础。

### PU cluster mask
动机：未标注 pair 和近同源蛋白可能是假负例。结果：保留。

### hard negative
动机：随机负例过于简单。结果：普通困难负例有价值；专用 hard-negative 模型和 curriculum 没有稳定替代主模型。

### Top-K surrogate
动机：直接优化实验预算边界。结果：对 K 敏感且易受假负例影响，保留研究代码，未进入主损失。

## 4. 外部预训练迁移

### Horizyn global MLNCE
动机：直接利用更大预训练空间。结果：未稳定超过 Broad，淘汰。

### 冻结 Horizyn 反应编码器
动机：固定外部反应空间，只适配另一侧。结果：淘汰。

### ESM-C → ProtT5 bridge
动机：解决蛋白表示空间不一致。结果：淘汰。

### DRFP + Horizyn 简单拼接
动机：同时利用本地和外部反应表示。结果：不稳定，淘汰。

### reaction distillation residual / exact-residual
动机：保留 Broad，只吸收外部模型增量。结果：exact-residual 曾在特定 R2E 预算保留；强化“外部模型只做增量”的原则。

## 5. TPS 局部机制分支

### motif context
动机：DDxxD、NSE/DTE、DxDD 等更接近催化机制。结果：局部辨别有价值，全库召回弱。

### motif RRF / motif residual
动机：限制 motif 只做二阶段修正。结果：开发有增益，冻结最优回到零修正。

### P2Rank / pocket-local
动机：口袋更接近底物选择。结果：独立模型弱，融合开发收益未通过冻结。

### 同前体、不同骨架困难负例
动机：针对 TPS 真正容易混淆的近邻。结果：语义合理，冻结未超过基线。

### mechanism auxiliary / skeleton metric
动机：显式监督 precursor、topology、oxidation 和骨架差异。结果：标签粒度不足，冻结不稳定。

### 粗骨架 / Morgan 反应簇 / 碳连接图
动机：逐步细化化学监督。结果：开发偶有新增命中，冻结权重回到零。

### TPS active-site Cross-Attention
动机：直接学习反应与活性位点 token 交互。结果：经历 hard pools、residual 和 HPO；后续发现评测对齐问题，未成为广域主线。

## 6. 图与非参数分支

### geometry alignment
动机：让反应内部几何与蛋白内部几何对齐。结果：多功能酶、不同折叠和标签稀疏破坏一一几何，未稳定进入生产。

### graph diffusion / multi-hop
动机：利用反应—蛋白关联图传播。结果：hub 放大且双冷缺入口，淘汰。

### candidate hub normalization
动机：抑制普遍高分候选。结果：多种归一化未形成稳定增益。

### dual kernel
动机：同时利用蛋白邻域、反应邻域和已知关联。结果：current-only 冻结失败；MARTS E2R Top-20 在限制预算后通过确认。

## 7. 融合与不确定性

### raw score addition
动机：直接融合模型。结果：分数尺度不兼容，淘汰。

### percentile / tied-rank fusion
动机：消除量纲。结果：局部保留。

### RRF
动机：只利用相对排名。结果：在特定 E2R 预算稳定通过确认。

### 三源固定融合
动机：Pfam、双核、多专家存在独占命中。结果：开发提高、冻结下降；说明互补性不等于固定权重可靠。

### ensemble / reliability / abstention
动机：降低随机性并暴露查询风险。结果：三 seed 和协议绑定的可靠性继续保留；原始模型分数不解释为活性概率。

### conformal retrieval sets
动机：固定 Top-K 无法表达查询难度。结果：保留为覆盖诊断旁路，不直接决定主排序。

### R2E↔E2R cycle consistency
动机：双向排名应提供相互支持。结果：权重网格和适用域门控没有产生确认新增命中，保留为诊断，不进入排序。

## 8. Broad 通用化与抗遗忘

### directional continuation
动机：吸收更广的酶—反应关系，同时保护 TPS 能力。结果：成为通用化试验基础。

### historical replay / embedding anchor
动机：减少 broad continuation 遗忘。结果：保留为 retention 思想。

### LwF / score distillation / Margin-MSE
动机：分别保护旧输出、旧排序和 pair margin。结果：均参与筛选，没有成为最终架构定义。

### RecAdam
动机：训练早期偏向旧模型，随后逐渐学习新域。结果：作为低遗忘 continuation 路线测试。

### checkpoint blending / WiSE-FT
动机：新域有收益但旧域下降时向 source 拉回。结果：可找到 Pareto 点，但没有解决多专家条件适用问题。

### Fisher / RegMean / TIES / AdaMerging
动机：从参数空间合并不同域/方向能力。结果：没有形成比排序级专家组织更稳定的主线。

### post-hoc domain routing
动机：不同模型可能只在某些查询有效。结果：首次明确指向“专家价值具有条件性”。

## 9. 新颖性与反应中心

### low-similarity novelty expert
动机：低相似反应使用专项模型。结果：novelty replay 和 routing 稳定性不足，淘汰。

### functional prototype residual
动机：用训练功能原型局部修正 Broad。结果：正式筛选拒绝。

### reaction-center V1
动机：只描述真正发生变化的原子和键。结果：hard-slice 未通过。

### reaction-center residual / identity-preserving residual
动机：只学习相对 Broad 的增量，缺失时严格回退。结果：继续推进。

### bounded reaction-center V3
动机：限制局部机制模型最大修正幅度。结果：通过确认；成为后续 bounded correction 的重要祖先。

## 10. EnzGFM 与局部精排

### EnzGFM native baseline
动机：利用已训练的大型酶—反应基础模型。结果：成为强基线和专家来源。

### EnzGFM + RDKit / RDKit+
动机：将蛋白基础表示与反应化学特征结合。结果：进入多专家候选。

### EnzGFM + reaction center
动机：加入局部机制变化。结果：成为重要专家源。

### Top-2000 pair reranker / residual / bounded residual
动机：Broad 已经能做全库排序，只在前部纠错。结果：有限前缀和受限修正思想保留。

### difficulty-aware / center gate
动机：只对需要且有证据的查询运行二阶段模型。结果：部分具体组合拒绝，查询级路由思想保留。

## 11. BiME-Rank

### R2E candidate union + LambdaRank
动机：专家先独立召回，再学习如何组合分数和 rank。结果：通过确认，成为 R2E 主干。

### R2E similarity router
动机：不同新颖度查询使用不同路线。结果：保留确定性 fallback。

### E2R four-expert LambdaRank
动机：复制 R2E 成功。结果：破坏强 EnzGFM 基础序，淘汰。

### baseline-anchored rescue / Anchored LambdaMART V3
动机：保护 EnzGFM Top-1，只在有限 union 内重排。结果：保留；形成“强基础序 + 有限修正”。

### CLIPZyme structure expert
动机：补充三维结构证据。结果：通过准入；缺失时严格回退。

### seed context
动机：已有阳性时利用任务上下文。结果：条件准入。

### homology context
动机：显式同源上下文。结果：外部 retention 未通过。

### reciprocal consistency expert
动机：将双向一致性升级为正式专家。结果：外部 retention 未通过。

### CAGE Top-20 generic expert
动机：把成熟结构模型重新接入通用系统。结果：内部 OOF 下降，通用准入失败。

### cost-aware hierarchy
动机：昂贵专家无法全库执行。结果：廉价专家广搜、昂贵专家 shortlist 执行的分层保留。

## 12. FIBRE 分支

### biological relation stratification / mechanistic strata / partial relation
动机：把二元 pair 扩展为更丰富的生物关系。结果：进入 FIBRE 统一建模探索。

### tensor-product field / interaction atlas
动机：把不同专家解释成同一催化交互空间的局部图册。结果：形成完整 FIBRE 几何路线。

### context-restricted domains / atlas gluing
动机：局部专家只在自己的域有效，并在重叠区拼接。结果：一致性假设过强，后续删除。

### catalytic kernel / conditional catalytic kinetics
动机：给几何模型更真实的催化解释。结果：理论增强，仍未解决实际排序稳定性。

### conditional modes
动机：一个通用模式加多个查询条件模式。结果：成为 FIBRE 第二阶段核心。

分支包括：完全对称联合势、双侧门控几何平均、对称线性混合、Gibbs 聚合、KL 重心、分子一体势、专家方差回退、规范化条件混合、维度缩放一致性、线性条件期望、Log-Mean-Exp、二阶方差修正、反应中心模式后验。多数方案在开发或冻结阶段因方向性能下降被否决；最终说明两个检索方向包含真实条件信息。

### scientific evidence layer
动机：把结构、机制和上下文作为锚定在 core 上的显式 evidence。结果：missing=zero、独立证据准入和 anchored additive correction 成为有效结论。

### HCM / query-adaptive mix
动机：将 EnzGFM、reaction center、CLIPZyme、seed context 放进统一条件计算图，并学习 query 级权重。结果：端到端 gate 伤害 R2E；冻结 core 的 post-hoc gate 改善不足以晋级。

### ERAM relational core + UniMol
动机：用广域关系学习方法替代大量手工几何。结果：完成训练和严格评测，整体仍未替代强 Broad。

### pluggable adapters / context plugin
动机：新证据无需重训统一核心。结果：插件化和缺失回退思想保留。

**FIBRE 结论：** 统一几何/关系核心停止；missing-neutral、插件化、query-conditioned applicability、保护 Broad 的思想进入 BRIDGE。

## 13. 回到 Broad + Experts

### expert as pair evidence
动机：专家只输出 query-candidate 增量证据。结果：保留。

### rebind evidence to Broad Core
动机：Broad 是更可靠的全局基础序。结果：成为最终结构基础。

### directional score evidence
动机：同一专家在 R2E/E2R 可能拥有不同权限。结果：EnzGFM E2R 可修改排序，R2E 只保留支持/置信作用。

### expert type 分层
动机：把具体专家整理为功能/基础、结构、机制、家族、上下文类型。结果：保留。

### dynamic router V4/V6
动机：按 query 决定哪些专家可以参与，并区分直接排序、响应探针和静音。结果：形成最终 query applicability 原则。

## 14. CAGE 与 TPS 回归

### Broad → generic CAGE
动机：检查 CAGE 的问题是否只来自旧候选池。结果：Broad 找回许多新阳性，但 CAGE 对大量开放候选没有可用输入，失败。

### P450 / phosphatase / terpene CAGE finetune
动机：通用 CAGE 失败不代表局部家族无价值。结果：各自测试出现局部提升，作为专项能力保留。

### TPS specialist
动机：重新利用项目最早的 TPS 专项积累。结果：通过 TPS manifold/applicability gate，只在很少的 TPS 查询中修正。

## 15. BRIDGE 收敛

```text
Broad Retrieval
├── 完整候选空间
├── 稳定基础顺序
└── 无专家也可独立运行
        ↓
Inference-Driven Applicability
├── 信息是否存在
├── 查询是否属于适用域
├── 该方向是否通过准入
└── 修正范围是否受限
        ↓
Gated Experts
├── 功能/基础
├── 结构
├── 机制
├── 上下文
├── 家族 CAGE
└── TPS specialist
        ↓
Bounded correction
```

核心公式：

\[
S_{\mathrm{BRIDGE}}=S_{\mathrm{Broad}}+\sum_k g_k(q)\Delta_k(q,e).
\]

工程史最终收敛到一条原则：**Broad 持有默认排序权，专家只在有依据时获得有限修正权。**
