# FIBRE 科学证据扩展

## 1. 当前实现

FIBRE 的扩展层采用**冻结核心排序 + 锚定科学证据**。

核心排序的系数固定为 1。每个已准入证据先在当前查询的有效候选中做尺度校准，再以非负强度加入：

$$
S(q,c)=S_0(q,c)+\sum_j \alpha_j E_j(q,c).
$$

证据缺失时该项严格为 0，其余证据和核心排序保持原值。不同证据不会争夺一个固定的概率质量。

当前代码：

- 融合内核：projects/active/fibre/kernel/evidence_fusion.py
- 插件接口：projects/active/fibre/runtime/scientific_evidence.py
- 准入拟合：reproducibility/bime_rank/support/evidence_admission.py
- 统一结果：reproducibility/bime_rank/records/FIBRE_SCIENTIFIC_EVIDENCE_V1_RESULT.json

## 2. 四类对象

### 潜在催化模式

FIBRE-Modes 内部端到端学习的局部分量。它们属于核心模型，不要求对应某个预先命名的生化机制。

### 科学证据

能够对“当前反应—酶配对是否更可信”提供额外信息的模块。当前接口把证据分成：

- 分子视角：另一种蛋白或反应表示；
- 机制证据：反应中心、催化残基、基序等；
- 结构证据：三维结构或口袋兼容性；
- 实验上下文：用户已知阳性等真实实验信息。

每个证据模块输出：

- score：候选级证据分数；
- available：该候选是否真的有这条证据；
- quality：可选，只有存在可验证质量定义时才使用。

### 候选生成器

能够独立找出候选的检索器。它可以同时也是科学证据模块，也可以只负责扩充候选集合。候选发现能力与分数融合分开记录。

### 实验约束

宿主、库存、表达性、长度、合成成本等条件。它们决定实验可行性，不改变催化相容分数。

## 3. 新证据怎样准入

默认准入不手调权重。

核心模型和已有证据全部冻结，对新证据生成训练外分数，然后只拟合非负证据系数。当前实现使用凸的成对逻辑排序目标，因此一个新模块只增加一个基础参数；有独立、可验证的质量量时可以再增加一个非负质量斜率。

准入比较的是**加入现有系统后的增量价值**。单独表现很强、但与现有证据高度重复的模块不会因为自身指标高就自动进入。

正式比较采用交叉拟合。外部或严格时间数据只用于冻结后的确认。

## 4. 已完成验证

### 结构证据

CLIPZyme 结构分数只学习一个非负系数。

三折内部交叉拟合中，MRR 从 0.09868 提升到 0.11487，Hit@10 从 23.75% 提升到 27.22%，Hit@50 从 42.30% 提升到 50.81%。

系数在三个留出折上分别约为 0.504、0.494、0.478。

固定全部内部数据得到的系数 0.49192 后，在 Rhea release 128→141 的 144 个严格双冷 R2E 查询、166,202 个共同候选上只确认一次：

- MRR：0.03626 → 0.06204
- MAP：0.03482 → 0.05865
- NDCG@10：0.03533 → 0.06083
- Hit@10：6.25% → 9.72%
- Hit@20：11.11% → 15.97%
- Hit@50：16.67% → 23.61%
- 最佳阳性中位名次：738 → 538

现有专门训练的结构学习排序器在该严格协议上的 MRR 为 0.08733。成熟证据源在值得投入额外复杂度时仍可采用专项融合；一参数证据接口承担低成本、统一的默认接入。

### 已知阳性上下文

用户提供一条已知阳性酶时，种子自身从输出中屏蔽，ESM-C 序列相似性作为上下文证据，只学习一个非负系数。

2,595 次内部交叉拟合试验中：

- MRR：0.13137 → 0.29881
- MAP：0.08286 → 0.17501
- NDCG@10：0.10433 → 0.23450
- Hit@10：29.83% → 63.35%
- Hit@20：38.57% → 73.10%
- Hit@50：48.67% → 78.88%

三个留出折学习到的系数约为 1.188、1.145、1.215。

### 反应中心机制证据

第三类验证使用已经训练过的反应中心有界残差。基础蛋白塔和反应塔保持冻结，历史训练过程中只有反应中心投影可以更新，因此“残差模型分数减基础模型分数”可以直接解释为这条机制信息对候选配对的增量证据。

三个 clean2023 留出折共包含 3,299 个反应查询、764,582 条显式候选行和 25,833 条阳性行。RXNMapper 成功时证据可用，映射置信度作为可选质量量。统一准入器只学习一个基础强度和一个非负质量斜率：

- fold 0：pairwise log loss 0.64887 → 0.63159；
- fold 1：0.64203 → 0.61841；
- fold 2：0.65010 → 0.62579。

三个留出折全部改善。全内部数据拟合得到基础强度 0.21333，质量斜率 0.11159。这说明同一个插件接口已经覆盖结构证据、用户已知阳性上下文和反应中心机制证据三种性质不同的信息源。

## 5. 自部署接入

一个本地模型只有在它提供新的催化关系信息时才进入科学证据接口。

最小接入信息是：

1. 名称与证据类别；
2. 支持 R2E、E2R 或两者；
3. 分数含义；
4. 什么情况下证据可用；
5. 训练数据和模型来源；
6. 候选级 score 与 available；
7. 若提供 quality，必须同时声明质量量的物理或统计含义。

系统随后在本地交叉拟合数据上学习非负证据强度。没有历史标签时，证据可以展示给用户，但当前准入流程不会自动让它改变主排序。

用户自己的宿主限制、库存和实验成本直接实现为候选约束或决策条件，不需要训练证据模块。

### 零代码表格接入

准入表至少包含：

\`query_id,candidate_id,core_score,evidence_score,label\`

可选列：

\`available,quality,fold\`

准入命令：

\`\`\`bash
python reproducibility/bime_rank/scripts/fit_scientific_evidence.py \\
  --csv local_evidence_training.csv \\
  --name local_structure_model \\
  --kind structural \\
  --direction r2e \\
  --score-semantics "higher means stronger structural support" \\
  --availability-semantics "a valid structure score exists for the pair" \\
  --provenance "laboratory frozen model; cross-fitted local assays" \\
  --output local_structure_admission.json
\`\`\`

准入通过以后，运行时可以把一个或多个证据表同时应用到冻结核心候选分数：

\`\`\`bash
python reproducibility/bime_rank/scripts/apply_scientific_evidence.py \\
  --core-csv core_candidates.csv \\
  --direction r2e \\
  --query-id RHEA:xxxxx \\
  --evidence-csv local_structure_scores.csv \\
  --admission-json local_structure_admission.json \\
  --top-k 20 \\
  --output ranked_with_local_evidence.csv
\`\`\`

输出同时保存每个证据模块对候选分数的实际贡献，便于审计本地信息怎样改变最终顺序。
