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

模块元数据同时声明 `score_direction`。高分更好的工具直接使用 `higher_is_better`；低分更好的距离、代价或能量型输出使用 `lower_is_better`，运行时统一转换成“数值越高越支持当前配对”的内部方向。

科学证据必须在同一查询下对候选产生区分。只描述查询本身、对所有候选给出同一个值的信息，经查询内校准后严格变成零贡献；它只有在与候选侧能力形成明确的配对兼容分数后，才会改变催化排序。

### 候选生成器

能够独立找出候选的检索器。它可以同时也是科学证据模块，也可以只负责扩充候选集合。候选发现能力与分数融合分开记录。

### 实验约束

宿主、库存、表达性、长度、合成成本等条件。它们决定实验可行性，不改变催化相容分数。

## 3. 新证据怎样准入

默认准入不手调权重。

核心模型和已有证据全部冻结，对新证据生成训练外分数，然后只拟合非负证据系数。当前实现使用凸的成对逻辑排序目标，因此一个新模块只增加一个基础参数；有独立、可验证的质量量时可以再增加一个非负质量斜率。准入结果同时绑定训练时的核心排序标识，因为证据系数的数值依赖核心分数尺度。

一个模块单独接入时可以独立准入。两个或更多模块同时改变同一排序时，必须在同一个核心上做**联合交叉拟合**，生产运行时会拒绝把多个独立准入结果直接相加。联合拟合默认使用非负 L1 正则；复制一条完全相同的证据不会因此降低正则代价或获得额外总权重。

准入比较的是**加入现有系统后的增量价值**。高度相关的模块在联合目标里共同定权，不把各自相对裸核心的收益重复计算。

正式准入同时要求两层稳定性：每个 held-out 折的成对逻辑损失都下降；把全部 OOF 查询逐个计算增益以后，平均增益的确定性 bootstrap 95% 下界也必须大于 0。外部或严格时间数据只用于冻结后的确认。

联合拟合解决的是“这些证据合在一起还有多少增量”，并不保证每个高度共线模块都能获得唯一解释。两条完全相同的证据在数学上只能识别总贡献，具体权重可以在二者之间分摊；因此对外解释时优先报告联合增量和候选级总贡献，不给共线模块制造虚假的独立机制含义。

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

三个留出折全部改善。全内部数据拟合得到基础强度 0.21333，质量斜率 0.11159。这说明同一个插件接口已经分别覆盖结构证据、用户已知阳性上下文和反应中心机制证据三种性质不同的信息源。

按当前更严格的准入器重放同一反应中心证据后，3,299 个 OOF 查询的平均成对损失增益为 0.02167，95% bootstrap 区间为 [0.01657, 0.02673]，因此仍通过正式准入。

这三组结果来自三个不同协议和核心分数来源，证明的是**接口可以承载三类信息**。它们目前没有在同一查询集合、同一核心分数上联合拟合，因此不构成一个已经验证的“三专家同时生产组合”。

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
  --baseline-id my-fibre-core-v1 \\
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
  --baseline-id my-fibre-core-v1 \\
  --evidence-csv local_structure_scores.csv \\
  --admission-json local_structure_admission.json \\
  --top-k 20 \\
  --output ranked_with_local_evidence.csv
\`\`\`

输出同时保存每个证据模块对候选分数的实际贡献，便于审计本地信息怎样改变最终顺序。

正式 FIBRE 检索命令也已经直接接入同一机制。证据在完整候选分数形成后、Top-K 截断前加入：

```bash
python -m projects.active.fibre.runtime.cli rank-enzymes \
  --reaction-id RHEA:xxxxx \
  --scientific-evidence-csv local_structure_scores.csv \
  --scientific-evidence-admission local_structure_admission.json \
  --scientific-evidence-baseline-id my-fibre-core-v1 \
  --top-k 20
```

多个模块同时使用时，先在同一核心训练表上做联合准入。`fit_scientific_evidence_bundle.py` 接收同一 `core-csv`、多个证据表和各自的单模块描述记录，联合学习一组非负系数；生产运行时重复传入 `--scientific-evidence-csv`，并用一个 `--scientific-evidence-bundle` 加载这组联合系数。多个独立 admission JSON 不能直接叠加。

```bash
python reproducibility/bime_rank/scripts/fit_scientific_evidence_bundle.py \
  --core-csv local_core_training.csv \
  --baseline-id my-fibre-core-v1 \
  --direction r2e \
  --evidence-csv local_structure_training.csv \
  --member-spec local_structure_descriptor.json \
  --evidence-csv local_mechanism_training.csv \
  --member-spec local_mechanism_descriptor.json \
  --output local_joint_bundle.json
```

生产使用时保持相同证据顺序，并显式声明同一个 baseline：

```bash
python -m projects.active.fibre.runtime.cli rank-enzymes \
  --reaction-id RHEA:xxxxx \
  --scientific-evidence-csv local_structure_scores.csv \
  --scientific-evidence-csv local_mechanism_scores.csv \
  --scientific-evidence-bundle local_joint_bundle.json \
  --scientific-evidence-baseline-id my-fibre-core-v1 \
  --top-k 20
```

输出中的 `scientific_evidence:模块名` 列记录每条证据对最终候选分数的实际贡献。候选生成器若发现核心候选宇宙之外的对象，需要先通过现有注册机制把对象及基础特征加入候选宇宙；科学证据层本身不会凭一条外部分数绕过核心候选注册。
