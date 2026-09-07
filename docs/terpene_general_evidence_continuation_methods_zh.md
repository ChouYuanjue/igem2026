# 从特化检索器到大规模证据域：低成本延拓方法路线

## 1. 问题定义

当前 Catalyst 检索器不是从零开始训练的新模型，而是已经在 TPS / MARTS 等目标域上完成特化、具有明确生产能力谱的双塔模型。新的训练目标是在不丢失这些能力的前提下，把约 24 万条更广泛的已记录蛋白–反应关系吸收到模型表示中，尤其改善 broad-universe 下的 zero-shot known-recovery。

因此，这不是普通的 domain adaptation。核心约束同时包括：

1. **Extension**：新证据域上的未见关系恢复必须实质提高；
2. **Retention**：既有 current-domain、historical/project-catalog、cold/few-shot 等能力不能因为新域训练而明显下降；
3. **Low cost**：优先采用一次短 continuation、参数高效更新或闭式模型合并，而不是重新做大规模从头训练；
4. **Bidirectional control**：R2E 与 E2R 分别对应 reaction tower 与 protein tower 的适配，允许方向独立训练和最终组合。

## 2. 调研结论

### 2.1 RecAdam：当前训练阶段主线

- 上游：<https://github.com/Sanyuan-Chen/RecAdam>
- 论文：Chen et al., *Recall and Learn: Fine-tuning Deep Pretrained Language Models with Less Forgetting*, EMNLP 2020.
- 许可证：Apache-2.0。

RecAdam 将目标域损失与“回到原始模型参数”的二次约束通过 objective shifting 组合。训练早期更强调 source recall，随后逐渐放开 target learning，因此比固定 L2-SP 更适合“先保住特化模型，再吸收大域关系”的顺序训练。

本项目直接复用上游 optimizer，不重新实现算法。Catalyst 只负责把当前方向可训练 tower 的参数与冻结 production checkpoint 中同名参数一一配对。现有 historical-query replay 与 frozen-model embedding anchor 继续保留，因此形成两层 retention：

- parameter-space recall：RecAdam；
- function/representation-space recall：historical embedding distillation。

### 2.2 L2-SP / LwF / DER / FDR：重要基线与机制来源

- THUML Transfer-Learning-Library：<https://github.com/thuml/Transfer-Learning-Library>，包含 L2-SP、LwF、DELTA、BSS 等迁移方法；
- Mammoth：<https://github.com/aimagelab/mammoth>，包含 DER/DER++、LwF、FDR、EWC、A-GEM 等大量 continual-learning 方法。

L2-SP 的核心是把权重拉回 fine-tuning 起点，而不是拉向零；RecAdam 可以理解为加入 objective shifting 的更适合 sequential adaptation 的版本。LwF/DER/FDR 则从函数输出或历史“dark targets”约束遗忘。本项目已有 embedding anchor，与 FDR/representation distillation 的思想接近；若仅约束 embedding 仍无法稳定旧排序，下一步优先增加 **historical score / ranking distillation**，而不是再堆一种同质参数正则。

Mammoth 本身是 class/domain-incremental benchmark 框架，直接把 Catalyst 的 full-candidate retrieval loss 塞进其训练器会带来较重适配，因此目前作为算法参考与对照来源，不强行替换现有训练 harness。

### 2.3 LoRA：真正低参数成本的并行对照

- 上游：<https://github.com/microsoft/LoRA>
- 许可证：MIT。

`loralib` 原生支持 `nn.Linear`，而 Catalyst projection tower 主要由 Linear 层组成，因此结构匹配度很高。LoRA 冻结原参数，只训练低秩增量，天然降低训练参数量和 checkpoint 成本，也给 source model 留下明确的可恢复基座。

它不自动保证旧 ranking 不漂移，所以仍需同样的 retention guard；但非常适合作为“参数高效延拓”对照。主线 RecAdam 稳定后，计划用少量 rank（如 4/8/16）测试一次，不做大网格。

### 2.4 WiSE-FT / Model Soups：训练后 consolidation

- WiSE-FT：<https://github.com/mlfoundations/wise-ft>
- Model Soups：<https://github.com/mlfoundations/model-soups>

WiSE-FT 在 source / zero-shot checkpoint 与 fine-tuned checkpoint 之间做 weight-space interpolation。当前 Catalyst 的单 seed 实验已经证明这一操作有效：full broad continuation 虽然大幅改善 unseen recovery，但产生遗忘；把 task vector 缩回 source checkpoint 后可以找到无历史退化的 Pareto 点。

因此权重插值不再被视为“补丁”，而是正式的 **post-training consolidation**。Model Soup 则适合在多个 seed / continuation 配置之间进一步平均，但只有在每个成员本身经过同口径评估后才使用。

### 2.5 RegMean / RegMean++：闭式、表示感知的 consolidation 候选

- FusionBench RegMean：<https://github.com/tanganke/fusion_bench>
- RegMean++：<https://github.com/nthehai01/RegMean-plusplus>

RegMean 对每个 Linear 层收集输入 activation 的 Gram matrix，并用线性回归的闭式解融合模型参数；它比单一 scalar interpolation 更能反映各域实际经过该层的表示几何。Catalyst 的 tower 只有 `LayerNorm → Linear → GELU → Linear`，因此尤其适合这一类方法。

RegMean++ 进一步考虑层间表示传播，2026 年 TMLR 结果显示其在 sequential merging 和分布偏移下比原始 RegMean 更稳定。但它的官方实现主要围绕 CLIP/FusionBench task pool；本项目先复用 FusionBench 的通用 Linear merge core，再决定是否有必要引入 RegMean++ 的逐层传播逻辑。

### 2.6 Fisher merging：高优先级高级 consolidation 候选

- 原论文：Matena & Raffel, *Merging Models with Fisher-Weighted Averaging*, 2022.
- FusionBench MIT 实现：<https://github.com/tanganke/fusion_bench>。

Fisher merging 用各参数在不同域上的 Fisher importance 作为融合权重。相比等权平均，它可以让 source-domain 关键参数更靠近 production checkpoint，而在 source 不敏感的方向吸收 broad-domain 更新。原论文还直接研究了 robust fine-tuning、intermediate-task training 与 domain-adaptive pretraining，并强调模型合并比重新做梯度迁移便宜。

这与 Catalyst 的目标非常吻合。计划复用 FusionBench 的 MIT-licensed Fisher merge core，只为双塔 retrieval loss 实现数据适配层；不会重新实现参数融合公式。

### 2.7 TIES / DARE / Task Arithmetic：暂不作为第一优先级

- TIES：<https://github.com/prateeky2806/ties-merging>
- MergeLM（含 DARE/TIES/RegMean/Fisher 实验实现）：<https://github.com/yule-buaa/MergeLM>

TIES/DARE 主要解决多个 task vector 之间的符号冲突和冗余。在当前 R2E/E2R directional continuation 中，两条 task vector 主要落在不同 tower，本来就近似解耦，因此收益空间不如 Fisher/RegMean 明确。等以后需要合并多个不同 broad-domain 专家时再提高优先级。

### 2.8 TAK / KFAC regularization：高级备选

Mammoth 已集成 ICLR 2026 的 Task Arithmetic with KFAC regularization（TAK）。该路线用 curvature/Jacobian-Gram 信息约束 task-vector 引起的表示漂移，并强调对 task-vector scaling 更稳健。理论上很适合解决当前“两个单塔分别安全、组合后仍可能相互作用”的问题。

但 KFAC 因子和当前 retrieval architecture 的适配工作显著高于 RecAdam / Fisher / RegMean，所以先作为第二阶段高级路线；只有当前三者仍无法同时满足 broad gain 与全套 retention gate 时再投入。

## 3. 当前方法叙事：Recall–Extend–Consolidate

现阶段最自然的整体方法可以概括为 **Recall–Extend–Consolidate (REC)**：

1. **Recall**
   - 从已经特化好的 production checkpoint 出发；
   - RecAdam 参数空间 recall；
   - historical query replay；
   - frozen-model representation distillation。
2. **Extend**
   - 在 broad recorded-association graph 上做 directional full-candidate contrastive continuation；
   - R2E 只更新 reaction tower，E2R 只更新 protein tower；
   - 一轮短 continuation，而不是重做原始训练。
3. **Consolidate**
   - 首先使用 WiSE-FT-style source/adapted interpolation；
   - 再比较 Fisher / RegMean 这类 importance- or representation-aware merge；
   - 按 seed 精确合并两个方向的 tower，最后恢复三 seed ensemble。
4. **Guard**
   - matched broad known-recovery；
   - historical/project-catalog 零退化 guard；
   - 原有 current-domain retention；
   - 再回归 frozen cold/few-shot/external 能力谱。

这条叙事的重点不是“把更多数据再训练一遍”，而是：**在保留领域特化能力的前提下，以 continual-learning regularization 和 model consolidation 将一个专门检索器低成本扩展到更大的证据域。**

## 4. 实验纪律

- 单 seed 只能用于快速方法筛选，最终结论必须恢复 `20260723/20260724/20260725` 三 seed ensemble；
- baseline 与 candidate 的 seed 集合、query ID、positive label 必须完全一致；
- broad unseen recovery 提升不能抵消旧指标下降；
- 任何新方法都同时接受新 known-recovery guard 和仓库已有 current/cold/few-shot 指标；
- 优先少量有理论依据的超参数点，不做为了单 seed 数字而设计的大规模网格；
- production 模型在完整审计前不替换。
