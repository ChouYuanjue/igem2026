# 萜类合酶候选酶推荐工作流阶段总结（校正版）

更新时间：2026-07-23。

本文替代此前以 gate matrix、reaction similarity 和 CAGE-aware RF/HGB rescue 为主体的阶段总结。旧结果保留为历史基线，但不再作为当前生产性能结论。完整实验与实现细节分别见：

- `results/terpene_research_iteration_report.md`
- `results/terpene_uniprot_expansion_report.md`
- `results/terpene_wetlab_execution_report.md`

## 1. 不变的任务目标

目标仍然是给定目标萜类反应，从候选 terpene synthases 中筛选最值得实验验证的酶。任务分为 reaction-only zero-shot 和带 1/2/3/5 个阳性酶的 few-shot seed expansion。

新版保留“zero-shot 首轮筛选 → 湿实验阳性 → few-shot 扩展”的总体流程，但不再把旧 gate + CAGE meta-ranker 当作最终主模型。

## 2. 旧方案为何需要修订

旧方案不是简单答案泄漏。`gate_matrix.py` 会排除目标 Rhea 本身以及相同 canonical reaction SMILES。其主要问题是评测口径：

- CAGE meta-ranker 只按 exact `reaction_id` 做 GroupKFold；
- 相似 reaction cluster 没有隔离；
- 50% protein sequence cluster 没有隔离；
- 同一候选蛋白可在其他反应的训练记录中出现；
- 大量模型、融合权重、gate 和 rescue slot 在同一批 OOF 结果上比较；
- 旧报告提到的 50 次 tune/test 验证目前没有对应的正式脚本和结果目录，`results/terpene_cage_fair/meta_ranker/` 也不在当前 server06 正式产物中。

因此旧 zero-shot Top10 约 39.6%、Top20 约 45.2% 应解释为当前数据库内的 exact-reaction-held-out completion，不是 unseen reaction family 与 unseen protein family 同时出现时的开放发现性能。

## 3. 旧 few-shot 高分的真实含义

旧 few-shot 将同一反应下的阳性酶随机拆分为 seed 和 hidden positives。新版稳定复现后，random-positive Top10 仍约为：

- 1 seed：73.7%
- 2 seeds：82.8%
- 3 seeds：87.1%
- 5 seeds：92.8%

这说明旧结果并非计算错误，但主要是同源家族补全。把 seed 与 hidden positives 按 50% identity cluster 隔离后，Top10 只有约 25.5%–29.6%，Top20 约 35.9%–42.9%。

旧脚本还使用 Python 内置 `hash()` 构造随机种子，跨进程不能保证完全复现。新版改为 BLAKE2b 稳定种子并保存逐 trial 结果。

## 4. 当前正式数据与模型

当前生产候选空间：

- 1391 条当前 TPS；
- 694 条注册 MARTS 外部蛋白；
- 2085 条 canonical protein candidates；
- 513 个当前 Rhea reactions；
- 240 个注册 MARTS 外部 reactions；
- 3439 条去重 current + MARTS 训练关联。

表示与模型：

- 蛋白：ESM-C 600M mean embedding，1152 维；
- 反应：DRFP + precursor/product-skeleton 类别，2115 维；
- 模型：multi-positive dual tower，256 维共享空间，三 seed ensemble；
- 训练：MARTS domain adaptation；
- false-negative 处理：同一 50% protein cluster 或 reaction cluster 的 unlabeled candidates 从 contrastive denominator 中移除，已知阳性始终保留。

## 5. 当前正式 zero-shot 结果

正式评测采用 5 个 protein-cluster folds × 5 个 reaction-cluster folds 的 25-cell Cartesian double-cold。每个 external positive pair 只评估一次。

Reaction → enzyme：

| 模型 | Hit@3 | Hit@10 | Hit@20 | MRR | 中位最佳正例排名 |
|---|---:|---:|---:|---:|---:|
| current production | 1.7% | 4.2% | 7.2% | 0.021 | 374 |
| MARTS adaptation | 3.8% | 11.4% | 18.1% | 0.044 | 158 |
| shared MARTS + PU | 3.4% | 12.7% | 18.1% | 0.046 | 149 |
| Top3/10 specialized | 4.6% | 12.7% | 17.7% | 0.047 | 158 |

当前路由：

- external reaction Top3/10：R2E loss weight 0.75 direct model；
- external reaction Top20：shared MARTS + PU direct model。

严格 external 下相对 current production：Top3 从 1.7% 到 4.6%，Top10 从 4.2% 到 12.7%，Top20 从 7.2% 到 18.1%。这些数字不能与旧 39.6%/45.2% 直接比较，因为候选、反应和冷启动条件不同。

## 6. 当前 few-shot 结论

同源家族补全时，ESM-C 或 3-mer seed similarity 仍是最强信号。跨 50% identity cluster 时，旧 78%–93% 不能作为预期。

对 external MARTS reaction，seed ESM-C ranking 的结果为：

| seed 数 | Hit@3 | Hit@10 | Hit@20 |
|---:|---:|---:|---:|
| 1 | 38.9% | 50.9% | 54.3% |
| 2 | 46.0% | 60.4% | 66.9% |
| 3 | 41.2% | 59.3% | 70.7% |

这组是 external open-world few-shot，但不是 seed/hidden protein-cluster-cold，二者应分开报告。

## 7. CAGE 的当前定位

旧结论“raw CAGE 不能作为主排序器”继续成立。进一步审计发现 CAGE sigmoid 分数大量饱和或近似并列，直接 probability fusion 会制造错误排序。

当前生产模型不依赖 CAGE。CAGE 只保留为：

- 可选结构证据；
- disagreement rescue；
- 非经典 TPS 人工复核。

旧 RF/HGB CAGE-aware rescue 尚未在严格 25-cell double-cold、raw logit、tie-aware 和 nested method selection 条件下重建，因此不部署。

## 8. UniProt 扩展与受控 rescue

五个 TPS 相关 Pfam 域经过长度/片段过滤、exact-sequence 去重、移除 current/MARTS、50% identity clustering 和证据分级后，得到 5672 条 named A–D candidates；822 条 domain-only candidates 暂不启用。

自由合并会使 strict external Hit@3/10/20 从 4.6%/12.7%/18.1% 降到 2.5%/6.3%/10.5%，只保留 54.5%/50.0%/58.1% 的原命中，因此淘汰。

部署策略：

- Top3：3 canonical + 0 UniProt；
- Top10：9 canonical + 1 UniProt tail slot；
- Top20：18 canonical + 2 UniProt tail slots。

对应原命中保留率为 100%、93.3%、97.7%。UniProt 不允许改变 canonical prefix。

## 9. Reaction-specific Pfam architecture contract

旧按 terpene carbon count 和粗粒度 domain family 判断兼容性的规则会混淆 PF13243-only class-II cyclase、PF13249-only fragment、PF13243+PF13249 complete OSC、PF00348 prenyltransferase 和 PF00494 family。

新版用 known-positive accession、exact sequence 和高覆盖 MMseqs nearest neighbor 建立 reaction-specific architecture contract：

- 240 个注册反应；
- 208 个支持五-Pfam rescue；
- 32 个不支持或无法可靠解析，保持 canonical-only；
- complete OSC 必须含 PF13243+PF13249；
- 单域片段不进入 rescue。

## 10. 湿实验执行

最终执行包含 4 块 canonical discovery plates 和 2 块 UniProt rescue plates，共 576 wells、480 protein assay wells、29 个不同反应、349 个候选 ID 和 348 条 exact-sequence-deduplicated constructs。

分板先使用 exact-capacity MILP，再在每个 reaction block 内做 Hungarian candidate-position randomization。

MILP 后：

- canonical class-II 从 3/1/0/0 变为 1/1/1/1；
- canonical 平均候选长度板间 range 从 178.2 aa 降到 26.3 aa；
- rescue B/C/D evidence counts 两板完全一致；
- rescue bacterial class-I 从 18/28 变为 23/23；
- rescue plant TPS full 从 10/16 变为 13/13；
- rescue complete OSC 从 20/4 变为 12/12；
- rescue 平均候选长度板间 range 从 100.1 aa 降到 1.5 aa。

孔位随机化后：

- mean role-slot entropy：0.201 → 0.974；
- maximum single-slot role share：100% → 33.3%；
- maximum role slot-count range：24 → 1；
- control/blank moved：0。

canonical 与 rescue 可以合并采购，但必须独立 QC。

## 11. 当前结论

保留的旧结论：

- similarity 是强 baseline；
- seed sequence 对近同源扩展有效；
- raw CAGE 不适合作为主排序器；
- 高召回 reservoir 与最终 TopK 应分开。

被修正的旧结论：

- 旧 zero-shot 39.6%/45.2% 不是严格开放发现性能；
- 旧 few-shot 78%–93% 主要对应同源家族补全；
- 旧 CAGE-aware meta-ranker 当前缺少完整可复现产物。

当前正式叙事：

```text
旧版证明相似度和 seed expansion 在现有数据库内有效；
新版把评测改为真正的 protein-cluster × reaction-cluster 双冷开放发现，
用 MARTS-adapted ESM-C/DRFP 双塔替代 gate + CAGE meta-ranker，
再以受控 UniProt rescue、Pfam architecture contract、MILP 分板和
Hungarian 孔位随机化完成湿实验落地。
```

## 12. 仍未解决的问题

- strict external R2E Top10/20 仍只有 12.7%/18.1%；
- UniProt stress test 只证明不破坏 known-positive ranking，不证明真实活性率；
- CAGE learned rescue 尚未在严格双冷下重建；
- R2E Top3/10 reliability calibrator 未通过时必须明确拒绝伪造 confidence；
- 最终价值仍需要真实湿实验阳性率、候选来源分层 hit rate 和下一轮反馈验证。
