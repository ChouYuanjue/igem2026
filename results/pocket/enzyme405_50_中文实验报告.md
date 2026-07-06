# EnzymeCAGE Pocket Baseline 实验报告：Enzyme-405 50-reaction slice

## 1. 实验目的

- 本实验不是完整复现论文 full benchmark。
- 目标是在公开可用资产和当前可运行条件下，比较不同 pocket 方案对 EnzymeCAGE 排序结果的影响。
- 统一使用 Enzyme-405 的 50 reaction slice。
- 只看 Top-5 和 Top-10。

## 2. 数据与规模

- n_reactions: `50`
- n_valid_reactions: `50`
- n_pairs: `1675`
- n_positive_pairs: `86`
- n_unique_enzymes: `1556`
- 每个 reaction 平均 candidate 数: `33.38`
- 是否有 reaction 无 positive: `0`
- 是否存在少于 10 candidates 的 reaction: `17` 个 reaction

## 3. Baseline 设计

- `official_precomputed_pocket`：官方预抽取 pocket，作为 anchor。
- `p2rank_top1`：从 AlphaFold full structure 重新抽取 P2Rank top1 pocket。
- `p2rank_topk_max`：P2Rank top5 pockets，取最大 CAGE score。
- `p2rank_topk_mean`：P2Rank top5 pockets，取均值。
- `p2rank_topk_rank_weighted`：按 P2Rank rank 加权。
- `p2rank_topk_softmax_pool`：按 CAGE score softmax pooling。
- `fpocket_top1`：几何 pocket detector 的 top1 pocket。
- `fpocket_topk_rank_weighted`：fpocket top5 pockets 的 rank-weighted 版本。
- `p2rank_fpocket_union_max`：P2Rank + fpocket pocket hypotheses 的并集后取最大分数。
- `p2rank_fpocket_union_source_weighted`：按 source weight 融合的并集版本。

## 4. 结果总表

| dataset_scale | baseline | pocket_source | pocket_selection | aggregation | status | n_reactions | n_valid_reactions | n_pairs | n_unique_enzymes | n_positive_pairs | n_pockets | top5_success | top10_success | delta_top5_vs_official | delta_top10_vs_official | mean_rank_shift_vs_official | median_rank_shift_vs_official | mean_score_shift_vs_official | best_pocket_rank_gt1_rate | blocked_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| enzyme405_50 | official_precomputed_pocket | official_precomputed | top1 | max | completed | 50 | 50 | 1675 | 1556 | 86 | 1550 | 0.54 | 0.68 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |
| enzyme405_50 | p2rank_top1 | p2rank | top1 | max | completed | 50 | 50 | 1675 | 1556 | 86 | 1548 | 0.54 | 0.68 | 0.0 | 0.0 | 0.030612244897959183 | 0.0 | 0.00031233038125946116 | 0.0 |  |
| enzyme405_50 | p2rank_topk_max | p2rank | topk | max | completed | 50 | 50 | 1675 | 1556 | 86 | 6221 | 0.54 | 0.68 | 0.0 | 0.0 | 0.030612244897959183 | 0.0 | 0.0008355983215491248 | 0.524609843937575 |  |
| enzyme405_50 | p2rank_topk_mean | p2rank | topk | mean | completed | 50 | 50 | 1675 | 1556 | 86 | 6221 | 0.54 | 0.68 | 0.0 | 0.0 | 0.030612244897959183 | 0.0 | 0.00037227173600707487 | 0.524609843937575 |  |
| enzyme405_50 | p2rank_topk_rank_weighted | p2rank | topk | rank_weighted | completed | 50 | 50 | 1675 | 1556 | 86 | 6221 | 0.54 | 0.68 | 0.0 | 0.0 | 0.030612244897959183 | 0.0 | 0.0003594141392546186 | 0.524609843937575 |  |
| enzyme405_50 | p2rank_topk_softmax_pool | p2rank | topk | softmax_pool | completed | 50 | 50 | 1675 | 1556 | 86 | 6221 | 0.54 | 0.68 | 0.0 | 0.0 | 0.030612244897959183 | 0.0 | 0.00041042352633316014 | 0.524609843937575 |  |
| enzyme405_50 | fpocket_top1 | fpocket | top1 | max | completed | 50 | 50 | 1675 | 1556 | 86 | 86 | 0.5 | 0.5 | -0.040000000000000036 | -0.18000000000000005 | 15.666666666666666 | 16.0 | 0.00525740139683673 | 0.0 |  |
| enzyme405_50 | fpocket_topk_rank_weighted | fpocket | topk | rank_weighted | completed | 50 | 50 | 1675 | 1556 | 86 | 428 | 0.5 | 0.5 | -0.040000000000000036 | -0.18000000000000005 | 15.703703703703704 | 16.0 | 0.0005949190104807719 | 0.6896551724137931 |  |
| enzyme405_50 | p2rank_fpocket_union_max | p2rank+fpocket | union_topk | max | completed | 50 | 50 | 1675 | 1556 | 86 | 6649 | 0.5 | 0.5 | -0.040000000000000036 | -0.18000000000000005 | 15.839506172839506 | 18.0 | 0.006191143206005459 | 0.4367816091954023 |  |
| enzyme405_50 | p2rank_fpocket_union_source_weighted | p2rank+fpocket | union_topk | source_weighted | completed | 50 | 50 | 1675 | 1556 | 86 | 6649 | 0.5 | 0.5 | -0.040000000000000036 | -0.18000000000000005 | 15.814814814814815 | 18.0 | 0.004219665425846192 | 0.4827586206896552 |  |

## 5. Top-5 / Top-10 结果解读

- `official_precomputed_pocket` 的 Top-5/Top-10 分别是 `0.54` / `0.68`。
- `p2rank_top1` 的 Top-5/Top-10 分别是 `0.54` / `0.68`。
- P2Rank top1 与 official_precomputed_pocket 在当前 slice 上几乎一致，没有观察到 Top-5 / Top-10 差异。
- P2Rank top-k 聚合里当前最好的 baseline 是 `p2rank_topk_max`，相对 `p2rank_top1` 的变化为 Top-5 `0`、Top-10 `0`。
- `best_pocket_rank > 1` 比例约为 `0.5246`，说明 pocket localization uncertainty 确实存在。
- 但 naive multi-pocket aggregation 没有把这种不确定性转化为检索命中率收益。
- fpocket 最好的完成结果是 `fpocket_top1`，相对 `p2rank_top1` 的变化为 Top-5 `-0.04`、Top-10 `-0.18`。
- fpocket 没有带来可见提升。
- union baseline 的最优完成结果是 `p2rank_fpocket_union_max`，相对 `p2rank_top1` 的变化为 Top-5 `-0.04`、Top-10 `-0.18`。
- union 方案没有带来可见提升。

## 6. Pocket uncertainty 分析

- 当前较好的 P2Rank top-k baseline 中，`best_pocket_rank > 1` 的比例约为 `0.5246`。
- 这说明不少样本的最高 CAGE score pocket 并不是 rank1，pocket localization uncertainty 的确存在。
- 如果这种现象并没有带来 Top-5 / Top-10 的提升，更合理的解释是：单纯扩大 pocket 搜索空间并不能自动修复检索排序。

## 7. 与论文结果的关系

- 论文在完整 Enzyme-405 benchmark 上报告更高表现。
- 当前实验不是 full benchmark 复现，因为公开 config 里引用了缺失的预计算 feature 路径，我们没有把这些缺失路径伪造成可用结果。
- 当前结果是基于公开资产、derived 50-reaction slice、重新组织输入和真实 EnzymeCAGE inference 的 pocket exploration。
- 这些结果只能说明当前可复现条件下 pocket 选择策略的相对表现，不能据此判断论文错误。

## 8. 结论

当前最佳 completed baseline 是 `official_precomputed_pocket`，相对 `p2rank_top1` 的提升为 Top-5 `0`、Top-10 `0`。
这意味着 pocket 设计仍然有影响，但收益取决于具体 detector / aggregation 组合。

## 9. 局限性

- 只用了 50 reaction slice。
- 不是完整论文 benchmark。
- 可能存在 candidate pool / feature reconstruction 差异。
- AlphaFold structures 是单独下载的。

## 10. 下一步建议

1. 扩展到更大 Enzyme-405 slice。
2. 请求作者提供 inference-ready feature bundle。
3. 加入 catalytic-residue-aware pocket prior，而不是继续盲目 top-k aggregation。

## 11. union 校准补充

- 额外的 source-balanced union 基线已经单独整理在 [`results/pocket/enzyme405_50_union_calibration_addendum.md`](/home/runnel/igem-pocket/results/pocket/enzyme405_50_union_calibration_addendum.md)。
- 这组补充实验没有改变当前 union baseline 的 Top-5 / Top-10 结论。
- 更重要的是，当前可评估的 valid reactions 只有 2 个，所以任何微小差异都会被压缩成 0.5 步长的粗粒度变化。
