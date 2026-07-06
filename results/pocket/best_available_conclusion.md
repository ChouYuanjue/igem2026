# 最佳可行 pocket 结果矩阵

## 执行摘要

- 我们使用当前已经拿到的官方资产、单独下载的 AlphaFold 结构，以及已经跑通的 P2Rank 流程，做了最大可行的 pocket exploration。
- 这不是完整论文复现；官方公开配置引用了缺失的预计算路径，所以我们把重点放在真实可复现的 derived smallset 上。
- 当前真正完成的最大规模是 `enzyme405_100`。
- 当前 completed baseline 在 Top-5 / Top-10 上基本并列；表中按排序给出一个代表性 baseline `enzyme405_100 / official_precomputed_pocket`，Top-5=0.5858585858585859，Top-10=0.7070707070707071。

## 数据集规模

- `enzyme405_50`: n_reactions=50, n_valid_reactions=50, n_pairs=1675, n_positive_pairs=86, completed=6, blocked=3, failed=1
- `enzyme405_100`: n_reactions=100, n_valid_reactions=99, n_pairs=3249, n_positive_pairs=154, completed=1, blocked=0, failed=0
- `enzyme405_all_feasible`: n_reactions=524, n_valid_reactions=519, n_pairs=15147, n_positive_pairs=894, completed=0, blocked=0, failed=0

## 主结果表

| dataset_scale | baseline | pocket_source | pocket_selection | aggregation | status | n_reactions | n_valid_reactions | n_pairs | n_unique_enzymes | n_positive_pairs | n_pockets | top5_success | top10_success | delta_top5_vs_official | delta_top10_vs_official | mean_rank_shift_vs_official | median_rank_shift_vs_official | mean_score_shift_vs_official | best_pocket_rank_gt1_rate | blocked_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| enzyme405_50 | official_precomputed_pocket | official_precomputed | top1 | max | completed | 50 | 50 | 1675 | 1556 | 86 | 1550 | 0.54 | 0.68 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |
| enzyme405_50 | p2rank_top1 | p2rank | top1 | max | completed | 50 | 50 | 1675 | 1556 | 86 | 1548 | 0.54 | 0.68 | 0.0 | 0.0 | 0.030612244897959183 | 0.0 | 0.00031233038125946116 | 0.0 |  |
| enzyme405_50 | p2rank_topk_max | p2rank | topk | max | completed | 50 | 50 | 1675 | 1556 | 86 | 6221 | 0.54 | 0.68 | 0.0 | 0.0 | 0.030612244897959183 | 0.0 | 0.0008355983215491248 | 0.524609843937575 |  |
| enzyme405_50 | p2rank_topk_mean | p2rank | topk | mean | completed | 50 | 50 | 1675 | 1556 | 86 | 6221 | 0.54 | 0.68 | 0.0 | 0.0 | 0.030612244897959183 | 0.0 | 0.00037227173600707487 | 0.524609843937575 |  |
| enzyme405_50 | p2rank_topk_rank_weighted | p2rank | topk | rank_weighted | completed | 50 | 50 | 1675 | 1556 | 86 | 6221 | 0.54 | 0.68 | 0.0 | 0.0 | 0.030612244897959183 | 0.0 | 0.0003594141392546186 | 0.524609843937575 |  |
| enzyme405_50 | p2rank_topk_softmax_pool | p2rank | topk | softmax_pool | completed | 50 | 50 | 1675 | 1556 | 86 | 6221 | 0.54 | 0.68 | 0.0 | 0.0 | 0.030612244897959183 | 0.0 | 0.00041042352633316014 | 0.524609843937575 |  |
| enzyme405_50 | fpocket_top1 | fpocket | top1 | max | blocked_fpocket_missing | 50 | 50 | 1675 | 1556 | 86 | NA | NA | NA | NA | NA | NA | NA | NA | NA | blocked_fpocket_missing |
| enzyme405_50 | fpocket_topk_rank_weighted | fpocket | topk | rank_weighted | failed | 50 | 50 | 1675 | 1556 | 86 | 428 | NA | NA | NA | NA | NA | NA | NA | NA | inference |
| enzyme405_50 | p2rank_fpocket_union_max | p2rank+fpocket | union_topk | max | blocked_missing_full_structure_for_p2rank | 50 | 50 | 1675 | 1556 | 86 | 6649 | NA | NA | NA | NA | NA | NA | NA | NA | blocked_missing_full_structure_for_p2rank |
| enzyme405_50 | p2rank_fpocket_union_source_weighted | p2rank+fpocket | union_topk | source_weighted | blocked_missing_full_structure_for_p2rank | 50 | 50 | 1675 | 1556 | 86 | 6649 | NA | NA | NA | NA | NA | NA | NA | NA | blocked_missing_full_structure_for_p2rank |
| enzyme405_100 | official_precomputed_pocket | official_precomputed | top1 | max | completed | 100 | 99 | 3249 | 2807 | 154 | 2796 | 0.5858585858585859 | 0.7070707070707071 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |
| enzyme405_100 | p2rank_top1 | p2rank | top1 | max | not_run | 100 | 99 | 3249 | 2807 | 154 | NA | NA | NA | NA | NA | NA | NA | NA | NA | resource_limited |
| enzyme405_100 | p2rank_topk_max | p2rank | topk | max | not_run | 100 | 99 | 3249 | 2807 | 154 | NA | NA | NA | NA | NA | NA | NA | NA | NA | resource_limited |
| enzyme405_100 | p2rank_topk_mean | p2rank | topk | mean | not_run | 100 | 99 | 3249 | 2807 | 154 | NA | NA | NA | NA | NA | NA | NA | NA | NA | resource_limited |
| enzyme405_100 | p2rank_topk_rank_weighted | p2rank | topk | rank_weighted | not_run | 100 | 99 | 3249 | 2807 | 154 | NA | NA | NA | NA | NA | NA | NA | NA | NA | resource_limited |
| enzyme405_100 | p2rank_topk_softmax_pool | p2rank | topk | softmax_pool | not_run | 100 | 99 | 3249 | 2807 | 154 | NA | NA | NA | NA | NA | NA | NA | NA | NA | resource_limited |
| enzyme405_100 | fpocket_top1 | fpocket | top1 | max | not_run | 100 | 99 | 3249 | 2807 | 154 | NA | NA | NA | NA | NA | NA | NA | NA | NA | blocked_fpocket_missing |
| enzyme405_100 | fpocket_topk_rank_weighted | fpocket | topk | rank_weighted | not_run | 100 | 99 | 3249 | 2807 | 154 | NA | NA | NA | NA | NA | NA | NA | NA | NA | blocked_fpocket_missing |
| enzyme405_100 | p2rank_fpocket_union_max | p2rank+fpocket | union_topk | max | not_run | 100 | 99 | 3249 | 2807 | 154 | NA | NA | NA | NA | NA | NA | NA | NA | NA | blocked_fpocket_missing |
| enzyme405_100 | p2rank_fpocket_union_source_weighted | p2rank+fpocket | union_topk | source_weighted | not_run | 100 | 99 | 3249 | 2807 | 154 | NA | NA | NA | NA | NA | NA | NA | NA | NA | blocked_fpocket_missing |
| enzyme405_all_feasible | official_precomputed_pocket | official_precomputed | top1 | max | resource_limited | 524 | 519 | 15147 | 8201 | 894 | 8137 | NA | NA | NA | NA | NA | NA | NA | NA | resource_limited |
| enzyme405_all_feasible | p2rank_top1 | p2rank | top1 | max | not_run | 524 | 519 | 15147 | 8201 | 894 | NA | NA | NA | NA | NA | NA | NA | NA | NA | resource_limited |
| enzyme405_all_feasible | p2rank_topk_max | p2rank | topk | max | not_run | 524 | 519 | 15147 | 8201 | 894 | NA | NA | NA | NA | NA | NA | NA | NA | NA | resource_limited |
| enzyme405_all_feasible | p2rank_topk_mean | p2rank | topk | mean | not_run | 524 | 519 | 15147 | 8201 | 894 | NA | NA | NA | NA | NA | NA | NA | NA | NA | resource_limited |
| enzyme405_all_feasible | p2rank_topk_rank_weighted | p2rank | topk | rank_weighted | not_run | 524 | 519 | 15147 | 8201 | 894 | NA | NA | NA | NA | NA | NA | NA | NA | NA | resource_limited |
| enzyme405_all_feasible | p2rank_topk_softmax_pool | p2rank | topk | softmax_pool | not_run | 524 | 519 | 15147 | 8201 | 894 | NA | NA | NA | NA | NA | NA | NA | NA | NA | resource_limited |
| enzyme405_all_feasible | fpocket_top1 | fpocket | top1 | max | not_run | 524 | 519 | 15147 | 8201 | 894 | NA | NA | NA | NA | NA | NA | NA | NA | NA | blocked_fpocket_missing |
| enzyme405_all_feasible | fpocket_topk_rank_weighted | fpocket | topk | rank_weighted | not_run | 524 | 519 | 15147 | 8201 | 894 | NA | NA | NA | NA | NA | NA | NA | NA | NA | blocked_fpocket_missing |
| enzyme405_all_feasible | p2rank_fpocket_union_max | p2rank+fpocket | union_topk | max | not_run | 524 | 519 | 15147 | 8201 | 894 | NA | NA | NA | NA | NA | NA | NA | NA | NA | blocked_fpocket_missing |
| enzyme405_all_feasible | p2rank_fpocket_union_source_weighted | p2rank+fpocket | union_topk | source_weighted | not_run | 524 | 519 | 15147 | 8201 | 894 | NA | NA | NA | NA | NA | NA | NA | NA | NA | blocked_fpocket_missing |


## 关键结论

- `official_precomputed_pocket` 的 Top-5/Top-10 分别是 `0.5858585858585859` / `0.7070707070707071`；`p2rank_top1` 分别是 `NA` / `NA`。
- 在当前可用数据上，P2Rank 没有明显落后于官方 precomputed pocket。
- `p2rank_topk_softmax_pool` 的 Top-5/Top-10 分别是 `NA` / `NA`；相对 `p2rank_top1` 的变化是 Top-5 `NA`、Top-10 `NA`。
- `max`、`mean`、`rank_weighted`、`softmax_pool` 之间的差异可直接从表格中比较；如果它们的 Top-5 / Top-10 基本一致，说明聚合策略在当前数据切片上的影响有限。

## 与论文的对照

- 论文报告的是完整 Enzyme-405 基准上的更高性能；我们这里没有做完整复现，也没有把缺失的预计算特征伪造出来。
- 这是基于公开资产、重新生成输入、以及真实 pocket intervention 得到的 best available reconstruction。
- 结果差异不能直接解释成论文错误，更合理的理解是：公开资产和公开配置本身就限制了可复现上限。

## 局限性

- 不是完整官方 benchmark 复现。
- 官方 config 仍然引用缺失的预计算路径。
- AlphaFold 结构是单独下载的。
- fpocket 目前仍然可能 blocked。
- candidate / feature reconstruction 可能与论文实现存在差异。

## 下一步

- 如果要继续推进，优先补齐更大规模的 Enzyme-405 slice。
- 如果能拿到作者提供的 inference-ready feature bundle，结果会更接近论文设置。
- 下一步最值得加的是 catalytic-residue-aware pocket prior。
