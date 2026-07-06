# Enzyme-405 50-reaction slice conclusion

- 数据规模：50 reactions、1675 pairs、86 positive pairs、1556 unique enzymes。

## Baseline Table

| baseline | status | top5_success | top10_success | best_pocket_rank_gt1_rate |
| --- | --- | --- | --- | --- |
| fpocket_top1 | completed | 0.5 | 0.5 | 0.0 |
| fpocket_topk_rank_weighted | completed | 0.5 | 0.5 | 0.6896551724137931 |
| official_precomputed_pocket | completed | 0.54 | 0.68 | 0.0 |
| p2rank_fpocket_union_max | completed | 0.5 | 0.5 | 0.4367816091954023 |
| p2rank_fpocket_union_source_weighted | completed | 0.5 | 0.5 | 0.4827586206896552 |
| p2rank_top1 | completed | 0.54 | 0.68 | 0.0 |
| p2rank_topk_max | completed | 0.54 | 0.68 | 0.524609843937575 |
| p2rank_topk_mean | completed | 0.54 | 0.68 | 0.524609843937575 |
| p2rank_topk_rank_weighted | completed | 0.54 | 0.68 | 0.524609843937575 |
| p2rank_topk_softmax_pool | completed | 0.54 | 0.68 | 0.524609843937575 |

## Top-5 / Top-10

- official_precomputed_pocket vs p2rank_top1: Top-5 `0.54` vs `0.54`, Top-10 `0.68` vs `0.68`.
- p2rank_top1 vs p2rank_topk aggregations: best completed top-k baseline is `p2rank_topk_max` with Top-5 delta `+0.0000` and Top-10 delta `+0.0000` vs p2rank_top1.
- best_pocket_rank > 1 ratio: `0.524609843937575`.
