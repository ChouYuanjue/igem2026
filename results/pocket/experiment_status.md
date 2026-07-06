# Pocket Experiment Status

## Environment

- smoke_test: present
- smoke_test_path: results/pocket/env_smoke_test.txt

## Official Full Eval

- official_eval_enzyme405: status=failed_missing_referenced_path, failed_step=official_eval
- official_eval_orphan335: status=failed_missing_referenced_path, failed_step=official_eval

Official full eval is not the main result because the official configs reference missing precomputed feature/data paths. The showable results use derived smallsets and real EnzymeCAGE inference.

## Completed Baselines

| dataset_scale | baseline | status | top5_success | top10_success | n_pairs | n_pockets |
| --- | --- | --- | --- | --- | --- | --- |
| enzyme405_50 | official_precomputed_pocket | completed | 0.54 | 0.68 | 1675 | 1550 |
| enzyme405_50 | p2rank_top1 | completed | 0.54 | 0.68 | 1675 | 1548 |
| enzyme405_50 | p2rank_topk_max | completed | 0.54 | 0.68 | 1675 | 6221 |
| enzyme405_50 | p2rank_topk_mean | completed | 0.54 | 0.68 | 1675 | 6221 |
| enzyme405_50 | p2rank_topk_rank_weighted | completed | 0.54 | 0.68 | 1675 | 6221 |
| enzyme405_50 | p2rank_topk_softmax_pool | completed | 0.54 | 0.68 | 1675 | 6221 |
| enzyme405_50 | fpocket_top1 | completed | 0.5 | 0.5 | 1675 | 86 |
| enzyme405_50 | fpocket_topk_rank_weighted | completed | 0.5 | 0.5 | 1675 | 428 |
| enzyme405_50 | p2rank_fpocket_union_max | completed | 0.5 | 0.5 | 1675 | 6649 |
| enzyme405_50 | p2rank_fpocket_union_source_weighted | completed | 0.5 | 0.5 | 1675 | 6649 |

## Blocked Baselines

No rows.

## Failed Baselines

No rows.

## Key Result Files

- best_available_result_matrix_csv: results/pocket/best_available_result_matrix.csv
- best_available_result_matrix_md: results/pocket/best_available_result_matrix.md
- best_available_conclusion_md: results/pocket/best_available_conclusion.md
- comparison_report: results/pocket/comparison/comparison_report.md
- enzymecage_patch: results/pocket/patches/enzymecage_path_fixes.patch

## Next Actions

- Install fpocket and rerun geometry baselines.
- Expand smallset size.
- Add catalytic-residue-aware pocket prior.
