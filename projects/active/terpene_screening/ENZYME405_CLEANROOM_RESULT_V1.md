# Enzyme-405 cleanroom confirmatory v1

The model recipe was selected exclusively on clean2023 strict double-cold folds 0/1/2, frozen in Git at `d5e9ce6`, and only then retrained on all 218,537 clean2023 pairs. Enzyme-405 labels did not participate in model selection. The primary result below is the biological `neural_score` only; the evaluator's benchmark-construction novelty shortcut is explicitly excluded from comparative claims.

| Metric | Catalyst frozen cleanroom | 95% query bootstrap | EnzymeCAGE paper-reported |
|---|---:|---:|---:|
| SR@1 | 14.24% | 10.51–18.31% | — |
| SR@3 | 26.78% | 21.69–31.86% | — |
| SR@5 | 37.63% | 32.20–43.39% | — |
| SR@10 | **50.17%** | 44.41–55.93% | **57.97%** |
| DCG@10 | **0.3836** | 0.3274–0.4450 | **0.4523** |
| EF@1% | **31.62** | 26.60–36.74 | **36.6031** |
| MRR | 0.2563 | 0.2197–0.2946 | — |
| MAP | 0.2499 | 0.2158–0.2860 | — |
| AUROC | 0.6733 | 0.6396–0.7071 | — |
| NDCG@10 | 0.2799 | 0.2417–0.3199 | — |

EnzymeCAGE remains ahead on its reported SR@10 and DCG@10. The EF@1% gap is less decisive under query resampling because the paper point estimate lies near the upper end of Catalyst's bootstrap interval. These intervals describe query-to-query uncertainty for Catalyst; they are not a paired significance test against EnzymeCAGE because author per-query predictions are not available here.

The earlier Catalyst Enzyme-405 numbers were based on individual fold0 development checkpoints. They are no longer the primary result. The current value is methodologically stronger because the recipe was selected by internal multifold development and then retrained once on the full clean2023 set before external reveal.

`author_valid_pocket` robustness was not rerun because the locally required EnzymeCAGE ESM-node feature mapping is absent. The full official reservoir result is complete and unaffected by this missing optional author-side asset.
