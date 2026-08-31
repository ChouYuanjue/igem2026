# Enzyme-405 cleanroom confirmatory v1

The Catalyst recipe was selected exclusively on clean2023 strict double-cold folds 0/1/2, frozen in Git at `d5e9ce6`, and only then retrained on all 218,537 clean2023 pairs. Enzyme-405 labels did not participate in model selection. The primary biological result is `neural_score`; the benchmark-construction novelty shortcut is excluded from comparative claims.

## Full official Enzyme-405 Catalyst result

| Metric | Catalyst frozen cleanroom | 95% query bootstrap |
|---|---:|---:|
| SR@1 | 14.24% | 10.51–18.31% |
| SR@3 | 26.78% | 21.69–31.86% |
| SR@5 | 37.63% | 32.20–43.39% |
| SR@10 | **50.17%** | 44.41–55.93% |
| DCG@10 | **0.3836** | 0.3274–0.4450 |
| EF@1% | **31.62** | 26.60–36.74 |
| MRR | 0.2563 | 0.2197–0.2946 |
| MAP | 0.2499 | 0.2158–0.2860 |
| AUROC | 0.6733 | 0.6396–0.7071 |
| NDCG@10 | 0.2799 | 0.2417–0.3199 |

These are complete Catalyst results on the 295-query official reservoir. We do **not** claim a complete author-equivalent EnzymeCAGE rerun on this full reservoir: the author-side precomputed feature set required for such a claim is incomplete locally.

## Primary reproducible EnzymeCAGE comparison: same reconstructed support

The strongest locally completed EnzymeCAGE reconstruction retained in this repository is `enzyme405_100 / official_precomputed_pocket`. Its original input can be deterministically reconstructed from `data/external/enzymecage_current/Enzyme-405.csv` with the historical `build_official_smallset.py` algorithm (`seed=42`, 100 reactions, at most 50 enzymes/reaction). The reconstructed support exactly matches the historical run metadata: **100 reactions, 99 reactions with a positive, 3,249 pairs, 2,807 unique enzymes, 154 positive pairs**. Catalyst was scored once with the already frozen `rankstrong_r2e98` model; this post-reveal scoring is descriptive only and cannot affect model selection.

| Same-support metric (99 valid reactions) | Catalyst frozen | locally reproduced EnzymeCAGE | Catalyst − CAGE |
|---|---:|---:|---:|
| SR@5 | **60.61%** | 58.59% | **+2.02 pp** |
| SR@10 | 69.70% | **70.71%** | **−1.01 pp** |

On the reproducible common support, neither system has a practically large advantage from these two retained CAGE metrics: Catalyst is slightly higher at Top-5 and EnzymeCAGE slightly higher at Top-10. This is the comparison that should be used when discussing what we can currently reproduce.

Catalyst additionally has MRR **0.5283**, MAP **0.5255**, AUROC **0.6566**, NDCG@10 **0.5508**, Hit@20 **77.78%**, Hit@50 **100%**, and median best-positive rank **2** on the same 99-reaction support. The historical raw EnzymeCAGE prediction rows are not retained on `nju-server-06`, so corresponding CAGE MRR/MAP/AUROC/NDCG values are deliberately **not imputed**.

## Paper-reported EnzymeCAGE values: context only

The paper reports SR@10 **57.97%**, EF@1% **36.6031**, and DCG@10 **0.4523** on its full Enzyme-405 protocol. These values remain useful for understanding the paper, but they are not a local rerun and are not used to compute a reproducible Catalyst-vs-CAGE delta. In particular, the previous statement that “EnzymeCAGE remains ahead” based only on the paper table is no longer treated as our reproducible conclusion.

## Metric contract

Repository evaluation documents define Hit/SR@K as whether at least one known positive appears in the first K candidates, MRR from the reciprocal rank of the best positive, and MAP/NDCG/AUROC as complementary ranking-quality views. They are comparable only under the same candidate support, split/information boundary, and query denominator. For this reason the direct local CAGE comparison above uses only the two metrics for which the historical CAGE reconstruction and frozen Catalyst share the exact same 99-reaction denominator.

The earlier Catalyst Enzyme-405 numbers from individual fold-development checkpoints are not primary. The current full-official Catalyst result is methodologically stronger because the recipe was selected by internal multifold development and retrained once on the full clean2023 set before the external reveal. The same-support reconstruction is a separate post-reveal reproducibility audit and does not reopen selection.
