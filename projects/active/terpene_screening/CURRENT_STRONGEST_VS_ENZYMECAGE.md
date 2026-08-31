# Clean-only strongest system vs EnzymeCAGE

This snapshot excludes retention, association-visible, or otherwise exposure-driven results from clean generalization claims. `R2E` = reaction -> enzyme; `E2R` = enzyme -> reaction. Repository metric definitions are authoritative: Hit/SR@K, MRR, MAP, NDCG and AUROC must be interpreted under an explicitly fixed candidate support and information boundary, and a single Hit@K number is not sufficient to promote a method.

## Evidence hierarchy for EnzymeCAGE

1. **Primary reproducible comparison:** a locally executed EnzymeCAGE run on a support that can be reconstructed exactly, compared with frozen Catalyst on that exact same support and denominator.
2. **Secondary local evidence:** real EnzymeCAGE score unions or partial reconstructions whose support differs from the current Catalyst benchmark; useful for capability/provenance, but not for raw cross-support deltas.
3. **Paper-reported values:** context for the original paper only. They are explicitly not a local rerun and must not be described as “our reproduced EnzymeCAGE result.”
4. Missing CAGE metrics are N/A; they are never imputed from Catalyst, from unsupported pairs, or from paper-level aggregate numbers.

## Clean / reproducible comparison matrix

| Scenario | Direction | Strongest defensible Catalyst result | Reproducible EnzymeCAGE evidence | Interpretation |
|---|---|---|---|---|
| Enzyme-405 **same reconstructed 100-reaction support** (99 valid queries) | R2E | SR@5 **60.61%**, SR@10 69.70%; MRR **0.5283**, MAP **0.5255**, AUROC **0.6566**, NDCG@10 **0.5508** | Local `enzyme405_100 / official_precomputed_pocket`: SR@5 58.59%, SR@10 **70.71%** | **Essentially matched on the retained same-support CAGE metrics:** Catalyst +2.02 pp at Top-5, CAGE +1.01 pp at Top-10. Catalyst scoring is post-reveal descriptive only. CAGE IR metrics are unavailable because the old raw prediction rows are not retained. |
| Enzyme-405 full official reservoir | R2E | SR@10 **50.17%**, SR@1 14.24%, SR@3 26.78%, SR@5 37.63%, EF@1% **31.62**, DCG@10 **0.3836**; MRR **0.2563**, MAP **0.2499**, AUROC **0.6733**, NDCG@10 **0.2799** | Complete author-equivalent local rerun: **N/A**. Paper values are context only (SR@10 57.97%, EF@1% 36.6031, DCG@10 0.4523). | Catalyst result is a valid frozen cleanroom reveal; the paper table alone does not establish our reproducible CAGE-vs-Catalyst ordering. |
| TPS native CAGE scored support | R2E | Cross-support Catalyst values are not used here | Real local CAGE union: 159,815 scored pairs, 86.22% positive-pair coverage; common-support MRR **0.0510**, MAP **0.0476**, AUROC **0.4712**, NDCG@10 **0.0513**, Hit@10 **12.35%** | Genuine local EnzymeCAGE scoring, but not Enzyme-405 or the full general universe; retain as secondary baseline evidence. |
| ReactZyme-projected strict double-cold, full candidate universe | R2E | Frozen novelty expert: Hit@10 **3.34%**, Hit@50 **8.51%**, MRR **0.00968**, MAP **0.00965**, AUROC **0.8801**, NDCG@10 **0.00944** | N/A | Clean R2E improvement; extreme reaction novelty remains difficult. |
| ReactZyme-projected strict double-cold, full candidate universe | E2R | Hit@10 **11.43%**, Hit@50 **25.64%**, MRR **0.04737**, AUROC **0.8330** | N/A | E2R is not an EnzymeCAGE-native retrieval direction. |
| Post-2020 temporal double-cold, full candidate universe | R2E | Frozen novelty expert: Hit@10 **4.32%**, Hit@50 **10.62%**, MRR **0.01859**, MAP **0.01669**, AUROC **0.9184** | N/A | Creation-date temporal extrapolation; already revealed and confirmatory only. |
| Post-2020 temporal double-cold, full candidate universe | E2R | Hit@10 **8.35%**, Hit@50 **19.01%**, MRR **0.03046**, AUROC **0.8558** | N/A | Same temporal caveat; no matching CAGE E2R protocol. |
| ReactZyme-projected protein-cold, train-seen reactions | R2E | Hit@10 **4.77%**, MRR **0.01923**, AUROC **0.8336** | N/A | Test proteins are entity-unseen; R2E remains weak. |
| ReactZyme-projected protein-cold, train-seen reactions | E2R | Hit@1 **60.99%**, Hit@10 **79.90%**, MRR **0.6771**, AUROC **0.9865** | N/A | Strong aggregate transfer; low-identity/no-hit slices must still be reported separately. |
| Deterministic broad reaction-cold, train-seen proteins | R2E | Frozen novelty expert: Hit@10 **13.58%**, Hit@50 **26.61%**, MRR **0.07023**, MAP **0.05360**, AUROC **0.9199** | N/A | Extreme `<0.3` reaction-similarity remains the main unresolved R2E regime. |
| Deterministic broad reaction-cold, train-seen proteins | E2R | Hit@10 **24.60%**, Hit@50 **43.22%**, MRR **0.08743**, AUROC **0.9317** | N/A | Secondary generalization evidence; no matching CAGE full-universe E2R protocol. |

## Top-2000 reranker / router: current internal confirmation

The original unrestricted pair-head V1 regressed badly and is rejected. The bounded residual V2 froze `residual_scale=0.03` using only internal folds 0/1/2, then passed a previously unseen fold3 confirmation. On fold3, coarse -> reranked changed MRR **0.10805 -> 0.11384**, MAP **0.08561 -> 0.09117**, NDCG@10 **0.10784 -> 0.11868**, Hit@10 **24.64% -> 27.68%**, while Hit@20 **33.36% -> 33.65%** and Hit@50 **41.42% -> 41.52%** did not regress; median best-positive rank improved **130 -> 105**.

A label-free difficulty router then froze the rule `use residual 0.03 iff max_train_drfp_tanimoto >= 0.9; otherwise exact coarse fallback` and passed a fresh salted strict double-cold fold6. There, MRR **0.10205 -> 0.11484**, MAP **0.08246 -> 0.09764**, NDCG@10 **0.10805 -> 0.12283**, Hit@10 **24.19% -> 25.95%**, Hit@20 **31.23% -> 33.43%**, Hit@50 **41.64% -> 44.43%**, and median best-positive rank **137 -> 90.5**. The `<0.3` reaction-similarity regime still receives the exact coarse fallback and remains a capability gap rather than being hidden by aggregate gains.

These are internal clean confirmations, not authorization to reuse an already revealed outer set. A future unbiased external claim requires a genuinely fresh frozen benchmark or a separately preregistered evaluation path.

## Current improvement target

The consistent bottleneck remains **R2E under extreme reaction novelty**, especially `<0.3` train-reaction similarity. Difficulty/capability reporting must keep the hard slices and their sample sizes visible rather than using aggregate MRR alone. Enzyme-405 and all already revealed outer cells remain confirmatory and must not be used for iterative model selection.
