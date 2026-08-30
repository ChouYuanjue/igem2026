# Clean-only strongest system vs EnzymeCAGE

This snapshot intentionally excludes retention, common-reservoir, association-visible, or otherwise exposure-driven results from the main comparison. Those results may remain as internal production diagnostics but must not be cited as evidence of generalization.

`R2E` = reaction -> enzyme. `E2R` = enzyme -> reaction.

## Clean comparison matrix

| Scenario | Direction | Strongest defensible Catalyst result | EnzymeCAGE | Interpretation |
|---|---|---|---|---|
| Enzyme-405 confirmatory benchmark | R2E | **SR@10 50.17%**, SR@1 **14.24%**, SR@3 26.78%, SR@5 37.63%, EF@1% **31.62**, DCG@10 **0.3836**; MRR **0.2563**, MAP **0.2499**, AUROC **0.6733**, NDCG@10 **0.2799** | **Paper-reported:** SR@10 **57.97%**, EF@1% **36.6031**, DCG@10 **0.4523** | **EnzymeCAGE still leads.** Catalyst recipe was selected only on three strict clean2023 double-cold folds, frozen in Git before Enzyme-405 reveal, then retrained on all **218,537** clean2023 pairs. Enzyme-405 labels were never used for model selection. |
| ReactZyme-projected strict double-cold, full candidate universe | R2E | Frozen novelty expert: Hit@10 **3.34%**, Hit@50 **8.51%**, MRR **0.00968**, MAP **0.00965**, AUROC **0.8801**, NDCG@10 **0.00944** | N/A | The expert recipe was frozen before outer reveal and materially improves clean R2E. Extreme reaction novelty remains difficult. |
| ReactZyme-projected strict double-cold, full candidate universe | E2R | Hit@10 **11.43%**, Hit@50 **25.64%**, MRR **0.04737**, AUROC **0.8330** | N/A | E2R is not an EnzymeCAGE-native retrieval direction. |
| Post-2020 temporal double-cold, full candidate universe | R2E | Frozen novelty expert: Hit@10 **4.32%**, Hit@50 **10.62%**, MRR **0.01859**, MAP **0.01669**, AUROC **0.9184** | N/A | Same pre-frozen expert improves most metrics; Hit@1 decreases slightly. Creation-date temporal extrapolation, not a strict historical snapshot. |
| Post-2020 temporal double-cold, full candidate universe | E2R | Hit@10 **8.35%**, Hit@50 **19.01%**, MRR **0.03046**, AUROC **0.8558** | N/A | Same temporal caveat; CAGE has no matching E2R protocol. |
| ReactZyme-projected protein-cold, train-seen reactions | R2E | Hit@10 **4.77%**, MRR **0.01923**, AUROC **0.8336** | N/A | Test proteins are entity-unseen. R2E remains weak. |
| ReactZyme-projected protein-cold, train-seen reactions | E2R | Hit@1 **60.99%**, Hit@10 **79.90%**, MRR **0.6771**, AUROC **0.9865** | N/A | Strong protein-to-reaction transfer, but low-identity/no-hit slices are reported separately. |
| Deterministic broad reaction-cold, train-seen proteins | R2E | Frozen novelty expert: Hit@10 **13.58%**, Hit@50 **26.61%**, MRR **0.07023**, MAP **0.05360**, AUROC **0.9199** | N/A | Strong clean gain under reaction cold. The extreme `<0.3` reaction-similarity regime remains the primary unresolved R2E failure mode. |
| Deterministic broad reaction-cold, train-seen proteins | E2R | Hit@10 **24.60%**, Hit@50 **43.22%**, MRR **0.08743**, AUROC **0.9317** | N/A | Useful secondary generalization evidence; CAGE has no matching E2R/full-universe protocol. |

## Evidence hierarchy

1. **Primary external comparison:** Enzyme-405. It currently favors EnzymeCAGE and remains a meaningful pressure test.
2. **Primary generalization evidence:** strict ReactZyme-projected double-cold plus post-2020 temporal double-cold.
3. **Factorized cold-start evidence:** protein-cold and reaction-cold, including low-similarity/low-identity slices rather than only aggregate metrics.
4. **Excluded from comparative claims:** old TPS common-reservoir, current known-relation retention, models trained on association graphs containing evaluation pairs, and any result whose advantage is plausibly explained by direct relation exposure.

## Current improvement target

The consistent bottleneck is **R2E under extreme reaction novelty**, especially the `<0.3` train-reaction-similarity slice. A train-only novelty continuation expert now improves aggregate R2E on strict double-cold, temporal double-cold, and reaction-cold clean protocols, but the first hard-threshold router is not robust across protocols and is not treated as a final system. Development is selected only on leakage-clean internal splits. Enzyme-405 and all outer cold-start cells are confirmatory and must not be used for iterative hyperparameter selection.

EnzymeCAGE reproduction is considered sufficient for comparison. Missing author-side precomputed features will not justify indefinite reproduction effort; paper-reported metrics remain explicitly marked as such.
