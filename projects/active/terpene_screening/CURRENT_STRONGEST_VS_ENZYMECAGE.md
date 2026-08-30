# Clean-only strongest system vs EnzymeCAGE

This snapshot intentionally excludes retention, common-reservoir, association-visible, or otherwise exposure-driven results from the main comparison. Those results may remain as internal production diagnostics but must not be cited as evidence of generalization.

`R2E` = reaction -> enzyme. `E2R` = enzyme -> reaction.

## Clean comparison matrix

| Scenario | Direction | Strongest defensible Catalyst result | EnzymeCAGE | Interpretation |
|---|---|---|---|---|
| Enzyme-405 confirmatory benchmark | R2E | **SR@10 50.51%**, SR@1 14.58%, EF@1% 30.74, DCG@10 0.3927; MRR 0.2597, MAP 0.2522 | **Paper-reported:** SR@10 **57.97%**, EF@1% **36.6031**, DCG@10 **0.4523** | **EnzymeCAGE currently leads.** Catalyst is trained from the leakage-controlled 2023 cleanroom source and Enzyme-405 labels are not used for checkpoint selection. |
| ReactZyme-projected strict double-cold, full candidate universe | R2E | Hit@10 **1.22%**, Hit@50 **8.21%**, MRR **0.00664**, AUROC **0.8688** | N/A | CAGE has no complete same-split/full-universe score matrix. This remains a major Catalyst weakness. |
| ReactZyme-projected strict double-cold, full candidate universe | E2R | Hit@10 **11.43%**, Hit@50 **25.64%**, MRR **0.04737**, AUROC **0.8330** | N/A | E2R is not an EnzymeCAGE-native retrieval direction. |
| Post-2020 temporal double-cold, full candidate universe | R2E | Hit@10 **3.09%**, Hit@50 **10.12%**, MRR **0.01685**, AUROC **0.9096** | N/A | Creation-date temporal extrapolation; not claimed as a strict historical database snapshot. |
| Post-2020 temporal double-cold, full candidate universe | E2R | Hit@10 **8.35%**, Hit@50 **19.01%**, MRR **0.03046**, AUROC **0.8558** | N/A | Same temporal caveat; CAGE has no matching E2R protocol. |
| ReactZyme-projected protein-cold, train-seen reactions | R2E | Hit@10 **4.77%**, MRR **0.01923**, AUROC **0.8336** | N/A | Test proteins are entity-unseen. R2E remains weak. |
| ReactZyme-projected protein-cold, train-seen reactions | E2R | Hit@1 **60.99%**, Hit@10 **79.90%**, MRR **0.6771**, AUROC **0.9865** | N/A | Strong protein-to-reaction transfer, but low-identity/no-hit slices are reported separately. |
| Deterministic broad reaction-cold, train-seen proteins | R2E | Hit@10 **10.22%**, Hit@50 **24.19%**, MRR **0.06085**, AUROC **0.9110** | N/A | For max train-reaction similarity `<0.3`, **Hit@10=0 and Hit@50=0**. This is a primary unresolved failure mode. |
| Deterministic broad reaction-cold, train-seen proteins | E2R | Hit@10 **24.60%**, Hit@50 **43.22%**, MRR **0.08743**, AUROC **0.9317** | N/A | Useful secondary generalization evidence; CAGE has no matching E2R/full-universe protocol. |

## Evidence hierarchy

1. **Primary external comparison:** Enzyme-405. It currently favors EnzymeCAGE and remains a meaningful pressure test.
2. **Primary generalization evidence:** strict ReactZyme-projected double-cold plus post-2020 temporal double-cold.
3. **Factorized cold-start evidence:** protein-cold and reaction-cold, including low-similarity/low-identity slices rather than only aggregate metrics.
4. **Excluded from comparative claims:** old TPS common-reservoir, current known-relation retention, models trained on association graphs containing evaluation pairs, and any result whose advantage is plausibly explained by direct relation exposure.

## Current improvement target

The consistent bottleneck is **R2E under reaction novelty**, especially strict double-cold and low-similarity unseen reactions. Development is selected only on nested leakage-clean splits. Enzyme-405, outer strict double-cold, temporal double-cold, and reaction-cold test results are confirmatory and must not be used for iterative hyperparameter selection.

EnzymeCAGE reproduction is considered sufficient for comparison. Missing author-side precomputed features will not justify indefinite reproduction effort; paper-reported metrics remain explicitly marked as such.
