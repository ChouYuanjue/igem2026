# Current strongest system vs EnzymeCAGE (2026-08-30)

This snapshot reports the strongest currently defensible Catalyst component/routed result for each scenario. It is **not** a test-label oracle: routed cells use development-only activation guards; cold benchmark models use frozen nested-dev-selected training; Enzyme-405 is shown as a confirmatory observation and is not used to select a final checkpoint.

`R2E` = reaction -> enzyme. `E2R` = enzyme -> reaction.

## Direct comparison

| Scenario | Direction | Current strongest Catalyst result | EnzymeCAGE baseline | Status |
|---|---|---|---|---|
| Enzyme-405, official canonical-reaction reservoir | R2E | **SR@10 50.51%**, SR@1 14.58%, EF@1% 30.74, DCG@10 0.3927; MRR 0.2597, MAP 0.2522 | **Paper-reported:** SR@10 **57.97%**, EF@1% **36.6031**, DCG@10 **0.4523** | **CAGE currently leads**. Catalyst gap: -7.46 pp SR@10. The current Catalyst number is a clean-room, target-label-free observation; do not select it by Enzyme-405 performance. |
| TPS / CAGE common scored reservoir, zero-shot retention | R2E | **Hit@10 98.60%, MRR 0.8932** (`direct:legacy`, route guard falls back) | Hit@10 17.32%, MRR 0.0721 | Catalyst far ahead; **retention/common-reservoir diagnostic, not clean generalization**. |
| TPS / CAGE common scored reservoir, zero-shot retention | E2R | **Hit@10 99.31%, MRR 0.9023** (`direct:legacy`) | Hit@10 17.92%, MRR 0.0825 | Catalyst far ahead; same caveat. |
| Few-shot hidden-positive common reservoir, 1 seed | R2E | **Hit@10 94.13%, MRR 0.8056** (guarded system falls back to backbone) | Hit@10 13.10%, MRR 0.0595 | Catalyst far ahead. CAGE is a fixed zero-shot scorer here; it does not consume the seed. |
| Few-shot hidden-positive common reservoir, 1 seed | E2R | **Hit@10 89.21%, MRR 0.7465** (backbone) | Hit@10 23.87%, MRR 0.1021 | Catalyst far ahead. |
| Few-shot hidden-positive common reservoir, 5 seeds | R2E | **Hit@10 94.85%, MRR 0.8171** (`rrf:direct+seed:e2r`, development-only activation guard) | Hit@10 11.47%, MRR 0.0522 | Catalyst far ahead; this is the one route cell currently passing the strict activation guard. |
| Few-shot hidden-positive common reservoir, 5 seeds | E2R | **Hit@10 74.41%, MRR 0.4990** (guarded fallback backbone) | Hit@10 16.47%, MRR 0.1260 | Catalyst far ahead, but current router correctly declines to activate an expert. |
| Broad recorded-association zero-shot, query-unseen, full 185,918-protein / 11,081-reaction universe | E2R | **Hit@10 63.24%, MRR 0.4925** (`replay=0.25`, stable SHA256 cohort; strong broad generalization while preserving retention) | **N/A** | EnzymeCAGE has no fair complete score matrix on this full candidate universe; its structure/pocket feature coverage is sparse and its paper task is R2E-centric. Do not treat unscored pairs as negatives. |
| Broad recorded-association zero-shot, query-unseen, full universe | R2E | **Hit@10 4.30%, MRR 0.0342** (`replay=0.25`) | **N/A** | Clear Catalyst weakness: broad R2E remains much harder than E2R. |
| Known-relation recovery / current-domain retention | E2R | **Hit@10 100%, MRR 0.9245** (`replay=0.25`, query-all-known) | **N/A** | CAGE does not provide complete scores over the same 1,391 × 513 known-domain matrix. |
| Known-relation recovery / current-domain retention | R2E | **Hit@10 99.81%, MRR 0.9148** (current production backbone, query-all-known) | **N/A** | Same reason; CAGE scored-support results are kept separately rather than filling missing pairs with negative labels. |
| ReactZyme-projected strict double-cold, full universe | R2E | **Hit@10 1.22%, Hit@50 8.21%, MRR 0.00664, AUROC 0.8688** | **N/A** | CAGE lacks a same-split, same-full-candidate-pool evaluation with complete required pocket features. This is a much stricter full-universe cold-start task. |
| ReactZyme-projected strict double-cold, full universe | E2R | **Hit@10 11.43%, Hit@50 25.64%, MRR 0.04737, AUROC 0.8330** | **N/A** | E2R is not an author-native EnzymeCAGE retrieval protocol. |
| Post-2020 temporal double-cold, full universe | R2E | **Hit@10 3.09%, Hit@50 10.12%, MRR 0.01685, AUROC 0.9096** | **N/A** | CAGE has no matching historical snapshot/cutoff + full-universe protocol. |
| Post-2020 temporal double-cold, full universe | E2R | **Hit@10 8.35%, Hit@50 19.01%, MRR 0.03046, AUROC 0.8558** | **N/A** | Same; E2R additionally is not CAGE-native. |
| ReactZyme-projected protein-cold, reactions seen | R2E | **Hit@10 4.77%, MRR 0.01923, AUROC 0.8336** | **N/A** | CAGE has no complete full-universe feature coverage. |
| ReactZyme-projected protein-cold, reactions seen | E2R | **Hit@1 60.99%, Hit@10 79.90%, MRR 0.6771, AUROC 0.9865** | **N/A** | Very strong Catalyst protein->reaction transfer; not a CAGE-native direction. Note this split is dominated by high-homology proteins, although 20–40% identity and no-hit slices also improve. |
| Broad reaction-cold, proteins seen | R2E | **Hit@10 10.22%, Hit@50 24.19%, MRR 0.06085, AUROC 0.9110** | **N/A** | Strong overall gain, but the hardest reaction-similarity `<0.3` slice still has **Hit@10=0 and Hit@50=0**; this remains a real weakness. |
| Broad reaction-cold, proteins seen | E2R | **Hit@10 24.60%, Hit@50 43.22%, MRR 0.08743, AUROC 0.9317** | **N/A** | CAGE comparison not directly available for E2R/full-universe protocol. |

## Interpretation

1. **Where the task overlaps EnzymeCAGE's scored-reservoir use case, Catalyst is already much stronger** on the old TPS/common-reservoir and few-shot settings. Those results are useful system/retention evidence, not clean cold-start evidence.
2. **Enzyme-405 is the important exception:** current clean-room Catalyst reaches SR@10 50.51%, while EnzymeCAGE reports 57.97%. This remains the most obvious external benchmark gap and must improve.
3. **The strict full-universe suite reveals a highly asymmetric model:** E2R is already very strong in protein-cold and broad query-unseen settings, while R2E remains the main bottleneck, especially for truly low-similarity unseen reactions.
4. The frozen 2-epoch cold-training recipe has generalized beyond one split: temporal double-cold improves 72/72 tracked metrics over the previous clean model; protein-cold improves 70/72 and ties 2/72 versus a paired 1-epoch control; reaction-cold improves 72/72 metrics versus the clean base.
5. The routed model should therefore be treated as the product-level model: it can activate specialists only when development evidence passes the non-degradation guard, and otherwise falls back to the backbone. At present only the 5-seed R2E few-shot cell passes the strict activation guard; routing still has substantial room to become more useful.

## EnzymeCAGE reproduction scope

Current-source provenance is now adequate for comparison work: the current Google Drive dataset/checkpoint archives were downloaded through the controller Clash proxy, validated locally, transferred by SSH, and hash-verified remotely. The Enzyme-405 CSV and the 18,463 minimal non-ESM files used locally are byte/CRC-identical to the current author package. The 2-UID `author_exact` ESM-C reconstruction smoke also succeeds. Because the original/public materials do not contain all precomputed author-side protein features, further effort will focus on model improvement rather than pursuing perfect bit-for-bit EnzymeCAGE reconstruction. Paper-reported Enzyme-405 metrics remain explicitly labelled as such.
