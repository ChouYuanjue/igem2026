# Authoritative external baseline comparisons v2

> **Historical baseline-alignment record only.** This file is no longer the canonical current external-evaluation document. Current production-relative novelty rules and headline results are in `CURRENT_RETRIEVAL_STATUS.md` and `CATALYST_EXTERNAL_EVALUATION_POLICY_V2.json`. In particular, old `broad_rhea` cold labels must not be interpreted as cold relative to the current clean2023 production training source.

This document preserves a historical same-support baseline-alignment record for the clean retrieval work. Direct deltas are reported only when both methods are evaluated on the same query/candidate support. Paper-only numbers are context, not local reproduction. No result in this document is allowed to select or retune a Catalyst model, router, threshold, or split.

## Baseline roles

| Capability | Formal comparison | Why this baseline | Allowed adaptation |
|---|---|---|---|
| reaction-novel general retrieval | official native CLIPZyme | CLIPZyme is a released reaction-conditioned enzyme retrieval model with native reaction and protein encoders | performance-blind input/support alignment only; no encoder substitution, Catalyst feature injection, target fine-tuning, or TIGER variant |
| Enzyme-405 novel-enzyme functional retrieval | EnzymeCAGE | EnzymeCAGE defines this benchmark and releases checkpoints/code | only the minimum support restriction required by author structural inputs; same pool for both systems |
| Orphan-335 orphan-reaction enzyme retrieval | author Selenzyme score | Selenzyme is directly executable on the immutable author retrieval pool and is an author-native task baseline | none beyond a shared canonical-pair evaluator for direct IR deltas |
| Orphan-335 EnzymeCAGE | protocol documented, local score N/A | the released config exists, but its RHEA-2025 full GVP/ESM-C/reaction feature snapshot is not present in the repository or current server assets | do not replace it with Enzyme-405 pocket features or fabricate missing structures |

TIGER is retained only as methodological/evaluation inspiration. It is not an executable formal baseline and its encoder-substituted CLIPZyme variants are not labeled as native CLIPZyme.

## Native CLIPZyme reaction-novel comparison

The original ReactZyme `reaction_smi` test contains 386 unordered molecule bags. Only **7** can be uniquely restored to a directed reaction through the independent registered reaction source while also satisfying native CLIPZyme prerequisites, below the frozen minimum of 50 queries. Therefore the direct unordered ReactZyme cell is not used.

The fallback order was frozen before scores. The first cell, `reactzyme_reaction_projected_double_cold`, passes the support gate and is therefore selected without looking at performance. The official Zenodo CLIPZyme screening asset was recovered from record 20673359; the source archive MD5 is `1ebd955e83fa480aea198c20c1a66381`, and the released protein embedding matrix is **261,907 × 1,280**.

After intersecting native CLIPZyme support with the frozen Catalyst universe, both methods are scored on exactly **158,665 protein candidates**, **4,222 reaction candidates**, and **4,915 positive pairs**. This gives **109 R2E queries** and **4,161 E2R queries**.

| Direction / metric | official CLIPZyme | Catalyst V3 | Catalyst − CLIPZyme |
|---|---:|---:|---:|
| R2E MRR | 0.1801 | **0.5029** | +0.3229 |
| R2E MAP | 0.1872 | **0.4298** | +0.2426 |
| R2E AUROC | 0.9126 | **0.9868** | +0.0742 |
| R2E NDCG@10 | 0.1748 | **0.4813** | +0.3065 |
| R2E Hit@10 | 23.85% | **71.56%** | +47.71 pp |
| R2E median best-positive rank | 184 | **3** | −181 |
| E2R MRR | 0.0841 | **0.4607** | +0.3765 |
| E2R MAP | 0.0857 | **0.4622** | +0.3764 |
| E2R AUROC | 0.7842 | **0.9921** | +0.2078 |
| E2R NDCG@10 | 0.1092 | **0.4962** | +0.3870 |
| E2R Hit@10 | 23.14% | **65.08%** | +41.94 pp |
| E2R median best-positive rank | 242 | **4** | −238 |

This is a **revealed descriptive fallback comparison**. It is strong evidence that the current system is competitive with a task-matched authoritative baseline on common native support, but it cannot be used for new model/router selection.

## Enzyme-405

Three layers are deliberately separated.

### A. Full official reservoir: current Catalyst vs paper context

The current V3 production model is evaluated on all **295 reactions / 15,524 unique reaction-enzyme pairs / 8,615 candidate UIDs**. Catalyst obtains **SR@10 47.12%**, **DCG@10 0.3901**, **EF@1% 28.15**, MRR **0.2469**, MAP **0.2386**, AUROC **0.6643**, and NDCG@10 **0.2634**. EnzymeCAGE reports **SR@10 57.97%**, **DCG@10 0.4523**, and **EF@1% 36.6031** in the paper. These are contextual side-by-side values, not a local same-support subtraction.

### B. Strict local apples-to-apples support

The old 99-query subset was only an early 100-reaction reconstruction and is no longer a headline result. The largest strict support that preserves each reaction's complete original candidate pool under locally executable author inputs contains **226 reactions / 11,665 canonical pairs**.

Five official EnzymeCAGE checkpoints (seeds 40–44) were rerun. EnzymeCAGE achieves **SR@10 51.33 ± 1.66%**, MRR **0.2517 ± 0.0148**, MAP **0.2521 ± 0.0197**, AUROC **0.6925 ± 0.0069**, NDCG@10 **0.2892 ± 0.0182**, DCG@10 **0.3887 ± 0.0232**, and EF@1% **32.78 ± 2.69**. Catalyst V3 on the identical support achieves **SR@10 49.12%**, MRR **0.2631**, MAP **0.2575**, AUROC **0.6741**, NDCG@10 **0.2855**, DCG@10 **0.3906**, and EF@1% **32.09**.

Thus the strict reproducible conclusion is **same-order performance with mixed metric wins**, not an implausibly large advantage in either direction: CAGE is +2.21 pp on SR@10 while Catalyst is slightly higher on MRR/MAP/DCG@10.

### C. Maximum-coverage pair intersection

For sensitivity analysis, restricting both systems to the maximum executable pair intersection preserves **295/295 reactions and 15,207 canonical pairs**, while 69 reactions no longer retain their full original candidate set. EnzymeCAGE seeds 40–44 give **SR@10 54.37 ± 1.72%**, MRR **0.2695 ± 0.0154**, MAP **0.2612 ± 0.0138**, AUROC **0.7013 ± 0.0066**, and NDCG@10 **0.2999 ± 0.0104**. Catalyst V3 gives **SR@10 48.14%**, MRR **0.2498**, MAP **0.2414**, AUROC **0.6620**, and NDCG@10 **0.2681**.

This third table is maximum-coverage sensitivity analysis, not a replacement for the stricter complete-candidate comparison.

## Orphan-335

The author retrieval procedure is retained exactly: for each orphan reaction, retrieve enzymes attached to the top-10 similar 2023 Rhea reactions. The resulting immutable author pool has **335 queries, 90,804 canonical reaction-protein pairs, and 44,889 candidate UIDs**. Under the 2025 truth, **233 queries** contain at least one positive in this retrieved pool and **102** contain none; all 335 remain in the denominator.

Author Selenzyme SR@1/3/5/10 is reproduced exactly. A unified canonical-pair evaluator is then applied to the unchanged Selenzyme score and Catalyst V3 score:

| Metric | Selenzyme | Catalyst V3 | Delta |
|---|---:|---:|---:|
| MRR | 0.2088 | **0.3007** | +0.0919 |
| MAP | 0.2113 | **0.2942** | +0.0829 |
| AUROC | 0.5140 | **0.7876** | +0.2735 |
| NDCG@10 | 0.2092 | **0.3124** | +0.1032 |
| SR / Hit@10 | 31.04% | **48.36%** | +17.31 pp |
| median best-positive rank | 14 | **4** | −10 |

The author raw-row DCG/EF implementation differs slightly from the project's canonical-pair evaluator because of duplicate-row handling; therefore direct Catalyst-vs-Selenzyme deltas use one shared evaluator, while author raw-row values remain provenance only.

The released EnzymeCAGE Orphan config points to a separate **RHEA 2025-02-05 full GVP + ESM-C pocket + reaction-feature snapshot**. Those assets are not in the checked repository or current server data. Reusing the Enzyme-405 structure bundle would cover only **2,259 / 44,889** candidate UIDs and leave just **17 complete-candidate queries**. That is not treated as a faithful Orphan-335 EnzymeCAGE reproduction. The score is therefore N/A rather than fabricated.

## Evidence boundary

All comparisons above are post-model-selection descriptive evidence. Revealed Enzyme-405, Orphan-335, ReactZyme, temporal, and Rhea128→141 labels are prohibited from selecting new representations, routes, thresholds, or hyperparameters. Missing baseline metrics are N/A, not imputed. Same-support restrictions are documented explicitly rather than presented as the original full benchmark.
