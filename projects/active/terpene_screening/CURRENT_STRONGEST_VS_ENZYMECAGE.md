# Catalyst clean retrieval mainline and EnzymeCAGE comparison

This is the canonical status document for the clean Catalyst retrieval work. `R2E` means reaction → enzyme and `E2R` means enzyme → reaction. The central story is now a single successful evolution of the clean retrieval system. Rejected experiments remain in their frozen audit files for reproducibility, but they are not treated as parallel mainlines.

All headline retrieval numbers below use an explicitly fixed candidate support and information boundary. Hit/SR@K, MRR, MAP, NDCG and AUROC are not mixed across incompatible denominators. Target-test labels are never used to select the model, representation, route threshold or confirmation split.

## Canonical successful evolution

| Stage | What changed | Strongest clean evidence | Current role |
|---|---|---|---|
| 1. RDKit+ reaction representation | Added chemically explicit RDKit descriptors on top of the registered DRFP/categorical reaction representation | On the 744-query reaction-cold/full-185,918-protein cell, Hit@10 **13.58% → 16.94%**, MRR **0.0702 → 0.0991**, MAP **0.0536 → 0.0727**, AUROC **0.9199 → 0.9288**, median best-positive rank **418.5 → 275.5** | Reaction-side base representation |
| 2. Direction-specific protein representation | R2E selected EnzGFM-650M; E2R selected equal-block ESM-C+EnzGFM, both with RDKit+ | Untouched `temporal_post2020_protein_cold`: R2E MRR **0.13355**, Hit@10 **29.18%**; E2R MRR **0.30250**, Hit@10 **53.77%** | Protein-cold direction experts |
| 3. Bounded Top-2000 precision refinement | Pair residual restricted to scale **0.03**, then label-free routed only when train DRFP similarity ≥ **0.9** | Fresh salted confirmation: MRR **0.10205 → 0.11484**, MAP **0.08246 → 0.09764**, Hit@10 **24.19% → 25.95%**, Hit@50 **41.64% → 44.43%**, median rank **137 → 90.5** | Confirmed high-similarity R2E precision module |
| 4. Identity-preserving reaction-center residual V2 | Kept the RDKit+ protein/reaction towers frozen and trained only zero-initialized `aux_to_hidden.weight` from a fixed 1280-d mapped reaction-center block | Fresh salted fold4: all-query MRR **0.10851 → 0.11501**; `<0.3` MRR **0.05839 → 0.06286**, MAP **0.04265 → 0.04711** | Established that explicit reaction-center information improves the hard R2E regime |
| 5. Geometry-bounded reaction-center residual V3 | Added a per-reaction hidden-residual norm cap while preserving exact zero-init identity; candidate caps **0.075 / 0.10 / 0.16** came only from clean2023 train geometry | All three preregistered caps passed development; **0.10** won the frozen ordering. On a separate fresh salted fold3, all frozen gates passed: all-query MRR **0.08863 → 0.09512**, MAP **0.07654 → 0.08197**, Hit@10 **22.01% → 24.38%**; `<0.3` MRR **0.03011 → 0.03347**, MAP **0.02241 → 0.02509**, Hit@10 **6.45% → 8.06%**, median rank **1972.5 → 1272** | **Current clean R2E representation mainline** |
| 6. Full-clean production package | Retrained the confirmed cap=0.10 residual on all **218,537** clean2023 pairs after confirmation | Exact identity initialization remains 0; only `aux_to_hidden.weight` is trainable; loader smoke passes. Production checkpoint SHA256 `1bc951373ff1c139d508c0ce2275cb57d892077e9d8b0c81b0638d8936e78688` | Deployable clean R2E core |
| 7. Fast similarity-conditioned model routing | Kept the bounded-center core for high train-reaction similarity and uses a full-clean EnzGFM+bounded-center expert only when `max_train_binary_drfp_tanimoto < 0.9`; the threshold was frozen before confirmation | Separate untouched fold6, 686 queries over all 185,918 proteins: MRR **0.09462 → 0.10163**, MAP **0.07917 → 0.08027**, AUROC **0.95809 → 0.97366**, NDCG@10 **0.10383 → 0.10525**, Hit@10 **23.76% → 25.36%**, Hit@50 **38.92% → 43.29%**, median best-positive rank **149 → 95.5**; every frozen check passed | Independently confirmed source selector; retained as the exact v4 tail fallback |
| 8. Automated LambdaRank source fusion | Reused the two confirmed full-clean R2E source models and let a mature LambdaRank learner automatically combine their score/rank signals on the union of their Top-100 candidates. Nineteen configurations were preregistered and selected only by nested OOF; no manual fusion-weight tuning was used | Fresh untouched salted fold6, 673 queries over all 185,918 proteins: MRR **0.10235 → 0.12167**, MAP **0.08324 → 0.10392**, AUROC **0.96462 → 0.96488**, NDCG@10 **0.10333 → 0.12289**, Hit@10 **25.41% → 26.60%**, Hit@20 **32.69% → 37.00%**, Hit@50 **42.50% → 49.18%**, median best-positive rank **93 → 53**; every frozen gate passed. Warm production median latency is only **1.26×** v3, p95 **1.23×**, with about **302 MiB** additional RSS | **Current production R2E route for eligible general-universe direct queries** |

The two full-clean source components remain `results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1` and `results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1`. Production v4 is now a controlled learned rank fusion: both sources score the full candidate universe, the frozen `cfg_07_392fe119` LambdaRank model reranks their Top-100 union into a learned Top-100 prefix, and every remaining candidate keeps the exact stage-7 similarity-router tail order (`<0.9` uses EnzGFM+center as fallback; `>=0.9` uses bounded-center).

The V3 confirmation is not only an aggregate improvement. Its post-decision 10,000-replicate paired bootstrap gives a 95% interval of **+0.00076 to +0.01224** for all-query MRR and **+0.00035 to +0.00808** for `<0.3` MRR. The corresponding MAP intervals are also above zero. The small 62-query hard slice still warrants cautious interpretation of discrete Hit@K changes, but the ranking improvement is no longer supported only by a handful of top-K events.

## Current routed-system contract

`Catalyst-Clean-Mainline-v1` is an evidence-scoped routed system, not a claim that every successful module has already been fused into one checkpoint.

- **Eligible general-universe external R2E direct queries:** score both full-clean source models, take the union of each source Top-100, apply the frozen LambdaRank `cfg_07_392fe119` to produce the learned Top-100 prefix, then append the exact stage-7 similarity-router ordering as the tail. The learned features contain source scores/ranks and train-only reaction similarity; target labels are not read at runtime.
- **Protein-cold R2E expert:** EnzGFM-650M + RDKit+, retained as the separately confirmed temporal protein-cold direction expert.
- **Protein-cold E2R expert:** equal-block ESM-C+EnzGFM + RDKit+, selected by the same frozen directional protocol.
- **High-similarity Top-2000 precision refinement:** remains independently confirmed but is **not** silently stacked on top of the new production route because its joint bounded-center experiment failed its separate gate.
- **Out-of-scope production requests:** TPS-specialized candidates, few-shot, masks, candidate subsets, taxonomy restrictions, temporary candidates and manual overrides keep their previous v2 deployments.

The earlier unconditional EnzGFM/equal-block + bounded-center comprehensive experiment still failed its preregistered material-gain gate; that history is not rewritten. The later R2E-only similarity router passed a separately frozen untouched confirmation, and the subsequent automated LambdaRank experiment asked a different question—whether the two already-confirmed full-clean R2E sources could be combined by learned candidate-level ranking. That learned fusion was selected only by nested OOF, then passed a new untouched confirmation and a pre-frozen runtime gate. We still do **not** claim an untested Top-2000 residual fusion with this production system.

The machine-readable source of truth is `CATALYST_CLEAN_MAINLINE_V1.json`; direction compatibility is enforced by `model_capability_registry.py`. `CATALYST_CLEAN_MAINLINE_CAPABILITY_V1.json` records the eight-stage evolution, and `CATALYST_BASELINE_PROVENANCE_V1.json` contains one provenance-safe EnzymeCAGE row—real or explicit N/A—for every registered scenario.

## Broad capability picture

| Scenario | Direction | Strongest frozen Catalyst evidence | Interpretation |
|---|---|---|---|
| LambdaRank untouched fold6, full universe | R2E | Frozen learned fusion: MRR **0.12167**, MAP **0.10392**, AUROC **0.96488**, NDCG@10 **0.12289**, Hit@10 **26.60%**, Hit@20 **37.00%**, Hit@50 **49.18%**, median best-positive rank **53** | Current internally confirmed production evidence; ranker/config were frozen before this fold, no external benchmark selected them, and a separately pre-frozen runtime gate also passed |
| Fast-router untouched fold6, full universe | R2E | Frozen `sim<0.9` route: MRR **0.10163**, MAP **0.08027**, AUROC **0.97366**, NDCG@10 **0.10525**, Hit@10 **25.36%**, Hit@20 **33.38%**, Hit@50 **43.29%**, median best-positive rank **95.5** | Independently confirmed predecessor and current v4 tail-fallback evidence |
| Fresh salted clean2023 strict double-cold mainline confirmation | R2E | V3 cap=0.10: MRR **0.09512**, MAP **0.08197**, AUROC **0.95348**, NDCG@10 **0.10376**, Hit@10 **24.38%**, Hit@50 **40.37%** | Primary internal confirmation for the current R2E representation |
| Same confirmation, reaction similarity `<0.3` | R2E | MRR **0.03347**, MAP **0.02509**, AUROC **0.90107**, NDCG@10 **0.02721**, Hit@10 **8.06%**, median best-positive rank **1272** | Extreme reaction novelty remains hard, but now has independently confirmed improvement rather than only better global ordering |
| Post-2020 temporal protein-cold, full universe | R2E | EnzGFM+RDKit+: MRR **0.13355**, MAP **0.12443**, AUROC **0.99232**, Hit@10 **29.18%**, Hit@50 **56.23%** | Strong protein-unseen R2E expert |
| Post-2020 temporal protein-cold, full universe | E2R | ESM-C+EnzGFM+RDKit+: MRR **0.30250**, MAP **0.26346**, AUROC **0.98733**, Hit@10 **53.77%**, Hit@50 **75.95%** | Strong protein-unseen E2R expert |
| Reaction-cold, train-seen proteins | R2E | RDKit+ frozen outer: Hit@10 **16.94%**, MRR **0.0991**, MAP **0.0727**, AUROC **0.9288** | Clear reaction-side representation gain |
| ReactZyme-projected strict double-cold | R2E | RDKit+ frozen outer: Hit@10 **3.95%**, Hit@50 **10.33%**, MRR **0.01329**, MAP **0.01302**, AUROC **0.8891** | Very hard joint novelty; retain as stress test rather than headline model-selection data |
| Post-2020 creation-date double-cold | R2E | RDKit+ frozen outer: Hit@10 **5.19%**, Hit@50 **12.10%**, MRR **0.02153**, AUROC **0.92497** | Temporal stress evidence; not a full historical source-snapshot claim |
| Enzyme-405 full official reservoir | R2E | Frozen pre-router V3: SR@10 **47.12%**, EF@1% **28.15**, DCG@10 **0.3901**, MRR **0.2469**, MAP **0.2386**, AUROC **0.6643** | Revealed descriptive external reference only; the new router is deliberately not rescored here for model selection |

## Native ReactZyme enzyme-similarity / EnzGFM-1.5B contract

This capability is now evaluated against one task-matched authoritative external baseline, **EnzGFM-1.5B**, rather than against an internal Catalyst predecessor. The local `enzyme_smi_split.zip` is byte-identical to the official ReactZyme archive (MD5 `e351fdb85830968fc9abe933c39f9eda`), and the Nature Communications paper defines the same standard MAP/AP, NDCG and Top-K semantics used here. Candidate selection was completed on a protein-disjoint train-only development split before the native test was scored; only the selected `dual_tower` candidate was revealed, and the alternative candidate was never scored on native test. This test is now permanently frozen against further model or router selection.

On the exact native support, E2R obtains **MAP 0.95580 vs 0.5156**, **NDCG@5 0.96466 vs 0.5152**, and **Top5 0.99221 vs 0.6636** for the EnzGFM-1.5B paper mean. R2E obtains **MAP 0.89161 vs 0.8211**, **NDCG@5 0.90705 vs 0.8484**, and **Top5 0.96503 vs 0.9425**. The full common paper metric set is MAP, NDCG@1, NDCG@5, Top1 and Top5; MRR, Hit@10 and NDCG@10 remain N/A for direct paper deltas rather than being silently substituted. Because Catalyst is one frozen run whereas the paper reports five-run mean ± SD, these are descriptive absolute same-split deltas, not a paired significance claim.

The large E2R margin was explicitly audited rather than accepted at face value. All **1,573/1,573** test reactions occur in association training, with a median of 21 training positives per test reaction. A zero-tuned diagnostic that simply averages **train-only EnzGFM-650M protein embeddings per reaction** already reaches E2R MAP **0.89222** and R2E MAP **0.85283**. The public archive also contains three exact train/test protein sequences despite the paper-level sequence-difference description; removing those three proteins post hoc leaves E2R MAP essentially unchanged (**0.95580 → 0.95584**). Therefore this benchmark is strong evidence for **sequence-divergent known-reaction retrieval**, but it must not be presented as reaction-novel discovery. The next external-baseline priority is a genuinely reaction-novel contract.

## Authoritative external comparisons

The comparison policy is intentionally conservative: use a task-matched authoritative baseline, adapt support rather than baseline capability, and report direct deltas only on identical query/candidate support. The full source of truth is `AUTHORITATIVE_BASELINE_COMPARISONS_V2.md`.

### Reaction-novel CLIPZyme

Native CLIPZyme is now the formal reaction-novel neural baseline; TIGER is methodology/reference only. Direct ReactZyme `reaction_smi` is not a valid native comparison because only **7/386** unordered reaction bags can be uniquely restored to a directed CLIPZyme input, below the frozen 50-query minimum. The first preregistered directed fallback, `reactzyme_reaction_projected_double_cold`, has enough native support after both reaction- and protein-side audits. Using the official Zenodo screening asset, the exact common universe contains **158,665 protein candidates**, **4,222 reaction candidates**, **4,915 positives**, **109 R2E queries** and **4,161 E2R queries**.

On this common support, R2E Catalyst V3 vs official CLIPZyme is MRR **0.5029 vs 0.1801**, MAP **0.4298 vs 0.1872**, NDCG@10 **0.4813 vs 0.1748**, Hit@10 **71.56% vs 23.85%**, with median best-positive rank **3 vs 184**. E2R is likewise MRR **0.4607 vs 0.0841**, MAP **0.4622 vs 0.0857**, NDCG@10 **0.4962 vs 0.1092**, Hit@10 **65.08% vs 23.14%**, with median rank **4 vs 242**. These are revealed descriptive baseline-alignment results and are forbidden from selecting a new model or router.

### Enzyme-405

The old 99-query reconstruction is no longer a headline comparison; it was an early 100-reaction subset, not the maximum fair support.

On the complete **295-query** official reservoir, current V3 obtains SR@10 **47.12%**, DCG@10 **0.3901** and EF@1% **28.15**. The EnzymeCAGE paper reports SR@10 **57.97%**, DCG@10 **0.4523** and EF@1% **36.6031**. This is full-reservoir Catalyst versus paper context, not a local same-support delta.

The strict local apples-to-apples comparison keeps every original candidate for each retained reaction and expands to **226 reactions / 11,665 canonical pairs**. Across official EnzymeCAGE seeds 40–44, CAGE obtains SR@10 **51.33 ± 1.66%**, MRR **0.2517 ± 0.0148**, MAP **0.2521 ± 0.0197**, AUROC **0.6925 ± 0.0069** and NDCG@10 **0.2892 ± 0.0182**. Current V3 on the identical support obtains SR@10 **49.12%**, MRR **0.2631**, MAP **0.2575**, AUROC **0.6741** and NDCG@10 **0.2855**. The defensible conclusion is same-order performance with mixed metric wins, rather than a large one-sided advantage.

A maximum-coverage pair-intersection sensitivity analysis keeps all **295 reactions / 15,207 canonical pairs** but no longer preserves every original candidate for 69 reactions. There CAGE reaches SR@10 **54.37 ± 1.72%** and MRR **0.2695 ± 0.0154**, versus V3 **48.14%** and **0.2498**. It is reported separately from the stricter 226-query result.

### Orphan-335

On the author's immutable retrieval pool (**335 queries / 90,804 canonical pairs / 44,889 candidate UIDs**), all 335 queries remain in the denominator, including the 102 for which the 2025 truth contains no positive inside the retrieved pool. Author Selenzyme SR@1/3/5/10 is reproduced exactly. Under one shared canonical-pair evaluator, Catalyst V3 vs Selenzyme is MRR **0.3007 vs 0.2088**, MAP **0.2942 vs 0.2113**, AUROC **0.7876 vs 0.5140**, NDCG@10 **0.3124 vs 0.2092**, and SR/Hit@10 **48.36% vs 31.04%**.

The released EnzymeCAGE Orphan config depends on a separate RHEA-2025 full GVP/ESM-C/reaction-feature snapshot that is absent from both the repository and current server assets. Reusing Enzyme-405 structure features would leave only **17 complete-candidate queries**, so no misleading local Orphan EnzymeCAGE score is reported. Its status is explicit N/A rather than an invented weak baseline.

## Historical evidence boundary

The older **unbounded** reaction-center residual V2 was tested once on a genuinely fresh Rhea release128→141 association-snapshot protocol and failed its frozen external promotion gate: global/deeper discrimination improved in places, but early ranking regressed. That result remains immutable and useful as a warning that unconstrained center updates can transfer poorly. It is **not** used to choose the V3 cap, V3 split, V3 gate or V3 production model, and bounded V3 has not been post-hoc rescored on that already revealed external set.

This is the intended level of prominence for failed side branches: they constrain what may be claimed, but they do not define the current system. The current mainline is determined by the successful preregistered chain ending in the freshly confirmed LambdaRank v4 R2E route, with bounded V3 and the direction-specific experts retained as its validated source/fallback components.

## Canonical artifacts

- Final routed/evidence manifest: `projects/active/terpene_screening/CATALYST_CLEAN_MAINLINE_V1.json`
- Mainline evolution/capability record: `projects/active/terpene_screening/CATALYST_CLEAN_MAINLINE_CAPABILITY_V1.json`
- Historical single-core production protocol/result: `CATALYST_CLEAN_MAINLINE_PRODUCTION_V1.json` / `CATALYST_CLEAN_MAINLINE_PRODUCTION_V1_RESULT.json`
- Historical similarity-routed production contract/result: `CATALYST_CLEAN_MAINLINE_PRODUCTION_V2.json` / `CATALYST_CLEAN_MAINLINE_PRODUCTION_V2_RESULT.json`
- Current learned-fusion production contract/result: `CATALYST_CLEAN_MAINLINE_PRODUCTION_V3.json` / `CATALYST_CLEAN_MAINLINE_PRODUCTION_V3_RESULT.json`
- LambdaRank protocol/development/confirmation/runtime: `CATALYST_R2E_LAMBDARANK_FUSION_V1.json`, `CATALYST_R2E_LAMBDARANK_FUSION_V1_DEVELOPMENT_RESULT.json`, `CATALYST_R2E_LAMBDARANK_FUSION_V1_CONFIRMATION_RESULT.json`, `CATALYST_R2E_LAMBDARANK_PRODUCTION_V1_RESULT.json`
- Fast similarity router protocol/result: `CATALYST_FAST_R2E_SIMILARITY_ROUTER_V1.json` / `CATALYST_FAST_R2E_SIMILARITY_ROUTER_V1_RESULT.json`
- V3 development/result/confirmation: `CLEANROOM_R2E_REACTION_CENTER_BOUNDED_RESIDUAL_V3*.json`
- Top-2000 precision confirmation: `CLEANROOM_R2E_TOP2000_DIFFICULTY_ROUTER_V1_CONFIRMATION.json`
- Directional EnzGFM frozen protocol: `CLEANROOM_ENZGFM_DIRECTIONAL_ROUTER_TEMPORAL_PROTEIN_COLD_V1.json`
- EnzymeCAGE provenance: `CATALYST_BASELINE_PROVENANCE_V1.json`
- Canonical authoritative external comparison matrix: `AUTHORITATIVE_BASELINE_COMPARISONS_V2.md`
- Enzyme-405 detailed external record: `ENZYME405_CLEANROOM_RESULT_V1.md`

At this point the clean retrieval track is coherent enough to treat as a stable learned-and-routed mainline: both R2E source components have full-clean production checkpoints; the simple `<0.9` train-similarity router remains an independently confirmed exact tail fallback; the automated LambdaRank fusion passed three-fold nested-OOF development, one new untouched strict double-cold confirmation, real low-/high-similarity product-engine smokes, and the pre-frozen latency/memory/determinism gate; protein-cold R2E/E2R retain their frozen direction-specific experts; the Top-2000 module remains separate rather than being post-hoc fused; and revealed external/baseline evidence remains explicitly separated from model selection.
