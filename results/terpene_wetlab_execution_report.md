# Terpene Synthase Wet-Lab Discovery Execution Report

## Scope and data contract

This report covers the persistent MARTS registry discovery run, candidate-panel construction, balanced core-TPS campaign, four canonical 96-well plates, two architecture-contract-filtered UniProt rescue plates, combined sequence procurement and the result-feedback workflow. Known MARTS enzyme-reaction associations are masked from every discovery ranking and appear only as explicitly labelled positive controls.

- Registered enzyme queries: 694
- Registered reaction queries: 240
- Known-association leakage found: 0
- Reliability after known-positive masking: not reused; masked discovery is explicitly marked uncalibrated.

## Candidate concentration audit

| Direction | Objective | Queries | Unique Top-1 | Largest single-candidate share | Top-10 candidates share | Effective Top-1 candidates | External Top-1 share |
|---|---|---:|---:|---:|---:|---:|---:|
| enzyme_to_reaction | top10 | 694 | 199 | 5.2% | 11.1% | 120.5 | 6.5% |
| enzyme_to_reaction | top20 | 694 | 217 | 3.6% | 7.6% | 139.1 | 11.0% |
| enzyme_to_reaction | top3 | 694 | 202 | 4.8% | 17.3% | 127.8 | 8.9% |
| reaction_to_enzyme | top10 | 240 | 156 | 4.6% | 9.5% | 123.6 | 50.8% |
| reaction_to_enzyme | top20 | 240 | 156 | 4.6% | 6.9% | 123.6 | 50.8% |
| reaction_to_enzyme | top3 | 240 | 156 | 4.6% | 15.0% | 119.9 | 48.3% |

The largest single candidate covers only approximately 4.5–5.0% of queries, so the discovery output is not dominated by one universal hub. Objective-specific Top-3/10/20 lists are not forced to be nested because the production routes differ by cutoff.

## Full reaction panels

- Reactions with panels: 240
- Discovery candidates per reaction: 12
- Total discovery assays represented: 2880
- Positive controls available: 240
- Allocation: 6 exploitation, 3 uncertainty, 3 ESM-C diversity candidates.
- Sequence-risk candidates removed from Top-20 pools: 107
- Eligibility rule: 200–1000 aa and canonical amino-acid alphabet; excluded candidates are replaced from the remaining masked Top-20 pool.

## Balanced core-TPS campaign

- Core reactions: 24
- Terpene-type distribution: {"di": 6, "tri": 6, "sesq": 5, "mono": 3, "sester": 3, "sesquar": 1}
- TPS-class distribution: {"1": 20, "2": 4}
- Unique positive-control IDs: 23 of 24 reactions
- Extended pathway slate: 8 reactions, {"sqs": 4, "psy": 3, "pt": 1}

The primary campaign excludes PSY/SQS/PT/tetraterpene pathway enzymes from the 24-reaction core slate and preserves them in a separate eight-reaction exploratory slate. The core selector guarantees type coverage, preserves at least four class-II reactions and penalizes repeated substrate and positive-control choices.

| Order | Reaction | Type | Class | Substrate | Product | Positive control |
|---:|---|---|---|---|---|---|
| 1 | MARTS_EXT_RXN_237f1c153ad6 | di | 1 | (2E,6E,10E)-GGPP | chitinol | A0A979G5X7 |
| 2 | MARTS_EXT_RXN_890006655690 | mono | 1 | (R,R)-chrysanthemyl diphosphate | (R,R)-chrysanthemol | P0C565 |
| 3 | MARTS_EXT_RXN_bb0c8b67ae81 | sesq | 1 | (2E,6E)-FPP | (+)-Isoitalicene | E5GAG1 |
| 4 | MARTS_EXT_RXN_92f6ef4d14a5 | sesquar | 1 | (R)-tetraprenyl-β-curcumene | sporulenol | Q796C3 |
| 5 | MARTS_EXT_RXN_c33f8bcece70 | sester | 1 | (2E,6E,10E,14E)-GFPP | aspergildiene B | P9WEP0 |
| 6 | MARTS_EXT_RXN_78f709de4b7c | tri | 1 | presqualene diphosphate | bisfarnesyl ether | G0Y287 |
| 7 | MARTS_EXT_RXN_0304c0087056 | tri | 2 | (S)-2,3-epoxysqualene | 21-beta-H-hopane-3-beta,22-diol | B0Y565 |
| 8 | MARTS_EXT_RXN_4c4ba4bc0549 | tri | 2 | (S)-2,3-epoxysqualene | achilleol A | P0C8Y0 |
| 9 | MARTS_EXT_RXN_000c8b710696 | tri | 2 | squalene | hop-22(29)-ene | P33247 |
| 10 | MARTS_EXT_RXN_455979525191 | tri | 2 | (S)-2,3-epoxysqualene | lupan-3β,20-diol | Q9C5M3 |
| 11 | MARTS_EXT_RXN_5e756bc9af81 | di | 1 | ent-neo-cis-trans-kolavenol | ent-neo-cis-trans-kolavelool | A0A345ZQ25 |
| 12 | MARTS_EXT_RXN_3d09cb5fb800 | sesq | 1 | (2Z,6E)-FPP | (6R,7S)-2,2,6-trimethyl-10-methylenebicyclo[5.4.0]undec-1(11)-ene | A0A140AZ63 |
| 13 | MARTS_EXT_RXN_22bda90ee992 | di | 1 | nerylneryl diphosphate | lycosantalene | G5CV51 |
| 14 | MARTS_EXT_RXN_96ac57c79326 | mono | 1 | (2E)-GPP | α-phellandrene | O23945 |
| 15 | MARTS_EXT_RXN_d4f733465fb3 | sesq | 1 | (2E,6E)-FPP | bicycloelemene | F1A1D4 |
| 16 | MARTS_EXT_RXN_cda281fba2bd | di | 1 | (2E,6E,10E)-GGPP | lobophytumin C | B5H135 |
| 17 | MARTS_EXT_RXN_71aa03bcb508 | sester | 1 | (2E,6E,10E,14E)-GFPP | atacamadiene | WP_109642683 |
| 18 | MARTS_EXT_RXN_d147768eaf20 | sesq | 1 | (2Z,6E)-FPP | (6R,7S)-himachala-9,11-diene | A0A140AZ63 |
| 19 | MARTS_EXT_RXN_0bc621c08854 | mono | 1 | (2E)-GPP | sylvestrene | marts_E00439 |
| 20 | MARTS_EXT_RXN_8d7f7dd54117 | di | 1 | (2E,6E,10E)-GGPP | chitinopinol | A0A979G646 |
| 21 | MARTS_EXT_RXN_27964350d2e0 | sester | 1 | (2E,6E,10E,14E)-GFPP | subrutilene B | WP_150521282 |
| 22 | MARTS_EXT_RXN_45be1b4caa76 | sesq | 1 | (2E,6E)-FPP | Capnellene | marts_E00826 |
| 23 | MARTS_EXT_RXN_23432a8b720a | tri | 1 | all-trans-hexaprenyl diphosphate(3−) | β-hexaprene | M5AW86 |
| 24 | MARTS_EXT_RXN_79368a2476ac | di | 1 | ent-copalyl diphosphate | 13-hydroxy-8(14)-ent-abietene | A0A8T0VMU7 |

## Construct procurement

- Candidate-ID constructs: 249
- Sequence-deduplicated constructs: 248
- Alias IDs collapsed: 1
- Total protein length: 127,052 aa
- Total coding length without stops: 381,156 nt
- Sequence-ready constructs: 249
- Constructs requiring manual sequence review: 0
- Protein FASTA is complete. No codon optimization has been performed because the expression host and vector architecture are not fixed.

## Four-plate layout

Each reaction occupies two adjacent columns. Candidates 1–8 occupy A–H of the first column; candidates 9–12 occupy A–D of the second. The remaining wells are positive control primary, positive control replicate, empty-vector negative and substrate/process blank.

| Plate | Reactions | Discovery wells | Positive-control wells | Empty-vector wells | Process blanks | Unique protein constructs |
|---|---:|---:|---:|---:|---:|---:|
| TPS_DISCOVERY_P01 | 6 | 72 | 12 | 6 | 6 | 77 |
| TPS_DISCOVERY_P02 | 6 | 72 | 12 | 6 | 6 | 68 |
| TPS_DISCOVERY_P03 | 6 | 72 | 12 | 6 | 6 | 77 |
| TPS_DISCOVERY_P04 | 6 | 72 | 12 | 6 | 6 | 72 |

## Reaction-to-plate balancing

Before assigning candidate wells, complete reaction blocks are redistributed across plates with an exact-capacity mixed-integer linear program. Canonical plates retain six reactions each and UniProt rescue plates retain twelve reactions each. The objective balances terpene type, TPS class, substrate, positive-control reuse, candidate sequence length, candidate-source fraction and, for the rescue campaign, evidence-tier and architecture counts.

Canonical balancing moves 12 of 24 reaction blocks. The summed per-category terpene-type range falls from 9 to 6; the TPS-class range falls from 8 to 0. The between-plate range of mean candidate median length falls from 146.8 aa to 6.8 aa.

UniProt rescue balancing moves 10 of 24 reaction blocks. The terpene-type range falls from 10 to 4, the TPS-class range from 6 to 2, and the between-plate mean candidate-length range from 100.1 aa to 4.9 aa. B/C/D evidence counts are equalized between the two rescue plates. Exact Pfam architecture imbalance is also removed: bacterial class-I range 10→0, plant-TPS-full 6→0, and complete OSC 16→0.

Every reaction retains exactly the same candidate and control set; only its plate/block assignment changes. Each resulting plate remains exactly 96 wells.

## Candidate-position randomization

The original generated layouts placed candidate-selection roles in fixed rows and local columns: canonical exploitation, uncertainty and diversity roles occupied disjoint slot sets, while UniProt evidence, homology, predicted and diversity candidates were fixed to A/B/C/D rows. This would confound selection strategy with plate-position effects.

The operational layouts therefore use deterministic within-reaction-block Hungarian assignment with seed `20260723`. Positive controls, empty-vector negatives and substrate/process blanks remain in their original wells; only candidate wells are reassigned.

| Balance diagnostic | Before | After |
|---|---:|---:|
| Mean normalized role-slot entropy | 0.201 | 0.974 |
| Maximum single-slot share for any role | 100.0% | 33.3% |
| Maximum role slot-count range | 24 | 1 |
| Candidate assignments | — | 384 |
| Control or blank wells moved | — | 0 |

The mean role-slot entropy increases from approximately 0.20 to 0.97, and the maximum count difference for a role across candidate slots falls from 24 to 1. The randomized manifests and matching result templates are the execution inputs; the original role-ordered layouts are retained only for provenance and audit.

## Combined six-plate procurement campaign

The four canonical plates and two UniProt rescue plates contain 576 wells across 39 distinct reactions. They contain 480 protein assay wells and 353 candidate IDs. Cross-campaign exact-sequence deduplication reduces these to 352 master constructs, including 12 sequences shared by both campaigns.

Total procurement size is 184,501 aa or 553,503 coding nucleotides without stop codons. The master FASTA contains protein sequences only; codon optimization remains deferred until the host and vector architecture are fixed.

| Master order | Scope | Plate | Reactions | Wells | Discovery wells | Positive controls | Negative controls | Process blanks | Unique constructs |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | canonical_discovery | TPS_DISCOVERY_P01 | 6 | 96 | 72 | 12 | 6 | 6 | 77 |
| 2 | canonical_discovery | TPS_DISCOVERY_P02 | 6 | 96 | 72 | 12 | 6 | 6 | 75 |
| 3 | canonical_discovery | TPS_DISCOVERY_P03 | 6 | 96 | 72 | 12 | 6 | 6 | 76 |
| 4 | canonical_discovery | TPS_DISCOVERY_P04 | 6 | 96 | 72 | 12 | 6 | 6 | 76 |
| 5 | uniprot_rescue | TPS_UNIPROT_RESCUE_P01 | 12 | 96 | 48 | 24 | 12 | 12 | 60 |
| 6 | uniprot_rescue | TPS_UNIPROT_RESCUE_P02 | 12 | 96 | 48 | 24 | 12 | 12 | 60 |

Procurement length tiers:

| Length tier | Usage | Constructs | Total aa | Total coding nt | Median length | Maximum length |
|---|---|---:|---:|---:|---:|---:|
| long_501_750aa | discovery_and_control | 3 | 2101 | 6303 | 732 | 738 |
| long_501_750aa | discovery_candidate | 93 | 58056 | 174168 | 620 | 742 |
| long_501_750aa | positive_control_only | 9 | 5379 | 16137 | 591 | 710 |
| standard_le_500aa | discovery_and_control | 7 | 2620 | 7860 | 379 | 465 |
| standard_le_500aa | discovery_candidate | 164 | 59829 | 179487 | 356 | 491 |
| standard_le_500aa | positive_control_only | 11 | 4074 | 12222 | 364 | 459 |
| very_long_751_1000aa | discovery_and_control | 1 | 757 | 2271 | 757 | 757 |
| very_long_751_1000aa | discovery_candidate | 58 | 46847 | 140541 | 778 | 998 |
| very_long_751_1000aa | positive_control_only | 6 | 4838 | 14514 | 806 | 844 |

The master manifest is for procurement and plate tracking only. Canonical and UniProt rescue feedback remain separate QC scopes, so a failed control or contamination event in one experimental batch cannot relabel assays from the other batch.

## Result feedback and iteration

The assay template contains expression status, soluble-expression status, assay/background signals, target-product detection, product-identity confidence, technical-issue flag and notes.

- Confirmed positive: reaction controls pass; target product is detected; identity confidence is at least the configured threshold; no technical issue.
- Expression-qualified negative: controls pass; target is not detected; expression is adequate/high, or low but soluble; no technical issue.
- Inconclusive: failed expression, failed controls, missing evidence or technical issue.
- Untested or unlabeled pairs are never converted into negatives.
- Failed-control reactions are routed to control/current-panel rerun rather than candidate expansion.
- Passed reactions receive an eight-candidate next panel: 4 outcome/model exploitation, 2 uncertainty and 2 diversity candidates.

## Canonical files

- `results/terpene_registry_batch/reaction_to_enzyme_rankings.csv`
- `results/terpene_wetlab_discovery_panels/campaign_reactions.csv`
- `results/terpene_wetlab_discovery_panels/campaign_discovery_candidates.csv`
- `results/terpene_wetlab_plate_manifest/assay_manifest.csv` (pre-randomization provenance)
- `results/terpene_wetlab_plate_balanced/canonical_balanced_assay_manifest.csv`
- `results/terpene_wetlab_plate_balanced/uniprot_balanced_assay_manifest.csv`
- `results/terpene_wetlab_plate_balanced/plate_balance_audit.csv`
- `results/terpene_wetlab_randomized_layout/canonical_randomized_assay_manifest.csv`
- `results/terpene_wetlab_randomized_layout/canonical_randomized_assay_results_template.csv`
- `results/terpene_wetlab_plate_manifest/sequence_deduplicated_constructs.fasta`
- `results/terpene_wetlab_plate_manifest/TPS_DISCOVERY_P01_layout.csv` through `P04`
- `results/terpene_uniprot_rescue_campaign/assay_manifest.csv` (pre-randomization provenance)
- `results/terpene_wetlab_randomized_layout/uniprot_randomized_assay_manifest.csv`
- `results/terpene_wetlab_randomized_layout/uniprot_randomized_assay_results_template.csv`
- `results/terpene_wetlab_randomized_layout/candidate_well_assignments.csv`
- `results/terpene_wetlab_randomized_layout/role_slot_balance_audit.csv`
- `results/terpene_combined_wetlab_campaign/master_assay_manifest.csv`
- `results/terpene_combined_wetlab_campaign/master_sequence_constructs.fasta`
- `results/terpene_combined_wetlab_campaign/feedback_scopes.csv`

## Remaining experimental decisions

Expression host, vector, tag, subcellular-targeting truncation policy, precursor-supply strategy, assay matrix, analytical detection limits and product-confirmation criteria remain wet-lab decisions. They are intentionally not inferred by the retrieval pipeline.
