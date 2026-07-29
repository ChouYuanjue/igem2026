# Terpene Synthase Retrieval — Iteration Report

## Current decision

The active production system uses direction- and objective-specific three-seed open-world ensembles. External E2R Top-10 is a locked reciprocal-rank fusion of two independently trained neural routes: a freeze-reaction-tower model with five-neighbor transfer and a hard-negative K=128 model with three-neighbor transfer. External E2R Top-20 now uses a separately confirmed reciprocal-rank fusion of the freeze-reaction route and a nonparametric dual-kernel collaborative-support route. The Top-20 auxiliary source combines reaction similarity, the training association graph and protein sequence similarity; its locked parameters are reaction-k 50, protein-k 5, temperature 0.03, degree power 1, primary weight 0.70, auxiliary weight 0.30 and RRF constant 60. It improved the independent confirmation split from 34.77% to 43.37% Hit@20 with a paired bootstrap 95% interval of +5.02 to +12.54 percentage points. R2E Top-10/20 uses the packaged Horizyn exact-residual reaction route, while R2E Top-3 remains the reaction-loss-0.75 shortlist model. External zero-shot queries expose seed disagreement, nearest-library novelty and bootstrap-gated empirical reliability. A 5,672-sequence UniProt TPS layer remains controlled rescue only and is never free-merged into canonical ranking.

## Data and deployment

- Current proteins: 1391
- Registered MARTS proteins: 694
- Canonical protein candidate space: 2085
- Controlled UniProt TPS rescue layer: 5672
- Architecture-contract-supported registered reactions: 208
- Architecture-contract-unsupported registered reactions: 32
- Current reactions: 513
- Registered MARTS reactions: 240
- Active reaction candidate space: 753
- Rehearsal associations: 3439
- Objective/direction-specific production checkpoints: R2E shared 3 + R2E Top-3 3 + R2E Top-10/20 exact-residual 3 + E2R primary 3 + E2R hard-negative secondary 3
- Persistent user registry: 694 proteins and 240 reactions
- Wet-lab execution: 6 plates, 576 wells and 352 sequence-deduplicated master constructs

## Strict external double-cold results

Every external-enzyme/external-reaction positive pair is evaluated exactly once across the 5 × 5 protein-cluster/reaction-cluster Cartesian split.

| Direction | Model | Hit@3 | Hit@10 | Hit@20 | MRR | Median best rank |
|---|---|---:|---:|---:|---:|---:|
| enzyme_to_reaction | Current production | 4.5% | 9.7% | 15.3% | 0.048 | 125.5 |
| enzyme_to_reaction | Shared MARTS + PU | 5.2% | 18.3% | 29.5% | 0.064 | 54.0 |
| enzyme_to_reaction | E2R specialized: frozen reaction tower | 5.6% | 18.7% | 29.9% | 0.073 | 53.5 |
| reaction_to_enzyme | Current production | 1.7% | 4.2% | 7.2% | 0.021 | 374.0 |
| reaction_to_enzyme | MARTS adaptation | 3.8% | 11.4% | 18.1% | 0.044 | 158.0 |
| reaction_to_enzyme | Shared R2E MARTS + PU | 3.4% | 12.7% | 18.1% | 0.046 | 149.0 |
| reaction_to_enzyme | R2E short-list specialized: loss weight 0.75 | 4.6% | 12.7% | 17.7% | 0.047 | 158.0 |

PU masking improves both directions without expanding the model: unlabeled candidates in the same 50% identity protein cluster or reaction cluster as a positive are removed only from the contrastive denominator.

## Routing selected after adaptation

| Direction | Objective | Selected route | Hit |
|---|---|---|---:|
| enzyme_to_reaction | Top-3 | freeze-reaction + 5-neighbor hybrid (direct 0.75) | 7.8% |
| enzyme_to_reaction | Top-10 | RRF: 0.35 freeze-route + 0.65 hard-negative route, c=60 | 25.4% |
| enzyme_to_reaction | Top-20 | RRF: 0.70 freeze-route + 0.30 dual-kernel collaborative support, c=60 | 39.2% |
| reaction_to_enzyme | Top-3 | reaction-loss-0.75 direct | 4.6% |
| reaction_to_enzyme | Top-10 | Horizyn exact-residual direct | 13.5% |
| reaction_to_enzyme | Top-20 | Horizyn exact-residual direct | 19.0% |

External E2R Top-10 RRF confirmation:

| Split assignment | Query-cells | RRF Hit@10 | Previous production Hit@10 | Delta | Cell-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| legacy development/evaluation split | 268 | 25.4% | 19.4% | +5.97 pp | [+1.33, +11.51] pp |
| confirmatory fold seed 20260724 | 278 | 22.7% | 19.4% | +3.24 pp | [+0.00, +6.25] pp |
| locked confirmatory fold seed 20260725 | 283 | 21.9% | 17.7% | +4.24 pp | [+0.00, +9.48] pp |

External E2R Top-20 dual-kernel RRF confirmation:

| Split role | Query-cells | Fused Hit@20 | Previous production Hit@20 | Delta | Diagnostic / cell-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| development cells (parameter selection) | — | 49.6% | — | — | MRR 0.087 |
| original frozen 16 cells | 153 | 34.0% | 28.8% | +5.23 pp | [-1.31, +11.76] pp |
| locked independent fold seed 20260726 | 279 | 43.4% | 34.8% | +8.60 pp | [+5.02, +12.54] pp |

The identifier `20260726` is the locked fold-seed value, not an execution date. The Top-20 parameters were selected before this split was generated and were not retuned on it.

Production `auto` routing uses the reaction-loss-0.75 direct model only for external R2E Top-3, the packaged Horizyn exact-residual model for external R2E Top-10/20, and the shared PU model for current-library reactions. External E2R Top-3 uses the freeze-reaction route with direct weight 0.75. External E2R Top-10 uses locked RRF between the freeze-reaction route (five neighbors, direct weight 0.5) and the hard-negative route (three neighbors, direct weight 0.9). External E2R Top-20 uses locked RRF between the freeze-reaction route (five neighbors, direct weight 0.75) and dual-kernel collaborative support (reaction-k 50, protein-k 5, temperature 0.03, degree power 1) with weights 0.70/0.30 and constant 60. Current-library enzymes remain direct; few-shot, masked-known-association and manual overrides bypass or invalidate external reliability annotation as appropriate.

## External-query reliability

Reliability is learned only from query-grouped predictions on the strict 25-cell external double-cold benchmark. The value is an empirical ranking-reliability score, not a biochemical activity probability. Deployment requires the bootstrap 95% ROC-AUC lower bound to exceed 0.5.

| Direction | Objective | Status | CV ROC-AUC | Bootstrap 95% CI | Overall hit | Highest-score quartile hit |
|---|---|---|---:|---:|---:|---:|
| enzyme_to_reaction | Top-10 | deployed | 0.711 | [0.632, 0.787] | 25.4% | 52.2% |
| enzyme_to_reaction | Top-3 | deployed | 0.874 | [0.802, 0.936] | 7.8% | 23.9% |
| reaction_to_enzyme | Top-10 | deployed | 0.626 | [0.519, 0.731] | 13.5% | 21.7% |
| reaction_to_enzyme | Top-20 | deployed | 0.610 | [0.514, 0.704] | 19.0% | 26.7% |
| reaction_to_enzyme | Top-3 | not deployed | 0.435 | [0.270, 0.622] | 4.6% | — |
| enzyme_to_reaction | Top-20 | deployed | 0.695 | [0.627, 0.758] | 39.2% | 65.7% |

E2R Top-3 uses nearest-train protein similarity alone; E2R Top-10 RRF and Top-20 use novelty plus ensemble agreement. R2E Top-10/20 exact-residual routes use ensemble agreement and both pass the bootstrap deployment threshold; R2E Top-3 remains uncalibrated. The reliability value is a ranking-evidence score rather than biochemical activity probability. The CLI supports annotation-only, require-calibrated, require-intermediate and require-higher policies.

## Controlled UniProt candidate expansion

The UniProt layer was built from five TPS-related Pfam domains, exact-sequence deduplicated, filtered against the current/MARTS universe and compressed to 50% identity representatives. Of 6,494 representatives, 5,672 named A–D evidence-tier sequences were embedded; 822 domain-only sequences remain inactive.

Free merging is rejected under strict double-cold stress. The added sequences are unlabelled in the benchmark, so this test measures preservation of known external positives rather than UniProt activity yield.

| Budget | Canonical hits | Free-merge hits | Original hits retained | Controlled slots: canonical + UniProt | Controlled retention |
|---:|---:|---:|---:|---:|---:|
| 3 | 11 | 6 | 54.5% | 3 + 0 | 100.0% |
| 10 | 30 | 15 | 50.0% | 9 + 1 | 93.3% |
| 20 | 43 | 25 | 58.1% | 18 + 2 | 97.7% |

A/B-only expansion preserves every Top-20 cutoff hit but still worsens ranks and MRR for most queries. Adding C-tier homologs causes the major collapse. Candidate mean centering, z-scoring and local-density hub correction were tested with training-reaction statistics and rejected. Known-positive accession, exact-sequence and high-coverage MMseqs evidence define a reaction-specific Pfam architecture contract: 208 registered reactions support the five-Pfam rescue layer, while 32 remain canonical-only. Complete OSCs require PF13243+PF13249; single-domain OSC and plant-TPS fragments are excluded. The deployed policy is canonical-only Top-3, nine canonical plus one UniProt at Top-10, and eighteen canonical plus two UniProt at Top-20 only for contract-supported reactions. The full decision record is `results/terpene_uniprot_expansion_report.md`.

## Wet-lab execution design

The four canonical and two UniProt rescue plates are procured through a shared exact-sequence-deduplicated master manifest containing 352 constructs (184,501 aa). Canonical and UniProt rescue results remain separate QC scopes.

Complete reaction blocks are first assigned to plates by exact-capacity MILP. Canonical balancing moves 12 reactions, reduces summed terpene-type imbalance from 9 to 6, eliminates TPS-class imbalance (8 to 0), and reduces the mean candidate-length range from 146.8 to 6.8 aa. Rescue balancing moves 10 reactions, reduces type/class imbalance from 10/6 to 4/2, equalizes B/C/D evidence counts, and removes exact Pfam architecture imbalance: bacterial class-I range 10→0, plant-TPS-full 6→0, complete OSC 16→0.

Within the balanced reaction blocks, the original role-ordered layouts were rejected because selection roles were perfectly or strongly coupled to rows and local columns. Deterministic Hungarian assignment with seed 20260723 raises mean normalized role-slot entropy from 0.201 to 0.974, reduces the maximum single-slot role share from 100.0% to 33.3%, and reduces the maximum role slot-count range from 24 to 1. All controls and blanks remain fixed.

The randomized manifests and matching result templates are the operational inputs. The original layouts remain only for provenance.

## Rejected ablations

- Aggregated MARTS mechanism-step transfer: Hit@10 0.6% versus 12.5% for adapted direct on mechanism-covered queries. It is not deployed.
- Multiview reaction fingerprints underperformed DRFP after MARTS adaptation and are not the production reaction tower.
- CAGE sigmoid probabilities were saturated; raw logits and rank diagnostics are retained, but CAGE remains an optional structural evidence channel rather than the main ranker.
- Embedding-anchor weights 0.01, 0.05 and 0.1 all reduced strict external Hit@10. Anchor weight remains zero.
- Freezing the protein tower and adapting only the reaction tower reduced Top-10. The reverse configuration was retained only for E2R.
- Free merging all 5,672 UniProt candidates lost 42–50% of canonical cutoff hits. C/D evidence tiers are rescue-only.
- Candidate mean centering, z-scoring and local-density correction did not repair full candidate-universe expansion.
- Carbon-count or coarse domain-family compatibility was too broad: it admitted PF13243/PF13249 fragments and reactions whose reference enzymes belong to PF00348 or PF00494. Production now uses known-positive architecture contracts.
- The original reaction-to-plate allocation was rejected because TPS class, terpene type and sequence length were unevenly concentrated across plates. Execution uses exact-capacity MILP block balancing.
- Role-ordered plate placement was rejected because candidate-selection strategies were confounded with row and column positions. Execution uses deterministic balanced randomization.

## Current-database retention sanity check

This is a training-retention check, not an unbiased cold-start estimate.

| Direction | Evaluation | ΔHit@3 | ΔHit@10 | ΔHit@20 |
|---|---|---:|---:|---:|
| enzyme_to_reaction | pair_leave_other_known_masked | +0.1 pp | +0.2 pp | +0.2 pp |
| enzyme_to_reaction | query_all_known_positives | -0.2 pp | +0.0 pp | +0.0 pp |
| reaction_to_enzyme | pair_leave_other_known_masked | -3.1 pp | -1.5 pp | -0.5 pp |
| reaction_to_enzyme | query_all_known_positives | -3.1 pp | -0.2 pp | +0.0 pp |

The adapted model slightly reduces current-library Top-1/3 memorization while preserving Top-10/20, which is accepted because strict external generalization improves substantially.

## Persistent extension workflow

```bash
.venv/bin/python projects/active/terpene_screening/manage_open_world_registry.py init --force

.venv/bin/python projects/active/terpene_screening/manage_open_world_registry.py add-enzymes \
  --enzyme-id NEW_ENZYME --sequence 'MSEQUENCE...'

.venv/bin/python projects/active/terpene_screening/manage_open_world_registry.py add-reactions \
  --reaction-id NEW_REACTION --reaction-smiles 'SUBSTRATE>>PRODUCT'
```

After registration, use `rank_open_world.py` with the new ID directly. A persistent duplicate-entity integration test placed a newly registered enzyme at rank 3 and a newly registered reaction at rank 5, then removed both and restored the registry baseline.

## Validation

Deployment status: R2E shared `valid`, R2E Top-3 `valid`, R2E exact-residual `valid`, E2R primary `valid`, E2R hard-negative secondary `valid`. Models: 3 + 3 + 3 + 3 + 3; protein input: 1152; base reaction input: 2115.
