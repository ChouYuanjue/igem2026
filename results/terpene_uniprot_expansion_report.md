# UniProt TPS Candidate Expansion — Evaluation and Deployment Decision

## Decision

The full UniProt TPS layer is not merged into the canonical production ranking. The active policy keeps the current+MARTS ranking as an unchanged prefix and exposes UniProt candidates only through validated tail slots: zero slots at Top-3, one at Top-10 and two at Top-20. A separate four-candidate-per-reaction wet-lab rescue campaign remains available for deliberate discovery experiments.

## Candidate construction

The source query uses the five TPS-related Pfam domains used by the MARTS curation workflow and excludes fragments and sequences outside 200–1000 aa.

| Stage | Count |
|---|---:|
| Raw UniProt rows | 46,064 |
| Valid normalized rows | 45,780 |
| Exact-sequence unique rows | 44,961 |
| Novel after existing ID/sequence removal | 43,812 |
| 50% identity clusters | 6,494 |
| Named primary embedding candidates | 5,672 |
| Domain-only rescue candidates | 822 |

Primary-layer evidence tiers:

- `D_named_predicted`: 3,061
- `C_homology_named`: 2,315
- `B_experimental_or_transcript_named`: 294
- `A_reviewed`: 2

ESM-C embeddings: 5,672/5,672 completed, 0 failed and 0 missing. The extractor uses length-bucketed low-level ESM-C transformer batches and is numerically aligned with the original SDK path.

## Free-merge strict double-cold stress test

The 5,672 UniProt candidates are treated as unlabelled decoys because the strict MARTS benchmark contains no labels for them. This test therefore measures preservation of known external positives, not UniProt activity yield.

| Budget | Canonical Hit | Free-expanded Hit | Original hits retained | Median positive rank: canonical → expanded |
|---:|---:|---:|---:|---:|
| 3 | 4.6% | 2.5% | 54.5% | 158 → 761 |
| 10 | 12.7% | 6.3% | 50.0% | 158 → 761 |
| 20 | 18.1% | 10.5% | 58.1% | 149 → 815 |

Free merging loses 42–50% of the canonical cutoff hits and increases the median true-positive rank by several hundred positions. It is rejected.

## Evidence-tier and architecture-contract ablation

| Budget | Evidence layer | Unconstrained retention | Contract retention, all strict queries | Contract retention, supported queries | Supported-query MRR ratio | Supported median rank inflation |
|---:|---|---:|---:|---:|---:|---:|
| 3 | A/B only | 90.9% | 90.9% | 88.9% | 0.939 | 33 |
| 3 | A/B/C | 63.6% | 63.6% | 55.6% | 0.533 | 326 |
| 3 | A–D | 54.5% | 54.5% | 44.4% | 0.457 | 563 |
| 10 | A/B only | 96.7% | 96.7% | 95.5% | 0.939 | 33 |
| 10 | A/B/C | 63.3% | 63.3% | 50.0% | 0.533 | 326 |
| 10 | A–D | 50.0% | 56.7% | 40.9% | 0.457 | 563 |
| 20 | A/B only | 100.0% | 100.0% | 100.0% | 0.899 | 30 |
| 20 | A/B/C | 65.1% | 74.4% | 60.7% | 0.519 | 339 |
| 20 | A–D | 58.1% | 67.4% | 50.0% | 0.435 | 553 |

On the 199 contract-supported strict queries, the architecture-constrained A/B layer retains all 28 Top-20 hits and 21/22 Top-10 hits. It still reduces Top-20 MRR to 0.899 of canonical and moves the median true positive back by 30 ranks, so it is not allowed to freely reorder the prefix. Adding C-tier homologs is the main failure boundary: contract-constrained A/B/C retains only 17/28 Top-20 hits on supported queries.

## Rejected hub-normalization methods

Candidate mean centering, candidate z-scoring and top-20 local-density correction were computed from training-reaction scores only. None improved the full A–D expansion enough for deployment.

| Budget | Best full-expansion normalization | Hit retention | MRR ratio to canonical | Median rank change |
|---:|---|---:|---:|---:|
| 3 | raw | 54.5% | 0.565 | 590 |
| 10 | candidate_mean_centered | 53.3% | 0.480 | 634 |
| 20 | raw | 58.1% | 0.538 | 648 |

## Reaction-specific architecture contracts

Known positive enzymes define the admissible Pfam architecture for each registered reaction. Accession matches are preferred, followed by exact sequence matches and then high-coverage MMseqs matches. The resulting contracts support 208 of 240 registered reactions; 32 reactions belong to enzyme families outside the five-Pfam expansion or lack sufficiently reliable mapping.

Complete OSCs require PF13243+PF13249. PF13249-only OSC fragments and single PF01397/PF03936 plant-TPS fragments are excluded. Unsupported reactions receive no UniProt tail slots and remain canonical-only.

## Controlled rescue quota

Canonical candidates occupy the ranking prefix and UniProt candidates occupy only reserved tail slots. The quota is applied only when the reaction architecture contract is supported.

| Budget | Canonical slots | UniProt slots | Strict hit retention | Resulting strict Hit |
|---:|---:|---:|---:|---:|
| 3 | 3 | 0 | 100.0% | 4.6% |
| 10 | 9 | 1 | 93.3% | 11.8% |
| 20 | 18 | 2 | 97.7% | 17.7% |

Batch validation across all registered reactions:

- Query/objective combinations: 720
- Contract-supported reactions: 208
- Contract-unsupported reactions kept canonical-only: 32
- Actual UniProt tail rows: 624
- Unique selected UniProt candidates: 285
- Maximum selected candidate usage: 9 reactions
- Known-association leakage: 0
- Canonical-prefix mismatches: 0

## Wet-lab rescue campaign

The separate campaign contains 96 UniProt candidates across 24 reactions, using 2 complete 96-well plates. It contains 93 unique candidates, and no candidate is used for more than 2 reactions.

Of the original 24 balanced targets, 19 were retained and 5 were replaced by supported reactions of the same terpene type. Before role selection, 421 unique candidates were excluded for conservative complete-architecture length, composition, hydrophobicity or residue risks.

The final 93 unique UniProt sequences contain 0 high-confidence sequence risks and 0 complete-architecture length risks. Motif absence remains annotation-only because the reviewed reference set does not support using exact motif regexes as a hard activity filter.

Selection balances an evidence anchor, a named homology candidate, a named predicted candidate and an ESM-C diversity candidate. This campaign is a discovery experiment and is not interpreted as calibrated probability output.

## Active artifacts

- `results/terpene_uniprot_controlled_rescue_batch/controlled_rankings.csv`
- `projects/active/terpene_screening/rank_uniprot_rescue.py`
- `results/terpene_uniprot_rescue_campaign/assay_manifest.csv`
- `results/terpene_uniprot_rescue_campaign/assay_results_template.csv`
- `results/terpene_uniprot_rescue_campaign/sequence_deduplicated_constructs.fasta`

## Final deployment rule

1. Top-3 remains canonical-only.
2. A contract-supported reaction may append one architecture-compatible UniProt candidate after nine canonical candidates at Top-10.
3. A contract-supported reaction may append two architecture-compatible UniProt candidates after eighteen canonical candidates at Top-20.
4. Contract-unsupported reactions remain canonical-only at every cutoff.
5. A/B/C/D evidence, exact Pfam architecture, contract provenance and historical hub frequency are reported for every UniProt row.
6. Reliability scores trained on the 2,085-candidate canonical universe are not reused after candidate expansion.
7. PF13249-only/PF01397-only/PF03936-only fragments and the 822 domain-only sequences remain outside the active rescue layer.
