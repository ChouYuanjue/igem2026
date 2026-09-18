# Correspondence-geometry current status

Pair labels are scarce; molecular observations are not. Molecular observations define label-free reaction/enzyme factor geometry; sparse known positive pairs define the biochemical correspondence; one scalar compatibility field on the product is read in either direction.

## MARTS / terpene canonical realization

See `CORRESPONDENCE_GEOMETRY_MAINLINE.md`.

Canonical implementation/assets:
- `product_correspondence_field.py`
- `evaluate_product_correspondence_marts_v1.py`
- `data/terpene_multiresolution_reaction_geometry_v1/`
- `data/terpene_multiresolution_protein_geometry_v4/`
- `results/terpene_product_correspondence_dev_v1/`

Internal double-cold development:
- E2R: MRR 0.07230, median rank 34, Hit@10 0.1826, Hit@20 0.3913.
- R2E: MRR 0.07469, median rank 115, Hit@10 0.1446, Hit@20 0.2410.
- All 9 development cells have zero train/test protein overlap and zero train/test reaction overlap.

Protein multiresolution geometry is canonical. Reaction-center W2 / transition-token geometry remains a catalytic-local witness rather than part of the current ranking metric because forcing it into the reaction factor causes a statistically clear E2R regression.

## Exact seed update

`add_positive_seed(...)` updates the same field by pointwise minima and exactly matches a full reconstruction. On the real MARTS audit the maximum absolute difference is 0.

Leave-one-seed development audit, excluding the seed pair itself:
- E2R: 553 improved / 1325 tied / 181 worsened; mean RR delta +0.0538.
- R2E: 371 improved / 884 tied / 55 worsened; mean RR delta +0.0302.

## Broad Rhea

The frozen broad-domain mainline remains product-flow v8 under the unchanged 1903-query / 185,918-candidate protocol. Do not reopen the spent external-retention set. Do not retroactively claim that v8 and the FIBRE correspondence defect on the MARTS application atlas are the same numerical operator; they currently share the scientific product-manifold object, not an identical solver.

## Strict inductive transfer audit

Authority: `results/terpene_product_correspondence_strict_inductive_v1/summary.json`.

For every double-cold cell, held-out proteins and reactions are removed before the
reference manifolds are built. They are then attached only from their own
query-to-reference molecular observations. All nine cells have zero train/test
protein overlap and zero train/test reaction overlap.

Current strict-inductive metrics:

- E2R: 115 queries, MRR 0.06236, median rank 38, Hit@10 0.1739, Hit@20 0.3478.
- R2E: 83 queries, MRR 0.06060, median rank 126, Hit@10 0.1325, Hit@20 0.2289.

This is a transfer audit, not a replacement for the canonical label-blind
transductive-side-information development numbers above.

## Production runtime realization

Current production reference package:
`data/terpene_correspondence_deployment_atlas_v2/`.

It is a self-contained read-only bundle for online query extension. It contains
frozen factor graph/geodesic information, canonical DRFP and ESM-C reference
statistics, local scales, reactant/product diffusion sufficient statistics,
entity order and aliases. Online requests do not rebuild the reference N x N
geometry.

Runtime authority:
`scripts/starase_navigator/retrieval/focused.py`.

The runtime supports the same field in both directions:

- registered/reference reaction or protein: exact factor-geodesic section;
- new reaction: canonical DRFP + reactant/product query-to-reference attachment;
- new protein: ESM-C query-to-reference attachment;
- verified positive seed: exact Omega pointwise-min update, including an external
  protein seed after out-of-sample attachment.

Planning and execution provenance are distinct. `planned_now` never means a
measurement affected the ranking; only `computed_now`, `provided` or
`reuse_cached` are available observations.

Verified production positives are not discarded merely because their protein
state lies outside the focused reference atlas. If a verified positive belongs
to the deployed general evidence universe and its exact sequence is already
available locally, production now retrieves that sequence through a persistent
byte-offset index, attaches the protein out of sample to the same protein
geometry, and includes it in the exact positive-set update. Multiple such
external positives are encoded in one ESM-C batch. Missing sequence still leaves
the association visible as database evidence; it simply cannot act as a
geometric seed.

## AI-native product routing

The production UI does not expose this correspondence implementation as a model,
atlas or candidate-universe choice.

The semantic planner receives the user's scientific goal and verified target
context and decides only scientific intent-level properties:

- `retrieval_scope = broad | application_domain`;
- `analysis_depth = standard | deep`.

`application_domain` is internally mapped to the strongest currently validated
focused realization; `broad` remains the open-domain route. Internal backend names
remain provenance, not user interaction concepts. Exact RHEA / UniProt / SMILES /
FASTA parsing remains deterministic because it is identity resolution, not intent
routing.

The frontend is conversation-first: no Standard/Deep selector, no route catalog,
and no architecture-facing backend choice. It may explain the actual AI-selected
search scope, evidence depth, rationale and measurements after the run. Public
`/api/rank*` requests cannot bypass this through hidden `route_mode` or
`observation_mode` fields; those legacy client fields are ignored at the HTTP
boundary.

## Current product validation

The AI-native product contract is enforced in tests: natural language and
verified target context choose broad versus application-domain retrieval and
interactive versus deep evidence acquisition; frontend/API callers do not choose
MARTS/TPS backends or observation-depth modes. The focused runtime also has
regressions for exact registered-state sections, external query attachment,
automatic verified-positive OOS extension, batched external-positive ESM-C
encoding, and candidate-side structure/pocket coverage provenance.

Live supervised-service smoke tests on 2026-09-18 also confirm the semantic
contract with the configured DeepSeek route planner. For RHEA:54512, a natural
request for the most reliable candidates selected `application_domain` and the
focused correspondence runtime; a natural request to search broadly across
enzyme families selected the 185,918-protein broad universe; and an explicit
request to use structure, pocket and mechanism information selected `deep`
evidence acquisition. None of those requests named an internal backend.

The service build identifier now includes a 12-hex SHA-256 fingerprint over the
runtime Starase Navigator source, relevant geometric query-extension code and
frontend assets, in addition to Git HEAD. This prevents an uncommitted but tested
research/product worktree from being misreported as the bare commit revision.

