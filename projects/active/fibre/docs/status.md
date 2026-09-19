# Correspondence-geometry current status

Pair labels are scarce; molecular observations are not. Molecular observations define label-free reaction/enzyme factor geometry; sparse known positive pairs define the biochemical correspondence; one scalar compatibility field on the product is read in either direction.

## MARTS / terpene canonical realization

See `method.md` and `operator_family.md`.

Canonical implementation/assets:
- `../geometry/correspondence.py`
- `../evaluation/correspondence.py`
- `data/terpene_multiresolution_reaction_geometry_v1/`
- `data/terpene_multiresolution_protein_geometry_v4/`
- `results/terpene_product_correspondence_dev_v1/`

Internal double-cold development:
- E2R: MRR 0.07230, median rank 34, Hit@10 0.1826, Hit@20 0.3913.
- R2E: MRR 0.07469, median rank 115, Hit@10 0.1446, Hit@20 0.2410.
- All 9 development cells have zero train/test protein overlap and zero train/test reaction overlap.

Protein multiresolution geometry is canonical. Reaction-center W2 / transition-token geometry does **not** enter the global reaction metric: even fixed-topology intrinsic-information tangent refinement regresses E2R from MRR 0.07230 to 0.04923 with a paired bootstrap interval entirely below zero. The observation is nevertheless retained inside FIBRE as a finer catalytic-local coordinate rather than discarded or turned into a separate expert.

The current biological hierarchy is global correspondence -> catalytic-pocket strata -> family-aware mechanistic charts. The catalytic layer reuses the same zero-temperature defect on reaction-center geometry and pocket-local ESM-C / pocket-3Di / pocket-OT coordinates. The third layer now also reuses that operator independently on class-I aspartate, NSE/DTE, DXDD and QW coordinates inside an already-observed catalytic parent; family applicability defines the chart, different charts are incomparable, and missing motif observations remain neutral. Development-only total-order refinements are encouraging but do not pass the strict-inductive non-degradation gate: pocket-only refinement gives 5 improved / 110 tied / 0 worsened E2R queries in development but 0 / 112 / 3 under strict factor-atlas rebuilding; Pareto consensus gives 5 / 109 / 1 in development and 1 / 110 / 4 under strict rebuilding. R2E is unchanged in these strict audits. Consequently the finer relations are current **model outputs but non-order-bearing**; the canonical deterministic rank remains the global defect.

## Stratified biological correspondence

Current implementation:

- ../geometry/stratified.py — coarse levels, local refinement and weight-free Pareto-consensus catalytic strata;
- ../pipelines/catalytic_consensus_geometry.py — one reaction-center manifold plus separate pocket-local ESM-C, pocket-3Di and pocket-OT geodesics;
- ../evaluation/consensus_stratified_correspondence.py — double-cold development audit;
- ../evaluation/strict_inductive_consensus.py — reference-only held-out factor rebuild and OOS-attachment audit;
- ../pipelines/mechanistic_chart_geometry.py — label-free family-aware motif coordinate atlases for the third resolution;
- ../evaluation/mechanistic_stratified_correspondence.py — double-cold development audit of the third relation;
- ../evaluation/strict_inductive_mechanistic.py — reference-only/OOS strict audit of the third relation;
- data/terpene_catalytic_consensus_geometry_v1/ — current second-resolution geometry asset;
- data/terpene_mechanistic_chart_geometry_v1/ — current generated third-resolution geometry asset.

Mechanistic development coverage is 8/115 E2R queries and 17/83 R2E queries with a refined mechanism chart. Under strict factor-atlas rebuilding, E2R retains the same 8/115 refined queries exactly, while R2E retains 11/83; all nine cells still have zero train/test protein and reaction overlap. The strict audit resolves no held-out positive at the third layer, so it supports a relation, not a new total-order rule.

Starase Navigator returns query.stratified_correspondence plus per-candidate fibre_resolution. These fields now distinguish coarse numerical level, whether the catalytic parent was observed, catalytic stratum when refined, family-applicable mechanistic chart, actually observed mechanistic coordinates, and mechanistic stratum when refined. They are explicitly marked non-order-bearing; the existing rank function still uses only the global correspondence score.

## Exact scalable sections

Current geometry/correspondence.py has dense reference and exact section
implementations. R2E/E2R can be evaluated from query-to-positive-support and
candidate-to-positive-support distances only, so the broad-domain implementation
does not require a dense reaction x protein field or factor all-pairs matrix.
Dense/section joint costs are exactly equal on the current atlas; final defect
differences are at most 1.78e-15 from floating subtraction order.

## Exact seed update

`add_positive_seed(...)` updates the same field by pointwise minima and exactly matches a full reconstruction. On the real MARTS audit the maximum absolute difference is 0.

Leave-one-seed development audit, excluding the seed pair itself:
- E2R: 553 improved / 1325 tied / 181 worsened; mean RR delta +0.0538.
- R2E: 371 improved / 884 tied / 55 worsened; mean RR delta +0.0302.

- Across 130 seeds, mean affected product-field fraction is about 1.08%.
  Anchor/product isolation has little association with benefit or harm, so the
  current evidence does not support density reweighting. Exact field influence
  footprint correlates with mean benefit (about 0.47 E2R / about 0.43 R2E) but
  not with worsen fraction; influence is therefore reported as stability
  provenance.

## Numerical level sets and applicability

Authority: results/fibre_levelset_uncertainty_dev_v1/summary.json.

- E2R: 29.6% of queries have nontrivial best-positive rank intervals; neutral
  tie-expected MRR 0.07010; median/p90 best level size 2/7.
- R2E: 18.1% nontrivial; neutral tie-expected MRR 0.06288; median/p90 best
  level size 3/12.
- Starase Navigator now exposes additive query.geometric_uncertainty provenance.
  Existing deterministic rank order is unchanged.

## Broad Rhea

The frozen broad-domain mainline remains product-flow v8 under the unchanged 1,903-query / 185,918-candidate protocol. A full internal-only matched audit of an exact scalable zero-temperature FIBRE section has now been completed on all three folds. Pooled zero-temperature metrics are MRR 0.12911, MAP 0.10716, NDCG@10 0.13605, Hit@10/20/50 0.30478/0.37940/0.44456 and median best-positive rank 121; frozen v8 is 0.14442, 0.12642, 0.15268, 0.34314/0.44666/0.58276 and rank 28. The paired MRR delta is -0.01530 with 95% bootstrap interval [-0.02768,-0.00274], and all three folds regress in MRR. Promotion is therefore rejected.

The exact broad implementation is retained because it proves that zero-temperature FIBRE can be evaluated exactly over the full 185,918-candidate universe without a dense reaction x protein field: identical train-reaction protein-support sets collapse to about 5.6k lossless groups, with GPU-vs-dense distance parity below 1.6e-6. The unresolved difference is numerical/operator-level, not a scalability excuse. Do not reopen the spent external-retention set and do not add a hybrid score or direction-specific gate to force numerical agreement.

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

