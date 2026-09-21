# Correspondence-geometry current status

Pair-specific, condition-resolved catalytic observations are much sparser than sequence and many molecular observations, while catalytically decisive state is itself incomplete. Molecular observations define label-free reaction/enzyme factor geometry; accepted positive pair observations define the current biochemical correspondence projection; the same correspondence object is read in either direction.

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

The current biological relation is **not a hierarchy**. It is a fixed-domain Pareto relation over four independently meaningful FIBRE defects: global correspondence, pocket-local ESM-C correspondence, pocket-3Di correspondence and pocket-OT correspondence. No coordinate has a hand-set weight or lexicographic priority. A candidate missing any declared coordinate is unordered in this relation rather than assigned a surrogate defect. Family-aware mechanistic coordinates (class-I aspartate, NSE/DTE, DXDD and QW) remain first-class mechanistic observations, but their sparse/applicability-specific common domain is not yet promoted into the current four-coordinate comparison family.

This change is supported by direct relation audits rather than by forcing a new total rank. In development, catalytic coordinates turn an average 68.7% of otherwise global-better comparisons into trade-offs for informative E2R queries and 86.3% for informative R2E queries. Under strict held-out-factor atlas rebuilding these values remain 68.0% and 85.7%. On queries informative in both audits, front-0 membership agrees 94.1% (E2R) and 96.3% (R2E). Strict median first-front size remains only 4.86% / 0.68% of complete candidates, so the effect is not caused by declaring nearly everything nondominated. Coverage is the limiting factor: only 34/115 E2R and 27/83 R2E strict queries contain a known positive complete on all four coordinates.

The deployed deterministic rank remains the global defect because the product still needs one stable list and previous attempts to linearize local information can regress strict-inductive retrieval. That linearization is an operational readout, not a claim that global correspondence is scientifically prior to catalytic-pocket correspondence.

The biological candidate state is now broader than the ranking coordinate family. It carries the molecular partial relation, absolute distance to accepted support, scoped catalytic-state evidence (pair mechanism plus protein-level catalytic/cofactor/site annotations), and pair-specific assay constraints when available. These objects are deliberately not fused into a new score. If a user explicitly requests an assay context and an exact canonical pair has a source-bound inactive/below-detection/no-conversion observation matching every requested dimension, that candidate is censored from eligibility rather than given a score penalty. Missing, mismatched, supporting or conflicting assay evidence cannot censor. Under the current strict source-binding policy only one canonical pair has a materialized pair-specific assay context and it is positive, so current production ranks are unchanged and condition-aware scalar ranking is not promoted.

## Partial biological relation and compatibility views

Current implementation:

- ../geometry/partial_relation.py — fixed-domain, weight-free Pareto relation across validated FIBRE coordinates;
- ../evaluation/partial_correspondence_relation.py — double-cold relation audit;
- ../evaluation/strict_inductive_partial_relation.py — reference-only held-out-factor relation audit;
- ../geometry/stratified.py — legacy-compatible coarse/local views retained for deployed provenance and historical experiments;
- ../pipelines/catalytic_consensus_geometry.py — one reaction-center manifold plus separate pocket-local ESM-C, pocket-3Di and pocket-OT geodesics;
- ../evaluation/consensus_stratified_correspondence.py — double-cold development audit;
- ../evaluation/strict_inductive_consensus.py — reference-only held-out factor rebuild and OOS-attachment audit;
- ../pipelines/mechanistic_chart_geometry.py — label-free family-aware motif coordinate atlases for the third resolution;
- ../evaluation/mechanistic_stratified_correspondence.py — double-cold development audit of the third relation;
- ../evaluation/strict_inductive_mechanistic.py — reference-only/OOS strict audit of the third relation;
- data/terpene_catalytic_consensus_geometry_v1/ — current second-resolution geometry asset;
- data/terpene_mechanistic_chart_geometry_v1/ — current generated third-resolution geometry asset.

Mechanistic development coverage is 8/115 E2R queries and 17/83 R2E queries with a refined mechanism chart. Under strict factor-atlas rebuilding, E2R retains the same 8/115 refined queries exactly, while R2E retains 11/83; all nine cells still have zero train/test protein and reaction overlap. The strict audit resolves no held-out positive at the third layer, so it supports a relation, not a new total-order rule.

Starase Navigator now returns `query.biological_relation` plus per-candidate `fibre_relation` as the primary scientific relation metadata. Online Pareto fronts are computed only **after** top-k selection and are explicitly scoped to the returned candidate set, so the Pareto relation itself cannot affect scoring, filtering or sorting. `query.assay_context_constraint` is separate: it can remove only a candidate carrying an exact pair-specific negative assay matched to an explicitly requested context, before deterministic ranking. `query.stratified_correspondence` / `fibre_resolution` remain available as backward-compatible provenance views for coarse numerical levels and mechanistic charts. Within the remaining eligible set, the deterministic numerical order still uses the global correspondence score plus the application-only same-level TPS refinement; no biological evidence field is scalar-fused into that score.

## Exact scalable sections

Current geometry/correspondence.py has dense reference and exact section
implementations. R2E/E2R can be evaluated from query-to-positive-support and
candidate-to-positive-support distances only, so the broad-domain implementation
does not require a dense reaction x protein field or factor all-pairs matrix.
Dense/section joint costs are exactly equal on the current atlas; final defect
differences are at most 1.78e-15 from floating subtraction order.

## Exact seed update

`add_positive_seed(...)` updates the same field by pointwise minima and exactly matches a full reconstruction. On the real MARTS audit the maximum absolute difference is 0. “Exact” refers to the accepted pair-level observation projection; it does not imply that annotations, weak activities and condition-resolved assays have identical biological evidential status.

Leave-one-seed development audit, excluding the seed pair itself:
- E2R: 553 improved / 1325 tied / 181 worsened; mean RR delta +0.0538.
- R2E: 371 improved / 884 tied / 55 worsened; mean RR delta +0.0302.

- Across 130 seeds, mean affected product-field fraction is about 1.08%.
  Anchor/product isolation has little association with benefit or harm, so the
  current evidence does not support density reweighting. Exact field influence
  footprint correlates with mean benefit (about 0.47 E2R / about 0.43 R2E) but
  not with worsen fraction; influence is therefore reported as stability
  provenance.
- Focused runtime now exposes `seed_update_stability`: every verified seed set
  reports the exact complete-section before/after influence footprint. Registered
  query/seed pairs additionally report intrinsic product novelty and exact
  canonical product-field single-seed influence. External seeds receive only
  the influence that is actually observable for their OOS section. Verified
  seeds are never attenuated or gated by these diagnostics.

## Numerical level sets and applicability

Authority: results/fibre_levelset_uncertainty_dev_v1/summary.json.

- E2R: 29.6% of queries have nontrivial best-positive rank intervals; neutral
  tie-expected MRR 0.07010; median/p90 best level size 2/7.
- R2E: 18.1% nontrivial; neutral tie-expected MRR 0.06288; median/p90 best
  level size 3/12.
- Starase Navigator now exposes additive query.geometric_uncertainty provenance.
  Existing deterministic rank order is unchanged.

## Enzymology stress audits

The first positive-only promiscuity stress test covers 422 enzymes and 1,555 leave-one-positive-out activities. Retaining the enzyme's other verified activities changes median expected rank from 8.0 to 5.5 and mean expected reciprocal rank from 0.248 to 0.396; 65.4% improve, 9.6% tie and 25.0% worsen. This supports exact positive-context updating as a useful promiscuity operation, but not as a monotone guarantee.

Observed mechanism-step types are available for about 90.1% of these held-out activities. On 633 high-overlap cases the median expected rank changes 6 to 4, but only two cases have disjoint observed step types. Mechanism is therefore retained as diagnostic/catalytic-state evidence rather than promoted into a gate or weighted reranker.

The geometric-locality assumption is now audited directly on the same 1,555 held-out positives. Median leave-one-out product-space distance to the remaining accepted relation is 0.823 (p90 1.021). Product distance correlates with warm expected rank at Spearman rho 0.664, while nearest-other-activity reaction distance correlates with positive-context rank improvement at rho -0.387. The nearest locality quartile improves 86.5% of held-out positives versus 48.9% in the farthest quartile. These are assumption diagnostics, not deployment thresholds or an estimate of a global biological Lipschitz constant.

A strict Rhea/UniProt cross-source audit independently corroborates 107 canonical positives with experimental UniProt catalytic evidence. Strict directed molecular identity maps only 213 canonical positive pairs to Rhea at all, so missing cross-source corroboration remains missing evidence rather than a penalty.

Context-shift, explicit-negative functional-cliff and pair-specific cofactor-holdout evaluation are currently blocked by evidence coverage rather than approximated with unknown-pair pseudo-negatives.

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

