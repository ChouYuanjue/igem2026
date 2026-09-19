# Product-Manifold Retrieval Evaluation Contract

## Object being evaluated

The method is one scalar compatibility field `F(r,e)` on the product manifold `M_R × M_E`. Known positive reaction–protein pairs define the sparse correspondence on that product. The canonical focused realization is the zero-temperature correspondence defect; kernel/heat or variational realizations are evaluated as members or candidate members of the operator family defined in `operator_family.md`, not silently treated as identical algorithms.

A change is mainline-compatible only if it changes the reaction metric, protein metric, sparse positive correspondence, or mathematically specified product-field operator of this same object. Routers, gates, score experts, fallback branches, fixed-prefix rerankers, and direction-specific models are not alternative evaluations of this object.

## Development protocol

- Source: `results/r2e_lambdarank_fusion_v1/clean_dev_v1/baseline_base/fold{0,1,2}` through the canonical loader in `run_r2e_lambdarank_fusion_v1.py`.
- Three frozen reaction-disjoint clean-dev folds.
- Query counts: fold0 `623`, fold1 `611`, fold2 `669`; pooled `1903`.
- Candidate universe: exactly `185,918` proteins for every query.
- Training positives may define the empirical measure; dev positives are used only after ranking for metric calculation.
- Required audit for every fold: reaction overlap `0`, exact reaction–protein pair overlap `0`.
- Full-support readout is mandatory: local numerical charts may approximate the field solve, but final ranking is over all `185,918` candidates. A fixed top-K reranking protocol is invalid.

## Metrics

Use the unchanged `evaluate_full_candidate_ranks` and `summarize_query_metrics` implementation from `broad_rhea_metrics.py`. Report at minimum pooled and per-fold:

- MRR
- MAP
- macro ROC-AUC
- NDCG@10
- Hit@10 / Hit@20 / Hit@50
- median best-positive rank


For new zero-temperature FIBRE claims, also report numerical-level-set companion
metrics whenever any best-positive level is nontrivial: optimistic, neutral
expected, and pessimistic best-positive rank; expected reciprocal rank; fraction
of queries with nontrivial rank intervals; and best-level size. The numerical
level is 64 float64 machine eps at whole-section scale. Candidate identifiers
may provide deterministic display order but must not be interpreted as
scientific score resolution.

Applicability reporting is threshold-free by default: query distance to positive
marginal support, best-level size/fraction, next distinct defect gap, and
candidate-support distance within the best level. These values are not
calibrated probabilities or OOD tiers.


Paired comparisons must align exactly on `(fold, query_id)`. The standard paired bootstrap uses `20,000` query bootstrap replicates with seed `20260917` and reports the point delta, 95% interval, and `P(delta > 0)`.

## Stratified-output promotion

A finer catalytic or mechanistic FIBRE coordinate may be reported without changing the canonical total rank. This is the default when the local relation is scientifically meaningful but has not passed the strict-inductive non-degradation gate.

Any claim that a local stratum is **order-bearing** must satisfy all of the following under the same matched protocol:

- cross-coarse-level order is invariant by construction;
- missing local observations are exactly neutral;
- no learned or hand-set scalar modality weight is introduced merely to obtain non-degradation;
- double-cold development is reported;
- held-out factor atlases are rebuilt reference-only and queries are attached out of sample;
- per-query improve/tie/worse counts and paired confidence intervals are reported in both directions.

Until those conditions pass, retrieval metrics are computed from the canonical coarse total rank, while catalytic strata are evaluated as additional resolution/coverage/stability outputs rather than silently linearized.

## Numerical validity

For the canonical zero-temperature defect, exact section evaluation must match the dense reference joint min-plus transform and differ only within the declared numerical-level tolerance after floating arithmetic. Candidate/support batching may change memory use only.

For smooth nonlinear variational/flow members of the operator family:

- convergence tolerance: `1e-5` relative change;
- `128` iterations is a safety cap, not an early-stopping hyperparameter;
- report converged query count, cap-hit count, median/p90 steps, and maximum final relative change;
- the numerical chart scale uses the intrinsic sampling rule `ceil(sqrt(N))` unless a run is explicitly marked diagnostic and excluded from final claims.

## Baseline authority

BiME-Rank is a matched-protocol reference, not an implementation constraint. The frozen clean-dev authority is:

`results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/development_oof_query_metrics.csv`

The previous unified geometric mainline is:

`results/geometric_product_flow_clean_dev_v1/development_oof_query_metrics.csv`

Comparisons must not silently substitute historical, quarantined, fixed-pool, top-K, different-support, or post-reveal evaluations.

## External retention

The one-time strict Rhea128→141 double-cold retention has already been spent. It is not a development set. No subsequent geometry, measure, solver, chart, or source choice may be selected using its labels or metrics. New mainline variants are selected on clean-dev only; external retention is left untouched unless a new genuinely final retention round is explicitly frozen in advance.

## Missing observations

Missing structure or missing geometric views mean missing observations. They must not create negative affinity, negative labels, penalties, or gate decisions. Known positives remain the only empirical pair observations.
