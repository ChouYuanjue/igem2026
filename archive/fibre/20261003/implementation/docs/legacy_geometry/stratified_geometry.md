# FIBRE stratified geometry — compatibility and historical view

This document records the earlier global-first implementation and the experiments
that motivated the current partial biological relation. It is retained because
Starase Navigator still exposes backward-compatible stratified provenance fields
and because the failed/safe linearization experiments are scientifically useful.

The current scientific definition is in partial_relation.md and method.md.

## What the stratified view did

The earlier implementation treated the global correspondence defect Delta_0 as
a coarse numerical relation. Machine-stable equal values of Delta_0 formed
coarse levels. Reaction-center × pocket correspondence could refine candidates
only inside one such level, and family-aware mechanistic coordinates could
refine still smaller observed cells.

This design had two important strengths:

- it guaranteed that adding local biological information could never change the
  deployed global rank;
- it made missing local observations neutral rather than negative.

But it also encoded a scientific assumption that is no longer accepted: global
correspondence became the semantic parent of every local coordinate. A candidate
that was globally better could never become incomparable with a candidate that
was catalytically better unless the two happened to be in the same numerical
global level.

## Historical catalytic consensus

The catalytic-local implementation used the same correspondence operator on a
reaction-center manifold and three active-site/pocket protein manifolds:

- pocket-local ESM-C sequence state;
- pocket 3Di structural state;
- pocket OT geometric state.

Inside one eligible coarse level, one candidate dominated another only when it
was no worse in every local coordinate and strictly better in at least one.
No scalar modality weight was introduced.

Development and strict-inductive experiments showed that this was safer than
scalar fusion but still unsuitable as a new total-order rule. The consensus
linearization produced E2R 5 improved / 109 tied / 1 worsened queries in
development and 1 / 110 / 4 after reference-only factor rebuilding, with a very
small negative strict mean RR delta. R2E remained unchanged.

These experiments remain valid evidence that local biological coordinates should
not be forced into the deployed rank merely because they exist.

## Historical mechanistic chart

Family-aware class-I aspartate, NSE/DTE, DXDD and QW coordinates were represented
as applicability-specific mechanistic charts. A non-applicable motif was outside
the chart; an applicable but unobserved motif remained missing.

On the nine double-cold cells, the chart relation refined 8/115 E2R and 17/83
R2E queries in development. Under reference-only factor rebuilding, E2R retained
the same 8/115 refined queries and R2E retained 11/83. No held-out positive was
mechanistically resolved under the strict audit.

This remains useful mechanistic provenance, but the sparse/family-specific
common domain is not currently part of the validated four-coordinate Pareto
comparison family.

## Why the current method moved beyond stratification

The current relation keeps the useful parts of the old design—same
correspondence operator, no score weights, explicit missingness—but removes the
global-first hierarchy.

For one query, the validated comparison family is:

    (Delta_global,
     Delta_pocket_ESMC,
     Delta_pocket_3Di,
     Delta_pocket_OT).

Candidate a dominates candidate b only when it is no worse in all four
coordinates and strictly better in at least one. Global/local disagreement
therefore becomes a genuine trade-off even across different global numerical
levels.

Under the strict held-out-factor audit, this relation converts an average 68.0%
of otherwise global-better comparisons into trade-offs for informative E2R
queries and 85.7% for informative R2E queries. On queries informative in both
development and strict audits, first-front membership agrees 94.1% and 96.3%.
The strict median first-front size is only 4.86% / 0.68% of complete candidates.

## Runtime compatibility

Starase Navigator still exposes stratified_correspondence and fibre_resolution.
These fields are compatibility/provenance views: global numerical levels, older
catalytic strata and mechanistic charts remain inspectable for reproducibility.

The primary current relation metadata is biological_relation plus per-candidate
fibre_relation. Online Pareto fronts are computed only after top-k selection and
are explicitly relative to the returned set. They never feed back into scoring,
filtering or sorting.

The deterministic product list remains the validated global correspondence
linearization. That is an operational readout, not the ontology of FIBRE.
