# FIBRE stability, level sets, and geometric applicability

FIBRE does not attach a separate confidence classifier to the compatibility
field. Stability and uncertainty are properties of the same geometric object.

## Numerical level sets are part of the contract

The canonical defect is a min-plus distance construction. Exact or
machine-scale-equal values are therefore expected: they are not evidence that
candidate identifiers should decide scientific preference.

For one query section, the numerical level tolerance is

    64 * eps_float64 * max(1, max |Delta|).

Candidates in the same level are scientifically unresolved by the scalar defect
at machine-stable precision. Deterministic product output may still use a stable
identifier order, but evaluation and provenance must distinguish that display
order from score resolution.

For a positive contained in a level with t candidates and m positives after b
strictly better candidates, FIBRE reports:

- optimistic best-positive rank b + 1;
- pessimistic best-positive rank b + (t-m) + 1;
- neutral expected best-positive rank b + (t+1)/(m+1);
- exact expected reciprocal rank under a uniform within-level permutation.

This is not randomization of the deployed result. It is an uncertainty
description of what the scalar field itself resolves.

### Current double-cold audit

Authority: results/fibre_levelset_uncertainty_dev_v1/summary.json.

- E2R: 115 queries; 29.6% have a nontrivial best-positive rank interval.
  Neutral expected MRR is 0.07010. Median best numerical level size is 2 and
  p90 is 7.
- R2E: 83 queries; 18.1% have a nontrivial best-positive rank interval.
  Neutral expected MRR is 0.06288. Median best numerical level size is 3 and
  p90 is 12.
- Exact section evaluation and dense reference evaluation have identical joint
  min-plus costs on the current atlas. Their final field differs by at most
  1.78e-15 from floating subtraction order, enough to permute candidates
  inside exact/near-exact levels. This is why level-set semantics are required
  for solver-independent scientific equivalence.

Frozen historical metrics are not retroactively replaced. New zero-temperature
FIBRE claims should report tie-aware companion metrics whenever level sets are
nontrivial.

## Geometric applicability is descriptive, not a threshold

One FIBRE section exposes the following threshold-free quantities:

- query distance to the observed positive marginal support;
- size and fraction of the best numerical defect level;
- gap to the next distinct defect level;
- nearest and median marginal-support distance among candidates in the best
  level.

These are returned by geometry/levelset.py and by the focused Starase Navigator
runtime under query.geometric_uncertainty.

They are not probabilities, OOD labels, or calibrated tiers. A future
calibration layer may be evaluated separately, but it must not be confused with
the intrinsic geometry.

## Positive-seed updates are exact but not globally monotone

Adding a verified positive updates the three min-plus transforms by pointwise
minima and exactly equals full reconstruction. That algebraic exactness does not
imply that every unrelated query ranking improves.

Authority:
results/terpene_correspondence_seed_update_audit_v1/summary.json.

Across 130 leave-one-seed interventions:

- the incremental/full-reconstruction field difference remains exactly zero;
- one seed changes about 1.08% of product-field entries on average;
- mean unrelated-query RR worsen fraction is about 8.1% for E2R and 4.4% for
  R2E, while mean RR change remains strongly positive overall;
- simple anchor isolation/density does not explain the side effects:
  product-isolation Spearman correlation with mean RR delta is about 0.005 for
  E2R and -0.130 for R2E;
- actual field influence footprint correlates with mean benefit
  (about 0.47 E2R and 0.43 R2E), but is essentially uncorrelated with worsen
  fraction.

Therefore the current evidence does not justify a density penalty or
hand-weighted positive anchor. Verified positives remain exact observations.
FIBRE instead records the update's intrinsic novelty and exact influence
footprint as stability provenance.

This provenance is now part of the focused runtime contract under
`query.seed_update_stability` and the public retrieval response under
`ranking.seed_update_stability`. For every applied seed set, the runtime compares
the complete candidate fibre before and after the exact pointwise-min update and
reports decreased/unchanged/increased defect counts, mean/max defect change, and
the affected-candidate fraction. When both the query and an individual seed are
registered in the focused atlas, the runtime additionally reports that seed's
reaction-, protein- and product-support novelty together with its exact global
product-field influence relative to canonical Omega. External/OOS seeds still
receive exact query-section provenance, but are not assigned fictitious
canonical global influence. Multiple-seed combined effects are reported from the
actual updated section rather than by adding single-seed influences.

## Reaction-local observations refine interpretation, not the current factor metric

Reaction-center W2 and explicit transition-token observations are chemically
valid local measurements. Multiple increasingly conservative attempts to place
them in the reaction factor have failed the matched development gate.

The current strongest attempt fixes the entire canonical global-reaction edge
set and lets local observations only add tangent energy to existing edges. View
reliability is the label-free local entropy deficit of that view; flat views
therefore fade automatically. It guarantees:

- no new long-range edge;
- no removed base edge;
- missing local observation is exactly neutral;
- duplicate identical refinement is idempotent;
- refinement cannot shorten a base edge.

Authority:
results/fibre_reaction_tangent_information_dev_v1/summary.json.

Even under these restrictions, E2R MRR falls from 0.07230 to 0.04923 with
paired 95% bootstrap CI for the RR delta entirely below zero. R2E also has no
reliable gain. Thus reaction-center observations do not enter the **global reaction geodesic**. The negative result is a design constraint, not an invitation to add another score expert.

They now re-enter the same FIBRE object at a different resolution. Inside a global numerical level, reaction-center geometry is paired with active-site/pocket protein geometry and evaluated with the same correspondence operator. The resulting catalytic strata are currently non-order-bearing because both scalar pocket refinement and Pareto-consensus refinement retain small strict-inductive E2R regressions. This distinction is important: the local biology is part of the model, while the conservative global linearization protects the validated total rank.

See stratified_geometry.md.
