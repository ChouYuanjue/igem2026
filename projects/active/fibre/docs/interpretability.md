# Biological Interpretability Contract for FIBRE

## Principle

FIBRE exposes one biochemical correspondence object through several molecular
coordinates. It does not attach independent biological experts after retrieval,
and it does not assume that those coordinates form a hierarchy.

The current scientific comparison family contains four correspondence defects:

1. global reaction-enzyme correspondence;
2. reaction-center × pocket-local ESM-C correspondence;
3. reaction-center × pocket-3Di correspondence;
4. reaction-center × pocket-OT correspondence.

The deterministic production rank still uses the global defect. That is an
operational linearization of the object, not a claim that global similarity is
biologically more authoritative than active-site compatibility.

Missing Pfam, motif, structure, reaction-center, pocket or assay-context
information is absence of observation, never evidence against catalysis.

## Global correspondence coordinate

The global reaction factor represents whole-reaction chemical state. The global
protein factor is a partially observed multiresolution molecular-state geometry
whose complete base is ESM-C sequence and whose observed structural coordinates
include whole-structure 3Di together with other validated molecular
measurements.

Known enzyme-reaction pairs define the sparse positive correspondence on the
product space. R2E and E2R are sections of the same correspondence defect; they
are not different ranking models.

Reaction-center transition geometry is deliberately not injected into the
global reaction metric. Matched experiments, including fixed-topology tangent
refinement, show a clear E2R regression when it is allowed to deform the global
reaction geodesic.

## Partial biological relation

Reaction-center information is nevertheless part of FIBRE. The same sparse
correspondence construction is evaluated on reaction-center geometry and three
active-site/pocket protein geometries.

For one query, let

    D(x) = (
        Delta_global(x),
        Delta_pocket_ESMC(x),
        Delta_pocket_3Di(x),
        Delta_pocket_OT(x)
    ).

All quantities are defect-like and lower is better. Candidate a dominates
candidate b only when a is no worse in every declared coordinate and strictly
better in at least one. If one candidate is globally better but locally worse,
the pair is a biological trade-off rather than a forced total-order decision.

This relation has three important constraints:

- no coordinate has a learned or hand-set scalar fusion weight;
- no coordinate has lexicographic priority;
- a candidate missing any declared coordinate is left unordered in this
  relation rather than imputed, penalized or compared on a pair-specific subset.

The fixed-domain rule matters mathematically: allowing every candidate pair to
choose a different coordinate intersection can make the comparison rule
non-transitive and can turn missingness itself into an ordering signal.

## What strict-inductive validation established

The new relation was audited on exactly the same double-cold query cells and
then rebuilt under the stronger held-out-factor protocol.

Development:
- E2R: 47/115 queries have a known positive complete on all four coordinates.
  Among queries with globally better complete candidates, an average 68.7% of
  those comparisons become trade-offs.
- R2E: 32/83 informative queries; the corresponding fraction is 86.3%.

Strict held-out-factor reconstruction:
- E2R: 34/115 informative queries; the blocked-global-dominance fraction remains
  68.0%.
- R2E: 27/83 informative queries; the fraction remains 85.7%.
- On queries informative in both audits, front-0 membership agrees 94.1% for
  E2R and 96.3% for R2E.
- The median first Pareto front contains only 4.86% of complete E2R candidates
  and 0.68% of complete R2E candidates.

Thus the result is not obtained by simply declaring nearly everything
incomparable, and the trade-off structure survives reference-only atlas
rebuilding and out-of-sample attachment.

Coverage remains the main limitation, which is why missing coordinates stay
explicitly unresolved.

## Mechanistic coordinates and assay context

Family-aware motif coordinates describe catalytic mechanism context rather than
a universal vote for activity. Current application coordinates include class-I
aspartate-rich context, NSE/DTE, DXDD and QW where biologically applicable.

These coordinates remain first-class observations of the same molecular object,
but they are not currently part of the common four-coordinate comparison family:
their applicable/observed domain is substantially sparser and family-specific.
A non-applicable motif is outside the chart; an applicable but unobserved motif
is missing. Neither becomes a penalty.

The same rule applies to cofactors, metals, pH, temperature, kinetic endpoints
and explicit inactivity. The observation layer records them with provenance,
but a field becomes an order-comparison coordinate only after coverage,
endpoint semantics and strict transfer have been established.

When assay context becomes sufficiently dense, the natural extension is a
conditioned correspondence object on reaction × enzyme × context, not an
additive pH/temperature score expert. Condition-scoped inactivity is then a
censored observation under that context rather than a permanent global negative
edge.

## Operational deterministic rank

Starase Navigator still needs a stable list for product use. The deployed rank
therefore remains the validated global correspondence order.

This does not make global correspondence the parent of the biological relation.
It is simply the conservative total-order readout that already passed the
existing retrieval validation. Previous attempts to use local geometry as a
total-order refinement can regress strict-inductive E2R, so no coefficient,
gate or lexicographic rule is introduced merely to force a new ranking.

Rank preservation is guaranteed by the runtime interface: the returned-set
partial relation is computed only after top-k selection and cannot feed back
into score computation, candidate filtering or sorting.

## Biological witnesses

A predicted pair may expose known positive precedents that minimize the same
joint product-space support cost. A witness is therefore a readable realization
of the model's existing support, not a separate scoring system.

Human-readable measurements may accompany the witness when available:
reaction-center similarity, sequence alignment, structural similarity,
pocket-local observations, source-linked UniProt annotations, MARTS mechanism
steps and family-aware catalytic motifs. Weak or missing observations are not
hidden.

The distinction is:

- correspondence coordinates say how the model compares hypotheses;
- the Pareto relation says which comparisons are actually justified jointly;
- witnesses say which known biochemical precedents support those coordinates;
- annotations say what biology is observed and where it came from.

## Applicability and uncertainty

The global zero-temperature defect naturally contains numerical level sets.
FIBRE reports rank intervals and intrinsic support geometry rather than
manufacturing a calibrated confidence score.

Current threshold-free quantities include query distance to positive marginal
support, best-level size/fraction, next-level defect gap and candidate-support
distance inside the best level. These remain useful even though the scientific
relation is no longer defined by nesting local geometry inside global levels.

geometric_uncertainty is not a probability or OOD classifier.

## Runtime contract

Starase Navigator exposes:

- biological_relation: the primary non-order-bearing relation metadata;
- fibre_relation: per-returned-candidate Pareto-front/completeness metadata;
- geometric_uncertainty: numerical resolution/support diagnostics;
- stratified_correspondence and fibre_resolution: backward-compatible
  provenance views for the earlier coarse/local/mechanistic implementation.

Online Pareto fronts are explicitly scoped to the returned candidate set and
must not be interpreted as the full-atlas front. Full scientific evaluation is
performed by the offline development and strict-inductive relation audits.

## Promotion rule for a new coordinate

A new biological coordinate may join the common order-comparison family only
when:

- it is a coordinate of the same correspondence object rather than an
  independent score expert;
- its biological endpoint and applicability domain are explicit;
- missing observations remain unordered rather than negative or imputed;
- no arbitrary scalar modality weight or lexicographic priority is introduced;
- matched double-cold evaluation is reported;
- reference-only held-out factor atlases are rebuilt and external entities are
  attached out of sample;
- the resulting relation remains informative rather than collapsing into one
  enormous nondominated set.

Promotion into the deterministic production rank is a separate question and is
not required for a coordinate to be scientifically part of FIBRE.

## What a biologist should read from one result

For one predicted pair:

1. inspect the global rank/defect as the current stable retrieval readout;
2. inspect fibre_relation: if two candidates trade off across global and
   catalytic coordinates, the model is explicitly declining to claim that the
   globally better one is biologically superior;
3. inspect coordinate completeness before interpreting a Pareto front;
4. inspect mechanistic/context observations to understand which catalytic
   mechanism and assay facts are actually present;
5. inspect positive witnesses, source provenance and geometric support.

None of these proves catalytic activity. Experimental validation remains the
final test.

## Method boundary

FIBRE does not claim that geometric proximity proves catalysis, that a predicted
structure is experimental evidence, that a motif is sufficient for function,
or that an intrinsic support coordinate is a calibrated probability.

Its interpretability claim is narrower: the same mathematical object that
produces retrieval also exposes which biochemical comparisons it can justify,
which comparisons remain genuine trade-offs, which molecular observations are
present, and which known positive precedents support them.
