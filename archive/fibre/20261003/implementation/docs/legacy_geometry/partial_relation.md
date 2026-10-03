# FIBRE partial biological relation

FIBRE's scientific output is not a hierarchy of evidence levels and is not a weighted ensemble of biological experts. It is one sparse biochemical correspondence object observed through several molecular coordinates.

## Fixed coordinate family

For a fixed query q and candidate x, let

    D(q,x) = (Delta_global, Delta_pocket_esmc, Delta_pocket_3di, Delta_pocket_ot, ...)

contain defect-like coordinates that have individually earned a place in the FIBRE object. The currently validated order-comparison family contains the global correspondence defect plus the three catalytic-pocket correspondence defects.

No coordinate is declared more important than another. In particular, global correspondence is not a lexicographic parent of pocket correspondence.

Candidate a dominates candidate b only when a is no worse in every coordinate of the declared family and is strictly better in at least one, using only the existing machine-scale numerical tolerance. Otherwise the candidates are either numerically equal or biologically incomparable on that family.

## Fixed domain and missing observations

Pareto comparison is performed on a fixed declared coordinate family. A candidate missing any coordinate in that family is not assigned an imputed defect and is not compared in that relation. This avoids pair-dependent coordinate intersections, hidden missingness penalties and non-transitive comparison rules.

Missing therefore means unresolved, not negative.

## Why this replaces the old hierarchy

The earlier stratified implementation allowed catalytic-pocket information to refine only one unresolved global numerical level. That interface was deliberately safe for a deployed total rank, but it made global correspondence a semantic parent by construction.

The partial relation removes that scientific priority. A candidate that is globally better but catalytically worse does not automatically remain better; the comparison becomes a trade-off. This is more faithful to the biological meaning of the coordinates and requires no hand-set score weight.

The old stratified fields and experiments remain useful as compatibility views and as evidence that forcing local coordinates into a total order can regress strict-inductive retrieval. They are not the current ontology of the method.

## Development and strict-inductive audit

Using the same double-cold splits and the same global/pocket geometry as the previous consensus experiments:

- development enzyme-to-reaction: 47/115 queries contain a known positive complete on the four-coordinate family; among queries with globally better complete candidates, an average 68.7% of those comparisons become trade-offs rather than certain global dominance;
- development reaction-to-enzyme: 32/83 informative queries; the corresponding average is 86.3%;
- strict held-out-factor reconstruction: 34/115 informative enzyme-to-reaction queries with 68.0% average blocked global dominance, and 27/83 informative reaction-to-enzyme queries with 85.7%;
- on queries informative in both development and strict audits, front-0 membership agrees 94.1% for enzyme-to-reaction and 96.3% for reaction-to-enzyme;
- strict median first-front size is only 4.86% of complete candidates for enzyme-to-reaction and 0.68% for reaction-to-enzyme.

Thus the relation is not produced by simply making a huge nondominated set, and the trade-off structure survives reference-only factor rebuilding.

## Operational total rank

Starase Navigator still needs a deterministic list. The deployed rank therefore remains the validated global correspondence linearization. That rank is an operational display/readout, not a claim that global correspondence is the biologically complete ordering.

The online service computes the partial relation only after top-k selection and exposes it as non-order-bearing metadata. It cannot feed back into score computation, filtering or sorting.

## Context and future coordinates

Assay context should enter by extending the observation domain, not by adding pH or temperature score experts. When enough source-linked observations are available, the natural object is a context-conditioned correspondence on reaction x enzyme x context.

Condition-scoped inactivity is then a censored observation at that context, for example activity below a detection limit, not a permanent global negative edge.

Cofactor, metal and mechanistic observations are already retained with provenance. They become order-comparison coordinates only when their endpoint semantics, coverage and strict transfer behavior justify a common comparison domain.
