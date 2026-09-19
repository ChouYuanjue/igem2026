# Biological Interpretability Contract for FIBRE

## Principle

FIBRE exposes one biochemical correspondence object at several resolutions. It
does not attach an independent biological expert after retrieval.

The current output has three nested parts:

1. the global correspondence defect, which supplies the validated deterministic
   total rank;
2. catalytic-pocket strata, defined only inside a machine-stable global level;
3. family-aware mechanistic coordinates, which describe catalytic context at a
   still finer resolution.

Missing Pfam, motif, structure, reaction-center or pocket information is absence
of observation, never evidence against catalysis.

## Global factor geometry

The global reaction factor represents whole-reaction chemical state. The global
protein factor is a partially observed multiresolution molecular-state geometry
whose complete base is ESM-C sequence and whose observed structural coordinates
include whole-structure 3Di together with other current molecular measurements.

Known enzyme-reaction pairs define the sparse positive correspondence on the
product space. R2E and E2R are sections of the same correspondence defect; they
are not different ranking models.

Reaction-center transition geometry is deliberately not injected into the
global reaction metric. Matched experiments, including fixed-topology tangent
refinement, show a clear E2R regression when it is allowed to deform the global
reaction geodesic.

## Catalytic-pocket resolution

Reaction-center information is not discarded. It enters at a finer resolution
where its biological role is more natural.

Within one unresolved global numerical level, FIBRE evaluates the same
correspondence operator on reaction-center geometry and on active-site/pocket
protein coordinates:

- pocket-local ESM-C;
- pocket 3Di;
- pocket OT.

The current consensus construction treats those pocket realizations as
independent local coordinates. One candidate dominates another only when it is
no worse in every available local coordinate and strictly better in at least
one. Thus disagreement remains unresolved rather than being hidden by a
hand-set weighted average.

This local relation cannot cross a global level boundary.

## Mechanistic coordinates

Family-aware motif coordinates describe catalytic mechanism context rather than
a universal vote for activity. Current application coordinates include class-I
aspartate-rich context, NSE/DTE, DXDD and QW where biologically applicable.

A non-applicable or unobserved motif is missing. It is never converted into a
penalty. Motif coordinates therefore tell a scientist which mechanistic chart
is observed for a candidate; they are not a second classifier.

## Why finer strata are currently non-order-bearing

A biologically meaningful relation may be part of FIBRE without automatically
being promoted into the deterministic total rank.

On current double-cold development, pocket-only refinement is non-degrading and
improves several E2R queries. The stronger strict-inductive test removes
held-out factor entities before building both global and local atlases and
attaches them only from query-to-reference molecular observations. Under that
protocol, pocket-only linearization retains three small E2R regressions.
Pareto-consensus local refinement is more conservative but also retains a tiny
negative strict-inductive mean delta. R2E remains unchanged.

Therefore the current contract is explicit:

- total rank comes from the global correspondence defect;
- catalytic strata and mechanistic coordinates are first-class FIBRE outputs;
- they are marked non-order-bearing until a future strict-inductive
  non-degradation gate is passed.

This is stronger than choosing a small coefficient that happens not to hurt a
benchmark. Rank preservation follows from the method interface.

## Biological witnesses

A predicted pair may also expose known positive precedents that minimize the
same joint product-space support cost. A witness is therefore a readable
realization of the model's existing support, not a separate scoring system.

Human-readable measurements may accompany the witness when available:
reaction-center similarity, sequence alignment, structural similarity,
pocket-local observations, Pfam/domain context and family-aware catalytic
motifs. Weak witnesses are not hidden.

The distinction is useful:

- correspondence strata say what the model itself resolves;
- witnesses say which known biochemical precedents support that relation;
- annotations say what biology is actually observed for those molecular
  states.

## Applicability and uncertainty

The zero-temperature defect naturally contains numerical level sets. FIBRE
reports rank intervals and intrinsic support geometry rather than manufacturing
a calibrated confidence score.

Current threshold-free quantities include query distance to positive marginal
support, best-level size/fraction, next-level defect gap and candidate support
distance inside the best level.

Starase Navigator exposes these under geometric_uncertainty and exposes the
nested scientific relation under stratified_correspondence / fibre_resolution.
Neither field changes the canonical total rank.

## Promotion rule

A finer biological relation may become order-bearing only if:

- it is a coordinate or relation of the same correspondence object rather than
  an independent score expert;
- cross-global-level order is invariant by construction;
- missing observations remain neutral;
- no arbitrary scalar modality weight is introduced to force non-regression;
- matched double-cold evaluation is reported;
- reference-only held-out factor atlases are rebuilt and external entities are
  attached out of sample;
- both directions report per-query improve/tie/worse counts and paired
  uncertainty.

Until then, the finer relation remains scientifically visible but the
conservative global linearization is retained.

## What the output means to a biologist

For one predicted pair, inspect:

1. global rank and correspondence defect: broad support from known biochemical
   precedents;
2. global numerical level: whether the scalar field really resolves the
   apparent ordering;
3. catalytic stratum: whether complete active-site/pocket geometry gives a
   consistent finer distinction;
4. mechanistic coordinates: which family-specific catalytic context is
   observed;
5. positive witnesses and support distances: which known systems and geometric
   regime support the hypothesis.

None of these proves activity. Experimental validation remains the final test.

## Method boundary

FIBRE does not claim that geometric proximity proves catalysis, that a predicted
structure is experimental evidence, that a motif is sufficient for function,
or that an intrinsic support coordinate is a calibrated probability.

Its interpretability claim is narrower: the same mathematical object that
produces retrieval also exposes its biochemical resolution, its known positive
precedents, the molecular observations that are present, and the distinctions
that remain unresolved.
