# FIBRE stratified multiresolution correspondence geometry

FIBRE is not a scalar retrieval model followed by external biological
annotation. Its natural output is a correspondence section resolved at several
biological scales.

## Resolution hierarchy

For a query q and candidate x, the first coordinate is the canonical global
correspondence defect

    Delta_0(q,x).

Machine-stable equal values of Delta_0 define coarse correspondence levels.
These levels are the global biochemical relation resolved by reaction chemistry
and the canonical multiresolution protein state.

A second, catalytic-local resolution is defined only inside one coarse level.
It uses the same correspondence operator on a reaction-center manifold and on
active-site/pocket protein manifolds. Current pocket coordinates are:

- pocket-local ESM-C sequence state;
- pocket 3Di structural state;
- pocket OT geometric state.

They are not added to Delta_0 and receive no hand-set fusion weight. Their role
is to refine an already-unresolved coarse equivalence class.

A third mechanistic resolution contains family-aware catalytic motif and residue
coordinates such as class-I aspartate, NSE/DTE, DXDD, and QW contexts. These
coordinates state which catalytic mechanism chart is actually observed. They
are sparse by biology: a motif that is not applicable or not observed is a
missing coordinate, never negative evidence.

## Partial order, not score fusion

The strongest current local construction treats each pocket realization as an
independent local coordinate of the same FIBRE operator. Inside one coarse
level, candidate a may precede candidate b only when a is no worse than b in
all jointly observed local coordinates and is strictly better in at least one.
This is Pareto dominance on local correspondence defects.

Consequently:

- no modality receives a learned or hand-set scalar weight;
- disagreement among pocket coordinates leaves candidates incomparable;
- a missing local coordinate leaves the whole comparison unresolved;
- local biology can never reverse two candidates that belong to different
  coarse correspondence levels;
- deterministic candidate identifiers are only display tie-breaks, not
  scientific evidence.

This makes local biological information part of the FIBRE object itself rather
than a profile attached after ranking.

## Promotion discipline

A local coordinate may exist scientifically without being order-bearing in the
deployed total ranking. Promotion of a local relation to a deterministic
display order requires matched development and strict-inductive evidence that
the relation is stable out of sample.

Current evidence already rules out two less coherent alternatives:

1. injecting reaction-center geometry directly into the global reaction factor
   causes a clear E2R regression;
2. allowing every local coordinate to rerank coarse levels produces mixed
   development behavior and strict-inductive regressions.

Pocket-only scalar refinement is much safer and is non-degrading on the current
double-cold development cells, but strict-inductive evaluation still shows a
small E2R regression. It is therefore not yet the canonical total-rank rule.

The consensus partial-order construction has now been tested under the same strict-inductive gate. On double-cold development it gives E2R 5 improved / 109 tied / 1 worsened with a positive paired mean; under reference-only factor rebuilding and OOS attachment it gives 1 improved / 110 tied / 4 worsened and a very small negative mean RR delta (about -3.96e-05). R2E remains unchanged.

Therefore catalytic strata are **not promoted to the canonical total-rank linearization**. They remain first-class FIBRE coordinates while the deterministic display rank stays Delta_0. This is not an external explanation layer: the scientific output is a stratified relation whose conservative coarse linear extension is deliberately protected until a future local relation passes the strict non-degradation gate.

## Biological interpretation

The hierarchy follows the distinction between broad catalytic compatibility and
local substrate/reaction specificity.

Global molecular state asks whether a reaction-enzyme pairing is supported by
the known biochemical correspondence geometry. Active-site/pocket coordinates
ask whether candidates that are globally unresolved remain compatible at the
local catalytic environment. Family-aware motif coordinates identify the
mechanistic chart in which that local comparison is biologically meaningful.

The method therefore uses more biological information by increasing resolution, not by attaching more score experts.

This hierarchy is also biologically aligned with how enzyme function is described outside the retrieval benchmark. M-CSA treats catalytic residues and reaction mechanism as explicit active-site objects, while recent geometric enzyme-retrieval work such as EnzymeCAGE explicitly models structure, catalytic function and reaction specificity together. Those precedents motivate giving active-site/pocket state a distinct local geometric role rather than treating it as a generic whole-protein bonus. Family-aware motifs then identify mechanistic context inside that local chart instead of serving as universal evidence.
