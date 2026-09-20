# Biological interpretation of FIBRE

This document explains why the current FIBRE construction is biologically
reasonable. It does not introduce a separate biological scoring layer. Reaction
chemistry, protein molecular state, catalytic-local structure and known positive
enzyme-reaction precedents are different resolutions of one correspondence
object.

## The biological question

The task is not generic embedding matching. A proposed enzyme-reaction pair says
that one protein molecular system may support one chemical transformation. For
experimental screening, three questions are naturally different:

1. Is the pair compatible with known biochemical precedents at a broad molecular
   level?
2. If several candidates remain globally indistinguishable, does the local
   catalytic environment support a more specific comparison?
3. Which catalytic mechanism or enzyme-family context is actually observed, and
   which evidence is still missing?

FIBRE represents those questions as nested resolutions instead of collapsing
them into independent score experts.

## Global correspondence: broad biochemical compatibility

Let reactions live on a reaction manifold M_R and proteins on a protein manifold
M_E. Known positive enzyme-reaction pairs form a sparse correspondence Omega on
M_R x M_E.

The canonical correspondence defect compares two explanations of a proposed
pair. One explanation requires the reaction and enzyme to be close to the same
known positive precedent. The other allows each side to resemble unrelated
known reactions and proteins independently. The excess cost of the shared
explanation is Delta_Omega; FIBRE uses F = -Delta_Omega.

Biologically, this asks whether the reaction and protein look jointly familiar,
not merely whether each looks familiar on its own. A protein that resembles one
enzyme family and a reaction that resembles a completely different family of
known chemistry do not receive the same support as a pair close to a shared
precedent.

R2E and E2R are therefore two sections of the same biochemical relation rather
than separate models.

## Protein state is multiresolution, not one sequence vector

The current 1,421-protein application-domain atlas is deliberately not reduced
to one representation. Its label-free molecular observations include:

- global ESM-C sequence state: 1,421 / 1,421 proteins;
- pocket-local ESM-C: 1,287 / 1,421;
- whole-structure 3Di: 1,287 / 1,421;
- pocket 3Di: 1,068 / 1,421;
- pocket OT geometry: 1,287 / 1,421;
- family-aware class-I aspartate context: 827 / 1,421;
- family-aware NSE/DTE context: 647 / 1,421;
- DXDD context: 142 / 1,421;
- QW context: 27 / 1,421.

These measurements describe one protein state at different scales. The canonical
global protein manifold uses a missing-neutral pullback geometry: a view
contributes only when both endpoints actually have that observation. Merely
having more modalities never produces a bonus, and an absent pocket or motif is
never interpreted as evidence against activity.

Raw structure similarities are first converted into graph/diffusion geometry
before entering the manifold; they are not simply called distances by fiat.

## Reaction state separates global chemistry from the catalytic event

The global reaction atlas captures whole-reaction chemical state and is the
current order-bearing reaction geometry.

Reaction-center transition Wasserstein distance and explicit transition-token
geometry are also valid measurements, but they answer a finer question. Directly
injecting them into the global reaction metric produces a clear E2R regression,
including under a fixed-topology tangent refinement. FIBRE therefore does not
pretend that local reaction-center chemistry should globally rewrite reaction
neighbourhoods.

Instead, reaction-center observations re-enter as catalytic-local
correspondence coordinates described below.

## Catalytic-pocket relation: specificity without a hierarchy

The local reaction side uses reaction-center transition geometry. The local
protein side is represented by three active-site/pocket coordinates:

- pocket-local ESM-C;
- pocket 3Di;
- pocket OT.

The same zero-temperature correspondence idea is evaluated independently on
these local factor geometries. Together with the global correspondence defect,
this gives the current four-coordinate FIBRE comparison family. There is no
extra classifier and no manually chosen coefficient such as 0.1 times pocket
score.

Candidate a dominates candidate b only when it is no worse on **all four**
declared correspondence coordinates and strictly better on at least one. If a
candidate is globally better but catalytically worse, the two hypotheses become
a trade-off rather than being forced into the global order. No coordinate has
lexicographic priority.

This is deliberately a partial order. Local biology is allowed to invalidate an
over-strong global comparison, but it is not forced to manufacture a scalar
preference.

## Why the partial relation does not currently replace total rank

The stronger test rebuilds each held-out protein/reaction factor atlas without
the held-out entities and attaches them only through their query-to-reference
molecular observations. The qualitative relation survives this strict-inductive
audit: among informative queries, catalytic coordinates turn an average 68.0%
of otherwise global-better comparisons into trade-offs for E2R and 85.7% for
R2E. On queries informative in both development and strict audits, front-0
membership agrees 94.1% and 96.3%, respectively.

The relation is also not trivially broad: in strict evaluation the median first
Pareto front contains only 4.86% of complete E2R candidates and 0.68% of
complete R2E candidates.

Starase Navigator nevertheless retains the global correspondence defect as its
deterministic list because previous attempts to **linearize** local information
can regress strict-inductive retrieval. This is an operational product readout,
not a biological hierarchy. The four-coordinate partial relation is a
first-class scientific output and is explicitly non-order-bearing online.

## Mechanistic coordinates: catalytic motifs are contextual, not universal votes

Family-aware motif coordinates remain explicit FIBRE observations. Current
coordinates include class-I aspartate-rich motifs, NSE/DTE, DXDD and QW context.
Family applicability defines the mechanism chart, and different charts are not
forced onto one universal numerical scale. Their current common comparison
domain is too sparse/family-specific to add them to the validated four-coordinate
Pareto family, so they remain mechanistic context rather than a hidden ranking
bonus.

These motifs are only defined when their enzyme-family context makes biological
sense. A non-applicable motif is outside the chart. An applicable but unobserved
motif leaves the comparison unresolved. Neither case becomes a negative score.
Motifs therefore answer a mechanism question rather than acting as universal
ranking rules.

This mirrors standard enzymology practice: catalytic residues and their
three-dimensional arrangement can be strongly mechanistic, while substrate
specificity also depends on the surrounding active-site/pocket environment.
M-CSA explicitly curates catalytic residues, their roles and reaction
mechanisms, and its EnzyMM extension searches three-dimensional catalytic motif
arrangements. Recent enzyme-retrieval work such as EnzymeCAGE likewise models
enzyme structure, catalytic function and reaction specificity jointly rather
than treating structure as a cosmetic explanation.

The resulting FIBRE object therefore contains **complementary coordinates, not a hierarchy**: a global correspondence coordinate, catalytic-pocket correspondence coordinates, and family-specific mechanistic/context observations. The first four validated global+pocket coordinates currently define the common Pareto comparison domain.

## Missing information remains neutral at every resolution

A key biological and statistical rule is that absence of an assay is not
evidence of absence of function.

- No structure: the structure coordinate is missing.
- No confidently observed pocket: the pocket coordinate is missing.
- Motif not applicable to the annotated family: that mechanistic coordinate is
  undefined.
- External query without the complete declared comparison family: the partial
  relation is unavailable for that candidate/query; no missing coordinate is
  converted into a penalty, while the operational global rank remains available.

This prevents experimental coverage and database completeness from becoming
hidden negative labels.

## Known positives are biochemical precedents, not a lookup table

A verified enzyme-reaction pair adds one exact point to Omega. FIBRE can update
the global correspondence field by pointwise minima without retraining, and that
incremental update is exactly equal to rebuilding the field from scratch.

The closest positive precedent can also be exposed as a witness, but the model
does not simply copy its label. The pair is evaluated through the geometry of
both axes and through the shared-precedent defect.

A new positive can improve many neighbouring queries and worsen a minority; the
current 130-seed audit shows that simple support density does not explain these
side effects. FIBRE therefore reports the seed's exact influence footprint
rather than downweighting an experimentally verified positive with an arbitrary
density correction.

## Applicability and uncertainty are geometric, not confidence theatre

The zero-temperature defect naturally produces numerical level sets. FIBRE
reports tie-aware rank intervals and threshold-free support geometry, including:

- distance from the query to positive marginal support;
- size of the best numerical level;
- gap to the next distinct defect level;
- candidate support distances within the best level.

These are not calibrated probabilities and are not labelled as such. They tell
a scientist whether the model is operating near known biochemical support and
whether the scalar field actually resolves one candidate from its neighbours.

## What a biologist should read from one result

A FIBRE result should be read as several complementary statements rather than a hierarchy:

1. The total rank is the current stable operational readout from the validated
   global correspondence coordinate.
2. The biological partial relation asks whether that global ordering is still
   justified once all available validated catalytic-pocket coordinates are
   considered jointly. A trade-off means the model is declining to claim that
   the globally better candidate is biologically superior.
3. Coordinate completeness determines whether a Pareto statement is available;
   missing local information leaves the relation unresolved rather than worse.
4. The mechanistic chart and other context observations show which
   family-specific catalytic facts are actually applicable/observed.
5. The positive witness and support distances show which known biochemical
   precedents and geometric regime support the hypothesis.

No single one of those items proves catalysis. Experimental validation remains
the final test.

## What FIBRE does not claim

FIBRE does not claim that geometric proximity proves catalytic activity, that a
predicted structure is an experimental structure, that a motif is sufficient
for function, or that a numerical stratum is a calibrated probability.

Its claim is narrower: sparse verified enzyme-reaction pairs can be propagated
through a biologically structured, partially observed molecular geometry, and
the model can preserve the distinction between broad compatibility, local
catalytic specificity and mechanistic context without inventing synthetic
negative evidence or a collection of ad-hoc score experts.

## External biological references

- Ribeiro et al., Mechanism and Catalytic Site Atlas (M-CSA): a database of
  enzyme reaction mechanisms and active sites, Nucleic Acids Research (2018),
  DOI 10.1093/nar/gkx1012.
- Liu et al., A geometric foundation model for enzyme retrieval with
  evolutionary insights (EnzymeCAGE), Nature Catalysis (2026),
  DOI 10.1038/s41929-026-01478-y.
