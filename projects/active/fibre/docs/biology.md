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

Instead, reaction-center observations re-enter at the catalytic-local
resolution described below.

## Catalytic-pocket resolution: specificity inside a coarse level

Machine-stable equal values of the global correspondence defect form coarse
correspondence levels. Only inside one such unresolved level does FIBRE inspect
a finer catalytic geometry.

The local reaction side uses reaction-center transition geometry. The local
protein side is represented by three active-site/pocket coordinates:

- pocket-local ESM-C;
- pocket 3Di;
- pocket OT.

The same zero-temperature correspondence operator is applied again at this
resolution. There is no extra classifier and no manually chosen coefficient
such as 0.1 times pocket score.

The most conservative current construction treats the three pocket
realizations as independent coordinates. Within one coarse level, one candidate
can dominate another only if its local correspondence defect is no worse in all
three pocket coordinates and strictly better in at least one. Disagreement means
that the candidates remain incomparable.

This is deliberately a partial order. Local biology is allowed to add
resolution, but not forced to manufacture a preference.

## Why the catalytic strata do not currently change total rank

Development-only experiments can make catalytic-pocket refinement look
attractive. Pocket-only scalar refinement gives five E2R improvements and no
development regressions; a Pareto-consensus construction also has a positive
mean development delta.

The stronger test rebuilds each held-out protein/reaction factor atlas without
the held-out entities and attaches them only through their query-to-reference
molecular observations. Under that strict-inductive audit, pocket-only
linearization still gives three small E2R regressions, and Pareto consensus gives
four small E2R regressions with a tiny negative mean delta. R2E is unchanged.

Therefore the scientifically honest current output is stratified:

- the global correspondence defect determines the deterministic total rank;
- catalytic-pocket strata are first-class finer coordinates;
- those strata are explicitly non-order-bearing until a future strict
  non-degradation gate is passed.

This guarantees that including the biologically necessary local information
cannot reduce the validated retrieval metric merely because it was included.
The information is genuinely inside the model object; what is withheld is only
the unjustified step of forcing every finer relation into a single total order.

## Mechanistic resolution: catalytic motifs are contextual, not universal votes

Family-aware motif coordinates form a still finer mechanistic chart. Current
coordinates include class-I aspartate-rich motifs, NSE/DTE, DXDD and QW context.

These motifs are only defined when their enzyme-family context makes biological
sense. A missing or non-applicable motif is an unobserved coordinate, not a
negative score. Motifs therefore answer a mechanism question rather than acting
as universal ranking rules.

This mirrors standard enzymology practice: catalytic residues and their
three-dimensional arrangement can be strongly mechanistic, while substrate
specificity also depends on the surrounding active-site/pocket environment.
M-CSA explicitly curates catalytic residues, their roles and reaction
mechanisms, and its EnzyMM extension searches three-dimensional catalytic motif
arrangements. Recent enzyme-retrieval work such as EnzymeCAGE likewise models
enzyme structure, catalytic function and reaction specificity jointly rather
than treating structure as a cosmetic explanation.

The resulting FIBRE hierarchy is therefore:

global biochemical correspondence
-> catalytic-pocket specificity
-> mechanistic motif context.

## Missing information remains neutral at every resolution

A key biological and statistical rule is that absence of an assay is not
evidence of absence of function.

- No structure: the structure coordinate is missing.
- No confidently observed pocket: the pocket coordinate is missing.
- Motif not applicable to the annotated family: that mechanistic coordinate is
  undefined.
- External query without a complete catalytic-local chart: the finer stratum is
  unavailable and the global correspondence rank remains unchanged.

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

A FIBRE result should be read in this order:

1. The total rank says which hypotheses are best supported by the validated
   global correspondence geometry.
2. The coarse numerical level says whether that apparent ordering is actually
   resolved by the scalar field.
3. The catalytic stratum says whether complete reaction-center/pocket geometry
   adds a consistent local distinction inside that coarse level.
4. The mechanistic coordinates say which family-specific catalytic context is
   observed.
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
