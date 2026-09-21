# Biological observation model: attributes, not evidence tiers

FIBRE should not invent a universal hierarchy such as level-1 / level-2 evidence. Enzymology does not provide a stable total ordering of evidence types: a quantitative assay at the wrong condition, a literature-reported product profile, a curated catalytic mechanism and a high-confidence structure answer different questions.

The current design therefore follows two community patterns:

- STRENDA / STRENDA Biocatalysis describes the experimental attributes needed to reproduce and interpret an enzymatic transformation. The catalogue is modular and not every attribute applies to every experiment.
- EnzymeML V2 represents proteins, small molecules, reactions, measurements, equations, parameters and units as composable typed objects.

FIBRE does not copy either standard into a new ontology. It keeps a thin observation index: enzyme/reaction identity, a sparse set of typed properties, and source provenance. A standard term is used only when the semantics match exactly; otherwise the property stays explicitly in the fibre namespace.

Pinned upstream authorities are recorded in ../standards/pinned_sources.json. Raw upstream snapshots are cached under results/fibre_standards_cache_v1/ and are not runtime inputs.

## What MARTS can say today

The raw MARTS source contains 2,833 enzyme-reaction rows. It already provides sequence, enzyme name, source species, substrate/product structures, TPS class, mechanism identifiers and publication links. 94.3% of rows carry a publication URL (801 unique publications), and 91.5% carry a UniProt identifier. A nonempty mechanism field does not imply an observed mechanism: 567 rows contain the explicit placeholder `no_mechanism`. Actual step-level mechanism records cover 2,266 rows (80.0%) and provide 3,395 individual steps with reaction type, source publication and the original MARTS evidence label.

It does not materialise assay pH, temperature, solvent, ionic strength, cofactor concentration, kinetic constants, detection limits, activity units or time-course measurements. The observation index therefore does not fabricate them.

One semantic distinction is important: MARTS species describes the biological origin of the enzyme. It maps naturally to STRENDA origin_organism, not to EnzymeML Protein.organism, whose V2 specification describes the expression host.

## Backward-compatible projection

The richer observation object does not replace the current FIBRE field yet. Exact sequence identity maps MARTS rows to the current protein atlas; exact reaction-signature identity maps rows to the current reaction atlas.

Of the 2,833 source observations, 2,526 map on both axes. After duplicate observations are collapsed at the pair level they recover exactly the 2,438 canonical pairs in the current Omega, with no extra and no missing pair. The remaining 307 observations are retained with their original reaction signature and provenance but have no current canonical reaction id.

This gives a strict compatibility rule: richer biological observations may extend the state available to FIBRE, but the old pair projection must remain identity-equivalent until a separately validated model change is promoted.

## Information acquisition order

Future enrichment should prefer direct structured sources before free-text inference:

1. existing MARTS molecular identity, mechanism and publication provenance;
2. identifier-resolved structured databases such as UniProt and Rhea, and kinetic resources when their programmatic interface is usable;
3. primary-publication extraction for attributes absent from structured databases;
4. computed molecular observations such as structure, pocket, motif and geometry, always marked as computed rather than measured.

Every imported property must retain its source URI or database record and the extraction route. Missing attributes stay missing. A model or language model may help extract a value from a paper, but the extracted value is not accepted without a recoverable source span or record and unit semantics.

## How the model should use future properties

The observation index itself has no ranking effect.

A property becomes a model coordinate only after three questions have been answered:

1. Coverage: is it observed often enough to support a meaningful comparison?
2. Semantics: is comparison valid across records, with the same endpoint and compatible units/conditions?
3. Transfer: under strict held-out-factor evaluation, does it add a stable biological distinction rather than dataset-source leakage?

This is deliberately different from assigning a quality score or hand weight. Applicability and presence determine whether a coordinate exists; validation determines whether that coordinate receives ordering authority.

## Measured external coverage

A local snapshot of official UniProtKB annotations was acquired for the MARTS identifiers. Of 1,337 requested identifiers, 1,332 resolved to UniProtKB entries. The structured snapshot contains catalytic-activity annotations for 878 entries, cofactor annotations for 1,043, binding-site annotations for 713 and explicit active-site annotations for 82. Catalytic annotations contain 603 distinct Rhea identifiers. ECO/source tokens are retained as provenance rather than converted into weights; for example, experimental ECO:0000269 and homology-derived ECO:0000250 remain distinct observations.

Cofactor coverage is high but not automatically discriminative. Mg(2+) dominates the current terpene-enzyme annotations (1,036 entries) and Mn(2+) is second (322). In addition, most cofactor records are inferred rather than directly measured. Cofactor state is therefore a useful biological/context observation, but it is not promoted to a ranking coordinate without a reaction- or assay-side counterpart.

Official Rhea directed-reaction SMILES were also pinned locally. Under strict directed structure identity, without name matching, hierarchy expansion, participant dropping or direction guessing, only 35 of 454 MARTS reaction rows (7.7%) map exactly to a Rhea directed reaction. Rhea is therefore an exact standard anchor when available, not the mandatory identity system for the MARTS terpene reaction space.

A primary-literature pilot used Europe PMC full text for five high-coverage MARTS publications. Four full texts were available, yielding 25 source-linked candidate paragraphs containing pH, temperature, metal/cofactor, kinetic or assay-context cues and linking back to 185 MARTS rows. These passages demonstrate that assay-context enrichment is feasible, but no extracted value is accepted until its enzyme/reaction/experiment scope and units are validated.

The source-bound extraction path is now materialized as an assay-context index rather than remaining only a text-mining pilot. Its promotion rule is deliberately strict: catalytic-assay paragraph, exact source-span verification, resolved experiment scope, exactly one explicit MARTS target, and an exact mapping to one canonical pair. Under that rule the current local corpus materializes **one** pair-specific assay context (pH 6.5). That number is scientifically useful because it prevents us from pretending that condition-aware ranking is already data-supported. pH/temperature/cofactor fields are now first-class state variables, but they do not receive scalar ranking authority at this coverage.

A separate catalytic-state index covers all 2,438 canonical positive pairs while preserving evidence scope. Pair-associated MARTS mechanism records are present for 2,200 pairs; 553 have at least one mechanism step labelled with experimental evidence. Protein-level UniProt state is available much more broadly: catalytic-activity annotations touch 1,430 positive pairs, cofactor annotations 1,804, binding-site annotations 1,270 and active-site annotations 100. These UniProt observations remain **protein-level** evidence; they are never silently upgraded into evidence that a particular enzyme-reaction pair was assayed under those conditions.

Explicit inactive/below-detection/no-conversion observations have a defined place in the model but none are currently materialized under the strict pair-specific source gate. If such a record is later available and the user explicitly requests the same fully matched context, it acts as a candidate-eligibility censor rather than a score penalty or permanent negative edge. Missing records, condition mismatches, positive-only support and conflicting positive/negative observations stay unresolved for exclusion purposes rather than becoming negative labels.

## Scientific relation: partial order, not layers

The previous global-first stratified view is retained only for compatibility. The scientific relation is now defined on a fixed family of FIBRE defect coordinates. The current validated family is global correspondence plus the three catalytic-pocket coordinates (pocket-local ESM-C, pocket 3Di and pocket OT).

A candidate dominates another only when it is no worse in every available coordinate of that declared family and strictly better in at least one. No coordinate receives lexicographic priority or a fitted fusion weight. A candidate missing any declared coordinate is left unordered in this relation rather than assigned a surrogate defect.

This distinction matters because local biological information need not make a candidate numerically "better" to be useful. It can invalidate an over-strong conclusion from global similarity: a candidate that is globally better but catalytically worse becomes a trade-off rather than automatically dominating another candidate.

On the existing double-cold development audit, global-plus-pocket comparison blocked 68.7% of otherwise global-better comparisons for informative enzyme-to-reaction queries and 86.3% for informative reaction-to-enzyme queries. Under strict held-out-factor reconstruction these values remained 68.0% and 85.7%, respectively. The median first Pareto-front size stayed small relative to the complete candidate set (4.86% and 0.68% in strict evaluation), so the relation is not simply declaring everything incomparable.

Coverage remains the main limitation. Under strict reconstruction only 29.6% of enzyme-to-reaction queries and 32.5% of reaction-to-enzyme queries contain at least one known positive with the complete four-coordinate family. Missing local coordinates therefore remain explicitly unresolved rather than being hidden by a fallback score.

## Biological stress tests and current promotion boundary

The first positive-only promiscuity holdout now tests a biological property rather than aggregate retrieval alone. Across 422 enzymes with at least two accepted activities, 1,555 verified positive edges are held out one at a time. A cold field removes every activity of the query enzyme; a warm field retains its other verified activities while keeping the held-out reaction hidden and removing already-known reactions from the returned candidate list. No unknown pair is labelled negative. The median expected held-out rank improves from 8.0 to 5.5 and mean expected reciprocal rank from 0.248 to 0.396; 65.4% of held-out activities improve, 9.6% tie and 25.0% worsen.

Mechanism annotations are used only to diagnose this behavior, not to tune the ranking. Mechanism-step overlap can be evaluated for about 90.1% of these held-out positives. The high-overlap subset contains 633 examples and improves from median expected rank 6 to 4, but the observed disjoint-step subset contains only two examples. That is not enough evidence to promote a mechanism gate or mechanism-weighted reranker.

Three previously proposed stress tests are therefore explicitly blocked rather than approximated with pseudo-labels:

- **assay-context shift:** only one pair-specific materialized assay context and no pair with two distinct contexts;
- **functional-cliff / explicit-negative contradiction:** no source-bound pair-specific inactive/below-detection observations yet;
- **cofactor holdout:** protein-side cofactor annotations are broad, but an equivalently scoped reaction/assay-side cofactor requirement is not available for enough canonical pairs.

This is the current model-development rule: implement the state and provenance needed to represent enzymology correctly, but grant ordering authority only where the available biological evidence can actually test it.
