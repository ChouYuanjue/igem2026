# Biological interpretation of the product-manifold method

> This document explains the biological meaning of the **existing product-manifold object**. It is not the primary method definition and it does not introduce an additional biological scoring layer. The primary story remains `M_R × M_E`, one positive measure, and one compatibility field `F(r,e)`.

## The biological question

The task is not treated as generic matching between two embeddings. An enzyme–reaction pair is a biochemical compatibility statement: a molecular system represented by a protein must support a particular chemical transformation. The method therefore represents **reaction chemistry**, **protein molecular state**, and **known enzyme–reaction precedents** in one object rather than combining several independent ranking rules.

The mathematical object is the scalar compatibility field `F(r,e)` on `M_R × M_E`. A point `(r,e)` is one candidate biochemical relationship. Experimentally curated or otherwise accepted positive pairs are observations of that field, not labels from which synthetic negatives are manufactured.

## Reaction chemistry is represented at several physical scales

The reaction factor `M_R` contains complementary descriptions of the same transformation:

- DRFP describes the global reaction change;
- RDKitPlus describes whole-reaction molecular and physicochemical context;
- atom-mapped reaction-centre geometry describes the local bond/atom changes closest to the catalytic event.

These are coordinates of one reaction geometry, not three voting models. Their local metric precision is determined by how much each coordinate resolves the current chemical neighbourhood. A coordinate that is locally uninformative contributes little; a coordinate that sharply distinguishes nearby transformations contributes more. No reaction class or manually chosen OOD threshold is required.

This matters biologically because two reactions can resemble one another globally while differing at the catalytic centre, or share a local transformation while occurring in different substrate contexts. The method keeps those scales visible instead of collapsing them into one fixed fingerprint similarity.

## The protein factor is a sequence–structure molecular-state geometry

The protein factor `M_E` currently uses dense ESM-C sequence representation and, where observed, CLIPZyme structure-dependent representation. For unit-normalized coordinates, the canonical average of their cosine similarities is equivalent up to scale to a direct sum of squared chordal distances. The protein factor can therefore be read as one sequence/structure atlas rather than a score ensemble.

Missing structure is not interpreted as evidence against activity. It means that only the observed sequence coordinate is available at that point.

Pfam architecture, terpene-synthase motifs, pockets, 3Di, and exact sequence alignments are biologically valuable but have a different role when their coverage is sparse. They are recorded as partial observations and evidence, not turned into universal gates merely because they are easy to explain.

## Known positives are biochemical precedents, not a lookup table

The empirical positive measure `mu` contains known enzyme–reaction pairs. It is transported only through the same reaction and protein geometries used by the field. The factor-exchange-invariant mollifier

`mu_tilde = 1/2 (K_R K_E + K_E K_R) mu`

has a simple interpretation: a known biochemical precedent can support a nearby candidate through both a chemically related reaction and a molecularly related enzyme, without declaring either axis to be primary.

The method does not copy the label of the nearest homolog. The positive measure is degree-normalized and spread continuously through local geometry. A highly annotated reaction or promiscuous protein therefore cannot dominate solely because it appears many times in the database.

## Why the nonlinear field is useful biologically

The final field is a robust variational extension of the prior compatibility field. Charbonnier conductance decreases as the local compatibility gradient becomes large. In practical terms, evidence is allowed to propagate through a locally coherent biochemical neighbourhood but is suppressed across a sharp compatibility boundary.

This is the role that a hard similarity threshold often tries to play, but here it is a property of the field itself. There is no rule of the form “if similarity < x, switch model” or “if structure is missing, fall back”.

The interpretation should remain conservative: a boundary in this learned/geometric field is evidence of incompatibility in the represented data geometry, not proof of a particular molecular mechanism.

## Every prediction carries its own support geometry

A prediction is accompanied by two continuous applicability coordinates computed from the same geometry:

- reaction-axis locality: how close the query reaction lies to its observed reaction neighbourhood;
- protein-axis locality: how close the candidate protein lies to proteins participating in nearby known positive pairs.

Each is reported relative to the intrinsic `sqrt(N)` neighbourhood boundary used by the numerical chart. Smaller ratios indicate denser local support. These quantities are **not thresholds and do not alter the rank**.

This separates biologically different situations that a single confidence score would obscure. A candidate may be chemically well supported but protein-side extrapolative, protein-side well supported but chemically novel, supported on both axes, or extrapolative on both axes.

## Biological witnesses make the support inspectable

For a predicted pair `(r,e)`, the system returns geometrically nearest known positive precedents `(r_i,e_i)` from the training relation. Witness selection uses the same reaction and protein geometry as the model and never uses the held-out label of `(r,e)`.

For each witness we report, when available:

- global, whole-reaction, and reaction-centre similarities;
- ESM-C and structure-dependent protein similarities;
- global and local BLOSUM62 sequence alignment identity and coverage;
- Pfam/domain annotation;
- family-specific catalytic motifs only when the protein has the corresponding family annotation;
- pocket/structural observations when available.

Several witnesses are reported, together with their consensus across distinct source reactions and proteins. This prevents the explanation from being reduced to one convenient homolog.

A weak witness is not hidden. For example, the unselected prediction `RHEA:10164 → A2SSV1` is supported only weakly by the nearest known positive pair: reaction-centre similarity is about `0.20`, DRFP similarity about `0.03`, and global sequence identity about `0.14`. The output exposes this as an extrapolative case rather than presenting it as mechanistically established.

Conversely, `RHEA:10048 → O31631` has a known positive precedent involving the same protein on a related reaction, while `RHEA:10020 → P0DPE4` has a nearest precedent with about `84%` sequence identity, `0.99` structure cosine, and substantial similarity on all three reaction coordinates. The same reporting format covers both cases.

## Mechanistic annotations are family-conditional

Catalytic motifs are not generic enzyme evidence. DDxxD, NSE/DTE, DxDD, and QW patterns are emitted only for proteins with an annotated terpene-synthase domain family. For arbitrary Rhea enzymes the report explicitly marks these observables as not applicable.

This rule is important: interpretability is not improved by attaching a familiar motif to a protein for which that motif has no biological meaning.

## Why sparse biological annotations are not forced into ranking

A biologically attractive feature is promoted into `g_R` or `g_E` only when it can define geometry on the relevant support, its missingness remains neutral, and the frozen internal evaluation shows no material regression.

Current pocket/P2Rank observations cover only about 0.69% of the canonical protein universe and the median canonical 432-node query chart contains zero pocket-observed proteins. Making pocket similarity a universal ranking term would therefore create a special-case branch for a small submanifold rather than a coherent protein geometry. It remains evidence until coverage improves.

The same discipline applies to Pfam, homology, and motifs. They can be strong biological evidence without being universally valid scoring rules.

## One field supports both retrieval directions

R2E fixes the reaction and reads `F(r, ·)`. E2R fixes the enzyme and reads `F(·, e)`. A separate biological story, separate expert ensemble, or separate notion of confidence is not required. The same reaction geometry, protein geometry, positive relation, field equation, witnesses, and applicability coordinates apply in both directions.

This bidirectional consistency is part of the method definition, not a post-hoc symmetry claim; the remaining engineering task is to complete the frozen full-support E2R fibre evaluation under its own established protocol.

## What the method does not claim

The method does not claim that geometric proximity proves catalysis, that an embedding dimension corresponds to a particular physical mechanism, or that a predicted enzyme is experimentally active. It produces a ranked biochemical hypothesis together with the known precedents and geometric regime supporting that hypothesis. Experimental validation remains the final test.

## Application-specific mechanism sheets

The general Rhea benchmark and the registered terpene application should not be conflated. The frozen general clean-development folds contain no TPS-family positives from the 6,494-protein TPS expansion, so they cannot validate TPS-specific catalytic motifs.

For the registered terpene application, 240 reactions have reaction-level architecture evidence assembled from known positive enzymes; 208 currently have mapped reference support suitable for biological annotation. This evidence is used as a mechanism sheet, not as a ranking gate.

For example, the registered geosmin reaction `(2E,6E)-FPP -> geosmin` has the experimentally established germacradienol/geosmin synthase `Q9X839` as a bacterial class-I reference. Candidate `A0A1E7JZ38` is independently annotated as PF19086 / bacterial class I. Its direct alignment to `Q9X839` gives about 51% global sequence identity and 53% local identity over most of both proteins, while both sequences expose the characteristic `DDHFLE` and NSE-like environments. These facts make the candidate biologically inspectable without changing its model rank.

The historical rule that converted architecture into `compatible` / `family_mismatch` routing is not part of the product-manifold mainline. The same information is more defensibly presented as observed biological precedent.
