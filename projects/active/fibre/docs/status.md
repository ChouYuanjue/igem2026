# Current FIBRE status

FIBRE now means **Factorized Interaction Basis for Reaction–Enzyme**. Its canonical mathematical object is an **interaction atlas**: several local catalytic interaction charts, built from different molecular representations, are glued by an availability-aware partition of unity into one global reaction–enzyme interaction.

## Implemented foundations

- Raw reaction chemistry and raw enzyme sequence retain the universal prediction path.
- Dual-tower models provide low-rank local interaction coordinates.
- Reaction multiview features already combine whole-reaction, substrate/product and signed-change information.
- Gated multi-expert prototypes already produce a global embedding, local expert embeddings and softmax expert gates; under the current theory these are approximations to a universal chart, local charts and partition weights.
- Structure-aware CLIPZyme, EnzGFM, reaction-centre, seed-context and TPS assets provide additional chart candidates with different support.
- kernel/atlas.py implements chart-specific bilinear forms, missing-neutral partition normalization, partition-of-unity gluing and overlap-consistency loss.
- kernel/interaction.py retains the train-free finite-rank observation update used to incorporate accepted positive pair evidence without retraining molecular encoders.
- Evidence, assay context and provenance remain attached to predictions.

## Multi-expert evidence already in the repository

The historical gated 8-expert MARTS experiment used one global channel plus eight local expert channels with softmax gates, balance regularization and expert-diversity regularization. MARTS-only adaptation improved the same strict double-cold multi-expert architecture from MRR 0.0191 to 0.0400 in R2E and from 0.0349 to 0.0539 in E2R on its frozen MARTS evaluation. These numbers are supporting internal evidence for family-specialized local charts, not new benchmark claims for the current atlas.

The current theory adds the missing mathematical requirement: experts that are simultaneously applicable should agree on the scalar catalytic interaction on chart overlaps. This is implemented by the gluing-consistency loss in kernel/atlas.py and is wired into the frozen multi-expert reproduction trainer behind an explicit glue-weight flag whose default is 0.

## Atlas-native reproduction closeout

The frozen atlas recipe is now recorded at
`reproducibility/bime_rank/configs/fibre_interaction_atlas_v1.yaml`.  It uses
the multiview reaction representation, one universal chart, eight local charts
and a weak overlap-consistency weight of 0.02.  The universal chart is included
in the same partition of unity as the local charts; this is algebraically the
same global/expert convex score used by the earlier directional implementation,
so the ontology change itself does not perturb scores.

On the fixed TPS reproduction assets, the whole atlas obtains:

- legacy-exact R2E: MRR 0.2370, Hit@10 35.87%, Hit@20 45.81%;
- strict 25-cell double-cold R2E: MRR 0.0449, Hit@10 10.78%, Hit@20 19.05%;
- strict 25-cell double-cold E2R: MRR 0.0723, Hit@10 19.02%, Hit@20 28.71%.

Against the historical MARTS-only eight-expert route on the same TPS lineage,
the atlas improves R2E MRR by 0.0049 and Hit@20 by 1.75 percentage points while
trading 1.45 points at Hit@10.  E2R improves MRR by 0.0184 and both Hit@10 and
Hit@20 by about 7.07 points.  This is treated as a whole-method result rather
than a requirement that every individual metric increase.

The TPS strict R2E queries were also scored against all 185,918 proteins in the
general_merged universe.  The full atlas reaches MRR 0.01245, Hit@10 3.22% and
Hit@20 5.60%, versus 0.00984, 2.52% and 4.62% for the same trained model's
universal chart alone.  Median best-positive rank improves from 8,998 to 5,488.
This same-model/same-split comparison is the current evidence that local
TPS/multiview specialization remains useful after expanding from a TPS-sized
candidate pool to the full general protein universe.

These TPS numbers are not pooled with the general BiME-Rank benchmarks.  The
unchanged general routes retain their frozen evidence, including strict
temporal R2E Hit@20/50 of 15.97%/22.22%, strict temporal E2R Hit@20/50 of
15.32%/18.55%, and Enzyme-405 Hit@10/MRR/MAP of
54.42%/0.2864/0.2847.  Orphan-335 and known-positive context results remain
separate protocol-specific evidence.  The complete non-collapsed capability
surface is hash-locked in
`reproducibility/bime_rank/records/FIBRE_INTERACTION_ATLAS_SCORECARD_V1.json`.

## TPS specialization

TPS is now modeled as a biochemical family chart, not as a dataset-specific candidate universe and not as an additive residual. Its applicability should be determined from TPS-relevant molecular/family state and compatible reaction chemistry.

Historical TPS-specific routes and assets remain available and reproducible. A broad-universe atlas integration must be validated under a frozen protocol before replacing any deployed route.

## Bilinear correction experiment

The earlier zero-initialized global bilinear correction \(I+B\) was evaluated once on three cleanroom strict double-cold folds and was not promoted because R2E was mixed/slightly negative while E2R improved. The result remains at reproducibility/bime_rank/records/FIBRE_CATALYTIC_INTERACTION_RESIDUAL_DEV_V1.json as a negative development record. It is not part of the current multi-expert geometry.

## Historical geometry

Product-manifold, correspondence-defect and Pareto-geometry experiments remain reproducibility records under docs/legacy_geometry and compatibility code. They are not the current FIBRE ontology.
