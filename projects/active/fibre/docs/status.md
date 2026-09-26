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

The current theory adds the missing mathematical requirement: experts that are simultaneously applicable should agree on the scalar catalytic interaction on chart overlaps. This is represented by the gluing-consistency loss in kernel/atlas.py.

## TPS specialization

TPS is now modeled as a biochemical family chart, not as a dataset-specific candidate universe and not as an additive residual. Its applicability should be determined from TPS-relevant molecular/family state and compatible reaction chemistry.

Historical TPS-specific routes and assets remain available and reproducible. A broad-universe atlas integration must be validated under a frozen protocol before replacing any deployed route.

## Bilinear correction experiment

The earlier zero-initialized global bilinear correction \(I+B\) was evaluated once on three cleanroom strict double-cold folds and was not promoted because R2E was mixed/slightly negative while E2R improved. The result remains at reproducibility/bime_rank/records/FIBRE_CATALYTIC_INTERACTION_RESIDUAL_DEV_V1.json as a negative development record. It is not part of the current multi-expert geometry.

## Historical geometry

Product-manifold, correspondence-defect and Pareto-geometry experiments remain reproducibility records under docs/legacy_geometry and compatibility code. They are not the current FIBRE ontology.
