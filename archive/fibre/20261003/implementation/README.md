# FIBRE research core

**FIBRE — Factorized Interaction Basis for Reaction–Enzyme** models catalytic compatibility with an interaction atlas.

Different molecular views define local coordinates for reaction demand and enzyme capability. Local experts estimate the same task-level catalytic compatibility in different biochemical regimes. An availability-aware partition of unity glues those local estimates; the current frozen R2E and E2R retrieval realizations may use different directional partitions rather than claiming pointwise-identical final scores.

Historical reaction/enzyme product-manifold geometry is retained only for reproducibility and compatibility.

## Current structure

- kernel/ — interaction-atlas primitives, chart gluing and train-free observation operators.
- core/ — candidate-universe, route and provenance contracts.
- runtime/ — deployed neural ranking primitives.
- application/ — application-only full-data and TPS-family assets.
- evidence/ — assay, mechanism and source-bound evidence.
- pipelines/ — deterministic evidence/data builders.
- evaluation/ — current and retained evaluation utilities.
- docs/ — current interaction-atlas method, evaluation and workflow documents.
- docs/history/ — the layered raw development narrative and the compressed scientific narrative.
- docs/theory/ — the Chinese LaTeX monograph for the current interaction-atlas mathematics and proofs.
- geometry/ and docs/legacy_geometry/ — retained historical/compatibility implementation.
- tests/ — scientific/runtime contract tests.

## Invariants

1. **Universal coverage.** A broad raw-input chart keeps every valid reaction/protein pair scoreable.
2. **Multi-view geometry.** Optional sequence, structure, pocket, reaction-centre and family views define additional local charts rather than mandatory inputs.
3. **Local experts, one interaction.** Experts estimate the same scalar catalytic interaction in different biochemical regimes.
4. **Missing-neutral gluing.** Unavailable charts receive zero mass and available chart weights renormalize.
5. **Coordinate independence.** Different charts may have unrelated dimensions; their local bilinear forms make their coordinates meaningful.
6. **Overlap consistency.** Charts that are simultaneously applicable are trained to agree on their shared physical interaction.
7. **TPS is a biochemical chart.** It is not defined by a TPS dataset or a restricted candidate universe.
8. **Train-free experimental updates.** Accepted pair observations may update chart-local empirical interaction operators without retraining molecular encoders.
9. **Research/application separation.** Only the frozen reproduction profile supports benchmark claims.

Start with docs/method.md and docs/catalytic_kernel_foundation.md for the current method. For project evolution, use docs/history/SCIENTIFIC_NARRATIVE_HISTORY_ZH.md first and docs/history/RAW_DEVELOPMENT_HISTORY_ZH.md for full traceability. For the mathematical derivation and proofs, see docs/theory/FIBRE_INTERACTION_ATLAS_THEORY_ZH.tex.
