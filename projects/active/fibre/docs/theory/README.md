# FIBRE interaction-atlas theory monograph

`FIBRE_INTERACTION_ATLAS_THEORY_ZH.tex` is the Chinese textbook-style mathematical treatment of the **current** FIBRE interaction-atlas model. It is not the legacy product-manifold ontology.

The monograph separates:

- context-dependent physical activity `A(r,e;c)` from task-level working compatibility `K*(r,e)`;
- local reaction/enzyme coordinate spaces from chart-local bilinear interaction forms;
- ideal pair-aware partition of unity from the current frozen directional R2E/E2R partitions;
- implemented train-free finite-rank interaction updates from application-level orchestration that is not yet claimed as complete;
- current atlas theory from the historical product-manifold/correspondence derivation.

## Build

The document requires XeLaTeX for reliable Chinese typesetting. From this directory:

```bash
xelatex -interaction=nonstopmode -halt-on-error FIBRE_INTERACTION_ATLAS_THEORY_ZH.tex
xelatex -interaction=nonstopmode -halt-on-error FIBRE_INTERACTION_ATLAS_THEORY_ZH.tex
```

The second pass resolves the table of contents and internal references. The generated PDF is a build artifact and is not required to be committed; the canonical source is the `.tex` file.

## Current verified build

The 2026-09-27 build corresponding to the current source renders as a 36-page A4 PDF. PDF preflight reports it as openable, non-encrypted and text-based. The final XeLaTeX log has no overfull boxes; the only remaining warning is a non-fatal `microtype` footnote patch warning.

The repository-facing mathematical invariants remain covered by the FIBRE tests and frozen reproduction records referenced inside the monograph.
