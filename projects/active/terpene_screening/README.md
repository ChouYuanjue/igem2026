# Enzyme–reaction retrieval research core

This directory contains the **current** scientific implementation used by Catalyst retrieval. It is intentionally organized by responsibility rather than by experiment lineage, paper nickname, or version number.

The current geometric mainline models reaction and protein molecular states as two factor spaces and represents their known association relation on the product space. Reaction-to-enzyme and enzyme-to-reaction retrieval are two sections of the same correspondence object. Missing structural or pocket measurements remain missing observations; they are not negative evidence.

## Authoritative structure

- `core/` — stable retrieval contracts, route loading, provenance, candidate scopes and evidence interfaces.
- `runtime/` — deployed ranking primitives and compatibility runtime required by the broad retrieval backend.
- `geometry/` — the current product-space geometry, partial-observation metric, out-of-sample extension and structural query extension.
- `pipelines/` — deterministic builders for the current molecular-state and deployment assets.
- `evaluation/` — current correspondence/inductive evaluation and audits.
- `evidence/` — biological witness and mechanism interpretation helpers.
- `docs/` — current scientific method, evaluation, product workflow and status documents.
- `tests/` — tests of the current scientific/runtime contracts only.

The default broad runtime route contract is `configs/production_routes/default.yaml`. Application-focused retrieval is selected by the Catalyst semantic planner and uses the deployment assets described in `docs/status.md`; users do not choose an internal backend.

## What is not current source

Historical BiME-Rank training/evaluation material is preserved under `reproducibility/bime_rank/`. It remains important for baseline reproduction and released evidence, but it is not the identity or directory layout of the current implementation.

Retired experiments and superseded branches are preserved under `archive/terpene_screening/`. Archive files are never used as runtime or current-research authority. Their presence is provenance, not an invitation to choose a method by filename.

## Research invariants

1. **One correspondence object.** R2E and E2R are sections of the same product-space relation.
2. **Positive evidence only.** Verified associations add support to the relation; synthetic absence is not silently converted into a negative label.
3. **Partial observation is neutral.** Sequence, structure, pocket and reaction-local coordinates contribute only where observed.
4. **Research/product separation.** Frozen research splits remain reproducible while production may use a separately versioned positive registry.
5. **No architecture-facing product switches.** The user states a scientific goal; semantic routing chooses search breadth and evidence depth.

Start with:

- `docs/method.md` — mathematical mainline;
- `docs/molecular_geometry.md` — factor geometry and missing-view semantics;
- `docs/evaluation.md` — evaluation contract;
- `docs/product_workflow.md` — research/product and AI-native workflow;
- `docs/status.md` — current implementation status.

## Validation

Current formal scientific tests live in this package:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q projects/active/terpene_screening/tests
```

Historical release reproduction is deliberately separate:

```bash
PYTHONPATH=. .venv/bin/python scripts/maintenance/run_reproduction_tests.py --tier release
```
