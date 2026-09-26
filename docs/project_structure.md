# Project structure

The repository separates the **current scientific implementation**, the **Starase Navigator product**, and **historical reproduction/lineage**. Currentness is defined by these responsibility boundaries, not by old experiment names, version suffixes, or timestamps.

## Top-level contract

| Path | Role | Authority |
| --- | --- | --- |
| `projects/active/fibre/` | FIBRE enzyme–reaction retrieval research core | Current scientific implementation |
| `scripts/starase_navigator/` | AI-native product/runtime, semantic planning, retrieval orchestration and observation acquisition | Current application implementation |
| `frontend/starase_navigator/` | Starase Navigator web interface | Current product UI |
| `configs/production_routes/` | Deployed route/runtime contracts | Current production configuration |
| `reproducibility/bime_rank/` | Frozen BiME-Rank baseline, canonical claims, source snapshots, historical scripts and release regression | Historical/reproduction authority only |
| `archive/terpene_screening/` | Retired research lineage and superseded branches | Audit only; never current authority |
| `scripts/maintenance/` | Release, manifest, asset and repository validation | Repository/release maintenance |
| `data/`, `results/` | Canonical assets plus local/derived machine data | Deny-by-default Git policy; explicit release manifests decide tracked assets |

## FIBRE scientific core

projects/active/fibre/ is organized by current responsibility while retaining explicit compatibility code:

- kernel/: current factorized catalytic-interaction primitives and bounded train-free updates.
- core/: candidate, routing and provenance contracts shared by deployed interaction routes.
- runtime/: deployed neural ranking primitives.
- application/: application-only full-data and TPS-family assets.
- evidence/ and pipelines/: provenance-bound biochemical evidence and deterministic builders.
- docs/: current factorized-interaction method, evaluation, status and workflow documents.
- docs/legacy_geometry/ and geometry/: retained product-manifold/correspondence lineage required only for historical replay or explicit compatibility calls.
- tests/: current scientific/runtime contracts plus explicitly marked legacy-compatibility tests.

**FIBRE** now means **Factorized Interaction Basis for Reaction–Enzyme**. Its research object is a catalytic interaction between reaction demand and enzyme capability learned from sparse paired observations and rich per-side molecular information. Reaction-to-enzyme and enzyme-to-reaction are two query directions over the same interaction problem; they do not require reactions and proteins to inhabit one product manifold.

Only the fibre-reproduction profile may support benchmark claims. The Starase application profile may use current full-data evidence and TPS specialization but cannot turn those outputs into benchmark evidence.

## Starase Navigator

`scripts/starase_navigator/` is also responsibility-based:

- `agent/`: model-led task resolution.
- `routing/`: semantic planning of retrieval/evidence work.
- `retrieval/`: retrieval gateway and focused retrieval service.
- `observations/`: progressive acquisition/inventory of molecular observations.
- `tests/`: product/runtime regression suite.

User-facing behavior does not expose historical backend names, candidate-universe switches, or experiment versions as product modes. The user states the scientific task; the planner selects the appropriate retrieval/evidence path.

## Historical BiME-Rank boundary

BiME-Rank remains an important frozen baseline and release/reproduction namespace, but it is **not the identity of the current implementation**. Its immutable protocols, records and model-lineage scripts live under `reproducibility/bime_rank/`. Frozen files keep their original bytes and hashes even when their historical internal paths refer to pre-refactor locations; `scripts/maintenance/repository_move_map.json` records the repository relocation.

Versioned names are therefore allowed for immutable artifacts and historical reproduction records. Current formal source/configuration names are responsibility-based and version-free.

## Archive rule

Retired experiments and superseded research branches live under `archive/terpene_screening/` (and explicitly marked experiment archives). Archive material cannot become runtime or evaluation authority merely because it exists on the server. If an old asset is still needed for a frozen reproduction claim, that dependency must be explicit in the reproduction manifests or move map.

## Validation

Use separate gates for separate responsibilities:

```bash
# Current scientific core
PYTHONPATH=. .venv/bin/python -m pytest -q projects/active/fibre/tests

# Starase Navigator product/runtime
PYTHONPATH=. .venv/bin/python -m pytest -q scripts/starase_navigator/tests

# Frozen BiME release and extended reproduction
PYTHONPATH=. .venv/bin/python scripts/maintenance/run_reproduction_tests.py --tier release
PYTHONPATH=. .venv/bin/python scripts/maintenance/run_reproduction_tests.py --tier extended
```

Plain `pytest` remains scoped to release-maintenance tests so archived or server-local lineage cannot silently re-enter the quality gate.

The rollback point immediately before this repository refactor is Git tag `pre-repository-refactor-20260918`.
