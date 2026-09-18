# Starase Atlas computational stack — FIBRE & Starase Navigator

This repository contains the computational part of **Starase Atlas**: the **FIBRE** enzyme–reaction retrieval method and the **Starase Navigator** scientific agent/runtime. Starase Atlas itself is the wider iGEM project and also includes the database experience, experimental work and Human Practices; this repository does not redefine that umbrella. The active implementation is organized by **software responsibility and scientific object**, not by the names or version numbers of historical experiments.

## Start here

| Need | Authority |
| --- | --- |
| Current scientific implementation | `projects/active/fibre/` |
| Current mathematical/mainline description | `projects/active/fibre/docs/method.md` |
| Current implementation status | `projects/active/fibre/docs/status.md` |
| Product/research workflow | `projects/active/fibre/docs/product_workflow.md` |
| Default broad runtime contract | `configs/production_routes/default.yaml` |
| Starase Navigator service | `scripts/starase_navigator/` |
| Starase Navigator frontend | `frontend/starase_navigator/` |
| Historical BiME-Rank reproduction | `reproducibility/bime_rank/` |
| Retired research lineage | `archive/terpene_screening/` |

## Current scientific object

The current research mainline uses one product-space correspondence between reaction molecular states and protein molecular states. Reaction → enzyme and enzyme → reaction are sections of that same object. Available sequence, reaction, structure and pocket observations refine the corresponding factor geometry; a missing view remains unobserved rather than becoming negative evidence.

The product is AI-native: users describe the scientific question. The semantic planner chooses broad versus application-focused search and interactive versus deeper evidence acquisition. Internal backend names, route IDs and historical model labels are provenance, not user-facing choices.

## Repository contract

```text
projects/active/fibre/
  core/         stable contracts and provenance
  runtime/      deployed broad-retrieval primitives
  geometry/     current correspondence geometry
  pipelines/    deterministic current asset builders
  evaluation/   current evaluations and audits
  evidence/     biological interpretation helpers
  docs/         current method/status/workflow documents
  tests/        current formal scientific tests

scripts/starase_navigator/
  agent/        model-led task resolution
  routing/      semantic retrieval planning
  retrieval/    retrieval gateway and focused service
  observations/ progressive molecular-observation handling
  tests/        product/runtime regression suite

reproducibility/bime_rank/
  historical baseline release, records, scripts and regression evidence

archive/
  retained lineage only; never current runtime/research authority
```

Versioned names are intentionally retained for **data/model artifacts and historical reproduction records**, because immutable artifact identity is part of scientific provenance. Formal source and production configuration filenames are responsibility-based and version-free.

## Validation

```bash
# Static/current scientific tests
PYTHONPATH=. .venv/bin/python -m pytest -q projects/active/fibre/tests

# Starase Navigator product tests
PYTHONPATH=. .venv/bin/python -m pytest -q scripts/starase_navigator/tests

# Historical release reproduction boundary
PYTHONPATH=. .venv/bin/python scripts/maintenance/run_reproduction_tests.py --tier release
PYTHONPATH=. .venv/bin/python scripts/maintenance/run_reproduction_tests.py --tier extended
```

Plain `pytest` remains limited to release-maintenance checks so archived or server-local historical files cannot silently become quality gates.

For release/data restoration policy, see `docs/RESEARCH_RELEASE.md` and `reproducibility/research_release_manifest.json`.
