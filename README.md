# iGEM 2026 Research Workspace

This repository is a modular workspace for iGEM 2026 enzyme retrieval and
terpene-screening experiments. It intentionally contains several research
blocks that are related at the iGEM strategy level but not always tightly coupled
at the implementation level.

The top-level structure separates **project code**, **operational scripts**,
**documentation**, **external dependencies**, **data**, and **results** so that
new directions do not get mixed into one large `explorations/` bucket.

## Repository Layout

```text
projects/
  active/
    terpene_screening/     Terpene synthase screening and production retrieval core.
  planned/
    candidate_retrieval/   Placeholder for candidate-pool construction work.
    mechanism_check/       Placeholder for mechanism/cofactor/failure checks.
    reaction_center/       Placeholder for reaction-center analysis.

scripts/
  catalyst_finder/         Current Catalyst service, model-led agent tools, route catalog, and API.
  terpene/                 Terpene screening controllers and status checks.
  setup/                   Dependency, asset, and environment setup.
  maintenance/             Cleanup and repository hygiene scripts.

docs/                      Cross-project documentation and prompts.
external_repos/            Read-only third-party repositories.
data/                      Local raw/intermediate data; mostly ignored by git.
results/                   Local runtime/model/experiment outputs; ignored by git.
```

See `docs/project_structure.md` for the full directory contract.

## Active Projects


### Catalyst Finder

Catalyst Finder is the current user-facing research service. It combines verified database evidence, bidirectional enzyme/reaction retrieval, literature and structure inspection, route design, and pathway compatibility through a model-led tool harness.

The current product source is isolated under:

```text
frontend/catalyst_finder/
scripts/catalyst_finder/
```

Retired portal/pocket implementations are not part of the tracked production source. Dynamic candidate-universe sizes and the deployed build revision are reported by `GET /api/status`; they are intentionally not duplicated as fixed numbers in this README.

See `frontend/catalyst_finder/README.md` for the current retrieval semantics, evidence sources, bilingual/session boundaries, runtime cache behavior, and deployment/test commands.

### `projects/active/terpene_screening/`

This block builds and evaluates terpene synthase candidate gates, including
reaction-only/few-shot CAGE-style reranking and wet-lab intention evaluation.

Typical entrypoint:

```bash
bash scripts/terpene/run_terpene_gate_matrix.sh
```

## Setup

Clone external repositories:

```bash
bash scripts/setup/clone_external_repos.sh
```

Synchronize the pinned database/frontend design reference separately:

```bash
bash scripts/setup/sync_igem_database_reference.sh
```

The pinned nested repository is read-only. Its exact commit and sparse-checkout
contract are tracked in `reproducibility/external_repos/igem_database.lock.json`.

Prepare a lightweight local environment for this repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

EnzymeCAGE itself may require its own environment. Keep that setup outside our
source edits and follow the upstream repository instructions.

## Repository Rules

- `external_repos/` is read-only dependency/reference space.
- Active project code belongs in `projects/active/<project>/`.
- Future directions belong in `projects/planned/<direction>/` until they have
  runnable code.
- Shared operational scripts belong in `scripts/<domain>/`.
- Intermediate data belongs in `data/`.
- Experiment outputs belong in `results/`.
- Documentation belongs in `docs/` or a project-specific `notes/` folder.
- `data/`, `results/`, local reports, model weights, raw databases, and generated artifacts
  are provisioned/rebuilt locally and must not be committed to git.
