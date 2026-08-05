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
    pocket_robustness/     EnzymeCAGE pocket-hypothesis robustness work.
    terpene_screening/     Terpene synthase screening and wet-lab candidate gates.
  planned/
    candidate_retrieval/   Placeholder for candidate-pool construction work.
    mechanism_check/       Placeholder for mechanism/cofactor/failure checks.
    reaction_center/       Placeholder for reaction-center analysis.

scripts/
  pocket/                  Pocket/EnzymeCAGE experiment controllers.
  terpene/                 Terpene screening controllers and status checks.
  setup/                   Dependency, asset, and environment setup.
  maintenance/             Cleanup and repository hygiene scripts.

docs/                      Cross-project documentation and prompts.
external_repos/            Read-only third-party repositories.
data/                      Local raw/intermediate data; mostly ignored by git.
results/                   Lightweight reports plus local generated outputs.
```

See `docs/project_structure.md` for the full directory contract.

## Active Projects

### `projects/active/pocket_robustness/`

This block studies how EnzymeCAGE retrieval changes under different pocket
hypotheses: official/P2Rank/fpocket sources, top-1 vs top-k selection, and
pocket-level to enzyme-level aggregation strategies.

Typical entrypoint:

```bash
python projects/active/pocket_robustness/runners/run_compare_baselines.py \
  --experiment_config projects/active/pocket_robustness/configs/demo_p2rank_top1.yaml
```

Or run the pocket baseline wrapper:

```bash
bash scripts/pocket/run_pocket_baselines.sh
```

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
- Large datasets, model weights, raw databases, and bulky generated artifacts
  should not be committed to git.
