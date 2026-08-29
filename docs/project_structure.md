# Project Structure

The repository is organized around independent research blocks rather than one
flat experiment directory. A block may share external assets and result folders,
but its source code, configs, notes, and tests should stay under one project
folder.

## Top-Level Contract

| Path | Purpose | Git policy |
| --- | --- | --- |
| `projects/active/` | Runnable research projects with code/configs/tests. | Commit source, configs, notes, lightweight tests. |
| `projects/planned/` | Future or parked directions. | Keep only README/design notes until active. |
| `scripts/catalyst_finder/` | Current Catalyst service and agent/API code. | Commit production source and tests. |
| `scripts/terpene/` | Shell entrypoints for terpene screening experiments. | Commit reproducible controllers/status scripts. |
| `scripts/setup/` | Dependency, asset, and environment setup. | Commit setup automation, not downloaded assets. |
| `scripts/maintenance/` | Cleanup and repo hygiene. | Commit safe cleanup tools. |
| `docs/` | Cross-project documentation and prompts. | Commit stable documentation. |
| `external_repos/` | Third-party repositories. | Treat as read-only; do not vendor large upstream code into commits. |
| `data/` | Raw/intermediate/generated runtime data. | Local/provisioned; ignored by git. |
| `results/` | Runtime models, reports, metrics, and generated outputs. | Local/provisioned; ignored by git. |

## Active Project Shape

A mature active project should generally look like this:

```text
projects/active/<project>/
  README.md
  configs/
  notes/
  adapters/       # if wrapping external tools or converting formats
  runners/        # if orchestrating experiment runs
  analysis/       # if producing metrics, reports, comparisons
  tests/
```

Not every project needs every folder. The rule is conceptual: code that belongs
to one research direction should stay with that direction.

## Current Blocks

### `terpene_screening`

Purpose: build terpene synthase candidate pools and evaluate wet-lab-oriented
screening gates.

Main code areas:

- data inspection and pair construction
- P2Rank/CAGE inference wrappers
- reaction-only and few-shot fair candidate evaluation
- gate-matrix generation for wet-lab decision support

### Planned blocks

`candidate_retrieval`, `mechanism_check`, and `reaction_center` are intentionally
kept under `projects/planned/` until they gain executable workflows. This avoids
mixing placeholders with currently active implementation code.

## Naming Rules

- Use project names that describe the research question, not the current script
  name. For example, `pocket_robustness` is better than `pocket`.
- Use script folders for operational domain: `scripts/catalyst_finder/`,
  `scripts/terpene/`, `scripts/setup/`, `scripts/maintenance/`.
- Prefer relative paths from repository root in configs and docs.
- Do not hard-code server-local roots such as `/home/.../igem2026` in committed
  source files.
