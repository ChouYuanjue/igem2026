# Repository Rules

## External Repositories Are Read-Only

`external_repos/` is dependency and reference space. Do not edit files under this
directory. In particular, do not add adapters, runners, notebooks, analysis
scripts, or patches inside `external_repos/EnzymeCAGE`.

If an experiment needs to wrap or adapt an external tool, write that wrapper in
this repository.

## Where Our Code Goes

- Exploration code: `explorations/<direction>/`
- Shared scripts: `scripts/`
- Documentation: `docs/`
- Direction-specific notes: `explorations/<direction>/notes/`
- Tests: `explorations/<direction>/tests/`

## Data, Results, Reports, and Configs

- Raw or downloaded data: `data/raw/`
- Small demo data: `data/demo/`
- Processed data: `data/processed/`
- Pocket working outputs: `data/pocket_runs/`
- Manifests: `data/manifests/`
- Predictions and metrics: `results/`
- Human-readable reports: `results/reports/`
- Experiment configs: `explorations/<direction>/configs/`

## Exploration Directory Shape

Each exploration direction should contain:

- `README.md`
- `notes/`
- `configs/`
- `adapters/`, `runners/`, or `analysis/` when needed
- `tests/`

## Required Experiment Artifacts

Each experiment run must produce:

- config copy
- command log
- `run_summary.json`
- predictions
- metrics or analysis results when applicable

If labels are unavailable, metrics can be skipped, but the run summary must say
why.

## Git Hygiene

Do not commit large files, model weights, raw databases, generated feature
stores, or bulky result directories. Keep reproducible commands, configs,
schemas, and lightweight tests in git.
