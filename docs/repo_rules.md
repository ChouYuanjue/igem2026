# Repository Rules

## External Repositories Are Read-Only

`external_repos/` is dependency and reference space. Do not edit files under this
directory. In particular, do not add adapters, runners, notebooks, analysis
scripts, or patches inside `external_repos/EnzymeCAGE`.

If an experiment needs to wrap or adapt an external tool, write that wrapper in
this repository.

Pinned nested references may have a tracked lock and verification script in the
parent repository. Their source worktrees remain ignored and read-only. Do not
convert one reference to a different management style without updating the shared
external-repository contract.

## Where Our Code Goes

- Active project code: `projects/active/<project>/`
- Shared scripts: `scripts/`
- Documentation: `docs/`
- Project-specific notes: `projects/active/<project>/notes/`
- Project tests: `projects/active/<project>/tests/`

## Data, Results, Reports, and Configs

- Raw or downloaded data: `data/raw/`
- Small demo data: `data/demo/`
- Processed data: `data/processed/`
- Manifests: `data/manifests/`
- Predictions and metrics: `results/`
- Human-readable generated reports: local `reports/` or `results/`; only frozen
  publication evidence explicitly listed in the research release manifest is versioned
- Experiment configs: `projects/active/<project>/configs/`

## Exploration Directory Shape

Each active project should contain:

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

`data/` and `results/` are deny-by-default, not universally forbidden. The scientific
release explicitly tracks project-owned model weights, canonical result evidence, and
the compact/public data assets listed in `reproducibility/research_release_manifest.json`.
Do not commit anything else from those roots merely because it exists locally.

Large derived feature matrices and multi-GB third-party foundation checkpoints stay out
of normal Git and must have a deterministic builder or fixed upstream source plus a
checksum. Local reports, private candidate libraries, downloads, caches, and retired
development-only code remain untracked.
