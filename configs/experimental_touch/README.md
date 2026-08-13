# Experimental-touch profiles

Each YAML profile is a complete switch point for a candidate source and its isolated runtime state.

Rules:

- candidate source: read-only input;
- `evidence_db`: only public/reusable evidence, under `allowed_evidence_root`;
- `run_root`: candidate-library/run-specific state, under `allowed_run_root`;
- ranking/focus CSV: optional read-only prioritization input;
- no profile may point runtime storage into `external_repos/igem_database`.

Use `doctor` before a new run. Copy `profiles/template.yaml` to add a library; do not change Python code to switch databases.
