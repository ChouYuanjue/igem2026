# Scripts

Shell entrypoints are grouped by operational domain.

- `pocket/`: EnzymeCAGE pocket robustness experiments, reports, status checks.
- `terpene/`: terpene screening pipelines, gate matrix, status checks.
- `setup/`: external repository cloning, asset download, environment setup.
- `maintenance/`: cleanup and repository hygiene.

Run scripts from the repository root, for example:

```bash
bash scripts/pocket/run_real_pocket_experiment.sh
bash scripts/terpene/run_terpene_gate_matrix.sh
```
