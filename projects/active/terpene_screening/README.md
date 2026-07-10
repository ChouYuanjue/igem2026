# Terpene Screening

This active project contains terpene synthase screening workflows and wet-lab
candidate-gate evaluation code.

## Scope

- inspect terpene synthase source data
- build reaction/enzyme candidate pairs
- download or locate structures
- run P2Rank/CAGE-style inference wrappers
- evaluate reaction-only, few-shot, and wet-lab intention workflows
- generate gate matrices for candidate prioritization

## Main Entrypoints

Run the gate matrix workflow:

```bash
bash scripts/terpene/run_terpene_gate_matrix.sh
```

Run the background screening pipeline:

```bash
bash scripts/terpene/run_terpene_screen_background.sh
```

Check current screening status:

```bash
bash scripts/terpene/check_terpene_screen_status.sh
```

## Outputs

- `data/terpene_cage_screen/`: local intermediate candidate and structure files.
- `data/terpene_gate_matrix/`: local gate candidate pools.
- `results/terpene_cage_screen/`: predictions, metrics, reports, logs.
- `results/wetlab_intentions/`: wet-lab intention evaluation outputs.

Large generated files should stay out of git unless they are intentionally small
summary artifacts.
