# Experiment Protocol

## Goal

Compare pocket hypotheses for EnzymeCAGE inference and measure whether enzyme
retrieval rankings are robust to pocket source, pocket selection, and
aggregation.

## Baseline Matrix

This project does not only compare top-1 against one top-k run. The maintained
matrix varies:

1. pocket source: official / P2Rank / fpocket / P2Rank + fpocket union
2. pocket selection: top-1 / top-k / union top-k
3. aggregation: max / mean / rank-weighted / softmax-pool / source-weighted
4. optional prior: catalytic residue / ScanNet residue prior

The machine-readable entry point is:

```bash
explorations/pocket/configs/baseline_matrix.yaml
```

Per-baseline configs are generated, not hand-maintained.

## Phase 1 Baselines

- `official_eval_enzyme405`
- `official_eval_orphan335`
- `pocket_smallset_from_enzyme405`
- `pocket_smallset_from_enzyme405_p2rank_topk_softmax`
- `pocket_smallset_from_orphan335_p2rank_top1`
- `pocket_smallset_from_orphan335_p2rank_topk_softmax`
- `official_or_p2rank_top1`
- `p2rank_top1`
- `p2rank_topk_max`
- `p2rank_topk_mean`
- `p2rank_topk_rank_weighted`
- `p2rank_topk_softmax_pool`
- `fpocket_top1`
- `fpocket_topk_max`
- `fpocket_topk_rank_weighted`
- `p2rank_fpocket_union_max`
- `p2rank_fpocket_union_source_weighted`

Scaffold-only initially:

- `p2rank_topk_catalytic_residue_prior`
- `p2rank_topk_scannet_prior`

## Required Records Per Experiment

Each experiment should save:

- config copy
- `commands.jsonl`
- logs
- `run_summary.json`
- pocket manifest
- pocket-level predictions when inference completes
- aggregated predictions when inference completes
- metrics if labels are available

If demo data or checkpoint files are missing, prediction steps must be marked
blocked and no synthetic prediction files should be created.

## Official Assets vs README Demo

The README mining demo references `dataset/demo` and
`checkpoints/pretrain/seed_42/best_model.pth`. The downloaded official Google
Drive assets currently contain official evaluation datasets and
`checkpoints/pretrain/seed_42/epoch_19.pth`, but not the demo mining paths.

Because of that mismatch, the first target is not the demo baseline. The
experiment has two layers:

1. `official_eval`: run official Enzyme-405 / Orphan-335 configs with the real
   `epoch_19.pth` checkpoint to validate the EnzymeCAGE environment and assets.
2. `derived_smallset`: build a small controllable slice from official evaluation
   data, then run pocket intervention only if full enzyme structures are
   available.

Do not rename, copy, or symlink `epoch_19.pth` as `best_model.pth`. Demo-based
baselines remain in the matrix, but they are unavailable until matching demo
assets exist.

## Analysis Checklist

- score distribution across baselines
- rank shift per reaction and enzyme
- rescued and harmed cases when labels exist
- pocket rank selected by aggregation
- whether top-ranked enzyme uses a non-top1 pocket
- whether fpocket and P2Rank select overlapping pockets
- whether union-source aggregation changes the best pocket source

## Interpretation

- `p2rank_top1` is a single predicted pocket hypothesis baseline.
- P2Rank top-k baselines test localization uncertainty within one detector.
- fpocket baselines test pocket detector choice.
- P2Rank + fpocket union baselines test heterogeneous pocket hypotheses.
- catalytic/ScanNet priors bridge toward mechanism-evidence-aware reranking.

## Follow-Up Directions

- install the EnzymeCAGE environment dependencies required by `infer.py`
- locate or download full enzyme structures for official evaluation entries
- supply EnzymeCAGE demo assets only if the README mining demo should be tested
- install P2Rank 2.5.1 and Java 17+
- build/install fpocket
- extend to curated failure cases
- add catalytic residue evidence and reaction-class-conditioned priors
