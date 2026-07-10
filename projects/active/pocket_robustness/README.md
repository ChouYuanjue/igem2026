# Pocket Robustness for EnzymeCAGE

## Background

EnzymeCAGE models the enzyme side through a catalytic pocket rather than the
whole protein structure. AlphaFill is the main pocket extraction route described
by the model, while P2Rank is the practical fallback/demo route. The pocket is
the local structural entry point for enzyme-reaction interaction.

This project does not train EnzymeCAGE and does not modify the EnzymeCAGE
repository. We intervene only at inference time by changing pocket hypotheses
and aggregation.

## Research Questions

- Is a single predicted pocket enough for reliable enzyme retrieval?
- Do different pocket detectors produce different rankings?
- How should multiple pocket hypotheses become one enzyme-level score?
- Can pocket confidence or residue evidence become a reranking prior?

## Why More Than Top-1 vs Top-K

Only comparing one top-1 pocket against one top-k run would mix several
questions. The maintained matrix separates four dimensions:

1. pocket source: official / P2Rank / fpocket / P2Rank + fpocket union
2. pocket selection: top-1 / top-k / union top-k
3. aggregation: max / mean / rank-weighted / softmax-pool / source-weighted
4. optional prior: catalytic residue / ScanNet residue prior

The single hand-maintained entry point is:

```bash
projects/active/pocket_robustness/configs/baseline_matrix.yaml
```

Generate per-baseline configs:

```bash
python projects/active/pocket_robustness/runners/generate_baseline_configs.py \
  --matrix projects/active/pocket_robustness/configs/baseline_matrix.yaml \
  --output_dir projects/active/pocket_robustness/configs/generated
```

## Baselines

- `official_eval_enzyme405`: first target for reproducing official Enzyme-405
  inference with `epoch_19.pth`.
- `official_eval_orphan335`: first target for reproducing official Orphan-335
  inference with `epoch_19.pth`.
- `pocket_smallset_from_*`: derived smallsets from official evaluation data,
  used only when full structures can be mapped for pocket intervention.
- `official_or_p2rank_top1`: nearest official EnzymeCAGE demo route.
- `p2rank_top1`: single predicted pocket hypothesis baseline.
- `p2rank_topk_*`: pocket localization uncertainty within P2Rank.
- `fpocket_top1/topk_*`: geometry-based detector comparison.
- `p2rank_fpocket_union_*`: heterogeneous pocket source union.
- `p2rank_topk_catalytic_residue_prior`: scaffold for catalytic evidence.
- `p2rank_topk_scannet_prior`: scaffold for ScanNet residue prior.

## Running

Inspect the downloaded official asset package:

```bash
bash scripts/pocket/inspect_enzymecage_assets.sh
bash scripts/pocket/inspect_enzymecage_configs.sh
```

Inspect EnzymeCAGE:

```bash
bash scripts/pocket/inspect_enzymecage.sh
```

Run one generated config:

```bash
python projects/active/pocket_robustness/runners/run_compare_baselines.py \
  --experiment_config projects/active/pocket_robustness/configs/generated/demo_p2rank_topk_max.yaml
```

Run the full matrix:

```bash
bash scripts/pocket/run_real_pocket_experiment.sh
```

The full controller writes:

- `results/pocket/enzymecage_inspection.json`
- `results/pocket/<run_id>/commands.jsonl`
- `results/pocket/<run_id>/run_summary.json`
- `results/pocket/comparison/comparison_report.md`
- `results/pocket/experiment_status.md`

If demo data, P2Rank/fpocket binaries, EnzymeCAGE dependencies, or checkpoints
are missing, the relevant baseline is marked blocked. Predictions are not
fabricated.

## Current Official-Asset Route

The downloaded EnzymeCAGE Google Drive assets do not match the README mining
demo route: `dataset/demo` and `checkpoints/pretrain/seed_42/best_model.pth`
are absent. The official assets do contain `checkpoints/pretrain/seed_42/epoch_19.pth`
and evaluation datasets such as Enzyme-405 and Orphan-335.

For that reason, demo-mining baselines are retained but treated as
`blocked_missing_demo_assets` until matching demo assets are supplied. The first
real target is `official_eval`, which validates the EnzymeCAGE environment,
official configs, and `epoch_19.pth` checkpoint without changing external repo
files. The second layer is derived-smallset pocket intervention on official data
slices. Those baselines require full structures; pre-extracted pocket PDBs are
not used as a fake replacement for full P2Rank/fpocket input.

Do not rename or copy `epoch_19.pth` to `best_model.pth`.

## Interpretation

- If `best_pocket_rank > 1` is common, top-1 pocket selection is unstable.
- If fpocket and P2Rank disagree strongly, pocket detector source matters.
- If union-source baselines change ranking, heterogeneous hypotheses are useful.
- If all baselines are unchanged on demo data, expand to curated failure cases
  before concluding that pocket hypotheses are irrelevant.
