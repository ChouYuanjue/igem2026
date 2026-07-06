# iGEM Enzyme Pocket Exploration

This repository is an early exploration workspace for iGEM enzyme retrieval work.
It is intended to stay modular: the current first direction is pocket hypothesis
exploration for EnzymeCAGE, and later directions may include reaction center
analysis, candidate retrieval, mechanism checks, and failure analysis.

## Current Focus

EnzymeCAGE uses an enzyme structure and a reaction to predict a catalytic
compatibility score. On the enzyme side, its inference route depends on a
predicted catalytic pocket rather than the whole protein structure. This is a
good modeling choice for reducing irrelevant structural context, but it creates
one important question:

**How robust is EnzymeCAGE enzyme retrieval to different pocket hypotheses?**

The first phase only performs inference-time intervention. We do not train
EnzymeCAGE and we do not modify the EnzymeCAGE source tree. Instead, we compare
pocket sources, pocket selection strategies, and pocket-level to enzyme-level
score aggregation outside the external repository.

## Repository Rules

- `external_repos/` is read-only dependency space.
- Do not edit code inside `external_repos/EnzymeCAGE` or any other external
  repository.
- Our code belongs in `explorations/`, `scripts/`, `docs/`, or `tests` under an
  exploration directory.
- Intermediate data belongs in `data/`.
- Experiment outputs belong in `results/`.
- Documentation belongs in `docs/` or an exploration-specific `notes/` folder.
- Every experiment must have a config file and produce a `run_summary.json`.
- Every executed command should be logged for reproducibility.
- Large datasets, model weights, databases, and generated artifacts should not
  be committed to git.

## Pocket Direction

EnzymeCAGE uses a predicted pocket as the local structural entry point for
enzyme-reaction interaction. The default route effectively uses one pocket
hypothesis, usually from AlphaFill or P2Rank top-1 fallback behavior. We test
whether alternative hypotheses change enzyme ranking:

- official or P2Rank top-1 pocket baseline
- P2Rank top-k pocket hypotheses
- enzyme-level aggregation from multiple pocket-level scores
- later extensions such as fpocket, ScanNet residue priors, and
  catalytic-residue-aware reranking

## Quick Start

Clone external repositories:

```bash
bash scripts/clone_external_repos.sh
```

Prepare a local lightweight environment for this exploration code:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

EnzymeCAGE itself may require its own environment. Keep that setup outside our
source edits and follow the upstream repository instructions.

Run the demo P2Rank top-1 runner:

```bash
python explorations/pocket/runners/run_compare_baselines.py \
  --experiment_config explorations/pocket/configs/demo_p2rank_top1.yaml
```

Run the demo P2Rank top-k runner:

```bash
python explorations/pocket/runners/run_compare_baselines.py \
  --experiment_config explorations/pocket/configs/demo_p2rank_topk.yaml
```

Or run both demo baselines:

```bash
bash scripts/run_pocket_baselines.sh
```

The first implementation is intentionally conservative. If EnzymeCAGE script
arguments cannot be inferred safely, the runner records a clear TODO in
`run_summary.json` instead of guessing.
