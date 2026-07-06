# Pocket Baseline Prompt

Implement the first phase of an iGEM enzyme retrieval exploration repository.
The current direction is pocket hypothesis exploration for EnzymeCAGE.

Key rules:

- `external_repos/` is read-only dependency space.
- Do not modify EnzymeCAGE or any external repository.
- Put adapters, runners, analysis code, tests, and docs in this repository.
- Store intermediate data in `data/` and results in `results/`.
- Every experiment must have a config, command log, copied config, and
  `run_summary.json`.
- If EnzymeCAGE script arguments are uncertain, write TODOs and clear warnings
  instead of guessing.

First-phase implementation:

- repository skeleton
- pocket manifest schema
- P2Rank top-k adapter scaffold
- aggregation from pocket-level scores to enzyme-level scores
- top-k evaluation
- rank-shift comparison
- conservative runner for top-1 and top-k baselines
- lightweight tests that do not require EnzymeCAGE or P2Rank

Research question:

How robust is EnzymeCAGE enzyme retrieval to different pocket hypotheses?
