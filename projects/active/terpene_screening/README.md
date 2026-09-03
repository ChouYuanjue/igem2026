# Catalyst bidirectional enzyme–reaction retrieval

This directory contains the current bidirectional retrieval system used by the iGEM 2026 Catalyst project. For live metrics and project boundaries, read **[`CURRENT_RETRIEVAL_STATUS.md`](CURRENT_RETRIEVAL_STATUS.md)** first. Historical experiment files are retained only when they still support an audit or reproducibility contract; their names do not imply that they are active routes.

## Current production surface

Production routing is defined only by `configs/production_routes/terpene_v1.yaml` (`terpene-production-routes-v5`).

- **R2E — reaction → enzyme:** eligible external `general_merged` queries use the confirmed two-source LambdaRank route (`cfg_07_392fe119`).
- **E2R — enzyme → reaction:** eligible registered external `general_merged` auto queries use Anchored LambdaMART V3.
- Current entities, few-shot requests, masks, candidate subsets, temporary candidates, manual overrides and scopes outside the confirmed learned-route contracts retain their existing fallback behavior.

The production API boundary is `projects.active.terpene_screening.core.engine.RetrievalEngine`; the CLI implementation is `rank_open_world.py`.

```bash
# Reaction -> enzyme
PYTHONPATH=. .venv/bin/python projects/active/terpene_screening/rank_open_world.py \
  rank-enzymes --reaction-id RHEA:10340 --top-k 10

# Enzyme -> reaction
PYTHONPATH=. .venv/bin/python projects/active/terpene_screening/rank_open_world.py \
  rank-reactions --enzyme-id Q4FNE4 --top-k 10
```

For programmatic server use, prefer `RetrievalEngine` rather than constructing model paths manually. Model/path overrides are intentionally not part of the normal public request surface.

## Candidate universes

The normal general candidate universe is `general_merged`:

- proteins: 185,918
- reactions: 11,081

A TPS-specialized universe still exists for explicitly specialized workflows, but it is a separate task scope. Scores from different candidate universes must not be compared as if they had the same denominator.

## Current external evaluation

The canonical policy is `CATALYST_EXTERNAL_EVALUATION_POLICY_V2.json`; the current result is `CATALYST_EXTERNAL_EVALUATION_V2_RESULT.json`.

The underlying external benchmark is the full **Rhea release128→141 Swiss-Prot strict double-cold v2** snapshot. Competition-facing main tables use the documented **best-of-8 budgeted presentation subset**; all eight seed results are retained and the full snapshot, when already computed, is a sanity-check/appendix result. The primary seed is explicitly selected after scoring for presentation only and is never used for model selection.

The full Rhea snapshot Relative to the exact current clean2023 production training source, protein, reaction and exact-pair overlap are all zero. All 1,122 test associations are retained. Evaluation reuses registered feature libraries and packaged production models; it performs no benchmark-specific representation training or large new encoding pass.

Do not infer current-production novelty from historical benchmark names. In particular, old `broad_rhea_fair_benchmarks_v1` labels such as `double_cold` refer to their own historical train partitions. Their current-clean2023 overlap is recorded in `CATALYST_LEGACY_BROAD_RHEA_CURRENT_TRAIN_AUDIT_V1.json`.

## Evaluation rules

A benchmark may be a primary broad-generalization result only after overlap is recomputed against the **exact training source of the model being evaluated**. The current admission thresholds are:

- query entity unseen fraction ≥ 30%
- positive target entity unseen fraction ≥ 30%
- exact positive pair unseen fraction ≥ 90%
- query-positive coverage ≥ 90%

When a benchmark passes, use the complete benchmark-defined dataset/cell after deterministic input-validity mapping. Do not choose a support subset from observed performance.

Asset reuse order is fixed:

1. existing IDs, cached features, frozen model scores or author scores;
2. deterministic ID/canonicalization alignment;
3. existing evaluator with path/config substitution;
4. small new encoding only when no admitted existing benchmark answers the same question;
5. large external-library encoding or benchmark-specific learned adapters are disabled by default.

Reproducing the strongest published baseline is **not** a blocking requirement. A readily reproducible same-task baseline, author score, or absolute external metric is sufficient when the alternative would create substantial new infrastructure. Direct model-vs-baseline deltas still require identical support and metric semantics.

## Active model artifacts

R2E:

- `results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1`
- `results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1`
- `results/catalyst_clean_mainline_v1/r2e_lambdarank_fusion_v1`

E2R:

- `results/catalyst_clean_mainline_v1/e2r_anchored_lambdamart_v3`

Machine-readable current model status: `CATALYST_CLEAN_MAINLINE_V1.json`.

## Development boundary

There are only two active retrieval model lines: **R2E LambdaRank** and **E2R Anchored LambdaMART V3**. Old residual searches, HPO sweeps, domain-adaptation branches, Top-K surrogate experiments, CAGE fusion attempts and benchmark-specific adapters are not alternative active mainlines.

External/revealed labels may describe a frozen system but may not select or retune it. If a new model family is ever needed, selection must use separate development evidence and an untouched confirmation source.

## Tests

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q projects/active/terpene_screening/tests
```

The full test suite is the preferred regression boundary after route/config changes.
