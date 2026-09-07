# BiME-Rank: bidirectional enzyme–reaction retrieval

**BiME-Rank** (Bidirectional Multi-Expert Learning-to-Rank) is the current retrieval system used by the iGEM 2026 Catalyst project. This directory contains the model/runtime implementation, benchmark protocols, admission tests, and application code. The user-facing Catalyst Finder service is a separate interface layer; it should not be confused with the retrieval model itself.

For machine-readable truth, use `reproducibility/bime_rank/canonical.json`. `CURRENT_RETRIEVAL_STATUS.md` is retained only as a compatibility redirect for older links. Historical filenames containing `current`, `production`, `V3`, or `V4` do not determine the active method.

## Production contract

The only production routing authority is:

`configs/production_routes/terpene_v1.yaml` (`bime-rank-production-routes-v2`)

The current system has four scientific layers:

1. **Bidirectional zero-shot base ranking.** R2E uses the frozen clean LambdaRank route; E2R uses the frozen Anchored LambdaMART route. These are the stable non-structural fallbacks.
2. **Availability-aware structural evidence.** A CLIPZyme-derived structural expert is fused only where its registered representation exists. Missing structural input means “expert unavailable”, never “low score”, and the exact frozen base route is preserved.
3. **Known-positive context.** If one or more verified positive enzymes/reactions are provided, the frozen context ranker acts after the zero-shot BiME-Rank order is formed. All supplied seeds are masked. Multi-seed use reuses the same frozen one-seed-trained rule; it is not a separately tuned model.
4. **Cost-aware execution.** Cached candidate-generating experts may score their full supported universe. Expensive uncached specialists are materialized only on a frozen shortlist. This is an execution policy and does not redefine expert admission or benchmark semantics.

Rejected homology, reciprocal-consistency, and EnzymeCAGE-Top20 expert variants remain in the evidence graph as negative results; they are not production experts.

## Candidate universes

The canonical general universe is `general-merged-v2`:

- **185,918 proteins** for R2E;
- **11,081 reactions** for E2R;
- **246,610 recorded associations** in the released general database.

`data/catalyst_candidate_universes/general_merged/manifest.json` is the candidate-universe authority. Benchmark-specific common supports (for example CLIPZyme) and the Enzyme-405 augmented universe are separate evaluation objects. TPS-specialized pools are application scopes. Percentages from different universes must not be compared as if they shared a denominator.

## Canonical evidence hierarchy

The release intentionally separates scientific questions instead of collapsing them into one headline score:

- **Strict external zero-shot generalization:** bidirectional CLIPZyme same-support temporal/double-cold evaluations.
- **Independent strong-baseline check:** Enzyme-405, where BiME-Rank and EnzymeCAGE are treated as the same statistical tier unless the paired interval supports a stronger statement.
- **Conditional known-positive retrieval:** one-seed retention and nested 1/2/3/5-seed scaling. These are not zero-shot metrics.
- **Historical precursor/application evidence:** TPS practical/strict Catalyst routes and the frozen Catalyst-V3 Selenzyme comparison remain available for provenance, but they are superseded/supplemental and are **not current BiME-Rank canonical claims**.
- **Expert admission and negative evidence:** promoted and rejected experts are recorded together so a rejected experiment cannot silently re-enter production.
- **Wet-lab decision package:** candidate recommendations and construct plans demonstrate how retrieval feeds experimental planning. They are predictions/plans, not biochemical activity measurements.
- **Execution evidence:** shortlist-retention and runtime measurements support the cost-aware execution layer, not a new ranking algorithm.

Every one of these claims resolves to an explicit primary in `reproducibility/bime_rank/canonical.json`. Judge-facing numeric presentation is locked by `BIME_RANK_RETRIEVAL_CAPABILITY_SCORECARD_V2.json` and checked by `scripts/maintenance/validate_bime_judge_report.py`.

## Current model assets

Base R2E components:

- `results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1`
- `results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1`
- `results/catalyst_clean_mainline_v1/r2e_lambdarank_fusion_v1`

Base E2R components:

- `results/catalyst_clean_mainline_v1/e2r_anchored_lambdamart_v3`

BiME-Rank expert/context heads:

- `results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/selected`
- `results/unified_safe_system_v1/e2r_clipzyme_anchored_lambdamart_v4_dev/selected`
- `results/bime_rank_unified_v1/r2e_seed_context_v1/selected`
- `results/bime_rank_unified_v1/e2r_seed_context_v1/selected`

These path names preserve development provenance. Publicly they form one BiME-Rank production system; they are not eight competing “current models”.

## Runtime entrypoints

Programmatic code should use `projects.active.terpene_screening.core.engine.RetrievalEngine` rather than constructing model paths manually. The CLI is `rank_open_world.py`:

```bash
# Reaction -> enzyme
PYTHONPATH=. .venv/bin/python projects/active/terpene_screening/rank_open_world.py \
  rank-enzymes --reaction-id RHEA:10340 --top-k 10

# Enzyme -> reaction
PYTHONPATH=. .venv/bin/python projects/active/terpene_screening/rank_open_world.py \
  rank-reactions --enzyme-id Q4FNE4 --top-k 10
```

## Selection and evaluation boundary

Expert admission is decided on clean development evidence. Frozen external labels may confirm or veto a candidate expert, but they may not select hyperparameters or retune it. Unsupported expert inputs must reproduce the frozen fallback exactly. Task-specific taxonomy, motif, expression, inventory, and wet-lab constraints remain downstream decision logic rather than universal retrieval features.

The canonical evidence graph also records rejected and superseded experiments. “Superseded” means “not a current claim source”; it does not automatically mean the file is disposable, because an evaluator may still depend on it for reproducibility.

The source tree follows the same rule. `reproducibility/bime_rank/source_roles.json` records the actual runtime import closure separately from canonical/rebuild evaluators, extended reproduction tests, and historical research source. `reproducibility/bime_rank/canonical_source_provenance.json` additionally states whether each canonical claim has a direct final generator, only retained upstream/component source, or an exact source snapshot. Lineage-only development tests are recorded in `reproducibility/bime_rank/historical_source_demotions.json` and do not ship in the public release clone. Internal names such as `dual_kernel`, `unified_safe`, `V3`, or `V4` therefore describe implementation lineage rather than public method identity.

## Validation

For the research-release regression boundary:

```bash
.venv/bin/python scripts/maintenance/validate_research_release.py --portable-only
.venv/bin/python scripts/maintenance/resolve_bime_asset.py --verify
.venv/bin/python scripts/maintenance/validate_bime_judge_report.py
.venv/bin/python -m pytest -q scripts/maintenance/tests/test_bime_asset_resolver.py
.venv/bin/python scripts/maintenance/run_bime_project_tests.py --tier release
```

For a provisioned research checkout, the retained extended reproduction tests can be run without discovering server-local lineage files:

```bash
.venv/bin/python scripts/maintenance/run_bime_project_tests.py --tier extended
```

Only tests listed by `source_roles.json` participate. Lineage-only tests may remain physically present on a development server but cannot silently re-enter the current quality gate.
