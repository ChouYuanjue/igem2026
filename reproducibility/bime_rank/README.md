# BiME-Rank asset map

This directory separates **current scientific authority** from reproducibility support, historical evidence, and local-only workspace state. It exists so neither humans nor scripts have to infer “the latest result” from filenames or mtimes.

## Authority order

1. **Production behavior:** `configs/production_routes/terpene_v1.yaml`.
2. **Claim-by-claim numeric authority:** `canonical.json`.
3. **Reviewed selection graph:** `selection.json`; `canonical.json` is the hash-locked build of that reviewed selection.
4. **Judge presentation contract:** `projects/active/terpene_screening/BIME_RANK_RETRIEVAL_CAPABILITY_SCORECARD_V2.json`.
5. **Judge narrative source:** `docs/release/bime_rank/`; narrative text is subordinate to canonical numeric evidence.
6. **Git/publication asset boundary:** `../research_release_manifest.json`.

Never select a result because a path contains `current`, `production`, `final`, a larger version number, or a recent timestamp. Legacy Catalyst/terpene/V3/V4 names remain in paths only where runtime or provenance compatibility requires them.

## Asset classes

### Current production

`canonical.json` claim `production` points to the active route and its production model dependencies. BiME-Rank is one system composed of stable R2E/E2R base rankers, availability-aware structural experts, optional known-positive context heads, and a cost-aware execution policy.

### Candidate universe and database

`candidate_universe` resolves to `data/catalyst_candidate_universes/general_merged/manifest.json`: 185,918 proteins and 11,081 reactions. The release manifest specifies which canonical tables are committed and which large feature matrices are rebuilt.

### Canonical evaluations

External/general, application-scope, conditional-context, expert-admission, and execution claims each have a distinct canonical ID. This prevents a TPS percentage, a known-positive score, and a general zero-shot score from being presented as the same task.

### Project-owned model and source assets

Learned project checkpoints/ranking heads below the ordinary Git blob limit are part of the research release. Large feature matrices are rebuildable data assets, not omitted model weights.

`source_roles.json` classifies tracked project Python into the current runtime import closure, canonical/rebuild reproduction source, the portable release regression boundary, extended reproduction tests, and historical research source. This is deliberately separate from filenames: a module with an old `V3`, `dual_kernel`, or training-era name may still be a real runtime dependency, while a newer-looking standalone evaluator may be historical. Historical classification does not authorize deleting the server file.

### External foundation and benchmark-data assets

CLIPZyme, EnzGFM, Horizyn, and other large third-party model payloads are not vendored. The Rhea release128/release141 Swiss-Prot association snapshots used to rebuild the current strict-temporal support are handled the same way: exact upstream archives, byte sizes, and SHA-256 values live in `reproducibility/research_release_manifest.json`. The original Rhea v2 protocol and builder remain byte-identical to the frozen historical record. For portable replay, `rebuild_rhea128_to141_strict_support_v2.py` reuses their mapping/selection logic with an alignment witness derived from tracked clean2023 and reproduces the frozen `test_pairs.csv` byte-for-byte without requiring the historical compact2023 cache.

### Wet-lab/application assets

`wetlab_success_first` records the frozen prediction/experiment-planning package. The canonical scope explicitly distinguishes predicted candidates and construct plans from measured activity. Private local laboratory libraries are not part of the public Git release.

### Rejected experts and negative evidence

Rejected experiments remain reproducibility evidence. They are deliberately preserved so an unsuccessful expert cannot be accidentally rediscovered and promoted by filename. Rejected does not mean disposable.

### Historical/superseded evidence

`canonical.json:superseded` maps historical claim sources to their replacement canonical claim. Human-readable historical documents live under `docs/archive/bime_rank/20260907/`. Historical result payloads may be stored under the local ignored archive and are indexed by `archive_moves.json`; the archive operation is move-only.

### Local operational/private state

Live service caches, downloads, private candidate libraries, and local archive payloads may exist on the server but are outside Git publication scope. `roles.json` and the research-release manifest describe these boundaries.

## Core files

- `canonical.json` — current hash-locked claim index.
- `selection.json` — reviewed semantic selection before hash locking.
- `roles.json` — asset-class policy.
- `dependency_graph.json` / `protected_files.json` — reproducibility dependencies that must not be treated as disposable merely because they are not headline results.
- `historical_references.json` — references to older result paths.
- `archive_plan.json`, `archive_moves.json`, `archive_candidates.json` — move-only historical archive audit.
- `validation.json` / `judge_report_validation.json` — deterministic validation records; they are evidence of checks, not new scientific results.
- `source_roles.json` — machine-readable current-runtime / canonical-reproduction / extended-test / historical-source classification.
- `canonical_source_provenance.json` — claim-by-claim record of direct generators, retained upstream source, exact one-off source snapshots, and explicitly missing finalization scripts.
- `model_assets.json` — route-derived inventory of current project-owned learned parameters/runtime support arrays and pinned external foundation models.
- `database_assets.json` — canonical database tables, row/hash locks, model-ready rebuild contracts, and historical multi-source assembly boundary.
- `rxnmapper_general_merged_v1.json` — normalized provenance for the compact mapped-reaction preprocessing asset shipped for reaction-center feature rebuilds.
- `historical_source_demotions.json` — exact hashes for lineage-only tests removed from public Git while preserved on the development server.
- `historical_research_source_demotions.json` — exact hashes for the narrowly audited legacy research auxiliaries removed from public Git; development-server copies remain local.
- `source_snapshots/` — frozen source copies needed to rebuild selected assets without depending on mutable historical scripts.

## Validation

```bash
.venv/bin/python scripts/maintenance/resolve_bime_asset.py --verify
.venv/bin/python scripts/maintenance/manage_bime_archive.py verify
.venv/bin/python scripts/maintenance/validate_bime_assets.py
.venv/bin/python scripts/maintenance/validate_bime_judge_report.py
.venv/bin/python scripts/maintenance/validate_research_release.py --portable-only
```
## Source-level reproducibility is claim-specific

A frozen result hash and a fully replayable command are not the same guarantee. `canonical_source_provenance.json` records that distinction for every canonical claim. Several current primaries have direct retained generators (for example Enzyme-405, multi-seed and seed-retention evaluations, and the candidate-universe builder). The exact CLIPZyme R2E/E2R external-confirmation evaluators and the reciprocal external-confirmation evaluator were recovered from ignored development-server result directories and are preserved byte-for-byte under `source_snapshots/`. Wet-lab finalization is likewise covered by an exact source snapshot. Historical MARTS route confirmations, the TPS legacy_exact Catalyst comparison, and the frozen Catalyst V3 Selenzyme comparison are documented separately and are not current canonical claims.

All 12 current canonical claims now have an explicit final replay boundary and `missing_final_generator=false`. That boundary may be a direct generator, an exact frozen source snapshot, the production runtime contract, or the judge presentation validator/contract. The two reviewed top-level aggregates also have deterministic assemblers and tracked component-support assets. Historical/superseded experiments are allowed to retain weaker provenance, but those gaps cannot silently re-enter the current canonical set.
