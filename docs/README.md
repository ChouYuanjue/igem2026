# Documentation map

The repository separates current scientific/release documentation from historical or local development notes. Do not infer authority from a filename containing `current`, `final`, a date, or a version number.

## Current release documentation

- `RESEARCH_RELEASE.md` — publication asset boundary, rebuildable matrices, third-party model pins, and validation tiers.
- `../CITATION.cff` — repository citation metadata.
- `../THIRD_PARTY_NOTICES.md` — project-license status plus vendored/external third-party license boundaries.
- `project_structure.md` — current Git/repository structure.
- `BIME_ASSET_AUDIT.md` — audit trail explaining how canonical and historical assets were separated.
- `../reproducibility/bime_rank/source_roles.json` — machine-readable runtime/reproduction/extended-test/historical-source classification.
- `../reproducibility/bime_rank/canonical_source_provenance.json` — exact source/replay boundary for each canonical claim, including the final replay boundary for every current canonical claim.
- `../reproducibility/bime_rank/model_assets.json` — production-route-derived current model weights/support arrays versus external checkpoint contracts.
- `../reproducibility/bime_rank/database_assets.json` — portable canonical database tables versus historical assembly provenance and model-ready rebuilds.
- `../reproducibility/bime_rank/historical_source_demotions.json` — integrity audit for lineage-only tests retained locally but removed from public Git.
- `../reproducibility/bime_rank/historical_research_source_demotions.json` — integrity audit for strictly isolated legacy research auxiliaries removed from public Git.
- `../reproducibility/bime_rank/historical_artifact_demotions.json` — integrity audit for isolated legacy machine protocol/result artifacts removed from the active Git surface.
- `../reproducibility/bime_rank/historical_runtime_asset_demotions.json` — exact hashes for legacy runtime-manifest assets kept for full-server compatibility provenance but no longer vendored in normal Git.
- `../reproducibility/bime_rank/enzgfm_stage2_530_timing_20260907.json` — provenance for the one deterministic timing rerun used as current cost-aware execution evidence.
- `TERPENE_REPRODUCIBILITY.md` — retained reproducibility detail for the terpene runtime lineage.
- `data_schema.md` — stable data conventions.
- `external_repos.md` — third-party repository handling.
- `repo_rules.md` — repository hygiene policy.
- `release/bime_rank/` — judge-facing TeX/Bib source; numeric claims are subordinate to canonical evidence.

## Method/protocol records retained for reproducibility

Some older method documents remain tracked because current runtime/evidence contracts or protocol history still reference them, for example:

- `terpene_competition_evidence_layer_20260805_zh.md`;
- `terpene_second_round_conformal_cycle_20260805_zh.md`;
- `terpene_retrieval_protocol_reassessment_zh.md`;
- `terpene_r2e_taxonomy_scope_20260809_zh.md`;
- `terpene_retrieval_scenarios.md`.

They are supporting methodology, not current numeric authority.

## Historical documentation

`archive/bime_rank/20260907/` contains superseded human-readable records preserved for audit/reproduction. The corresponding current claim source is resolved through `reproducibility/bime_rank/canonical.json`.

## Local development notes

Planning documents, run-tracking requirements, old frontend plans, stage summaries, and superseded generated reports may remain on a developer/server workspace but are excluded from Git when they are not part of a current reproducibility contract. Public deny-by-default rules live in `.gitignore`; exact Git-only demotions are recorded by the machine `historical_*_demotions.json` audits. Development-server copies may additionally be hidden with local `.git/info/exclude` entries, which are intentionally not part of the public repository.

## CI validation bundle

Every successful `master` run uploads `bime-rank-release-validation-<commit>`. The compact bundle includes the current manifests, the three historical-demotion audits, timing/RXNMapper provenance, citation/license metadata, and `release-status.json`; it is validation evidence rather than a duplicate model/database artifact tree.
