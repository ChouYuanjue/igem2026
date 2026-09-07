# Documentation map

The repository separates current scientific/release documentation from historical or local development notes. Do not infer authority from a filename containing `current`, `final`, a date, or a version number.

## Current release documentation

- `RESEARCH_RELEASE.md` — publication asset boundary, rebuildable matrices, third-party model pins, and validation tiers.
- `project_structure.md` — current Git/repository structure.
- `BIME_ASSET_AUDIT.md` — audit trail explaining how canonical and historical assets were separated.
- `../reproducibility/bime_rank/source_roles.json` — machine-readable runtime/reproduction/extended-test/historical-source classification.
- `../reproducibility/bime_rank/canonical_source_provenance.json` — exact source/replay boundary for each canonical claim, including explicit finalization gaps.
- `../reproducibility/bime_rank/historical_source_demotions.json` — integrity audit for lineage-only tests retained locally but removed from public Git.
- `../reproducibility/bime_rank/historical_research_source_demotions.json` — integrity audit for strictly isolated legacy research auxiliaries removed from public Git.
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

Planning documents, run-tracking requirements, old frontend plans, stage summaries, and superseded generated reports may remain on a developer/server workspace but are excluded from Git when they are not part of a current reproducibility contract. Exact ignored paths are documented in `.gitignore`.
