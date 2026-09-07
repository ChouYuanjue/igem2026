# Research-release project structure

The Git repository is organized around a public scientific release, not around the full server workspace. Files under ignored roots may still exist locally; their presence does not make them part of the publication.

## Top-level contract

| Path | Scientific role | Git policy |
| --- | --- | --- |
| `projects/active/terpene_screening/` | BiME-Rank model/runtime, protocols, evaluators, application logic, tests | Track production/reproducibility source; historical source may remain only when required by evidence dependencies |
| `configs/production_routes/` | Current runtime routing contract | Track and hash-lock |
| `reproducibility/bime_rank/` | Canonical claim graph, roles, provenance, archive/audit indexes | Track; this is the asset-selection authority |
| `reproducibility/` | Publication/runtime manifests and golden route fixtures | Track |
| `scripts/catalyst_finder/` | Catalyst Finder user-facing service/tool harness | Track production source/tests |
| `scripts/maintenance/` | Release/asset validators and safe archive tooling | Track |
| `data/` | Canonical/public tables plus local/derived data | Deny-by-default; force-track only explicit release assets |
| `results/` | Project weights, canonical evidence, local experiment output | Deny-by-default; force-track only explicit release assets |
| `docs/release/bime_rank/` | Judge-facing source | Track; numeric claims must validate against canonical evidence |
| `docs/archive/` | Historical human-readable records | Track only when useful for audit; never current authority |
| `external_repos/` | Third-party reference repositories | Read-only; do not vendor large upstream worktrees |

## Scientific layers

The public release should be read in this order:

1. **Method:** BiME-Rank bidirectional multi-expert retrieval (`projects/active/terpene_screening/README.md`).
2. **Runtime:** one production routing contract (`configs/production_routes/terpene_v1.yaml`).
3. **Database:** one canonical general candidate universe plus explicitly scoped benchmark/application universes.
4. **Evidence:** claim-specific canonical primaries (`reproducibility/bime_rank/canonical.json`).
5. **Application:** wet-lab candidate/construct planning and Catalyst Finder integration.
6. **Reproduction:** directly committed assets, rebuildable matrices, external checkpoints, and validation gates (`docs/RESEARCH_RELEASE.md`).

Historical development names are provenance, not public architecture. `Catalyst clean mainline`, `V3`, `V4`, old TPS bundles, and similar names may remain as dependency paths when changing them would break hashes or runtime compatibility.

## Data/results rule

`data/` and `results/` are deliberately mixed local roots on the development server, so Git uses an explicit publication whitelist rather than tracking whole directories. The machine-readable whitelist is `reproducibility/research_release_manifest.json`.

A release asset must be one of:

- project-owned learned weights needed by the reported system;
- canonical public tables/manifests;
- canonical claim evidence;
- compact derived features whose inclusion materially improves reproducibility;
- deterministic builders/protocols/tests needed to recover larger assets.

Private local candidate libraries, downloads, caches, ad-hoc experiment runs, and historical archive payloads remain outside Git.

## Source and test release boundary

`reproducibility/bime_rank/source_roles.json` is the machine-readable source map. Current runtime source, canonical/rebuild source, portable regression tests, and extended reproduction tests remain in the release. Research-lineage source may remain tracked when useful for method provenance, but lineage-only development tests are excluded from public Git and audited in `reproducibility/bime_rank/historical_source_demotions.json`. Local historical files are preserved on the development server and are not discovered by current quality gates.

Historical machine-readable protocol/result files are not kept in `projects/active/` merely because they once looked like a mainline artifact. When a top-level legacy artifact has no current dependency or reverse reference, it is removed from the Git active surface and hash-audited in `reproducibility/bime_rank/historical_artifact_demotions.json`; the development-server copy remains untouched.

A legacy asset may still remain listed in `reproducibility/terpene_runtime_manifest.json` after it leaves normal Git. That manifest is a compatibility/provenance contract, not a declaration that every historical TPS asset is part of the current BiME-Rank scientific release. Such intentional exceptions are audited in `historical_runtime_asset_demotions.json`; current production model assets are never eligible for this demotion.

## Naming rules

- Public method identity is **BiME-Rank**, not an internal expert count or V-number.
- `master` is the release branch.
- Prefer relative repository paths in current configs/docs.
- Do not infer currentness from path names or timestamps.
- Do not rename frozen historical/model assets solely for aesthetics when the path is part of a hash/provenance/runtime contract; present them through the asset map instead.
