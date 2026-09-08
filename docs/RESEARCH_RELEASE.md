# Research release contract

The publication branch is **`master`**. BiME-Rank is the scientific retrieval/model identity; Catalyst Finder is the user-facing service that consumes it. Internal `Catalyst`, `terpene`, and V-numbered paths are retained only where they encode runtime/provenance compatibility.

This repository is a scientific release, not a mirror of the server workspace.
The authoritative machine-readable release inventory is
`reproducibility/research_release_manifest.json`. Claim-by-claim numeric authority is separate and lives in `reproducibility/bime_rank/canonical.json`; this prevents asset packaging decisions from silently changing scientific claims.

For the human asset-class map, read `reproducibility/bime_rank/README.md`.

## What is committed

The release deliberately commits the project-owned material needed to inspect,
validate, and reproduce the reported system:

- self-trained production checkpoints and frozen ranking heads used by the current
  BiME-Rank route;
- the earlier TPS production bundles still referenced by the current route;
- canonical result summaries for every claim in
  `reproducibility/bime_rank/canonical.json`;
- public/canonical protein sequences, reaction tables, association tables, candidate
  ordering metadata, feature schemas, calibration assets, and evaluation summaries;
- lightweight feature stores that fit ordinary Git and materially reduce the cost of
  reproduction;
- deterministic builders, protocol contracts, source/version hashes, tests, and the
  judge-facing TeX release source.

`data/` and `results/` remain ignored *by default*. Release files below those roots are
tracked explicitly from the manifest. This prevents a developer's arbitrary local run
from becoming part of the publication merely because it lives beside a released asset.

## What is rebuilt instead of committed

Large derived arrays are not normal Git blobs when doing so would make the repository
fragile or exceed GitHub's single-object limit. Their exact role, builder, expected
shape/metadata, and input contracts are recorded under `rebuildable_assets` in the
release manifest.

The main examples are:

1. **General ESM-C protein matrix** (`185,918 x 1,152`). Rebuild from the committed
   `general_merged/protein_sequences.tsv` with
   `extract_esmc_embeddings.py`. The observed ESM-C Hugging Face snapshot is pinned in
   the release manifest.
2. **General EnzGFM protein matrix** (`185,918 x 2,048`). Rebuild with
   `build_enzgfm_protein_features.py` from the same committed sequence registry and the
   fixed EnzGFM-650M encoder. The historical exact union contract remains in the tracked
   feature manifest and can be reconstructed with `merge_protein_feature_libraries.py`.
3. **DRFP / RDKit+ / reaction-center matrices.** Rebuild from the committed reaction
   table, the committed TPS reference feature assets, and the tracked deterministic
   feature builders.
4. **CLIPZyme R2E candidate matrix** (`166,207 x 1,280`). Rebuild with the frozen source
   snapshot `reproducibility/bime_rank/source_snapshots/build_clipzyme_r2e_candidate_asset.py`.
   The project-native extension embeddings needed by that merge are committed; the
   official released CLIPZyme screening asset remains an upstream third-party asset.

These are derived feature databases, not learned project checkpoints. The model weights
that determine the project ranking heads are committed whenever they fit ordinary Git.

## External foundation models

Large third-party checkpoints are not vendored into project Git. They are pinned by
upstream repository/record and checksum in `reproducibility/research_release_manifest.json`:

- Horizyn v1.0 development checkpoint: Dayhoff Labs repository commit plus Zenodo DOI
  and SHA-256;
- CLIPZyme released checkpoint and screening data: official repository commit, Zenodo
  record, and project-verified checksums;
- EnzGFM-650M: official DeepBxM repository commit, Zenodo record, archive MD5, and local
  encoder SHA-256.

This distinction is intentional: a 2--3 GB third-party foundation checkpoint is an
external dependency, while our trained retrieval heads are a publication artifact and
are therefore committed.

## External benchmark data

The current CLIPZyme strict-temporal comparisons share a frozen Rhea release128→141 v2 support construction. The original protocol `CLEANROOM_R2E_RHEA128_TO141_EXTERNAL_V2.json` and its original builder are preserved byte-for-byte; release packaging does not rewrite a frozen scientific protocol. The official Rhea release128 and release141 archives are pinned by extracted `rhea2uniprot_sprot.tsv` byte sizes/SHA-256 values. A separate deterministic release assembler, `rebuild_rhea128_to141_strict_support_v2.py`, reuses the original mapping/selection implementation but constructs the old compact-alignment witness from the already tracked clean2023 boundary. Thus the historical `rhea_2023_compact.csv.gz` cache is not a hard dependency of the portable release.

A release-time deterministic audit reproduced the frozen 1,122-pair / 208-reaction upstream support with `test_pairs.csv` SHA-256 `9a53a465e6327e2c04a4fdd6171abd7d076aec2a3441a34955bf0f4526bc3334` and byte-identical output. The small frozen split is also shipped as an evaluation-support artifact for exact replay; it is not a numeric authority. Final CLIP fair-support evaluators apply their additional mutual-train-cold and model-input support projection (144 R2E reaction queries / 166,202 proteins; 248 E2R protein queries / 10,131 reactions).

## Database reconstruction

The canonical candidate universe is `general-merged-v2`:

- 185,918 protein entities;
- 11,081 reaction entities;
- 246,610 recorded associations.

The release commits the final sequence/reaction/association tables and their manifest,
so database reconstruction does not depend on an undocumented server directory. The
builder `projects/active/terpene_screening/build_general_candidate_universe.py` also
records how the universe was originally assembled and deduplicated from upstream layers.
Feature databases are rebuilt from these canonical tables, rather than redefining the
candidate universe during reproduction.

For exact provenance, use the SHA-256 values in the general-universe, EnzGFM, CLIPZyme,
and research-release manifests. Do not infer provenance from a directory called
`current`, `production`, or from modification time.

## Database release boundary

The release distributes the **canonical database tables themselves**: 185,918 protein sequences/metadata entries, 11,081 reactions, and 246,610 protein–reaction associations, with exact row counts and SHA-256 values in `reproducibility/bime_rank/database_assets.json`. These tracked tables are the portable database authority. Large ESM-C/EnzGFM/reaction feature matrices are model-ready derivatives rebuilt from the canonical tables with the declared builders and pinned model/runtime inputs.

`build_general_candidate_universe.py` remains the provenance recipe for the original multi-source assembly. Its manifest records 13 exact source-file hashes; some historical public-derived intermediate sources are intentionally not vendored. Exact replay of the *original assembly process* therefore requires reacquiring those hash-matched intermediates, but use and reproduction of the released canonical database do not. This distinction avoids turning historical caches into hidden release dependencies.

Reaction-center features additionally require deterministic atom correspondence. The 11,081-row RXNMapper registry is small enough to ship directly as a fixed preprocessing asset. `reproducibility/bime_rank/rxnmapper_general_merged_v1.json` records its canonical input hashes, 10,839 successful mappings, 242 explicit failures, RXNMapper version/configuration, and historical model hashes.

## Model asset boundary

`reproducibility/bime_rank/model_assets.json` is generated from the canonical production route. It distinguishes project-owned learned parameters and compact runtime support arrays, which must be Git-tracked and hash-locked, from large third-party foundation checkpoints, which remain external and are restored by exact repository/version/hash contracts. A locally protected historical checkpoint is not a current release model merely because it still exists on a development server.

The model index is regenerated in CI together with the release and source-role manifests. This makes route changes fail closed: adding or changing a production model bundle without committing the required project asset or updating the external contract produces release drift.

## Claim source provenance

`reproducibility/bime_rank/canonical_source_provenance.json` separates three questions that older project directories tended to conflate: whether a result is frozen and hash-verifiable, whether its underlying runtime/model/component source is retained, and which exact generator/evaluator/runtime contract replays the final current claim.

All **current** canonical claims now fail closed on `missing_final_generator`: the release validator rejects any current claim whose final replay boundary is still missing. Direct evaluators/builders are retained where available; formerly ignored one-off evaluators/finalizers are preserved as exact source snapshots; reviewed top-level aggregates (`expert_admission` and `cost_aware`) now have deterministic assemblers over hash-locked component evidence. Superseded historical experiments may still have weaker provenance, but they cannot be promoted into the current canonical set merely because their result file exists.

The compact component JSON/summary files needed by the two canonical assemblers are declared as `aggregate_support_assets` in the research-release manifest. They are shipped directly because they are small and scientifically meaningful; large private candidate libraries and transient caches are not pulled into Git merely to support an aggregate.

## Validation tiers

A clean clone should pass the portable release gate without downloading multi-GB
foundation models or running inference:

```bash
python scripts/maintenance/validate_research_release.py --portable-only
python scripts/verify_terpene_runtime.py --portable-only
python scripts/maintenance/validate_bime_judge_report.py
python scripts/maintenance/resolve_bime_asset.py --verify
```

The GitHub workflow `.github/workflows/terpene-ci.yml` runs these checks and the source-role-defined portable regression suite. Project test membership is derived from `reproducibility/bime_rank/source_roles.json` rather than discovered by globbing the physical server directory. In a portable clone, the single test that opens the full general ESM-C/DRFP matrices is explicitly skipped until those rebuildable matrices are provisioned; on a fully provisioned server the full-asset-only checks execute as well.

A second **extended reproduction** tier retains tests that directly import the current runtime or canonical/rebuild source; run it with `python scripts/maintenance/run_bime_project_tests.py --tier extended`. Tests that cover only retired research branches are recorded with exact hashes in `reproducibility/bime_rank/historical_source_demotions.json` and removed from public Git. Their development-server copies are preserved in place, but they cannot be rediscovered accidentally by the current quality gate.

Historical source is pruned more conservatively than tests. `reproducibility/bime_rank/historical_research_source_demotions.json` contains only legacy auxiliary scripts that passed all three release audits: no tracked references/importers, no overlap with canonical/release asset paths, and no builder/preparer/trainer/serve/export/rank/download/validate/audit/wet-lab role. Standalone scientific tooling is intentionally retained even when it has no current caller. Demotion is Git-index-only; the audited development-server files remain in place and are hash-checked when present.

Top-level historical machine artifacts use the same non-destructive rule. `reproducibility/bime_rank/historical_artifact_demotions.json` records legacy protocol/result files that were still sitting under `projects/active/terpene_screening/` but had no canonical dependency, direct-release role, current source-role membership, or tracked reverse reference. They are removed from the public active Git surface without being deleted from the development server; Git history plus the audit remains the public historical record.

Legacy runtime compatibility assets are handled separately. `reproducibility/terpene_runtime_manifest.json` remains the frozen full-server compatibility/provenance contract even when a legacy asset is no longer vendored in normal Git. `reproducibility/bime_rank/historical_runtime_asset_demotions.json` records such cases with exact hashes. Portable runtime verification skips these untracked legacy entries, while a fully provisioned server still verifies the preserved local copies. A runtime-demoted learned weight is forbidden from being a current production model asset.

On a fully provisioned research server, omit `--portable-only` to additionally validate
locally restored external and rebuildable assets.

## CI validation artifacts

A successful `master` run of `.github/workflows/terpene-ci.yml` uploads `bime-rank-release-validation-<commit>`. The bundle contains the canonical/research-release/runtime/model/database/source-role manifests, judge validation metadata, all four Git-only demotion audits, the verified EnzGFM timing provenance, RXNMapper preprocessing provenance, Git commit, Python version, resolved dependencies, `CITATION.cff`, and `THIRD_PARTY_NOTICES.md`. It also includes `release-status.json`, a compact machine-readable summary of claim closure, asset counts, source counts, and project-license status. It is deliberately small: project weights and canonical databases stay at their authoritative repository paths, while multi-GB third-party models remain external by checksum contract.

## Reproduction boundary

No model training, benchmark rerun, test-label-based model selection, or scientific retuning is performed as part of release packaging. One narrowly scoped deterministic reproducibility rerun was performed after audit discovered that the old `35.82 s` EnzGFM materialization number had no independent timing log: the same 530-protein EnzGFM-650M feature materialization was rerun on the recorded RTX 4090 environment in `37.33 s`, and the regenerated `(530, 2048)` embedding file was byte-identical to the original. `reproducibility/bime_rank/enzgfm_stage2_530_timing_20260907.json` records the command, hardware, environment and hashes. This is execution/timing evidence, not a new model-quality benchmark.

Private/local candidate libraries, downloads, historical experiment payloads, ad-hoc reports, and machine-specific caches are never part of the Git release. They may remain on a developer machine without being deleted.
