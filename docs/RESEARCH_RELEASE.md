# Research release contract

This repository is a scientific release, not a mirror of the server workspace.
The authoritative machine-readable release inventory is
`reproducibility/research_release_manifest.json`.

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

## Validation tiers

A clean clone should pass the portable release gate without downloading multi-GB
foundation models or running inference:

```bash
python scripts/maintenance/validate_research_release.py --portable-only
python scripts/verify_terpene_runtime.py --portable-only
python scripts/maintenance/validate_bime_judge_report.py
python scripts/maintenance/resolve_bime_asset.py --verify
```

The GitHub workflow `.github/workflows/terpene-ci.yml` runs these checks and the frozen
44-test release regression suite. It intentionally does **not** run every exploratory
research test in the repository, because those tests may require GPUs, external
benchmarks, or non-portable development assets.

On a fully provisioned research server, omit `--portable-only` to additionally validate
locally restored external and rebuildable assets.

## Reproduction boundary

No training, inference, benchmark rerun, or test-label-based model selection is performed
as part of release packaging. Release preparation only freezes existing project outputs,
tracks the required assets, records reconstruction contracts, and checks their integrity.

Private/local candidate libraries, downloads, historical experiment payloads, ad-hoc
reports, and machine-specific caches are never part of the Git release. They may remain
on a developer machine without being deleted.
