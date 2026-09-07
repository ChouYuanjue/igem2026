# iGEM 2026 — BiME-Rank Research Release

This repository is the scientific release for **BiME-Rank (Bidirectional Multi-Expert Learning-to-Rank)** and its iGEM 2026 enzyme-discovery application. BiME-Rank is the retrieval/model layer; **Catalyst Finder** is the user-facing research service built around that layer, database evidence, and downstream scientific tools.

The repository is not a mirror of the development server. Git contains the project-owned code and learned weights, canonical evidence, public database material, and reproducibility contracts required for the release. Large derived matrices are rebuilt from pinned inputs, while multi-GB third-party foundation checkpoints are restored from fixed upstream releases. Private laboratory candidate libraries and historical workspace payloads are not part of the public Git release.

## Start here

Use these files instead of choosing results by filename, modification time, or a directory called `current`/`production`:

| Question | Authoritative entry |
| --- | --- |
| What is deployed? | `configs/production_routes/terpene_v1.yaml` |
| Which result is canonical for each claim? | `reproducibility/bime_rank/canonical.json` |
| How are release assets classified? | `reproducibility/bime_rank/README.md` |
| What is the public scientific story? | `projects/active/terpene_screening/README.md` |
| What may be committed/rebuilt/downloaded? | `reproducibility/research_release_manifest.json` and `docs/RESEARCH_RELEASE.md` |
| What numbers may appear in judge-facing material? | `projects/active/terpene_screening/BIME_RANK_RETRIEVAL_CAPABILITY_SCORECARD_V2.json` |
| Judge-facing source | `docs/release/bime_rank/` |
| Historical cleanup/audit record | `docs/BIME_ASSET_AUDIT.md` |

## Scientific system

BiME-Rank retrieves in both directions:

- **R2E — reaction → enzyme** over the general protein universe;
- **E2R — enzyme → reaction** over the general reaction universe.

The production system combines frozen learned-to-rank base routes with experts that are admitted on clean development evidence. Structural CLIPZyme signals are availability-aware: when the required representation is absent, the exact frozen non-structural fallback is preserved. When verified positive examples are supplied, a separate frozen context stage reranks only after the zero-shot BiME-Rank order has been formed and masks all supplied seed IDs. Cost-aware execution changes when expensive features are materialized, not the scientific ranking contract.

The canonical general universe contains **185,918 proteins** and **11,081 reactions**. TPS-specific pools and benchmark-specific common supports remain separate evaluation scopes and must not be compared as if they had the same denominator.

## Repository layout

```text
projects/active/terpene_screening/   BiME-Rank runtime, training/evaluation code, protocols, tests
configs/production_routes/           Frozen production routing contracts
reproducibility/bime_rank/           Canonical claim graph, asset roles, provenance, archive records
reproducibility/                      Research-release and runtime manifests
data/                                 Deny-by-default; explicit public release assets are force-tracked
results/                              Deny-by-default; explicit weights/evidence are force-tracked
scripts/catalyst_finder/              Catalyst Finder service/tooling
frontend/catalyst_finder/             User-facing interface
docs/release/bime_rank/               Judge-facing TeX/Bib source validated against canonical evidence
docs/archive/                         Historical documents; never current numeric authority
```

See `docs/README.md` for the documentation map and `docs/project_structure.md` for the directory contract.

## Reproduce and validate

Create the project environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

A clean clone should pass the portable release gate without multi-GB third-party models or model inference:

```bash
python scripts/maintenance/build_research_release_manifest.py
git diff --exit-code -- reproducibility/research_release_manifest.json
python scripts/maintenance/validate_research_release.py --portable-only
python scripts/verify_terpene_runtime.py --portable-only
python scripts/maintenance/resolve_bime_asset.py --verify
python scripts/maintenance/validate_bime_judge_report.py
```

For database reconstruction, third-party model pins, and full-server validation, read `docs/RESEARCH_RELEASE.md`.

## Git policy

- `master` is the research-release branch.
- `data/` and `results/` are deny-by-default, not universally forbidden. Only files explicitly admitted by the release contracts are versioned.
- Project-owned learned weights required by the reported system are committed when they fit ordinary Git.
- Large derived feature databases use deterministic builders plus pinned manifests/input hashes.
- Large third-party checkpoints are referenced by upstream version/checksum rather than vendored.
- Private candidate libraries, downloads, local caches, exploratory runs, and historical payloads may remain on a development machine but are not part of the Git release.
- Legacy names such as `Catalyst`, `terpene`, `V3`, or `V4` may remain inside compatibility/runtime paths. They are not alternative public method identities.
