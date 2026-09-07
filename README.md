> BiME-Rank 当前资产与结果入口：`reproducibility/bime_rank/canonical.json`；评委 release 源：`docs/release/bime_rank/`；审计说明：`docs/BIME_ASSET_AUDIT.md`。旧实验目录、旧 PDF 和旧版本名不能用于自动选择当前结果。

# iGEM 2026 Research Release

This repository is the scientific release for the iGEM 2026 enzyme-retrieval and
terpene-screening work. It contains the source, project-owned model weights,
canonical evidence, reproducibility contracts, and the public database material
needed to reconstruct the reported system.

The release boundary is machine-readable in
`reproducibility/research_release_manifest.json` and explained in
`docs/RESEARCH_RELEASE.md`. Large derived feature matrices and multi-GB third-party
foundation checkpoints are reconstructed from pinned inputs instead of being
vendored as ordinary Git blobs.

The top-level structure separates **project code**, **operational scripts**,
**documentation**, **external dependencies**, **data**, and **results** so that
new directions do not get mixed into one large `explorations/` bucket.

## Repository Layout

```text
projects/
  active/
    terpene_screening/     Terpene synthase screening and production retrieval core.
scripts/
  catalyst_finder/         Current Catalyst service, model-led agent tools, route catalog, and API.
  terpene/                 Terpene screening controllers and status checks.
  setup/                   Dependency, asset, and environment setup.
  maintenance/             Cleanup and repository hygiene scripts.

docs/                      Cross-project documentation, release sources, and prompts.
  release/bime_rank/        Judge-facing BiME-Rank TeX/Bib locked to canonical evidence.
external_repos/            Read-only third-party repositories.
data/                      Local data root; release-whitelisted tables/assets are tracked explicitly.
results/                   Local result root; release-whitelisted weights/evidence are tracked explicitly.
```

See `docs/project_structure.md` for the full directory contract.

## Active Projects


### Catalyst Finder

Catalyst Finder is the current user-facing research service. It combines verified database evidence, bidirectional enzyme/reaction retrieval, literature and structure inspection, route design, and pathway compatibility through a model-led tool harness.

The current product source is isolated under:

```text
frontend/catalyst_finder/
scripts/catalyst_finder/
```

Retired portal/pocket implementations are not part of the tracked production source. Dynamic candidate-universe sizes and the deployed build revision are reported by `GET /api/status`; they are intentionally not duplicated as fixed numbers in this README.

See `frontend/catalyst_finder/README.md` for the current retrieval semantics, evidence sources, bilingual/session boundaries, runtime cache behavior, and deployment/test commands.

### `projects/active/terpene_screening/`

This block builds and evaluates terpene synthase candidate gates, including
reaction-only/few-shot CAGE-style reranking and wet-lab intention evaluation.

Typical entrypoint:

```bash
bash scripts/terpene/run_terpene_gate_matrix.sh
```

## Setup

Clone external repositories:

```bash
bash scripts/setup/clone_external_repos.sh
```

Synchronize the pinned database/frontend design reference separately:

```bash
bash scripts/setup/sync_igem_database_reference.sh
```

The pinned nested repository is read-only. Its exact commit and sparse-checkout
contract are tracked in `reproducibility/external_repos/igem_database.lock.json`.

Prepare a lightweight local environment for this repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For the exact research-release asset policy, database reconstruction commands,
third-party model checksums, and clean-clone validation commands, see
`docs/RESEARCH_RELEASE.md`.

EnzymeCAGE itself may require its own environment. Keep that setup outside our
source edits and follow the upstream repository instructions.

## Repository Rules

- `external_repos/` is read-only dependency/reference space.
- Active project code belongs in `projects/active/<project>/`.
- Shared operational scripts belong in `scripts/<domain>/`.
- Intermediate data belongs in `data/`.
- Experiment outputs belong in `results/`.
- Documentation belongs in `docs/` or a project-specific `notes/` folder.
- `data/` and `results/` are ignored by default, but files explicitly listed in
  `reproducibility/research_release_manifest.json` are part of the publication and
  are force-tracked. Unlisted local runs, caches, downloads, private candidate
  libraries, and generated reports must not be committed.
