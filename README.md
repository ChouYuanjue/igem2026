# Starase Atlas software — FIBRE and Starase Navigator

This repository contains the software component of the **NJU-China iGEM 2026 Starase Atlas** project. The current scientific method is **FIBRE**, a bidirectional enzyme–reaction correspondence method, and the user-facing product is **Starase Navigator**, a scientific agent and web interface built on top of FIBRE.

For iGEM judging, the authoritative source copy must live in the team's official repository on **iGEM GitLab** (`gitlab.igem.org`). Other development mirrors are not the evaluation authority. This project therefore ships a deterministic source-only exporter that creates a clean Git repository below the 50 MiB award limit.

## What this software does

FIBRE represents reactions and enzyme molecular states as two factor geometries and uses known positive enzyme–reaction associations as a correspondence on their product space. Reaction → enzyme and enzyme → reaction are two sections of the same scientific object. Missing molecular observations remain missing rather than being treated as negative evidence.

Starase Navigator turns that method into an application. A user can ask about a reaction, enzyme, candidate search, evidence, pathway or related scientific question in natural language. The agent resolves the requested entity and intent, calls deterministic scientific tools, and returns ranked hypotheses separately from database-recorded evidence.

The **application profile** intentionally uses more information than the frozen benchmark profile. For the current terpene-synthase application domain it uses all accepted canonical positive associations, the promoted global molecular geometry, catalytic-pocket and mechanistic resolutions, provenance-bound evidence, and an application-only TPS-domain coordinate learned from the historical pair-supervised TPS model. That TPS coordinate refines candidates only *within* a primary FIBRE numerical level; it does not replace the global correspondence or become benchmark evidence.

## Who this is for

This software is intended for:

- iGEM teams and researchers who want to search enzyme ↔ reaction hypotheses in Starase Navigator;
- computational biology / enzyme-engineering users who want an auditable application workflow;
- researchers who want to reproduce the reported FIBRE/Starase computational results; and
- future teams who want to rebuild the FIBRE method on another enzyme–reaction database.

No knowledge of the repository's historical experiments is required to use the current method or application.

## Repository map

| Need | Start here |
| --- | --- |
| User-facing Starase Navigator | `scripts/starase_navigator/` and `frontend/starase_navigator/` |
| Current FIBRE implementation | `projects/active/fibre/` |
| Current method description | `projects/active/fibre/docs/method.md` |
| Portable new-dataset workflow | `workflow/Snakefile` + `config/README.md` |
| Release profiles | `projects/active/fibre/release/profiles/` |
| Frozen result reproduction | `reproducibility/bime_rank/` + FIBRE evaluation authorities |
| Dependency lock | `requirements.lock.txt` |
| iGEM submission validator | `scripts/maintenance/validate_igem_submission.py` |
| <50 MiB GitLab source exporter | `scripts/maintenance/build_igem_submission.py` |

Historical experimental lineage is not part of the current method/runtime authority.

## Install

The tested runtime uses **Python 3.12**. The committed dependency contract is `requirements.lock.txt`. PyTorch is platform-specific, and DRFP 0.3.6 has stale upstream metadata referring to `rdkit-pypi`, so the supported installer handles those two cases explicitly.

```bash
git clone <THE-OFFICIAL-IGEM-GITLAB-CLONE-URL>
cd <repository>
python3.12 -m venv .venv
./scripts/bootstrap_terpene_runtime.sh
```

For a CPU-only machine, installing the pinned CPU PyTorch wheel first is recommended:

```bash
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.0
./scripts/bootstrap_terpene_runtime.sh
```

The bootstrap verifies the local runtime after installation. Large scientific data, model weights and generated results are deliberately not stored in the official Git history; see **Large assets and the 50 MiB rule** below.

## Run Starase Navigator

After the required application assets have been restored or built:

```bash
# Build the full-information Starase application references.
PYTHONPATH=. .venv/bin/python -m projects.active.fibre.application.build_full_data \
  --output-dir results/fibre_application/full_data

# Start the web/API service.
./scripts/starase_navigator/manage.sh start

# Inspect the runtime and verify application_profile.status == "ready".
curl -fsS http://127.0.0.1:8791/api/status | .venv/bin/python -m json.tool
```

The normal user-facing route is the **Starase application profile** for a query already inside the current application domain. Broad/general queries remain available and use the broad candidate universe instead of silently forcing a TPS-only scope.

To stop the service:

```bash
./scripts/starase_navigator/manage.sh stop
```

## Run FIBRE on a new database

The portable method package does not depend on the Starase database or project-trained TPS weights. Prepare tables following `config/README.md`, copy `config/fibre.example.yaml`, then run:

```bash
PYTHONPATH=. .venv/bin/python -m projects.active.fibre.portable.build_plan \
  --config config/fibre.example.yaml

PYTHONPATH=. .venv/bin/python -m snakemake \
  --snakefile workflow/Snakefile \
  --configfile config/fibre.example.yaml \
  --cores 1
```

A small clean-room fixture under `.test/` demonstrates the complete build without any project database.

## Reproduce the main results

Reproduction and application are intentionally isolated.

**FIBRE Reproduction Bundle** freezes the exact claim-specific splits, candidate supports, evaluators, model assets and result authorities used for reported numbers. It never reads the full-data application coordinate.

In the full research workspace, the validator is strict: every frozen claim authority and every project-owned reproduction asset must be present and hash-correct.

```bash
PYTHONPATH=. .venv/bin/python scripts/maintenance/validate_fibre_release_profiles.py
PYTHONPATH=. .venv/bin/python scripts/maintenance/run_reproduction_tests.py --tier release
PYTHONPATH=. .venv/bin/python scripts/maintenance/run_reproduction_tests.py --tier extended
```

The official source-only iGEM repository intentionally excludes large weights/results. There the same validator automatically enters `source_only_contract` mode (or can be invoked explicitly with `--source-only`): it validates package isolation, exact asset hashes/sizes and the externalization boundary, and reports `full_reproduction_assets_materialized: false` until those assets have been restored. This is not presented as a completed metric reproduction.

The current FIBRE result authorities include the frozen broad product-flow result, the terpene double-cold correspondence result, its strict-inductive audit, and the current partial-relation audits. Exact authority paths are machine-readable in:

```text
projects/active/fibre/release/profiles/reproduction.yaml
projects/active/fibre/release/manifests/fibre-reproduction.json
reproducibility/research_release_manifest.json
reproducibility/bime_rank/model_assets.json
```

To run the portable clean-room reproduction test:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  projects/active/fibre/tests/test_portable.py \
  projects/active/fibre/tests/test_portable_reaction_center.py

rm -rf .test/output
PYTHONPATH=. .venv/bin/python -m snakemake \
  --snakefile workflow/Snakefile \
  --configfile .test/config.yaml \
  --cores 1
```

## Three release profiles

The same research workspace serves three different purposes, but their inputs and claims are not interchangeable.

| Profile | Purpose | Pair-supervised project weights | Benchmark claims |
| --- | --- | --- | --- |
| **FIBRE Method Kit** (`fibre-method`) | Transfer FIBRE to a new database | forbidden | no |
| **FIBRE Reproduction Bundle** (`fibre-reproduction`) | Reproduce frozen reported metrics | only when required by that frozen claim | yes, frozen |
| **Starase Application Bundle** (`starase-application`) | Best-information user-facing deployment | allowed | forbidden |

`projects/active/fibre/release/asset_catalog.yaml` is only a content-addressed asset registry; it is not a fourth software package.

## Large assets and the 50 MiB rule

The iGEM software repository must remain below **50 MiB**, so large datasets, checkpoints, embeddings and generated result matrices do not belong in the official Git history. The source repository contains the software needed to build/use them plus immutable paths, hashes, provenance and restoration contracts.

Project-owned and third-party model/data inventories are recorded in the release manifests. In particular, `reproducibility/research_release_manifest.json` describes the full research workspace and may report large historical/direct scientific assets; its `direct_git_bytes` field is **not** the size of the source-only iGEM submission repository. Project-owned reproduction assets remain hash/size-addressed in `reproducibility/bime_rank/model_assets.json`; a public asset location can be attached later without changing the source/reproduction identity. Until those externalized assets are restored, the source-only repository validates the release contract but does not claim that frozen benchmark metrics have been reproduced there.

The source tree can be staged as a clean official-repository checkout with:

```bash
PYTHONPATH=. .venv/bin/python scripts/maintenance/build_igem_submission.py \
  --output dist/igem-gitlab
```

The command initializes a fresh Git history and fails if either the clean checkout or Git history reaches 50 MiB, or if a source file larger than 5 MiB has accidentally crossed the asset boundary.

The current large research workspace itself is **not** the iGEM submission history. Do not push its historical `.git` directory to the official iGEM repository.

## iGEM GitLab publication

Before the award freeze, the clean tree at `dist/igem-gitlab` must be pushed to the team's official 2026 iGEM GitLab repository. Authentication and the team-specific clone URL are intentionally not hard-coded into the software.

The submission contract can be checked locally with:

```bash
PYTHONPATH=. .venv/bin/python scripts/maintenance/validate_igem_submission.py
```

When the checkout's Git remote points to `gitlab.igem.org`, this validator also enforces that its Git history is below 50 MiB.

## Validation

Useful release checks are:

```bash
# Current scientific tests
PYTHONPATH=. .venv/bin/python -m pytest -q projects/active/fibre/tests

# Starase Navigator product tests
PYTHONPATH=. .venv/bin/python -m pytest -q scripts/starase_navigator/tests

# Release/profile boundary
PYTHONPATH=. .venv/bin/python scripts/maintenance/validate_fibre_release_profiles.py

# iGEM checklist
PYTHONPATH=. .venv/bin/python scripts/maintenance/validate_igem_submission.py

# Git whitespace/hygiene
git diff --check
```

The official source repository also includes `.gitlab-ci.yml`, which runs the portable tests and submission-size gate in GitLab CI.

## License

Project source code is released under the **MIT License**. See `LICENSE`.

Third-party software, pretrained models and datasets remain under their respective upstream licenses. See `THIRD_PARTY_NOTICES.md` and the machine-readable asset manifests for exact provenance.

## Citation

Citation metadata is provided in `CITATION.cff`. Starase Atlas is the wider NJU-China iGEM 2026 project; FIBRE and Starase Navigator are its software/method components.
