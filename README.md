# Starase Atlas software — FIBRE and Starase Navigator

This repository contains the software component of the **NJU-China iGEM 2026 Starase Atlas** project. The current scientific method is **FIBRE — Factorized Interaction Basis for Reaction–Enzyme**. The user-facing product is **Starase Navigator**.

## Scientific idea

Enzyme discovery has an asymmetric data problem. Molecular information is relatively abundant: a protein can be represented from sequence and, when available, structure, pocket and family context; a reaction can be represented from substrate/product chemistry, fingerprints and reaction-centre information. Reliable enzyme–reaction pairs are much scarcer.

FIBRE starts from that asymmetry. It builds an **interaction atlas**: reactions are described through catalytic-demand coordinates, enzymes through catalytic-capability coordinates, and different molecular views define local interaction charts. Sparse verified pairs teach the chart-specific interaction forms and how overlapping charts are glued into one global catalytic interaction.

This gives the system three practical properties:

- an unseen reaction and unseen enzyme remain scoreable from raw molecular inputs;
- optional structure/pocket/family information can improve a prediction without determining whether a prediction exists;
- a new verified wet-lab positive can enter as a bounded finite-rank interaction update without retraining the global encoders.

The historical product-manifold/correspondence geometry experiments are preserved for reproducibility, but they are no longer the definition of FIBRE.

## Multi-view and family-specialized charts

The broad raw-input chart supplies universal coverage. Reaction-centre, structure, pocket, family and verified-context experts are local charts of the same interaction and can use different latent dimensions.

Terpene-synthase specialization is a biochemical family chart. Its applicability is determined by TPS-relevant molecular state rather than membership in a particular dataset, so the specialization is compatible with broad candidate universes.

## Evidence and uncertainty

Starase separates prediction from evidence. Results can expose contributing model views, source-linked known activities, sequence/structure/pocket/mechanistic observations, assay context and provenance. Missing optional information remains missing rather than becoming a negative.

A probability-like confidence is reported only when a matching calibration authority exists. Otherwise the system reports model support, view agreement, coverage and provenance directly.

## Repository map

| Need | Start here |
| --- | --- |
| FIBRE method | projects/active/fibre/docs/method.md |
| Mathematical basis | projects/active/fibre/docs/catalytic_kernel_foundation.md |
| Interaction primitives | projects/active/fibre/kernel/ |
| Current FIBRE implementation | projects/active/fibre/ |
| Starase Navigator | scripts/starase_navigator/ and frontend/starase_navigator/ |
| Frozen method/benchmark reproduction | reproducibility/bime_rank/ |
| Release profiles | projects/active/fibre/release/profiles/ |
| Production route contract | configs/production_routes/default.yaml |
| Dependency lock | requirements.lock.txt |

## Install

The tested runtime uses Python 3.12.

    git clone <THE-OFFICIAL-IGEM-GITLAB-CLONE-URL>
    cd <repository>
    python3.12 -m venv .venv
    ./scripts/bootstrap_terpene_runtime.sh

Large datasets, checkpoints, embeddings and generated result matrices are deliberately kept outside the official small Git history and are restored through the release manifests.

## Run Starase Navigator

    ./scripts/starase_navigator/manage.sh start
    curl -fsS http://127.0.0.1:8791/api/status | .venv/bin/python -m json.tool

Natural language is the primary control surface. The semantic planner chooses broad versus supported family-specialized search and the appropriate evidence depth; users do not choose internal model IDs.

## Reproduce FIBRE results

Research reproduction and application are intentionally isolated.

**FIBRE Reproduction Bundle** freezes claim-specific data snapshots, splits, candidate supports, model assets, scripts and evaluators. Only this profile may support benchmark-performance claims.

    PYTHONPATH=. .venv/bin/python scripts/maintenance/validate_fibre_release_profiles.py
    PYTHONPATH=. .venv/bin/python scripts/maintenance/run_reproduction_tests.py --tier release

A previous global bilinear-correction experiment is retained as a negative development record and is not part of the current interaction-atlas architecture.

**Starase Application Bundle** may use all accepted current molecular information, verified positive pairs, provenance-bound live evidence and TPS specialization. It is optimized for current use, not benchmark claims.

## Three release profiles

| Profile | Purpose | Project/full-data specialization | Benchmark claims |
| --- | --- | --- | --- |
| **FIBRE Method Kit** | interaction model, update rule and scientific contracts | forbidden | no |
| **FIBRE Reproduction Bundle** | frozen reported evaluation | only as frozen by the claim | yes |
| **Starase Application Bundle** | best-information deployment | allowed | forbidden |

## Validation

    PYTHONPATH=. .venv/bin/python -m pytest -q projects/active/fibre/tests
    PYTHONPATH=. .venv/bin/python -m pytest -q scripts/starase_navigator/tests
    PYTHONPATH=. .venv/bin/python scripts/maintenance/validate_fibre_release_profiles.py
    git diff --check

## License and citation

Project source code is released under the MIT License. Third-party software, pretrained models and datasets retain their upstream licenses; see THIRD_PARTY_NOTICES.md.

Citation metadata is in CITATION.cff. Starase Atlas is the wider NJU-China iGEM 2026 project; FIBRE and Starase Navigator are its method/software components.
