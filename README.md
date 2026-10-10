# BRIDGE + COMPASS

This repository contains the modeling and agent stack developed by **NJU-China iGEM 2026** for enzyme–reaction discovery and pathway design.

- **BRIDGE** — **Broad Retrieval with Inference-Driven Gated Experts** — is the current enzyme–reaction retrieval method.
- **COMPASS** — **Conversational Orchestration for Molecular Pathway and Enzyme Search System** — is the conversational research agent built on top of BRIDGE and the project evidence stack.

## Overview

BRIDGE separates universal retrieval from specialist reasoning.

1. **Broad Retrieval** provides a stable ranking over the open candidate universe.
2. **Inference-Driven Gating** decides which experts are applicable to the current query and direction.
3. **Bounded Expert Corrections** allow validated specialists to refine the ranking without overwriting the Broad base order.
4. Missing or inapplicable expert evidence is treated as neutral rather than negative.

For registered enzyme-to-reaction discovery with known associations excluded, the production ranker uses documented catalyst-neighbor evidence and reaction-neighbor prototypes over the first 1,000 Broad candidates. The candidate universe remains all 11,081 registered reactions.

A compact form of the ranking rule is:

```text
final_score = broad_score + sum(applicability_gate_k * bounded_correction_k)
```

## Shared evaluation protocol

The current broad retrieval experiment uses **218,537 training relations**, **5,216 validation relations**, and **21,505 shared R2E/E2R test relations**. R2E candidates comprise 185,918 proteins; E2R candidates comprise 11,081 reactions. Full MRR, Hit@K, novelty-stratified Balanced metrics, and four expert ablations are in the current evaluation report at projects/active/bridge/docs/evaluation.md.

The canonical evaluation reports results on the shared 21,505 enzyme-reaction relations, using one 5,216-relation validation cohort and complete directional candidate universes.

## COMPASS

COMPASS is the user-facing scientific agent. It translates natural-language research questions into bounded retrieval, evidence, pathway, and compatibility operations while keeping verified evidence separate from model hypotheses.

COMPASS can:

- retrieve enzyme candidates for a reaction;
- retrieve reaction candidates for an enzyme;
- inspect verified Rhea and UniProt evidence;
- use literature, structure, assay, mechanism, and family-specific evidence when available;
- design and compare molecular pathways;
- evaluate whole-pathway enzyme compatibility;
- explain which search and evidence steps were used for a request.

Internal model choices are not exposed as product switches. COMPASS selects the search scope, BRIDGE route, expert depth, and evidence operations from the scientific task.

## Repository map

| Area | Path |
| --- | --- |
| BRIDGE overview | `projects/active/bridge/README.md` |
| Current method | `projects/active/bridge/docs/method.md` |
| Current evaluation | `projects/active/bridge/docs/evaluation.md` |
| Uniform 21,505-relation main tables | projects/active/bridge/docs/evaluation_tables_v3.md |
| Current implementation status | `projects/active/bridge/docs/status.md` |
| Reproducibility rules | `projects/active/bridge/docs/reproducibility.md` |
| Current BRIDGE claim map | `reproducibility/bridge/canonical.json` |
| COMPASS backend | `scripts/starase_navigator/` |
| COMPASS frontend | `frontend/starase_navigator/` |
| Production routing | `configs/production_routes/default.yaml` |

## Installation

The tested environment uses Python 3.12.

```bash
git clone <THE-OFFICIAL-IGEM-GITLAB-CLONE-URL>
cd <repository>
python3.12 -m venv .venv
./scripts/bootstrap_terpene_runtime.sh
```

Large databases, foundation models, embeddings, and derived matrices are restored through pinned release manifests instead of being treated as ordinary Git source files.

## Run COMPASS

```bash
./scripts/starase_navigator/manage.sh start
curl -fsS http://127.0.0.1:8791/api/status | .venv/bin/python -m json.tool
```

The runtime directory still uses the historical `starase_navigator` path for deployment compatibility. The current product name is COMPASS.

## Run BRIDGE directly

Reaction to enzyme:

```bash
PYTHONPATH=. .venv/bin/python projects/active/bridge/runtime/cli.py \
  rank-enzymes --reaction-id RHEA:54512 --top-k 10 \
  --output /tmp/bridge-r2e.csv
```

Enzyme to reaction:

```bash
PYTHONPATH=. .venv/bin/python projects/active/bridge/runtime/cli.py \
  rank-reactions --enzyme-id 7S5L_A --top-k 20 \
  --output /tmp/bridge-e2r.csv
```

## Release and reproducibility

The current release boundary has three roles:

| Package | Role | May support benchmark claims |
| --- | --- | --- |
| `bridge-method` | Method definition, routing, and expert contracts | No |
| `bridge-reproduction` | Frozen data, weights, evaluation, and claim evidence | Yes |
| `starase-application` | COMPASS application bundle and full-information research workflow | No |

Validation:

```bash
PYTHONPATH=. .venv/bin/python scripts/maintenance/validate_bridge_release_profiles.py --source-only
PYTHONPATH=. .venv/bin/python scripts/maintenance/build_bridge_release_manifests.py
PYTHONPATH=. .venv/bin/python -m pytest -q projects/active/bridge/tests
```

BiME-Rank frozen regression remains available:

```bash
.venv/bin/python reproducibility/bime_rank/run_reproduction_tests.py --tier release
.venv/bin/python reproducibility/bime_rank/run_reproduction_tests.py --tier extended
```

Some immutable result files still use historical `FIBRE_*` filenames because filenames, schemas, and hashes are part of experiment provenance. Current BRIDGE claims refer to those records through `reproducibility/bridge/canonical.json` without rewriting the original evidence.

## Historical boundaries

- **BiME-Rank** is the direct predecessor and remains a frozen comparison/reproduction line.
- **FIBRE** is a retired research branch. Its interaction-atlas, conditional-mode, relational-core, and related implementation history is archived under `archive/fibre/20261003/`.
- Historical names that remain in immutable result files or compatibility schemas do not define the current method identity.

## iGEM software access

**What this software does:** BRIDGE ranks open enzyme–reaction candidates using Broad embeddings, gated biochemical evidence, and documented catalytic-neighbor relationships; COMPASS provides the conversational research interface.

**Who this is for:** Synthetic biology teams prioritizing enzymes and reaction candidates for experimental validation.

**Install:** Follow the Python 3.12 setup in Installation and restore the pinned external scientific assets.

**Run the software:** Start COMPASS with the command under Run COMPASS, or invoke the BRIDGE CLI under Run BRIDGE directly.

**Reproduce the main results:** Use the frozen association graph and externalized release assets with the records in reproducibility/bime_rank/records and the validation commands under Release and reproducibility.

**iGEM GitLab:** The team's 2026 software submission is maintained as a separately initialized BRIDGE–COMPASS release repository, with its own `main` history, source-only CI, immutable scientific records, and external SHA-256 asset locks. The old in-tree `dist/igem-gitlab` staging workflow is retired and must not be used. The independent release is staged on nju-server-06; publication requires the team's dedicated official software GitLab remote, which is not yet configured.

## License and citation

Project source code is released under the MIT License. Third-party models, datasets, and software remain subject to their upstream licenses. See `THIRD_PARTY_NOTICES.md` and `CITATION.cff` for details.
