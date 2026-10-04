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

A compact form of the ranking rule is:

```text
final_score = broad_score + sum(applicability_gate_k * bounded_correction_k)
```

The current method grew through the following engineering line:

```text
TPS candidate screening
    -> CAGE-centered ranking
    -> open candidate retrieval
    -> Broad Retrieval
    -> BiME-Rank multi-expert organization
    -> query-level expert applicability
    -> BRIDGE
```

TPS and CAGE both remain useful, but their roles changed. TPS became a sparsely activated domain specialist. CAGE moved from an early primary ranker to family-specific structural expertise. BiME-Rank is preserved as the direct predecessor and frozen baseline. The retired FIBRE line is archived as engineering history.

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

## Engineering history

The complete decision tree is documented in:

`projects/active/bridge/docs/engineering.md`

It includes the successful route to BRIDGE and the major rejected or redirected branches, including candidate-pool expansion, TPS-specific biochemical features, external pretrained-model transfer, graph methods, continual-learning/model-merging methods, reaction-center residuals, BiME-Rank variants, and the full FIBRE detour.

## Repository map

| Area | Path |
| --- | --- |
| BRIDGE overview | `projects/active/bridge/README.md` |
| Current method | `projects/active/bridge/docs/method.md` |
| Complete engineering tree | `projects/active/bridge/docs/engineering.md` |
| Current evaluation | `projects/active/bridge/docs/evaluation.md` |
| Current implementation status | `projects/active/bridge/docs/status.md` |
| Reproducibility rules | `projects/active/bridge/docs/reproducibility.md` |
| Current BRIDGE claim map | `reproducibility/bridge/canonical.json` |
| Frozen BiME-Rank predecessor | `reproducibility/bime_rank/` |
| Retired FIBRE archive | `archive/fibre/20261003/` |
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

## License and citation

Project source code is released under the MIT License. Third-party models, datasets, and software remain subject to their upstream licenses. See `THIRD_PARTY_NOTICES.md` and `CITATION.cff` for details.
