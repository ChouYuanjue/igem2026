# BRIDGE

**BRIDGE — Broad Retrieval with Inference-Driven Gated Experts** is the current enzyme–reaction retrieval method in this repository.

BRIDGE keeps global retrieval and specialist evidence separate. Broad Retrieval produces the stable base order over the open candidate universe. Optional experts are then admitted only when their evidence is available, their domain is applicable to the current query, and their direction-specific ranking permission has been validated. Expert corrections are bounded, and missing evidence is neutral.

## Core design

```text
Open candidate universe
        |
        v
Broad Retrieval
        |
        v
Stable base ranking
        |
        +------------------------------+
        |                              |
        v                              v
Query applicability               Evidence availability
        |                              |
        +--------------+---------------+
                       |
                       v
                  Gated experts
                       |
                       v
                Bounded corrections
                       |
                       v
                  Final ranking
```

The conceptual scoring rule is:

```text
final_score = broad_score + sum(applicability_gate_k * bounded_correction_k)
```

The formula is a design abstraction. Individual production routes may use rank-based, residual, anchored, or shortlist-specific implementations as long as they obey the same authority contract: Broad owns the default order and optional experts receive only validated local correction rights.

## Why BRIDGE exists

The method grew through a long engineering sequence:

```text
TPS candidate screening
    -> candidate pool + CAGE
    -> open candidate expansion
    -> Broad Retrieval
    -> multi-expert BiME-Rank
    -> expert applicability becomes query-dependent
    -> BRIDGE
```

Two historical loops are important:

- **TPS** began as the whole task and now returns as a sparsely activated domain specialist.
- **CAGE** began as a major pool-internal structural ranker, failed as a universal expert, and returned as family-specific structural expertise where local evidence supports it.

BiME-Rank remains the direct frozen predecessor. FIBRE is a retired research branch whose unified interaction-geometry / relational-core hypothesis was tested and rejected as the universal ranking core.

## Repository structure

- `core/` — candidate universes, routing contracts, applicability, provenance, and production invariants.
- `runtime/` — Broad ranking, BiME-derived production baselines, expert runtime interfaces, and route-specific scoring.
- `evidence/` — structural, mechanistic, contextual, and family evidence adapters.
- `application/` — full-information application assets and TPS specialization.
- `pipelines/` — data and evidence construction.
- `portable/` — portable feature/data reconstruction tools.
- `release/` — isolated method, reproduction, and application release contracts.
- `docs/` — method, Engineering history, evaluation, status, and reproducibility documentation.

## Documentation

- `docs/method.md` — current BRIDGE method narrative.
- `docs/engineering.md` — complete engineering decision tree, including rejected branches.
- `docs/evaluation.md` — current evaluation boundaries and headline results.
- `docs/status.md` — current implementation status.
- `docs/reproducibility.md` — relationship between current BRIDGE claims and immutable historical evidence.

## Historical boundaries

The retired FIBRE implementation is archived at:

`archive/fibre/20261003/`

Frozen BiME-Rank and late-stage experimental evidence remain under:

`reproducibility/bime_rank/`

Some immutable result records still use `FIBRE_*` filenames. They keep their original names and hashes for provenance. Current BRIDGE claim names map to those records through `reproducibility/bridge/canonical.json`.

## COMPASS

The conversational agent built on BRIDGE is **COMPASS — Conversational Orchestration for Molecular Pathway and Enzyme Search System**.

COMPASS lives under the historical runtime paths `scripts/starase_navigator/` and `frontend/starase_navigator/` for deployment compatibility, but the current user-facing agent name is COMPASS.

## Validation

```bash
PYTHONPATH=. .venv/bin/python -m compileall -q projects/active/bridge scripts
PYTHONPATH=. .venv/bin/python -m pytest -q projects/active/bridge/tests
PYTHONPATH=. .venv/bin/python scripts/maintenance/validate_bridge_release_profiles.py --source-only
```
