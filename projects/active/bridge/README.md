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

## Biochemical evidence

The ranking system combines EnzGFM functional compatibility, CLIPZyme structural evidence, reaction-center mechanisms, documented clean2023 catalytic relations, and query-applicable family and TPS experts. E2R also uses 16 documented catalyst neighbors and eight reaction-neighbor prototypes.

The single shared validation corpus contains **5,216 verified enzyme–reaction associations**. R2E and E2R are evaluated on the same **21,505 training-unseen associations**. The R2E relation authority and EnzGFM scaling are fitted on the shared validation corpus; runtime few-shot support authority is estimated using grouped positive-support episodes from that same corpus. See the current evaluation report at docs/evaluation.md for exact denominators, ablations, and limitations.

## Repository structure

- `core/` — candidate universes, routing contracts, applicability, provenance, and production invariants.
- `runtime/` — Broad ranking, expert runtime interfaces, and route-specific scoring.
- `evidence/` — structural, mechanistic, contextual, and family evidence adapters.
- `application/` — full-information application assets and TPS specialization.
- `pipelines/` — data and evidence construction.
- `portable/` — portable feature/data reconstruction tools.
- `release/` — isolated method, reproduction, and application release contracts.
- `docs/` — method, evaluation, status, and reproducibility documentation.

## Documentation

- `docs/method.md` — current BRIDGE method narrative.
- `docs/evaluation.md` — current evaluation boundaries and headline results.
- `docs/status.md` — current implementation status.
- `docs/reproducibility.md` — relationship between current BRIDGE claims and immutable historical evidence.

## COMPASS

The conversational agent built on BRIDGE is **COMPASS — Conversational Orchestration for Molecular Pathway and Enzyme Search System**.

COMPASS lives under the runtime paths `scripts/starase_navigator/` and `frontend/starase_navigator/` for deployment compatibility, but the current user-facing agent name is COMPASS.

## Validation

```bash
PYTHONPATH=. .venv/bin/python -m compileall -q projects/active/bridge scripts
PYTHONPATH=. .venv/bin/python -m pytest -q projects/active/bridge/tests
PYTHONPATH=. .venv/bin/python scripts/maintenance/validate_bridge_release_profiles.py --source-only
```
