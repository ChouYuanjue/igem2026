# BRIDGE Engineering Atlas

Public route:

`https://nju-igem.runnelzhang.com/engineering/`

## Presentation model

The page is a **hierarchical Design–Build–Test–Learn atlas**. It does not treat every experiment or every engineering loop as an equal-sized timeline step.

The current macro structure is intentionally weighted by the final scientific story:

- **A — bounded ranking → open retrieval**: two compact early loops establish the candidate-coverage problem and Broad Retrieval.
- **B — Broad stabilization**: one medium loop protects the strong open-world base while testing generalization, external transfer and anti-forgetting strategies.
- **C — expertization → BiME-Rank**: three same-level loops develop functional evidence, structural/mechanistic evidence and expert fusion/routing before converging into BiME-Rank.
- **D — FIBRE detour**: a narrower nested subprocess containing relational formulation, scientific-evidence admission and relational-core replacement experiments. It visibly returns to Broad rather than becoming an equal final pillar.
- **E — BRIDGE formation**: the dominant final stage. Query permissions, family-specific CAGE, TPS specialization and bounded integration develop in parallel and converge into BRIDGE.

Canonical history remains complete: the generator validates that every lineage record belongs to a macro stage or a parallel application track.

## Cycle visual

Each important engineering loop uses a four-part radial process diagram:

- Design is the upper arc;
- Build is the right arc;
- Test is the lower arc;
- Learn is the left arc.

The D/B/T/L letters sit directly on the matching colored arc. The short phase explanation sits beside that same quadrant. The center contains only the compact `DBTL` mark; long explanatory text is never placed inside the ring.

The loop outcome appears below the cycle title, while exact evidence and canonical records are available through the cycle detail action.

## Full record

`Full record` is organized for reading rather than lookup:

1. stage;
2. engineering loop;
3. Design / Build / Test / Learn;
4. key evidence attached to each phase;
5. lower-priority experiments grouped under `Additional experiments`;
6. records shared across multiple loops appear under `Shared evidence & supporting work`.

This keeps all failed and local experiments without presenting the user with a flat family-sorted list.

## Current BRIDGE architecture

After the engineering atlas, the page switches from history to current composition:

`Broad Retrieval → query applicability / permission → optional expert field → bounded pair-evidence correction`

The ranking rule is rendered with native MathML in the page rather than as a plain text formula.

## Source of truth

Historical inventory:

`scripts/engineering_lineage/lineage_data.py`

Presentation generator:

`scripts/engineering_lineage/build_lineage.py`

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 scripts/engineering_lineage/build_lineage.py
```

Generated outputs:

- `frontend/engineering_lineage/engineering/data.js`
- `projects/active/bridge/docs/engineering.md`

Current schema:

`bridge-engineering-atlas-v7`
