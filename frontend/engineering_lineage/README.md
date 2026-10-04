# BRIDGE Engineering Story

Public route:

`https://nju-igem.runnelzhang.com/engineering/`

## Reading model

The Engineering page is deliberately **progressive rather than panoramic**. It does not show the whole project as one giant graph.

The user reads five consecutive scenes. Each scene shows only:

1. the current system stack;
2. one primary Design–Build–Test–Learn loop;
3. a few local secondary loops;
4. the lesson that changes the next design.

This preserves nested and parallel engineering work without forcing every cycle into the first view.

## Story scenes

### 01 — Candidate eligibility

The original bounded candidate system uses EnzymeCAGE / meta-ranking above a similarity-based gate. Pocket, full-library and reaction-transfer experiments reveal that candidate eligibility itself creates a hard recall ceiling.

### 02 — Broad below CAGE

Broad first enters conservatively as the lower retrieval layer while CAGE is still expected to remain the ideal upper reranker. Layered evaluation then exposes a second ceiling: Broad can reach many more positives than a generic CAGE upper layer can natively support. This is the point where ranking responsibility begins moving into Broad itself.

Retention and generalization work remains available here as secondary/supporting engineering, but it is not promoted into an independent era.

### 03 — Broad base order + experts

Once Broad has a meaningful order, functional, structural/mechanistic, contextual and fusion work develops around it. BiME-Rank is the first explicit system for protecting the incumbent Broad ranking while admitting additional capabilities as experts.

### 04 — FIBRE side experiment

FIBRE is shown only in its own scene and explicitly as a replacement hypothesis branching from the working Broad/BiME stack. Its useful interfaces survive; its global replacement claim does not.

### 05 — BRIDGE authority model

The final scene asks which expert may change Broad's order for the current query and direction. Family-specific CAGE, TPS, context and other evidence become gated specialists under query applicability / permission and bounded correction.

## Cycle visual

Only the primary loop in each scene receives a full radial DBTL diagram.

The radial layout uses a fixed 3×3 geometry:

- Design above the ring;
- Build to the right;
- Test below;
- Learn to the left;
- the ring itself remains isolated in the center cell.

This prevents explanatory text from overlapping the cycle. Direct subloops use compact mini-loop markers. Selecting one opens a new DBTL loop in the same dialog; that loop can contain further subloops, so the hierarchy is recursive rather than limited to two levels.

## Full record

`Full record` opens the same recursive loop navigator at the scene root. Each view shows one DBTL loop, its direct child loops, its outcome and `Why next`. A breadcrumb allows moving back up the hierarchy. Raw experiment records are treated as evidence and stay collapsed at container loops; leaf loops expose their evidence by default.

All canonical history remains represented. The builder validates coverage before generating the public data.

## Current architecture

After the five scenes, BRIDGE is shown separately as the current system composition:

`Broad Retrieval → query applicability / permission → optional expert field → bounded pair-evidence correction`

The ranking formula is rendered using native MathML.

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

`bridge-engineering-scenes-v9`
