# Atlas Engineering

Public route:

`https://nju-igem.runnelzhang.com/engineering/`

## Reading model

The Engineering page is a **fixed causal-loop map**, not a free canvas.

The public interaction rules are deliberately strict:

1. there is no pan, zoom, drag or force-directed layout;
2. one hierarchy level is visible at a time;
3. clicking a loop with children replaces the current layer with its direct subloops;
4. breadcrumbs and the Parent control return to higher levels;
5. clicking a leaf loop opens its DBTL detail inline below the same fixed map;
6. raw historical experiment records remain secondary evidence and may open in a simple detail surface.

This keeps the system readable on both desktop and mobile while still allowing a deep recursive engineering history.

## Causal semantics

Every loop is represented as a circular Design–Build–Test–Learn object with four stable phase ports.

The important cross-loop relation is not just “A is related to B”. It is a typed phase handoff, normally:

`Loop A / Learn → Loop B / Design`

These handoffs encode a concrete engineering statement: a learned constraint from one loop changed the design of another loop.

The current generated schema is:

`atlas-engineering-loops-v10`

Its main causal structure lives in:

- `atlas.root` — recursive loop hierarchy;
- `atlas.handoffs` — phase-level causal edges;
- `atlas.systems` — EDGE / BRIDGE / COMPASS authority labels.

The historical `crossLinks` inventory is still preserved as source-history metadata, but the public causal map is driven by `atlas.handoffs`.

## Top-level Atlas view

The overview intentionally contains only seven large loops:

- **EDGE** — build and expose the known biochemical graph;
- **BRIDGE** — rank the open candidate frontier;
- **COMPASS** — maintain verified research state;
- **EDGE × BRIDGE** — dynamic knowledge-boundary evaluation;
- **BRIDGE × COMPASS** — intent-to-ranking contract;
- **EDGE × COMPASS** — canonical evidence orchestration;
- **Atlas Knowledge Frontier** — the final coupled system loop.

The overview uses fixed grid positions. It does not rearrange itself according to physics or user gestures.

## Deep evaluation branch

The EDGE × BRIDGE knowledge-boundary program contains seven sequential DBTL loops:

1. fixed-graph missing-edge recovery;
2. cold-start endpoint splits;
3. new-entity arrival events;
4. official Rhea historical graph difference;
5. continuous release / curation events;
6. directional query × candidate exposure;
7. candidate-cardinality decomposition.

Each Learn phase explicitly triggers the next Design phase.

This branch explains how evaluation moved from static graph completion to real temporal graph growth and why the final deployment protocol keeps the complete candidate universe fixed.

## COMPASS branch

The COMPASS program also contains seven sequential loops:

1. research intent enters retrieval;
2. bounded scientific agent;
3. verified identity before reasoning;
4. reusable research workspace objects;
5. explicit evidence state;
6. observation lifecycle;
7. invariant / metamorphic agent evaluation.

The story is about preserving scientific state under interaction, not about accumulating UI features.

## BRIDGE branch

The existing five-scene BRIDGE engineering spine is embedded unchanged as the BRIDGE program:

1. candidate-system recall ceiling;
2. Broad below CAGE;
3. Broad base order plus experts;
4. FIBRE replacement experiment;
5. BRIDGE local authority model.

Its existing recursive subloops and historical evidence remain available below those five loops.

## Cross-system loops

Cross-system programs are first-class DBTL loops rather than decorative edges.

Examples:

- EDGE knowledge-graph learning and COMPASS evidence-state learning jointly motivate canonical evidence orchestration;
- dynamic graph evaluation learns that candidate cardinality is an independent difficulty axis, which directly changes the BRIDGE × COMPASS retrieval contract;
- BRIDGE learns a query-gated authority model while COMPASS learns that intent should not own scientific execution, producing a clean intent-to-ranking boundary.

## Desktop behavior

Desktop uses a fixed grid.

- circular loop nodes stay in deterministic positions;
- an SVG overlay draws only the phase-level causal handoffs for the currently visible layer;
- no canvas transform is ever applied;
- selecting a deeper loop rerenders the same fixed region rather than zooming the old layer.

## Mobile behavior

At `max-width: 760px`:

- the SVG causal overlay is removed;
- loops become one stable vertical sequence;
- each source loop prints its outgoing `L → D` causal handoff as compact text below the loop;
- the same click-to-focus hierarchy is retained;
- there is no horizontal engineering canvas to pan.

This is a semantic reflow, not a miniaturized desktop graph.

## Source of truth

Historical inventory:

`scripts/engineering_lineage/lineage_data.py`

Presentation / Atlas schema generator:

`scripts/engineering_lineage/build_lineage.py`

Generated public data:

`frontend/engineering_lineage/engineering/data.js`

Rebuild:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 scripts/engineering_lineage/build_lineage.py
```

The generator still validates that all canonical BRIDGE historical records remain represented before writing the public payload.
