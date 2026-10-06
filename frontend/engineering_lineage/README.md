# Atlas Engineering

Public route:

`https://nju-igem.runnelzhang.com/engineering/`

## Reading model

Atlas Engineering is presented as **three parallel engineering storylines**:

- **EDGE · Database** — the evolving known biochemical graph and evidence store;
- **BRIDGE · Model** — open-world enzyme–reaction inference;
- **COMPASS · Agent** — scientific orchestration and reusable research state.

Atlas EDGE, Atlas BRIDGE and Atlas COMPASS are shown as parallel engineering functions. Their vertical positions provide stable orientation rather than a hierarchy of importance.

There is no fourth “shared”, “evaluation”, or “integration” tree. Work that spans systems appears as a **junction inside one of the three primary storylines**, with explicit causal handoffs to the other line.

Examples:

- temporal graph-growth evaluation belongs to the EDGE storyline and crosses into BRIDGE because BRIDGE is the system being tested against the changing graph;
- canonical database evidence belongs to EDGE and crosses into COMPASS when it becomes reusable research state;
- the intent-to-ranking contract belongs to COMPASS and crosses into BRIDGE when the agent requests a model operation.

## Global overview

The overview is a compact storyline map, not a literal hierarchy tree.

It shows:

- EDGE as the upper horizontal storyline;
- Atlas BRIDGE as the middle storyline for enzyme–reaction inference;
- COMPASS as the lower horizontal storyline;
- major DBTL loops as stations on each line;
- nested subloops as small branches growing away from their parent station;
- cross-system handoffs as light connector curves between storylines;
- the current loop and its ancestor path as the highlighted route.

The overview contains no DBTL prose. Its only job is to reveal the overall shape of the engineering history and show where the current local focus sits inside the three-line system.

The layout follows a storyline/metro-map principle: persistent lines remain easy to follow, crossings are kept sparse, and detailed causal explanations are deferred to the local view.

## Local focus

The main panel always shows only one local hierarchy level.

There is no pan, zoom, drag, force-directed layout or free canvas.

Interaction rules:

1. the overview always keeps the three global storylines visible;
2. clicking a loop with children replaces the local panel with only its direct subloops;
3. breadcrumbs and the Parent control return to higher levels;
4. clicking a leaf loop opens its Design–Build–Test–Learn details inline;
5. the overview does not zoom — it only changes which route is highlighted;
6. the overview itself is navigable: trunk lines open EDGE / BRIDGE / COMPASS, visible stations and branch segments open the corresponding loop, and cross-system connector lines follow their target junction;
7. every loop with an outgoing Learn→Design handoff exposes an explicit `Next` action, independent of entering its sub-loops.

## Causal semantics

Every loop is represented as a Design–Build–Test–Learn object.

The important cross-loop relation is:

`Loop A / Learn → Loop B / Design`

A handoff means that a learned constraint from one loop changed another loop’s next design.

These handoffs may stay within one storyline or cross between EDGE, BRIDGE and COMPASS.

The generated schema is:

`atlas-engineering-storylines-v11`

Key fields:

- `atlas.root` — contains exactly the three public trunks: EDGE, BRIDGE, COMPASS;
- `atlas.storylines` — records their visual order and relative trunk weight;
- `atlas.handoffs` — phase-level causal edges;
- `atlas.systems` — authority labels for EDGE / BRIDGE / COMPASS.

The generator rejects any public root that adds a fourth top-level system.

## EDGE storyline

EDGE is substantially deeper than a database frontend story.

Its main evolution is:

### Build a biochemical graph people can actually enter

Subloops:

- heterogeneous files → persistent biochemical entities;
- persistent entities → first-class graph relations;
- relation graph → parsed search and pathway entry points.

### Scale the graph without erasing provenance

Subloops:

- Swiss-Prot + TrEMBL source segmentation;
- persistent identity across source refreshes;
- scope-aware search / BLAST / indexing at large scale.

### Turn the graph into a canonical evidence service

Subloops:

- shared entity/bundle assembly;
- server-side SMILES identity resolution;
- reproducible source-segmented data assets.

### Evaluate a graph that keeps growing

This is the evaluation program. It belongs inside EDGE and crosses into BRIDGE.

Its subloops are:

- missing relations inside a fixed graph;
- unseen endpoints;
- entity-arrival events;
- official Rhea historical graph replay;
- continuous curation events;
- directional query/candidate exposure;
- candidate-cardinality decomposition.

The final Learn is that endpoint exposure and search-space size are separate axes, so the deployment candidate universe must remain fixed.

### Make graph evidence reusable inside scientific reasoning

This is the EDGE→COMPASS junction.

Subloops:

- direct canonical evidence service for COMPASS;
- molecular structures exposed progressively;
- database provenance preserved as reusable workspace state.

## Atlas BRIDGE storyline

The existing Atlas BRIDGE engineering story remains the inference storyline and is not rewritten.

Its six primary loops are:

- Can better ranking rescue a bounded candidate system?
- Can Broad handle recall while CAGE keeps ranking authority?
- How should heterogeneous evidence modify a strong Broad order?
- Can one relational core replace the expert stack and Broad order?
- Who may alter Broad’s order for this query?
- How should known positives influence a model that may already have seen them?

The sixth station is the current dual-memory context loop. Training-time `clean2023` relations are treated as read-only long-term recall and cannot vote again as runtime seeds; relations that arrive after training become episodic support, are pooled in the frozen Broad space, and receive a validation-frozen query-specific trust value. Registered and open-world supports use the same support-only semantics and do not change the candidate universe.

The storyline handoff is explicit: the fifth loop learns that ranking authority can be assigned per query, then the sixth loop asks how context authority should depend on provenance. All existing recursive BRIDGE subloops and the 232 historical engineering records remain represented beneath those stations; the dual-memory station is a current presentation layer over retained historical records rather than a rewrite of the old seed-context history.

## COMPASS storyline

COMPASS now starts from its original product motivation instead of appearing fully formed as a research agent.

### Hide model-routing complexity from the user

The initial role of the agent is simply to remove the need for scientists to memorize model routes and control vocabulary.

Subloops:

- scientific request → explicit model operation;
- conversational scope → inspectable retrieval parameters.

### Turn the portal into a bounded scientific agent

Subloops:

- typed tool actions;
- layered and isolated context;
- action recovery and failure-safe state.

### Turn conversations into a reusable research workspace

Subloops:

- reusable workspace objects;
- local route patching;
- derived-route lineage.

### Require verified identity before multi-step reasoning

Subloops:

- stereochemical normalization;
- verified compound binding;
- pathway ambiguity resolved by verified references.

### Represent facts, inferred candidates and unsupported claims separately

Subloops:

- evidence counts and provenance;
- database facts vs model candidates as different epistemic layers.

### Manage the lifecycle and cost of scientific observations

Subloops:

- budget-aware evidence acquisition;
- verified observation reuse;
- source-bound extracted observations.

### Evaluate whether the agent preserves scientific invariants

Subloops:

- long-horizon agent history evaluation;
- metamorphic perturbation tests.

### Bind orchestration to the complete BRIDGE model

This is the COMPASS→BRIDGE junction.

Subloops:

- semantic scope separated from candidate cardinality;
- complete BRIDGE runtime owns ranking authority;
- verified research context shapes the next model call.

This closes the narrative arc: COMPASS begins as a convenience portal over model routing and ends as a persistent scientific workflow around BRIDGE and EDGE.

## Desktop behavior

Desktop keeps the three-storyline overview visible beside the local branch.

The three system lines use comparable visual weight; their distinction comes from function, color and stable position.

The local panel uses fixed grid positions and renders only the current sibling loops. Cross-loop causal arrows are drawn only for that local layer.

## Tablet behavior

The global storyline map moves above the local panel. The three lines remain intact and no horizontal pan is introduced.

Local loops reflow into fewer fixed columns.

## Mobile behavior

The three-storyline overview remains visible at the top.

The local causal SVG is removed and loops become one stable vertical sequence. Each outgoing Learn→Design transition is rendered as compact text under its source loop.

The mobile version is a semantic reflow, not a miniature draggable canvas.

## Source of truth

Historical BRIDGE inventory:

`scripts/engineering_lineage/lineage_data.py`

Atlas storyline and loop generator:

`scripts/engineering_lineage/build_lineage.py`

Generated public data:

`frontend/engineering_lineage/engineering/data.js`

Rebuild:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 scripts/engineering_lineage/build_lineage.py
```

The generator verifies that all 232 canonical BRIDGE historical records remain represented while also enforcing the three-trunk public Atlas architecture.
