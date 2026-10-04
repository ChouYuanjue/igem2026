# BRIDGE Engineering Atlas

Public route:

`https://nju-igem.runnelzhang.com/engineering/`

## Representation

The Engineering Atlas is one continuous **multi-level project graph**. It does not collapse research programs into one-level summaries and does not open branches in separate views.

The canonical project route now includes the historically important lines that were missing from earlier visual drafts:

- EnzymeCAGE → pocket robustness → full-library structural screening → reaction-similarity gate → closed candidate pool → measured recall ceiling → open-world retrieval;
- open candidate registries, TPS specialization, uncertainty/evidence, graph alternatives, broad representation learning, domain adaptation and wet-lab decision design as real parallel subtrees;
- Broad generalization/anti-forgetting, foundation/structure/mechanistic experts, cleanroom evaluation and rank fusion converging into BiME-Rank;
- the full FIBRE research detour with its own deep geometry, conditional-mode, scientific-evidence and relational-core branches;
- the return to a protected Broad base, query-level applicability, bounded family specialists and final BRIDGE;
- the real user-driven product lineage from semantic scope switching through the bounded agent and persistent research workspace to Starase Navigator and COMPASS;
- discovery panels, MILP plate balancing, Hungarian well-position randomization and wet-lab feedback as the experimental-execution branch.

Database work is intentionally not inserted into the scientific lineage.

## Graph semantics

- **Solid parent-child edges** are direct technical descent.
- **Thicker dark edges** show the causal route that survived into BRIDGE.
- **Gold dashed links** show ideas or evidence reused across otherwise separate branches.
- Major transitions carry short causal annotations such as the candidate-gate recall ceiling, the move to molecular-input open retrieval, expert decomposition, the FIBRE replacement attempt, and the later return to Broad.

All descendants are present in the same SVG scene at the same time. Clicking a node only inspects its reasoning; it never changes the graph topology.

## Navigation

The graph is deliberately larger than the viewport.

- drag to pan;
- wheel or pinch to zoom;
- `Root` returns to EnzymeCAGE;
- `Fit tree` gives a bird's-eye topology view;
- search moves the camera directly to any method, experiment, failure or decision;
- double-click/double-tap focuses a node;
- inheritance links can be toggled without affecting primary descent.

On mobile, edges and nodes live in the same transformed SVG scene. Primary connections therefore remain visible during pan/zoom instead of being lost by a separate responsive layout.

## Node hierarchy

Visual weight reflects project role rather than descendant quantity:

- milestones: larger serif cards;
- research programs: medium branch cards;
- concrete experiments: compact cards;
- rejected, historical and considered routes: distinct restrained treatments;
- EnzymeCAGE and BRIDGE: root/current-method endpoints.

At very distant zoom levels, low-level labels fade while the nodes and all edges remain present. Zooming in restores the exact experiment labels; this is semantic zoom, not a separate layer or collapsed branch.

## Single source of truth

The lineage is maintained in:

`scripts/engineering_lineage/lineage_data.py`

Run:

```bash
PYTHONPATH=. python3 scripts/engineering_lineage/build_lineage.py
```

The builder validates the primary graph and generates both:

- `frontend/engineering_lineage/engineering/data.js`
- `projects/active/bridge/docs/engineering.md`

The website and repository Engineering narrative therefore share one causal graph.

## Deployment

The page remains served by the isolated `engineering-lineage.service` on `127.0.0.1:8866` and routed through the existing `nju-igem` Cloudflare tunnel at `/engineering/`. It remains separate from COMPASS and database frontend processes.
