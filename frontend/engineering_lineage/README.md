# BRIDGE Engineering Atlas

Public route:

`https://nju-igem.runnelzhang.com/engineering/`

## Display model

The atlas contains the complete **219-node** BRIDGE engineering lineage, rendered as a **semantic-zoom research constellation graph** rather than a directory tree or a scale-to-fit mega-diagram.

The graph has two visual scales:

1. **Landscape overview**
   - a dark, curved design spine traces the main EnzymeCAGE → BRIDGE causal line;
   - intermediate spine steps use short semantic labels rather than shrinking full experiment names;
   - major parallel research programs appear as colored constellations around the spine;
   - constellation size/halo communicates that a branch contains deeper history;
   - branch placement is deterministic and collision-audited rather than force-directed at runtime.

2. **Local constellation view**
   - selecting a research constellation keeps its parent on the left as context;
   - the focused branch occupies the center;
   - direct experiments spread along one or two bowed graph arcs with equal vertical spacing;
   - branches with children can be opened recursively without changing label scale;
   - Back and Overview preserve navigation context.

Search is global across all 219 nodes. A search hit jumps directly to the local graph containing that experiment.

Cross-branch inheritance remains distinct from primary descent. Gold dashed links are drawn only when both endpoints are present in the current semantic view; all inheritance relations remain available in the detail panel.

## Why this representation

The Engineering history needs both topology and legibility. Displaying all 219 labels simultaneously made the topology visible but the text unreadable. Replacing the graph with a directory fixed legibility but destroyed the visual meaning of parallel exploration and convergence.

The current approach uses **semantic zoom + focus/context**: the visual representation changes with the level of attention instead of geometrically shrinking the same labels. The overview answers “what were the major research directions and where did they attach?”; the focused view answers “what exact experiments were inside this branch?”

## Lineage semantics

- **EnzymeCAGE is the single root.**
- Primary edges are direct design descent.
- Sibling constellations are parallel or competing research programs.
- FIBRE is a large side branch from BiME-Rank, not a mandatory step to BRIDGE.
- Cross-links record ideas later reused by another branch.
- `CONSIDERED` separates serious design/literature exploration from completed frozen experiments.

## Single source of truth

The canonical lineage is maintained in:

`scripts/engineering_lineage/lineage_data.py`

Run:

```bash
PYTHONPATH=. python3 scripts/engineering_lineage/build_lineage.py
```

The builder validates the primary tree and generates both:

- `frontend/engineering_lineage/engineering/data.js`
- `projects/active/bridge/docs/engineering.md`

The website and repository Engineering narrative therefore share the same 219-node source.

## Deployment

The page is served by the isolated `engineering-lineage.service` on `127.0.0.1:8866`, routed through the existing `nju-igem` Cloudflare tunnel at `/engineering/`. It remains separate from COMPASS and the database frontend processes.
