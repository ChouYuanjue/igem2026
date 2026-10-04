# BRIDGE Engineering Atlas

Public route:

`https://nju-igem.runnelzhang.com/engineering/`

## Display model

The atlas contains the complete **219-node** BRIDGE engineering lineage, but it no longer scales every label into a single giant SVG viewport.

The current interface uses **overview + progressive disclosure**:

- a fixed-size **15-step primary backbone rail** keeps the main EnzymeCAGE → Broad → BiME-Rank → BRIDGE story readable;
- the lineage reader opens with only the **24 structural nodes** required to understand the topology;
- every node keeps normal-size text and reports how many real descendants sit below it;
- `+` reveals the next actual experimental layer in place;
- the detail panel can open an entire local branch when the user wants exhaustive history;
- search exposes a hidden experiment together with only the ancestor path needed to understand where it belongs;
- `Show all 219` remains available for exhaustive scrolling without shrinking typography;
- cross-branch inheritance is represented as explicit `↗` relations in node details instead of drawing 34 overlapping lines across the page.

This preserves the single-tree model while separating **global orientation** from **local reading**.

## Lineage semantics

- **EnzymeCAGE is the single root.**
- Primary parent-child relations are direct design descent.
- Sibling branches are parallel or competing research programs.
- FIBRE is a large side branch from BiME-Rank, not a mandatory step to BRIDGE.
- Cross-links record ideas reused by another branch.
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
