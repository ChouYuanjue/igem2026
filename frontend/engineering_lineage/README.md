# BRIDGE Engineering Atlas

Public route:

`https://nju-igem.runnelzhang.com/engineering/`

## Display model

The complete 219-node lineage is presented as a **vertical semantic graph**.

- EnzymeCAGE is the root at the top.
- BRIDGE is the terminal milestone at the bottom.
- The primary design spine grows vertically.
- Parallel research programs branch left and right from the spine.
- Opening a branch replaces the overview with a vertical local subgraph: parent context at the top, the focused branch below it, and concrete experiments continuing downward in alternating left/right leaves.
- Search still covers all 219 nodes.
- Cross-branch inheritance remains distinct through gold dashed links and the detail panel.

## Node visual hierarchy

The page deliberately uses different node forms instead of one universal card:

- **milestones**: numbered circular markers plus a compact title block;
- **waypoints**: small colored dots with short labels;
- **research programs**: side-leaf cards with a colored stem and descendant count;
- **experiments**: compact branch cards used only inside focused views;
- **focused branch**: a larger central marker that anchors the local graph.

On screens below 720 px, the same topology is recomputed for the available width. The central spine remains fixed, side leaves become narrower, branch distance and vertical spacing change, and details open as a dismissible bottom sheet. No horizontal scrolling is required.

## Validation

Geometry checks cover phone and desktop widths. The overview has zero card collisions at 360, 390, 430, 719, 768, 1024, and 1400 px. Dense focused branches, including the 16-child generalization branch and 14-child FIBRE conditional-mode branch, also have zero card collisions across the tested widths.

## Single source of truth

The lineage itself remains in:

`scripts/engineering_lineage/lineage_data.py`

Run:

```bash
PYTHONPATH=. python3 scripts/engineering_lineage/build_lineage.py
```

The builder generates both the website data and `projects/active/bridge/docs/engineering.md`.

## Deployment

The page remains served by the isolated `engineering-lineage.service` on `127.0.0.1:8866` and routed through the existing `nju-igem` Cloudflare tunnel at `/engineering/`. It is separate from COMPASS and the database frontend.
