# BRIDGE Engineering Atlas

Public route:

`https://nju-igem.runnelzhang.com/engineering/`

## Purpose

The atlas is a single continuous project graph. Its job is to let a reader understand **what problem appeared, what was tried, what the experiment showed, and why the next branch existed** with as little on-screen text as possible.

The graph is grounded in the repository's development histories, frozen evaluation records, current method documents, runtime architecture and relevant commit chronology. Commit order is supporting evidence, not the organizing principle.

## Display architecture

The frontend uses two coordinated layers:

- **HTML/CSS DOM cards** for node text and interaction;
- a high-DPI **Canvas edge layer** for primary descent and cross-branch inheritance.

Both use the same world coordinates and camera transform. This keeps browser-native text readable while making edge redraw cheap and reliable during desktop wheel zoom, drag navigation and mobile pinch zoom.

No branch opens a separate interface. Search and detail links only move the camera.

## Compact layered layout

The layout is calculated per causal depth rather than by total descendant-leaf width.

- the surviving EnzymeCAGE → BRIDGE route stays on the vertical centerline;
- siblings remain grouped around their actual parent;
- parent groups are packed at each depth with collision-free spacing;
- shallow stages therefore stay compact even when one branch later develops many experiments;
- only genuinely dense deeper levels expand horizontally.

This avoids the earlier failure mode where a shallow node inherited the full width of all remote descendants and produced extremely long visual arms.

## Node copy

Visible cards contain only:

1. a semantic method/decision name;
2. a short evidence-backed outcome.

Internal experiment version names and implementation labels are removed from display names. Examples include replacing reaction-center version labels with `Direct reaction-center fusion` / `Bounded reaction-center correction`, replacing numerical shortlist implementation names with `Shortlist pair reranking`, and replacing router version numbers with `Query-conditioned expert routing` / `Expert permission levels`.

Full motivation, result and surviving lesson remain available in the detail panel.

## Graph semantics

- dark thick edge: surviving technical descent;
- thin solid edge: direct experiment/alternative;
- amber solid edge: turning-point branch;
- gold dashed edge: an idea or evidence source reused across branches.

Major transitions also receive short causal notes, such as the candidate-gate recall ceiling, open-world molecular-input requirement, expert decomposition, FIBRE replacement attempt, return to Broad and query-specific expert applicability.

## Canonical source

The project graph is maintained in:

`scripts/engineering_lineage/lineage_data.py`

Generate synchronized website data and Engineering documentation with:

```bash
PYTHONPATH=. python3 scripts/engineering_lineage/build_lineage.py
```

Generated outputs:

- `frontend/engineering_lineage/engineering/data.js`
- `projects/active/bridge/docs/engineering.md`

## Deployment

The page is served by the isolated `engineering-lineage.service` on `127.0.0.1:8866` and routed through the existing `nju-igem` Cloudflare tunnel at `/engineering/`. It remains separate from COMPASS and the database frontend.
