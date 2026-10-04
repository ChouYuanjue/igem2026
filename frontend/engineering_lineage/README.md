# BRIDGE Engineering Atlas

This directory contains the interactive **complete engineering lineage** for BRIDGE.

Public route:

`https://nju-igem.runnelzhang.com/engineering/`

## Representation

The visualization is intentionally more expressive than a timeline or a simple flowchart.

- **EnzymeCAGE is the single root.**
- **Solid parent-child edges** represent direct design descent: one problem or method directly produced the next branch.
- **Sibling branches** represent parallel research programs or competing solutions; commit dates may interleave.
- **Dashed grafts** represent ideas that later crossed into another branch without making the source branch part of the primary ancestry.
- **FIBRE is a large side branch from BiME-Rank.** It is not placed on the mandatory path to BRIDGE. Surviving ideas such as explicit scientific evidence, plug-in experts, frozen-core gating, and missing-neutral fallback are grafted back to the final line.
- **CONSIDERED** nodes distinguish serious literature/design exploration from experiments that were actually implemented and frozen.

The current canonical inventory contains **219 nodes, 151 leaves, and 34 cross-branch inheritance links**.

## Single source of truth

The lineage is maintained in:

`scripts/engineering_lineage/lineage_data.py`

Run:

```bash
PYTHONPATH=. python3 scripts/engineering_lineage/build_lineage.py
```

The builder validates the primary tree and generates both:

- `frontend/engineering_lineage/engineering/data.js` for the interactive page;
- `projects/active/bridge/docs/engineering.md` for the complete text representation.

This prevents the website and repository Engineering narrative from silently diverging.

## Interaction

The page supports:

- pan and zoom over the full tree;
- full-tree, backbone, and FIBRE-focused views;
- search across method names, motivations, results, and legacies;
- filters by outcome and research family;
- a detail panel for motivation, result, surviving idea, parent/child relations, and cross-branch inheritance;
- optional display of dashed graft links.

## Deployment

The static page is served by an isolated user service on `127.0.0.1:8866` using:

`scripts/engineering_lineage/engineering-lineage.service`

The existing `nju-igem` Cloudflare tunnel routes `/engineering` to this service before the COMPASS catch-all. The Engineering Atlas does not share the COMPASS application process or database frontend process.
