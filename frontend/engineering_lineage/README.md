# BRIDGE Engineering Atlas

Public route:

`https://nju-igem.runnelzhang.com/engineering/`

## Visual model

The page uses a hierarchical Design–Build–Test–Learn atlas rather than a linear timeline or a single giant tree.

Four macro stages carry different structural meanings:

- **A — serial loops:** early EnzymeCAGE / open-retrieval work is intentionally compressed into two sequential loops.
- **B — parallel loops:** generalization, functional evidence, structural/mechanistic evidence and fusion/routing develop in parallel and converge into BiME-Rank.
- **C — nested loops:** FIBRE is one large redesign loop containing relation-geometry, conditional-formulation, scientific-evidence and relational-core subloops.
- **D — convergence:** query permission, family-specific CAGE, TPS specialization and bounded integration develop as sibling loops and merge into BRIDGE.

The visual weight therefore follows the project: early work is compact; the later expert/FIBRE/BRIDGE work occupies most of the page.

## Information hierarchy

Primary view:

- macro stage;
- cycle title;
- all four D/B/T/L summaries;
- the Learn outcome.

Secondary detail:

- selecting a cycle Learn center opens the exact key records for its four phases;
- `Full record` exposes every canonical historical attempt assigned to that stage;
- wet-lab execution and COMPASS remain small parallel tracks.

The canonical record is not reduced: the generator validates that all historical records remain represented.

## Typography

The page deliberately avoids display-scale typography. Desktop sizes are approximately:

- page title: 28 px;
- macro-stage title: 20 px;
- cycle title: 14 px;
- phase/result text: 11–13 px.

Mobile uses the same hierarchy with 24/18/14/11–13 px scales.

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

`bridge-engineering-atlas-v6`
