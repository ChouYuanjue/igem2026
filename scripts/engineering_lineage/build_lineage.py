from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from scripts.engineering_lineage.lineage_data import CROSSLINKS, FAMILY_LABELS, NODES

ROOT = Path(__file__).resolve().parents[2]
DATA_JS = ROOT / 'frontend' / 'engineering_lineage' / 'engineering' / 'data.js'
DOC = ROOT / 'projects' / 'active' / 'bridge' / 'docs' / 'engineering.md'


def validate() -> tuple[dict[str, dict], dict[str, list[str]]]:
    by = {node['id']: node for node in NODES}
    if len(by) != len(NODES):
        raise ValueError('duplicate lineage node id')
    children: dict[str, list[str]] = defaultdict(list)
    roots = []
    for node in NODES:
        parent = node['parent']
        if parent is None:
            roots.append(node['id'])
        else:
            if parent not in by:
                raise ValueError(f"unknown parent {parent!r} for {node['id']!r}")
            children[parent].append(node['id'])
    if roots != ['enzymecage']:
        raise ValueError(f'unexpected roots: {roots}')
    for link in CROSSLINKS:
        if link['source'] not in by or link['target'] not in by:
            raise ValueError(f'bad cross-link: {link}')
    # Cycle audit on the primary tree.
    for node_id in by:
        seen = set()
        cur = node_id
        while cur is not None:
            if cur in seen:
                raise ValueError(f'cycle through {node_id}')
            seen.add(cur)
            cur = by[cur]['parent']
    return by, children


def write_data() -> None:
    payload = {
        'nodes': NODES,
        'crossLinks': CROSSLINKS,
        'families': FAMILY_LABELS,
        'meta': {
            'schema': 'bridge-engineering-lineage-v2',
            'root': 'enzymecage',
            'current': 'bridge',
            'semantics': {
                'solid': 'primary design descent',
                'dashed': 'idea reused across branches',
                'considered': 'documented/researched but not promoted into a complete frozen experiment',
            },
        },
    }
    DATA_JS.write_text(
        'window.LINEAGE_DATA = ' + json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + ';\n',
        encoding='utf-8',
    )


def write_doc(by: dict[str, dict], children: dict[str, list[str]]) -> None:
    lines = [
        '# BRIDGE Engineering Lineage',
        '',
        'The current Engineering map is a **rooted design lineage with explicit cross-links**.',
        'Its root is **EnzymeCAGE**. Solid parent–child edges represent direct design descent;',
        'dashed cross-links represent ideas that were reused later by a different branch.',
        'This structure intentionally preserves parallel research programs and the large FIBRE detour instead of forcing the history into a linear timeline.',
        '',
        'Live interactive atlas: `https://nju-igem.runnelzhang.com/engineering/`',
        '',
        'Status vocabulary: `KEEP`, `LOCAL`, `TURN`, `REJECT`, `HISTORICAL`, `CONSIDERED`.',
        '',
        '## Primary lineage tree',
        '',
    ]

    def emit(node_id: str, depth: int) -> None:
        node = by[node_id]
        status = node['status'].upper()
        prefix = '  ' * depth + '- '
        lines.append(f"{prefix}**{node['label']}** `[{status}]` — {node['why']} **Result:** {node['result']} **Legacy:** {node['legacy']}")
        for child in children.get(node_id, []):
            emit(child, depth + 1)

    emit('enzymecage', 0)
    lines += ['', '## Cross-branch inheritance', '']
    for link in CROSSLINKS:
        lines.append(f"- **{by[link['source']]['label']}** → **{by[link['target']]['label']}** — {link['label']}.")
    lines += [
        '',
        '## Reading principle',
        '',
        'The tree is organized by technical causality rather than strict commit order. A sibling relationship means the routes were parallel alternatives or independent supporting programs.',
        'FIBRE is therefore displayed as a large branch beside the protected-Broad line, with surviving ideas grafted back through cross-links.',
        'A rejected node remains visible when it materially constrained the next design.',
        '',
        'The machine-readable source of this document and the live visualization is `scripts/engineering_lineage/lineage_data.py`.',
    ]
    DOC.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    by, children = validate()
    write_data()
    write_doc(by, children)
    leaves = [node_id for node_id in by if not children.get(node_id)]
    depth = {}
    for node_id in by:
        d = 0
        cur = by[node_id]['parent']
        while cur is not None:
            d += 1
            cur = by[cur]['parent']
        depth[node_id] = d
    print(json.dumps({
        'nodes': len(NODES),
        'cross_links': len(CROSSLINKS),
        'leaves': len(leaves),
        'max_depth': max(depth.values()),
        'data_js': str(DATA_JS.relative_to(ROOT)),
        'doc': str(DOC.relative_to(ROOT)),
    }, indent=2))


if __name__ == '__main__':
    main()
