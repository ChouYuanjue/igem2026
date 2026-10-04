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
    for node_id in by:
        seen = set()
        cur = node_id
        while cur is not None:
            if cur in seen:
                raise ValueError(f'cycle through {node_id}')
            seen.add(cur)
            cur = by[cur]['parent']
    for link in CROSSLINKS:
        if link['source'] not in by or link['target'] not in by:
            raise ValueError(f'bad cross-link: {link}')
    return by, children


def subtree(root: str, children: dict[str, list[str]]) -> set[str]:
    out: set[str] = set()

    def walk(node_id: str) -> None:
        out.add(node_id)
        for child in children.get(node_id, []):
            walk(child)

    walk(root)
    return out


def before(root: str, cut_roots: list[str], children: dict[str, list[str]]) -> set[str]:
    selected = subtree(root, children)
    for cut in cut_roots:
        selected -= subtree(cut, children) - {cut}
    return selected


def ordered(ids: set[str]) -> list[str]:
    return [node['id'] for node in NODES if node['id'] in ids]


def build_cycle_sets(children: dict[str, list[str]]) -> tuple[list[set[str]], dict[str, set[str]]]:
    c1 = before('enzymecage', ['open_problem'], children)
    c2 = (subtree('candidate_program', children) - subtree('wetlab_program', children)) | {'open_problem'}
    c3 = (
        before('representation_program', ['broad'], children)
        | subtree('tps_mech_program', children)
        | subtree('evidence_program', children)
        | subtree('graph_program', children)
        | {'open_problem', 'broad'}
    )
    c4 = (
        subtree('budget_routing', children)
        | subtree('generalization_program', children)
        | subtree('expert_program', children)
        | subtree('stress_program', children)
        | {'broad', 'fusion_program'}
    )
    c5 = (subtree('fusion_program', children) | subtree('bime', children))
    c5 -= subtree('fibre', children)
    c5 -= subtree('return_broad', children)
    c5 |= {'fusion_program', 'bime'}
    c6 = subtree('fibre', children)
    c7 = subtree('return_broad', children)
    tracks = {
        'wetlab': subtree('wetlab_program', children),
        'compass': subtree('user_semantic_routing', children),
    }
    return [c1, c2, c3, c4, c5, c6, c7], tracks


def phase(title: str, summary: str, key_ids: list[str]) -> dict:
    return {'title': title, 'summary': summary, 'keyIds': key_ids}


def build_cycles(by: dict[str, dict], children: dict[str, list[str]]) -> tuple[list[dict], list[dict]]:
    cycle_sets, track_sets = build_cycle_sets(children)
    cycles = [
        {
            'id': 'closed-cage',
            'number': '01',
            'title': 'Make structure ranking work in a real library',
            'change': 'Candidate space, not pocket choice, became the first bottleneck.',
            'outcome': 'Gate recall is the bottleneck',
            'phases': {
                'design': phase('Design', 'Start from EnzymeCAGE and test whether better pocket use can fix retrieval.', ['enzymecage', 'pocket_audit']),
                'build': phase('Build', 'Run full-library structural scoring, then add reaction-neighbour candidate transfer.', ['full_library_structure', 'reaction_transfer', 'closed_pool']),
                'test': phase('Test', 'Stress the structural ranker on TPS reactions and measure candidate-pool coverage.', ['full_library_structure', 'gate_coverage_ceiling']),
                'learn': phase('Learn', 'Pair compatibility helps only after the right proteins enter the pool; a hard gate creates an unrecoverable ceiling.', ['gate_coverage_ceiling', 'open_problem']),
            },
            'recordIds': ordered(cycle_sets[0]),
            'evidence': [
                {'label': 'Full-library structure', 'value': '0 Top-10 hits · MRR ≈ 0.0037'},
                {'label': 'Relation gate coverage', 'value': '720 / 1,640 = 43.98%'},
            ],
            'trackIds': [],
        },
        {
            'id': 'open-candidates',
            'number': '02',
            'title': 'Open the candidate universe',
            'change': 'Candidate generation became an explicit engineering object instead of hidden preprocessing.',
            'outcome': 'Gates still define scoreability',
            'phases': {
                'design': phase('Design', 'Allow unseen proteins and reactions to enter without relying on one fixed relation table.', ['open_problem', 'candidate_program']),
                'build': phase('Build', 'Add registries, few-shot expansion, UniProt quotas, Pfam constraints and semantic scope.', ['registry', 'fewshot', 'uniprot', 'pfam']),
                'test': phase('Test', 'Compare free expansion, controlled tails, homolog and cross-cluster seeds, and taxonomy constraints.', ['uniprot_free', 'uniprot_quota', 'fewshot_crosscluster', 'pfam_hier']),
                'learn': phase('Learn', 'Heuristics can widen coverage, but every gate still decides who is scoreable. Retrieval itself must become continuous.', ['candidate_program', 'representation_program']),
            },
            'recordIds': ordered(cycle_sets[1]),
            'evidence': [
                {'label': 'Key lesson', 'value': 'wider candidate pools ≠ universal scoreability'},
            ],
            'trackIds': ['wetlab'],
        },
        {
            'id': 'broad-retrieval',
            'number': '03',
            'title': 'Learn one full-space ranking',
            'change': 'Broad Retrieval became the stable order that works even when optional evidence is absent.',
            'outcome': 'Broad becomes the base order',
            'phases': {
                'design': phase('Design', 'Represent reaction demand and enzyme capability continuously in both retrieval directions.', ['representation_program', 'esm_c', 'drfp', 'dual_tower']),
                'build': phase('Build', 'Train a bidirectional dual tower with false-negative protection, hard negatives and domain adaptation.', ['dual_tower', 'pu_mask', 'hard_negatives', 'hard_curriculum', 'marts_adapt']),
                'test': phase('Test', 'Compare Top-K objectives, Horizyn transfer, graph kernels, mechanism cues and reliability controls.', ['topk_surrogate', 'horizyn_mlnce', 'dual_kernel_marts', 'cycle']),
                'learn': phase('Learn', 'The broad dual tower survives across candidate universes and becomes the universal fallback order.', ['broad', 'budget_routing']),
            },
            'recordIds': ordered(cycle_sets[2]),
            'evidence': [
                {'label': 'Broad Core', 'value': 'MRR 0.2093 · Hit@10 36.87% · Hit@100 59.41%'},
            ],
            'trackIds': ['compass'],
        },
        {
            'id': 'conditional-evidence',
            'number': '04',
            'title': 'Add evidence without destroying Broad',
            'change': 'Extra evidence was useful only under the right support, direction and query.',
            'outcome': 'Evidence must be conditional',
            'phases': {
                'design': phase('Design', 'Expand domains and specialist evidence while preserving the strong Broad order.', ['generalization_program', 'expert_program']),
                'build': phase('Build', 'Try retention methods plus EnzGFM, reaction-center, CLIPZyme and shortlist rerankers.', ['replay_anchor', 'score_distill', 'enzgfm', 'center_v3', 'clipzyme']),
                'test': phase('Test', 'Freeze comparisons on external, temporal and novelty stress tests instead of training-set wins.', ['stress_program', 'enzyme405', 'orphan335', 'rhea_transfer', 'temporal']),
                'learn': phase('Learn', 'No extra signal deserves unconditional authority; the next model must route a portfolio around Broad.', ['posthoc_router', 'fusion_program']),
            },
            'recordIds': ordered(cycle_sets[3]),
            'evidence': [
                {'label': 'External generalization', 'value': 'Broad beats generic CAGE on Enzyme-405'},
                {'label': 'Orphan retrieval', 'value': 'Broad MRR 0.2605 vs Selenzyme 0.2088'},
            ],
            'trackIds': [],
        },
        {
            'id': 'bime',
            'number': '05',
            'title': 'Route and protect experts explicitly',
            'change': 'BiME showed that global expert admission is still too coarse.',
            'outcome': 'Global admission is too coarse',
            'phases': {
                'design': phase('Design', 'Build a portfolio with explicit candidate union, routing, fallback and direction-specific protection.', ['fusion_program', 'portfolio', 'r2e_lambdarank']),
                'build': phase('Build', 'Use learned fusion for R2E and anchored learning-to-rank for strong E2R baselines.', ['r2e_lambdarank', 'e2r_branch', 'e2r_anchor', 'admission']),
                'test': phase('Test', 'Admit seed, structural and contextual experts only after frozen confirmation; test generic CAGE as an expert.', ['bime_clip', 'seed_context', 'generic_cage_expert', 'cost_hierarchy']),
                'learn': phase('Learn', 'An expert can be globally valid yet locally harmful. Expert rights must become query- and direction-specific.', ['bime', 'generic_cage_expert', 'fibre', 'return_broad']),
            },
            'recordIds': ordered(cycle_sets[4]),
            'evidence': [
                {'label': 'BiME confirmation', 'value': 'R2E MRR 0.1024 → 0.1217'},
                {'label': 'Generic CAGE expert', 'value': 'OOF MRR 0.2678 → 0.2648'},
            ],
            'trackIds': [],
        },
        {
            'id': 'fibre',
            'number': '06',
            'title': 'Try replacing the stack with one relational core',
            'change': 'FIBRE simplified the abstraction on paper, but could not justify replacing Broad.',
            'outcome': 'Unified core cannot replace Broad',
            'phases': {
                'design': phase('Design', 'Unify biological relation, mechanism and heterogeneous evidence in one relational formulation.', ['fibre', 'bio_relation', 'context_domain', 'tensor_field']),
                'build': phase('Build', 'Implement conditional modes, explicit scientific evidence, adaptive mixtures and an ERAM relational core.', ['conditional_modes', 'scientific_evidence', 'query_mix_posthoc', 'eram']),
                'test': phase('Test', 'Use strict temporal and double-cold evaluation to ask whether the relational core can own ranking.', ['relational_main', 'temporal_relational']),
                'learn': phase('Learn', 'Keep evidence admission, frozen-core gating, plugins and missing-neutral fallback; restore Broad as ranking authority.', ['plugins', 'open_fallback', 'return_broad']),
            },
            'recordIds': ordered(cycle_sets[5]),
            'evidence': [
                {'label': 'Replacement test', 'value': 'strict temporal / double-cold did not justify replacing Broad'},
            ],
            'trackIds': [],
        },
        {
            'id': 'bridge',
            'number': '07',
            'title': 'Allocate ranking authority per query',
            'change': 'BRIDGE = Broad base order + gated pair evidence + bounded local correction.',
            'outcome': 'Bounded local authority',
            'phases': {
                'design': phase('Design', 'Protect the Broad order and reinterpret every expert as optional pair evidence with explicit rights.', ['return_broad', 'pair_evidence', 'rebind_broad']),
                'build': phase('Build', 'Add query applicability, permission levels, family CAGE specialists and a TPS specialist.', ['dynamic_v4', 'dynamic_v6', 'query_applicability', 'cage_family', 'tps_correction']),
                'test': phase('Test', 'Validate specialists only in their own applicability domain and compare the complete layered system.', ['p450_cage', 'phosphatase_cage', 'terpene_cage', 'layered_cage_eval']),
                'learn': phase('Learn', 'Missing evidence is neutral, Broad remains globally valid, and specialists earn bounded correction rights only where they help.', ['integrated_specialists', 'bridge']),
            },
            'recordIds': ordered(cycle_sets[6]),
            'evidence': [
                {'label': 'BRIDGE', 'value': 'MRR 0.2369 · Hit@10 41.54% · Hit@100 64.92%'},
                {'label': 'P450 CAGE', 'value': 'MRR 0.0370 → 0.0705'},
                {'label': 'Phosphatase CAGE', 'value': 'MRR 0.2522 → 0.3169'},
                {'label': 'Terpene CAGE', 'value': 'MRR 0.0189 → 0.0387'},
            ],
            'trackIds': [],
        },
    ]

    tracks = [
        {
            'id': 'wetlab',
            'label': 'Wet-lab execution',
            'recordIds': ordered(track_sets['wetlab']),
        },
        {
            'id': 'compass',
            'label': 'COMPASS workflow',
            'recordIds': ordered(track_sets['compass']),
        },
    ]

    for cycle in cycles:
        for phase_data in cycle['phases'].values():
            unknown = [node_id for node_id in phase_data['keyIds'] if node_id not in by]
            if unknown:
                raise ValueError(f"unknown phase records in {cycle['id']}: {unknown}")
    covered = set().union(*(set(c['recordIds']) for c in cycles), *(set(t['recordIds']) for t in tracks))
    missing = set(by) - covered
    if missing:
        raise ValueError(f'cycle presentation lost records: {sorted(missing)}')
    return cycles, tracks


def build_architecture() -> dict:
    return {
        'formula': 'S_BRIDGE(q,e) = S_Broad(q,e) + Σ_k g_k(q) Δ_k(q,e)',
        'base': {
            'title': 'Broad Retrieval',
            'body': 'Universal candidate generator and stable global ranker. It remains valid when every optional expert is silent.',
        },
        'control': {
            'title': 'Applicability + permission',
            'body': 'For each query and direction, decide whether an expert is relevant, supported and allowed to modify rank.',
        },
        'experts': [
            {'id':'functional','title':'Functional / evolutionary','members':['EnzGFM with reaction features'],'body':'General learned evidence; admitted only where it adds clean ranking information.'},
            {'id':'structural','title':'Structural','members':['CLIPZyme','cached pocket / structure support'],'body':'Structure is bounded evidence with explicit support limits.'},
            {'id':'mechanistic','title':'Mechanistic','members':['bounded reaction-center correction','TPS specialist'],'body':'Mechanistic signals act only when their applicability tests pass.'},
            {'id':'context','title':'Context','members':['known-positive seed context','multi-seed context'],'body':'Optional query evidence; no seed means a silent channel.'},
            {'id':'family','title':'Family-specific CAGE','members':['P450','phosphatase','terpene'],'body':'Reaction-only generic→family response and family agreement activate a matching fine-tuned CAGE specialist.'},
        ],
        'correction': {
            'title': 'Bounded correction',
            'body': 'Experts act as pair evidence with direction-specific rights; outside their scope, Broad order is preserved.',
        },
        'policyNotes': [
            'Unavailable or inapplicable evidence contributes exactly zero.',
            'Family-finetuned CAGE is a specialist signal, never global CAGE authority.',
            'Outside the reranked shortlist, preserve Broad order exactly.',
        ],
    }


def write_data(cycles: list[dict], tracks: list[dict], architecture: dict) -> None:
    payload = {
        'nodes': NODES,
        'crossLinks': CROSSLINKS,
        'families': FAMILY_LABELS,
        'cycles': cycles,
        'tracks': tracks,
        'architecture': architecture,
        'meta': {
            'schema': 'bridge-engineering-cycles-v5',
            'root': 'enzymecage',
            'current': 'bridge',
            'presentation': 'dbtl-spine',
        },
    }
    DATA_JS.write_text('window.LINEAGE_DATA = ' + json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + ';\n', encoding='utf-8')


def write_doc(by: dict[str, dict], cycles: list[dict], tracks: list[dict], architecture: dict) -> None:
    lines = [
        '# BRIDGE Engineering Cycles',
        '',
        'The Engineering story is organized around seven Design → Build → Test → Learn cycles.',
        'Each cycle records the question, the implementation, the evidence and the lesson that changed the next design.',
        'The full canonical experiment inventory is retained below each cycle, while the public page keeps it behind progressive disclosure.',
        '',
    ]
    for cycle in cycles:
        lines += [f"## Cycle {cycle['number']} · {cycle['title']}", '', f"**What changed:** {cycle['change']}", '']
        for key in ['design','build','test','learn']:
            p = cycle['phases'][key]
            lines += [f"### {p['title']}", '', p['summary'], '']
        if cycle['evidence']:
            lines += ['### Key evidence', '']
            for item in cycle['evidence']:
                lines.append(f"- **{item['label']}:** {item['value']}")
            lines.append('')
        lines += ['### Full record', '']
        for node_id in cycle['recordIds']:
            node = by[node_id]
            lines.append(f"- **{node['label']}** `[{node['status'].upper()}]` — {node['result']} _{node['legacy']}_")
        lines.append('')

    lines += ['## Parallel tracks', '']
    for track in tracks:
        lines += [f"### {track['label']}", '']
        for node_id in track['recordIds']:
            node = by[node_id]
            lines.append(f"- **{node['label']}** — {node['result']}")
        lines.append('')

    lines += ['## BRIDGE today', '', f"`{architecture['formula']}`", '']
    lines.append(f"- **Base:** {architecture['base']['title']} — {architecture['base']['body']}")
    lines.append(f"- **Control:** {architecture['control']['title']} — {architecture['control']['body']}")
    for expert in architecture['experts']:
        lines.append(f"- **{expert['title']}:** {', '.join(expert['members'])}. {expert['body']}")
    lines.append(f"- **Correction:** {architecture['correction']['title']} — {architecture['correction']['body']}")
    lines += ['', '## Source of truth', '', 'The canonical historical inventory remains in `scripts/engineering_lineage/lineage_data.py`.']
    DOC.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    by, children = validate()
    cycles, tracks = build_cycles(by, children)
    architecture = build_architecture()
    write_data(cycles, tracks, architecture)
    write_doc(by, cycles, tracks, architecture)
    represented = set().union(*(set(c['recordIds']) for c in cycles), *(set(t['recordIds']) for t in tracks))
    print(json.dumps({
        'canonical_records': len(NODES),
        'represented_records': len(represented),
        'cycles': len(cycles),
        'parallel_tracks': len(tracks),
        'schema': 'bridge-engineering-cycles-v5',
        'data_js': str(DATA_JS.relative_to(ROOT)),
        'doc': str(DOC.relative_to(ROOT)),
    }, indent=2))


if __name__ == '__main__':
    main()
