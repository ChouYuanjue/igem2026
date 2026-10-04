from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from scripts.engineering_lineage.lineage_data import CROSSLINKS, FAMILY_LABELS, NODES

ROOT = Path(__file__).resolve().parents[2]
DATA_JS = ROOT / 'frontend' / 'engineering_lineage' / 'engineering' / 'data.js'
DOC = ROOT / 'projects' / 'active' / 'bridge' / 'docs' / 'engineering.md'


def validate() -> tuple[dict[str, dict], dict[str, list[str]]]:
    by = {n['id']: n for n in NODES}
    if len(by) != len(NODES):
        raise ValueError('duplicate lineage node id')
    children: dict[str, list[str]] = defaultdict(list)
    roots = []
    for n in NODES:
        if n['parent'] is None:
            roots.append(n['id'])
        else:
            if n['parent'] not in by:
                raise ValueError(f"unknown parent {n['parent']} for {n['id']}")
            children[n['parent']].append(n['id'])
    if roots != ['enzymecage']:
        raise ValueError(f'unexpected roots: {roots}')
    return by, children


def subtree(root: str, children: dict[str, list[str]]) -> set[str]:
    out: set[str] = set()
    def walk(i: str) -> None:
        out.add(i)
        for c in children.get(i, []):
            walk(c)
    walk(root)
    return out


def before(root: str, cuts: list[str], children: dict[str, list[str]]) -> set[str]:
    out = subtree(root, children)
    for cut in cuts:
        out -= subtree(cut, children) - {cut}
    return out


def ordered(ids: set[str]) -> list[str]:
    return [n['id'] for n in NODES if n['id'] in ids]


def phase(label: str, short: str, key_ids: list[str]) -> dict:
    return {'label': label, 'short': short, 'keyIds': key_ids}


def cycle(
    cycle_id: str,
    title: str,
    outcome: str,
    phases: dict[str, dict],
    record_ids: set[str] | list[str],
    *,
    size: str = 'medium',
    micro: list[dict] | None = None,
) -> dict:
    return {
        'id': cycle_id,
        'title': title,
        'outcome': outcome,
        'size': size,
        'phases': phases,
        'recordIds': ordered(set(record_ids)) if not isinstance(record_ids, list) else record_ids,
        'micro': micro or [],
    }


def build_atlas(by: dict[str, dict], children: dict[str, list[str]]) -> tuple[list[dict], list[dict], dict]:
    wetlab = subtree('wetlab_program', children)
    compass = subtree('user_semantic_routing', children)

    stage_a_records = before('enzymecage', ['broad'], children) - wetlab
    stage_a_records.add('broad')
    broad_descendants = subtree('broad', children) - subtree('fibre', children) - subtree('return_broad', children) - compass
    stage_d_records = subtree('fibre', children)
    stage_e_records = subtree('return_broad', children)

    early_cycle = before('enzymecage', ['open_problem'], children)
    open_cycle = (stage_a_records - early_cycle) | {'open_problem', 'broad'}

    generalization = subtree('generalization_program', children) | {'budget_routing'}
    stress = subtree('stress_program', children)
    stage_b_records = generalization | stress | {'broad'}
    stage_c_records = broad_descendants - stage_b_records | {'broad', 'fusion_program', 'bime'}

    functional = subtree('enzgfm', children) | {'functional_proto'}
    structural = subtree('reaction_center', children) | subtree('top2000', children) | subtree('clipzyme', children) | subtree('reactzyme', children)
    fusion = subtree('fusion_program', children) - subtree('fibre', children) - subtree('return_broad', children)

    relation_geometry = subtree('bio_relation', children) | subtree('context_domain', children) | subtree('tensor_field', children) | subtree('catalytic_kernel', children)
    conditional = subtree('conditional_modes', children)
    scientific = subtree('scientific_evidence', children)
    relational = subtree('hcm', children) | subtree('eram', children) | subtree('plugins', children)

    permissions = {'pair_evidence','rebind_broad','score_evidence','expert_types','dynamic_v4','dynamic_v6','query_applicability'}
    family = subtree('cage_family', children)
    tps = subtree('tps_correction', children)
    integration = subtree('integrated_specialists', children)

    stages = [
        {
            'id': 'open-retrieval',
            'index': 'A',
            'title': 'From bounded structure ranking to open retrieval',
            'summary': 'Two compact loops establish the problem and the Broad retrieval base. The early work stays intentionally small in the atlas.',
            'weight': 'compact',
            'layout': 'serial',
            'entry': 'EnzymeCAGE',
            'exit': 'Broad Retrieval',
            'cycles': [
                cycle('a1-library-scale','Can structure ranking scale to a real library?','Hard candidate gates cap recall',{
                    'design': phase('Design','audit pocket robustness',['enzymecage','pocket_audit']),
                    'build': phase('Build','full-library scoring + transfer',['full_library_structure','reaction_transfer','closed_pool']),
                    'test': phase('Test','rank + gate coverage',['full_library_structure','gate_coverage_ceiling']),
                    'learn': phase('Learn','missed candidates stay lost',['gate_coverage_ceiling','open_problem']),
                }, early_cycle, size='small'),
                cycle('a2-open-world','Can unseen reactions and proteins enter ranking directly?','Broad becomes the open-world base order',{
                    'design': phase('Design','molecular-input scoring',['open_problem','candidate_program','representation_program']),
                    'build': phase('Build','registry + dual tower',['registry','dual_tower','pu_mask','hard_negatives']),
                    'test': phase('Test','double-cold stress tests',['topk_surrogate','dual_kernel_marts','cycle']),
                    'learn': phase('Learn','protect Broad order',['broad','budget_routing']),
                }, open_cycle, size='medium'),
            ],
            'recordIds': ordered(stage_a_records),
            'trackIds': [],
        },
        {
            'id': 'broad-stability',
            'index': 'B',
            'title': 'Stabilize Broad before adding more capability',
            'summary': 'Broad is now the incumbent. The next loop asks whether it can expand to new domains without losing the ranking behavior that made open retrieval work.',
            'weight': 'medium',
            'layout': 'serial',
            'entry': 'Broad Retrieval',
            'exit': 'Protected Broad core',
            'cycles': [
                cycle('b1-retention','Can Broad adapt without forgetting?','A strong base should be protected, not repeatedly rewritten',{
                    'design': phase('Design','extend without forgetting',['generalization_program','directional_cont']),
                    'build': phase('Build','replay + retention methods',['replay_anchor','checkpoint_blend','score_distill']),
                    'test': phase('Test','external + temporal tests',['enzyme405','orphan335','temporal','rhea_transfer']),
                    'learn': phase('Learn','route around the incumbent',['posthoc_router']),
                }, stage_b_records, size='large'),
            ],
            'evidenceIds': ordered(stress),
            'recordIds': ordered(stage_b_records),
            'trackIds': ['wetlab','compass'],
        },
        {
            'id': 'expertization',
            'index': 'C',
            'title': 'Turn heterogeneous evidence into explicit experts',
            'summary': 'Functional, structural and ranking loops run in parallel. Their shared result is BiME-Rank: Broad stays protected while additional capabilities are admitted as experts.',
            'weight': 'major',
            'layout': 'parallel',
            'entry': 'Protected Broad core',
            'exit': 'BiME-Rank',
            'cycles': [
                cycle('c1-functional','Can functional models add broad evidence?','Functional evidence helps, but remains optional',{
                    'design': phase('Design','add functional evidence',['enzgfm']),
                    'build': phase('Build','EnzGFM + reaction features',['enzgfm_rdkit','enzgfm_rdkitplus']),
                    'test': phase('Test','frozen retrieval tests',['enzyme405','orphan335']),
                    'learn': phase('Learn','expert, not base ranker',['enzgfm_rdkitplus']),
                }, functional),
                cycle('c2-structure','Can structure and mechanism improve ranking safely?','Structure and mechanism help only inside supported regions',{
                    'design': phase('Design','add structural evidence',['reaction_center','clipzyme']),
                    'build': phase('Build','bounded local corrections',['center_v1','center_identity','center_v3','clip_fallback','bounded_top2000']),
                    'test': phase('Test','support + fresh transfer',['clip_support','rhea_transfer']),
                    'learn': phase('Learn','gate by applicability',['center_v3','clip_fallback']),
                }, structural),
                cycle('c3-fusion','Can multiple experts improve ranking without breaking Broad?','BiME stabilizes a portfolio, but global admission is still too coarse',{
                    'design': phase('Design','protect Broad + add experts',['fusion_program','portfolio']),
                    'build': phase('Build','union + anchored LambdaRank',['candidate_union','r2e_lambdarank','e2r_anchor']),
                    'test': phase('Test','frozen expert admission',['admission','generic_cage_expert','seed_context']),
                    'learn': phase('Learn','query-specific usefulness',['bime','generic_cage_expert']),
                }, fusion, size='large'),
            ],
            'recordIds': ordered(stage_c_records),
            'trackIds': [],
        },
        {
            'id': 'fibre-detour',
            'index': 'D',
            'title': 'FIBRE detour',
            'summary': 'A concentrated attempt to replace the expert stack with one relational core. The detour contributes useful principles, then returns to Broad.',
            'weight': 'detour',
            'layout': 'nested',
            'entry': 'BiME-Rank',
            'exit': 'Return to Broad',
            'macro': {
                'design': 'unify relation, mechanism and evidence',
                'build': 'relational geometry + evidence + adaptive core',
                'test': 'strict temporal / double-cold replacement',
                'learn': 'keep interfaces; restore Broad authority',
            },
            'cycles': [
                cycle('d1-relations','Relational formulation','Useful geometry emerges, but forced symmetry and many aggregations fail',{
                    'design': phase('D','formalize relations',['bio_relation','context_domain']),
                    'build': phase('B','geometry + conditionals',['tensor_field','catalytic_kernel','conditional_modes']),
                    'test': phase('T','compare formulations',['linear_expectation','logmeanexp','second_order']),
                    'learn': phase('L','retain directional evidence',['linear_expectation','scientific_evidence']),
                }, relation_geometry | conditional, size='medium', micro=[
                    {'label':'symmetric / Gibbs / KL / mixture','result':'rejected'},
                    {'label':'directional conditional expectation','result':'local winner'},
                ]),
                cycle('d2-evidence','Scientific evidence','Structure, context and reaction-center signals work better as admitted evidence',{
                    'design': phase('D','anchor explicit evidence',['scientific_evidence']),
                    'build': phase('B','structure + context + center',['structure_evidence','known_context','rc_evidence']),
                    'test': phase('T','calibrate evidence support',['evidence_admission']),
                    'learn': phase('L','missing evidence stays neutral',['evidence_admission']),
                }, scientific),
                cycle('d3-relational-core','Can the relational core replace Broad?','No: retain plugins and fallback, abandon global replacement',{
                    'design': phase('D','query-adaptive relations',['hcm','query_mix']),
                    'build': phase('B','post-hoc gate + ERAM + plugins',['query_mix_posthoc','eram','plugins']),
                    'test': phase('T','strict replacement test',['relational_main','temporal_relational']),
                    'learn': phase('L','restore Broad authority',['open_fallback','return_broad']),
                }, relational, size='large'),
            ],
            'recordIds': ordered(stage_d_records),
            'trackIds': [],
        },
        {
            'id': 'bridge-formation',
            'index': 'E',
            'title': 'Build BRIDGE from locally valid expert authority',
            'summary': 'This is the main final stage. Query permission, family CAGE, TPS specialization and bounded integration develop as sibling loops, then converge into BRIDGE.',
            'weight': 'final',
            'layout': 'converge',
            'entry': 'Return to Broad',
            'exit': 'BRIDGE',
            'cycles': [
                cycle('e1-permission','Query applicability and permission','Expert authority becomes query- and direction-specific',{
                    'design': phase('D','rebind experts to Broad',['pair_evidence','rebind_broad']),
                    'build': phase('B','directional router + expert types',['score_evidence','expert_types','dynamic_v4']),
                    'test': phase('T','availability ≠ usefulness',['dynamic_v6','query_applicability']),
                    'learn': phase('L','only applicable experts act',['query_applicability']),
                }, permissions, size='large'),
                cycle('e2-family-cage','Family-specific CAGE','Generic CAGE fails globally; family CAGE works locally',{
                    'design': phase('D','specialize CAGE by family',['cage_family']),
                    'build': phase('B','fine-tune family specialists',['p450_cage','phosphatase_cage','terpene_cage']),
                    'test': phase('T','family-scoped evaluation',['p450_cage','phosphatase_cage','terpene_cage']),
                    'learn': phase('L','activate local family expert',['cage_family']),
                }, family, size='large', micro=[
                    {'label':'P450','result':'MRR 0.0370 → 0.0705'},
                    {'label':'phosphatase','result':'MRR 0.2522 → 0.3169'},
                    {'label':'terpene','result':'MRR 0.0189 → 0.0387'},
                ]),
                cycle('e3-tps','TPS specialist','Mechanistic TPS evidence stays sparse and local',{
                    'design': phase('D','reuse TPS mechanism evidence',['tps_correction','tps_foundation']),
                    'build': phase('B','gate TPS correction',['tps_correction']),
                    'test': phase('T','matched TPS queries only',['layered_cage_eval']),
                    'learn': phase('L','silent outside TPS niche',['tps_correction']),
                }, tps, size='medium'),
                cycle('e4-integration','Integrated bounded correction','All specialists merge through one bounded correction interface',{
                    'design': phase('D','combine admitted evidence',['integrated_specialists']),
                    'build': phase('B','preserve Broad outside shortlist',['integrated_specialists']),
                    'test': phase('T','full-suite comparison',['layered_cage_eval']),
                    'learn': phase('L','global Broad + local gains',['bridge']),
                }, integration, size='large'),
            ],
            'recordIds': ordered(stage_e_records),
            'trackIds': [],
        },
    ]

    tracks = [
        {'id':'wetlab','label':'Wet-lab execution','recordIds':ordered(wetlab)},
        {'id':'compass','label':'COMPASS workflow','recordIds':ordered(compass)},
    ]

    for stage in stages:
        for c in stage['cycles']:
            for p in c['phases'].values():
                unknown = [i for i in p['keyIds'] if i not in by]
                if unknown:
                    raise ValueError(f"unknown phase ids in {c['id']}: {unknown}")
    covered = set().union(*(set(s['recordIds']) for s in stages), *(set(t['recordIds']) for t in tracks))
    missing = set(by) - covered
    if missing:
        raise ValueError(f'presentation lost canonical records: {sorted(missing)}')

    architecture = {
        'formula':'S_BRIDGE(q,e) = S_Broad(q,e) + Σ_k g_k(q) Δ_k(q,e)',
        'base':'Broad Retrieval',
        'control':'Query applicability + permission',
        'experts':['Functional / evolutionary','Structural','Mechanistic / TPS','Context','Family-specific CAGE'],
        'correction':'Bounded pair-evidence correction',
    }
    return stages, tracks, architecture


def write_data(stages: list[dict], tracks: list[dict], architecture: dict) -> None:
    payload = {
        'nodes': NODES,
        'crossLinks': CROSSLINKS,
        'families': FAMILY_LABELS,
        'stages': stages,
        'tracks': tracks,
        'architecture': architecture,
        'meta': {
            'schema':'bridge-engineering-atlas-v7',
            'root':'enzymecage',
            'current':'bridge',
            'presentation':'hierarchical-dbtl-atlas',
        },
    }
    DATA_JS.write_text('window.LINEAGE_DATA = ' + json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + ';\n', encoding='utf-8')


def write_doc(by: dict[str, dict], stages: list[dict], tracks: list[dict], architecture: dict) -> None:
    lines = ['# BRIDGE Engineering Atlas','', 'The Engineering history is organized as five macro stages. FIBRE is a detour between BiME and the dominant final BRIDGE-formation stage.','']
    for stage in stages:
        lines += [f"## {stage['index']} · {stage['title']}",'',stage['summary'],'']
        for c in stage['cycles']:
            lines += [f"### {c['title']}",'',f"**Learn:** {c['outcome']}",'']
            for key in ['design','build','test','learn']:
                p=c['phases'][key]
                lines.append(f"- **{p['label']}:** {p['short']}")
            lines.append('')
        lines += ['### Full record','']
        for i in stage['recordIds']:
            n=by[i]
            lines.append(f"- **{n['label']}** `[{n['status'].upper()}]` — {n['result']} _{n['legacy']}_")
        lines.append('')
    lines += ['## Parallel tracks','']
    for t in tracks:
        lines += [f"### {t['label']}",'']
        for i in t['recordIds']:
            n=by[i]; lines.append(f"- **{n['label']}** — {n['result']}")
        lines.append('')
    lines += ['## BRIDGE today','',f"`{architecture['formula']}`",'']
    DOC.write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8')


def main() -> None:
    by, children = validate()
    stages, tracks, architecture = build_atlas(by, children)
    write_data(stages, tracks, architecture)
    write_doc(by, stages, tracks, architecture)
    represented=set().union(*(set(s['recordIds']) for s in stages),*(set(t['recordIds']) for t in tracks))
    print(json.dumps({'canonical_records':len(NODES),'represented_records':len(represented),'stages':len(stages),'cycles':sum(len(s['cycles']) for s in stages),'schema':'bridge-engineering-atlas-v7'},indent=2))


if __name__ == '__main__':
    main()
