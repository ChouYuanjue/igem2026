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
    stage_b_records = subtree('broad', children) - subtree('fibre', children) - subtree('return_broad', children) - compass
    stage_c_records = subtree('fibre', children)
    stage_d_records = subtree('return_broad', children)

    early_cycle = before('enzymecage', ['open_problem'], children)
    open_cycle = (stage_a_records - early_cycle) | {'open_problem', 'broad'}

    generalization = subtree('generalization_program', children) | {'budget_routing'}
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
            'summary': 'The early work is compressed into two loops: first expose the candidate-gate ceiling, then remove that ceiling with molecular-input retrieval.',
            'weight': 'compact',
            'layout': 'serial',
            'entry': 'EnzymeCAGE',
            'exit': 'Broad Retrieval',
            'cycles': [
                cycle('a1-library-scale','Can structure ranking scale to a real library?','The hard candidate gate, not pocket choice, limits recall',{
                    'design': phase('Design','verify pocket robustness',['enzymecage','pocket_audit']),
                    'build': phase('Build','score full library + transfer candidates',['full_library_structure','reaction_transfer','closed_pool']),
                    'test': phase('Test','measure rank and pool coverage',['full_library_structure','gate_coverage_ceiling']),
                    'learn': phase('Learn','missed candidates cannot be recovered',['gate_coverage_ceiling','open_problem']),
                }, early_cycle, size='small'),
                cycle('a2-open-world','Can unseen reactions and proteins enter ranking directly?','Broad becomes the open-world base order',{
                    'design': phase('Design','score from molecular inputs',['open_problem','candidate_program','representation_program']),
                    'build': phase('Build','open registry + bidirectional dual tower',['registry','dual_tower','pu_mask','hard_negatives']),
                    'test': phase('Test','double-cold + mechanism + graph alternatives',['topk_surrogate','dual_kernel_marts','cycle']),
                    'learn': phase('Learn','protect the full-space Broad order',['broad','budget_routing']),
                }, open_cycle, size='medium'),
            ],
            'recordIds': ordered(stage_a_records),
            'trackIds': ['wetlab','compass'],
        },
        {
            'id': 'conditional-experts',
            'index': 'B',
            'title': 'Broad meets heterogeneous evidence',
            'summary': 'Four engineering loops ran in parallel. Their common lesson was that extra evidence is useful only under the right support, direction and query.',
            'weight': 'major',
            'layout': 'parallel',
            'entry': 'Broad Retrieval',
            'exit': 'BiME-Rank',
            'cycles': [
                cycle('b1-retention','Can Broad adapt without forgetting?','Parameter updates alone cannot preserve every regime',{
                    'design': phase('Design','extend Broad while preserving old behavior',['generalization_program','directional_cont']),
                    'build': phase('Build','replay, blending, regularization, distillation',['replay_anchor','checkpoint_blend','score_distill']),
                    'test': phase('Test','freeze old and new-domain retrieval',['temporal','rhea_transfer']),
                    'learn': phase('Learn','route domain capability instead of forcing one state',['posthoc_router']),
                }, generalization),
                cycle('b2-functional','Can functional models add broad evidence?','Functional evidence helps, but should remain optional',{
                    'design': phase('Design','add learned functional/evolutionary evidence',['enzgfm']),
                    'build': phase('Build','augment EnzGFM with reaction features',['enzgfm_rdkit','enzgfm_rdkitplus']),
                    'test': phase('Test','compare only on frozen retrieval tasks',['enzyme405','orphan335']),
                    'learn': phase('Learn','keep EnzGFM as an expert, not the base ranker',['enzgfm_rdkitplus']),
                }, functional),
                cycle('b3-structure','Can structure and mechanism improve ranking safely?','Structure and mechanism only help inside supported regions',{
                    'design': phase('Design','add reaction-center and structural evidence',['reaction_center','clipzyme']),
                    'build': phase('Build','bounded center correction + CLIPZyme + shortlist reranking',['center_v1','center_identity','center_v3','clip_fallback','bounded_top2000']),
                    'test': phase('Test','audit support and fresh transfer',['clip_support','rhea_transfer']),
                    'learn': phase('Learn','specialists need explicit applicability',['center_v3','clip_fallback']),
                }, structural),
                cycle('b4-fusion','Can multiple experts be combined without breaking Broad?','BiME stabilizes a portfolio, but global admission is still too coarse',{
                    'design': phase('Design','combine experts around a protected incumbent',['fusion_program','portfolio']),
                    'build': phase('Build','candidate union + LambdaRank + anchored E2R',['candidate_union','r2e_lambdarank','e2r_anchor']),
                    'test': phase('Test','admit experts only after frozen confirmation',['admission','generic_cage_expert','seed_context']),
                    'learn': phase('Learn','expert usefulness is query- and direction-dependent',['bime','generic_cage_expert']),
                }, fusion, size='large'),
            ],
            'evidenceIds': ordered(subtree('stress_program', children)),
            'recordIds': ordered(stage_b_records),
            'trackIds': [],
        },
        {
            'id': 'fibre-detour',
            'index': 'C',
            'title': 'FIBRE: can one relational core replace the expert stack?',
            'summary': 'This was one large redesign loop containing several nested loops. Useful principles survived; the replacement model did not.',
            'weight': 'major',
            'layout': 'nested',
            'entry': 'BiME-Rank',
            'exit': 'Return to Broad',
            'macro': {
                'design': 'unify relation, mechanism and evidence',
                'build': 'geometry + conditional modes + evidence + relational core',
                'test': 'strict temporal / double-cold replacement',
                'learn': 'keep evidence interfaces; restore Broad authority',
            },
            'cycles': [
                cycle('c1-geometry','Relation geometry','Geometry organizes the question, but does not solve ranking alone',{
                    'design': phase('D','formalize biological relation',['bio_relation','context_domain']),
                    'build': phase('B','tensor/product and catalytic geometry',['tensor_field','catalytic_kernel']),
                    'test': phase('T','check whether geometry yields stable conditional ranking',['interaction_atlas','atlas_gluing']),
                    'learn': phase('L','move more scientific evidence out of latent geometry',['scientific_evidence']),
                }, relation_geometry, size='small'),
                cycle('c2-conditional','Conditional formulations','Many elegant aggregations failed; directional conditional expectation was the local survivor',{
                    'design': phase('D','make asymmetric experts mathematically compatible',['conditional_modes']),
                    'build': phase('B','test symmetric, Gibbs, KL, mixture and variance forms',['symmetric_potential','gibbs','kl_bary','normalized_mix']),
                    'test': phase('T','compare on frozen directional ranking',['linear_expectation','logmeanexp','second_order']),
                    'learn': phase('L','keep directional conditioning, drop forced symmetry',['linear_expectation']),
                }, conditional, size='medium', micro=[
                    {'label':'symmetric / Gibbs / KL / mixture','result':'rejected'},
                    {'label':'directional conditional expectation','result':'local winner'},
                ]),
                cycle('c3-evidence','Scientific evidence','Structure, context and reaction-center signals work better as admitted evidence',{
                    'design': phase('D','anchor experts in explicit scientific evidence',['scientific_evidence']),
                    'build': phase('B','structure + known-positive + reaction-center channels',['structure_evidence','known_context','rc_evidence']),
                    'test': phase('T','calibrate support before allowing evidence to act',['evidence_admission']),
                    'learn': phase('L','evidence needs admission and missing-neutral semantics',['evidence_admission']),
                }, scientific),
                cycle('c4-relational','Adaptive relational core','A modern relational core still cannot justify replacing Broad',{
                    'design': phase('D','learn query-adaptive expert relations',['hcm','query_mix']),
                    'build': phase('B','frozen post-hoc gate + ERAM core + plugins',['query_mix_posthoc','eram','plugins']),
                    'test': phase('T','strict temporal and double-cold replacement test',['relational_main','temporal_relational']),
                    'learn': phase('L','retain plugins and fallback; abandon global replacement',['open_fallback','return_broad']),
                }, relational, size='large'),
            ],
            'recordIds': ordered(stage_c_records),
            'trackIds': [],
        },
        {
            'id': 'bridge-formation',
            'index': 'D',
            'title': 'BRIDGE: converge only the loops that earned local authority',
            'summary': 'The final stage is a convergence. Several small loops run in parallel, then merge into one protected Broad + gated specialist architecture.',
            'weight': 'major',
            'layout': 'converge',
            'entry': 'Return to Broad',
            'exit': 'BRIDGE',
            'cycles': [
                cycle('d1-permission','Query applicability and permission','Expert authority becomes query- and direction-specific',{
                    'design': phase('D','rebind every expert to Broad',['pair_evidence','rebind_broad']),
                    'build': phase('B','directional evidence + expert types + router',['score_evidence','expert_types','dynamic_v4']),
                    'test': phase('T','separate availability from usefulness',['dynamic_v6','query_applicability']),
                    'learn': phase('L','only applicable experts may move rank',['query_applicability']),
                }, permissions, size='large'),
                cycle('d2-family-cage','Family-specific CAGE','Generic CAGE fails globally; family CAGE works locally',{
                    'design': phase('D','turn CAGE into a family specialist',['cage_family']),
                    'build': phase('B','fine-tune P450, phosphatase and terpene specialists',['p450_cage','phosphatase_cage','terpene_cage']),
                    'test': phase('T','evaluate each family only in its applicability domain',['p450_cage','phosphatase_cage','terpene_cage']),
                    'learn': phase('L','family response should activate a specialist, not global CAGE authority',['cage_family']),
                }, family, size='large', micro=[
                    {'label':'P450','result':'MRR 0.0370 → 0.0705'},
                    {'label':'phosphatase','result':'MRR 0.2522 → 0.3169'},
                    {'label':'terpene','result':'MRR 0.0189 → 0.0387'},
                ]),
                cycle('d3-tps','TPS specialist','Mechanistic TPS evidence stays sparse and local',{
                    'design': phase('D','reuse TPS-specific mechanistic evidence',['tps_correction','tps_foundation']),
                    'build': phase('B','gate a bounded TPS correction',['tps_correction']),
                    'test': phase('T','activate only on matched TPS queries',['layered_cage_eval']),
                    'learn': phase('L','keep the specialist silent outside its niche',['tps_correction']),
                }, tps, size='medium'),
                cycle('d4-integration','Integrated bounded correction','All specialists merge only through a bounded correction interface',{
                    'design': phase('D','combine admitted pair evidence',['integrated_specialists']),
                    'build': phase('B','preserve Broad outside the reranked shortlist',['integrated_specialists']),
                    'test': phase('T','run layered full-suite comparison',['layered_cage_eval']),
                    'learn': phase('L','Broad remains global; local specialists provide the gain',['bridge']),
                }, integration, size='large'),
            ],
            'recordIds': ordered(stage_d_records),
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
            'schema':'bridge-engineering-atlas-v6',
            'root':'enzymecage',
            'current':'bridge',
            'presentation':'hierarchical-dbtl-atlas',
        },
    }
    DATA_JS.write_text('window.LINEAGE_DATA = ' + json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + ';\n', encoding='utf-8')


def write_doc(by: dict[str, dict], stages: list[dict], tracks: list[dict], architecture: dict) -> None:
    lines = ['# BRIDGE Engineering Atlas','', 'The Engineering history is organized as four macro stages containing serial, parallel and nested DBTL loops.','']
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
    DOC.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main() -> None:
    by, children = validate()
    stages, tracks, architecture = build_atlas(by, children)
    write_data(stages, tracks, architecture)
    write_doc(by, stages, tracks, architecture)
    represented=set().union(*(set(s['recordIds']) for s in stages),*(set(t['recordIds']) for t in tracks))
    print(json.dumps({'canonical_records':len(NODES),'represented_records':len(represented),'stages':len(stages),'cycles':sum(len(s['cycles']) for s in stages),'schema':'bridge-engineering-atlas-v6'},indent=2))


if __name__ == '__main__':
    main()
