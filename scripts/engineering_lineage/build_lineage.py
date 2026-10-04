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
        parent = n['parent']
        if parent is None:
            roots.append(n['id'])
        else:
            if parent not in by:
                raise ValueError(f"unknown parent {parent!r} for {n['id']!r}")
            children[parent].append(n['id'])
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


def phase(label: str, text: str, key_ids: list[str]) -> dict:
    return {'label': label, 'text': text, 'keyIds': key_ids}


def loop(loop_id: str, title: str, phases: dict[str, dict], outcome: str, why_next: str, record_ids: set[str] | None = None) -> dict:
    return {
        'id': loop_id,
        'title': title,
        'phases': phases,
        'outcome': outcome,
        'whyNext': why_next,
        'recordIds': ordered(record_ids or set()),
    }


def mini(loop_id: str, title: str, outcome: str, record_ids: set[str], *, children: list[dict] | None = None) -> dict:
    return {
        'id': loop_id,
        'title': title,
        'outcome': outcome,
        'recordIds': ordered(record_ids),
        'children': children or [],
    }


def build_scenes(by: dict[str, dict], children: dict[str, list[str]]) -> tuple[list[dict], list[dict], dict]:
    wetlab = subtree('wetlab_program', children)
    compass = subtree('user_semantic_routing', children)

    early = before('enzymecage', ['open_problem'], children)
    open_world = before('open_problem', ['broad', 'wetlab_program'], children)
    open_world.add('broad')

    broad_desc = subtree('broad', children) - subtree('fibre', children) - subtree('return_broad', children) - compass
    retention = subtree('generalization_program', children) | subtree('stress_program', children) | {'budget_routing', 'broad'}
    experts = broad_desc - retention | {'broad', 'fusion_program', 'bime'}

    fibre = subtree('fibre', children)
    bridge = subtree('return_broad', children)

    functional = subtree('enzgfm', children) | {'functional_proto'}
    structural = subtree('reaction_center', children) | subtree('top2000', children) | subtree('clipzyme', children) | subtree('reactzyme', children)
    context = subtree('seed_context', children) | subtree('multi_seed_context', children) if 'seed_context' in by and 'multi_seed_context' in by else set()
    fusion = subtree('fusion_program', children) - fibre - bridge

    fibre_relation = subtree('bio_relation', children) | subtree('context_domain', children) | subtree('tensor_field', children) | subtree('catalytic_kernel', children)
    fibre_conditional = subtree('conditional_modes', children)
    fibre_evidence = subtree('scientific_evidence', children)
    fibre_core = subtree('hcm', children) | subtree('eram', children) | subtree('plugins', children)

    permission = {'pair_evidence','rebind_broad','score_evidence','expert_types','dynamic_v4','dynamic_v6','query_applicability'}
    family = subtree('cage_family', children)
    tps = subtree('tps_correction', children)
    integration = subtree('integrated_specialists', children)

    scenes = [
        {
            'id': 'candidate-ceiling',
            'number': '01',
            'eyebrow': 'Candidate eligibility',
            'title': 'The first failure was upstream of ranking.',
            'lead': 'EnzymeCAGE could judge pairs inside a bounded candidate system, but a positive excluded by the gate could never be recovered downstream.',
            'stack': {
                'upper': {'title': 'EnzymeCAGE / meta-ranker', 'note': 'owns in-pool ranking'},
                'lower': {'title': 'Similarity candidate gate', 'note': 'decides who can be scored'},
            },
            'cageRole': 'primary upper ranker',
            'broadRole': None,
            'primary': loop(
                'candidate-system',
                'Can better ranking rescue a bounded candidate system?',
                {
                    'design': phase('Design','Audit pocket choice and structural ranking.',['enzymecage','pocket_audit']),
                    'build': phase('Build','Score the full library and add reaction-similarity transfer.',['full_library_structure','reaction_transfer','closed_pool']),
                    'test': phase('Test','Measure full-library rank and candidate-gate coverage.',['full_library_structure','gate_coverage_ceiling']),
                    'learn': phase('Learn','Coverage is a hard ceiling: missed positives never reach the upper ranker.',['gate_coverage_ceiling','open_problem']),
                },
                'Candidate eligibility, not another CAGE tweak, became the bottleneck.',
                'Replace the fixed gate with retrieval from molecular inputs.',
                early,
            ),
            'secondary': [
                mini('pocket-audit','Pocket robustness','Pocket choice was not the dominant failure.', subtree('pocket_audit', children)),
                mini('reaction-transfer','Reaction-neighbour rescue','Transfer improved ranks but remained a closed candidate strategy.', subtree('reaction_transfer', children)),
            ],
            'recordIds': ordered(early),
            'trackIds': [],
        },
        {
            'id': 'broad-under-cage',
            'number': '02',
            'eyebrow': 'Recall layer changes first',
            'title': 'Broad entered below CAGE before it became the base order.',
            'lead': 'The initial goal was conservative: let Broad open the candidate universe while CAGE remained the ideal upper reranker. That architecture exposed a second ceiling in the upper layer itself.',
            'stack': {
                'upper': {'title': 'CAGE upper reranker', 'note': 'still expected to make the final judgement'},
                'lower': {'title': 'Broad Retrieval', 'note': 'opens the candidate universe'},
            },
            'cageRole': 'intended upper reranker',
            'broadRole': 'broad recall layer → increasingly meaningful order',
            'primary': loop(
                'broad-cage-stack',
                'Can Broad handle recall while CAGE keeps ranking authority?',
                {
                    'design': phase('Design','Replace the hard gate, not the upper ranker.',['open_problem','candidate_program','representation_program']),
                    'build': phase('Build','Train open bidirectional retrieval with false-negative and hard-negative protection.',['dual_tower','pu_mask','hard_negatives','marts_adapt']),
                    'test': phase('Test','Compare Broad candidate reach with what generic CAGE can actually score.',['broad','layered_cage_eval']),
                    'learn': phase('Learn','Broad reaches far more positives than a generic CAGE upper layer can support.',['layered_cage_eval','clip_support']),
                },
                'Broad could no longer be treated as recall-only; part of ranking responsibility had to move downward.',
                'Keep Broad\'s meaningful order and make extra evidence conditional instead of universally authoritative.',
                open_world | retention,
            ),
            'secondary': [
                mini('broad-training','Open Broad retrieval','Dual-tower training, hard negatives and domain adaptation built a usable full-space order.', before('representation_program',['broad'],children) | {'broad'}),
                mini('retention-guardrails','Retention guardrails','Replay, blending and distillation protected the incumbent while capabilities were added.', retention),
            ],
            'evidence': [
                {'label':'Broad Top-1000 positive-query coverage','value':'≈77.91%'},
                {'label':'generic CAGE scoreable-positive coverage','value':'≈25.08%'},
            ],
            'recordIds': ordered(open_world | retention),
            'trackIds': ['wetlab','compass'],
        },
        {
            'id': 'experts-bime',
            'number': '03',
            'eyebrow': 'Ranking responsibility is shared',
            'title': 'Broad became the base order; other capabilities became experts.',
            'lead': 'Once Broad itself had a meaningful order, the engineering problem changed from “replace the ranker” to “decide when additional evidence deserves to modify that order.”',
            'stack': {
                'upper': {'title': 'BiME expert layer', 'note': 'admission, routing, context, cost'},
                'lower': {'title': 'Broad base order', 'note': 'protected incumbent ranking'},
            },
            'cageRole': 'generic expert candidate',
            'broadRole': 'global base order',
            'primary': loop(
                'expert-delegation',
                'How should heterogeneous evidence modify a strong Broad order?',
                {
                    'design': phase('Design','Keep Broad as incumbent and externalize extra capabilities.',['fusion_program','portfolio']),
                    'build': phase('Build','Add candidate union, learned fusion and anchored E2R protection.',['candidate_union','r2e_lambdarank','e2r_anchor']),
                    'test': phase('Test','Admit experts only after frozen direction-specific confirmation.',['admission','generic_cage_expert','seed_context']),
                    'learn': phase('Learn','Expert usefulness is conditional on query, direction and support.',['bime','generic_cage_expert']),
                },
                'BiME organized experts, but global expert admission was still too coarse.',
                'Ask whether one relational object can explain and replace the expert stack.',
                experts,
            ),
            'secondary': [
                mini('functional-evidence','Functional / evolutionary evidence','EnzGFM helps as optional evidence, not as the universal base.', functional),
                mini('structural-evidence','Structure / mechanism evidence','Reaction-center and CLIPZyme gains depend on support and applicability.', structural),
                mini('context-evidence','Context evidence','Known-positive seed context helps only when context exists.', context),
                mini('fusion-routing','Fusion and routing','RRF → LambdaRank → anchored E2R → BiME.', fusion),
            ],
            'recordIds': ordered(experts),
            'trackIds': [],
        },
        {
            'id': 'fibre-detour',
            'number': '04',
            'eyebrow': 'Side experiment',
            'title': 'FIBRE tested a cleaner abstraction without replacing the main line.',
            'lead': 'BiME made the stack work, but left a scientific question: what common relation are all these experts estimating? FIBRE explored that question as a side branch, then returned useful principles to the Broad-centered design.',
            'stack': {
                'upper': {'title': 'BiME experts', 'note': 'main line remains intact'},
                'lower': {'title': 'Broad base order', 'note': 'replacement target, not discarded'},
                'side': {'title': 'FIBRE relational core', 'note': 'detour / replacement hypothesis'},
            },
            'cageRole': 'one evidence family among many',
            'broadRole': 'incumbent challenged by FIBRE',
            'primary': loop(
                'fibre-replacement',
                'Can one relational core replace the expert stack and Broad order?',
                {
                    'design': phase('Design','Model enzyme–reaction matching as one relational problem.',['fibre','bio_relation','context_domain']),
                    'build': phase('Build','Explore geometry, conditional modes, evidence admission and ERAM-style cores.',['conditional_modes','scientific_evidence','query_mix_posthoc','eram']),
                    'test': phase('Test','Use strict temporal and double-cold replacement tests.',['relational_main','temporal_relational']),
                    'learn': phase('Learn','Keep evidence admission, plugins and neutral fallback; restore Broad authority.',['plugins','open_fallback','return_broad']),
                },
                'The replacement failed, but its interfaces clarified how optional evidence should enter the final system.',
                'Return to Broad and allocate expert authority locally rather than globally.',
                fibre,
            ),
            'secondary': [
                mini('fibre-relation','Relation geometry','Geometry clarified the object, but did not solve ranking alone.', fibre_relation | fibre_conditional),
                mini('fibre-evidence','Scientific evidence','Structure, context and reaction-center signals worked better as admitted evidence.', fibre_evidence),
                mini('fibre-core','Adaptive relational core','The modern relational core still could not justify replacing Broad.', fibre_core),
            ],
            'recordIds': ordered(fibre),
            'trackIds': [],
            'detour': True,
        },
        {
            'id': 'bridge-authority',
            'number': '05',
            'eyebrow': 'Final authority model',
            'title': 'BRIDGE decides who may change Broad\'s order, where, and by how much.',
            'lead': 'The final design keeps Broad globally valid and gives specialists bounded ranking rights only when the current query and direction support them.',
            'stack': {
                'upper': {'title': 'Gated specialist evidence', 'note': 'family CAGE, TPS, context, functional, structural'},
                'middle': {'title': 'Applicability + permission', 'note': 'query- and direction-specific authority'},
                'lower': {'title': 'Broad base order', 'note': 'global default remains valid'},
            },
            'cageRole': 'family-specific specialist',
            'broadRole': 'global default order',
            'primary': loop(
                'local-authority',
                'Who may alter Broad\'s order for this query?',
                {
                    'design': phase('Design','Rebind every expert to Broad as optional pair evidence.',['pair_evidence','rebind_broad']),
                    'build': phase('Build','Add query-conditioned routing, permission levels and specialist gates.',['dynamic_v4','dynamic_v6','query_applicability']),
                    'test': phase('Test','Validate family CAGE and TPS specialists only inside their applicability domains.',['p450_cage','phosphatase_cage','terpene_cage','layered_cage_eval']),
                    'learn': phase('Learn','Missing evidence is neutral; local experts earn bounded correction rights.',['integrated_specialists','bridge']),
                },
                'Broad stays global; specialists act locally. This is BRIDGE.',
                'The engineering story closes in the current architecture.',
                bridge,
            ),
            'secondary': [
                mini('permission','Query applicability / permission','Availability and usefulness are separated per query and direction.', permission),
                mini('family-cage','Family-specific CAGE','Generic CAGE failed globally; P450, phosphatase and terpene specialists succeed locally.', family, children=[
                    {'label':'P450','result':'MRR 0.0370 → 0.0705'},
                    {'label':'phosphatase','result':'MRR 0.2522 → 0.3169'},
                    {'label':'terpene','result':'MRR 0.0189 → 0.0387'},
                ]),
                mini('tps','TPS specialist','TPS returns as one gated biochemical specialist rather than the whole task.', tps),
                mini('integration','Bounded integration','All admitted specialists merge through one bounded correction interface.', integration),
            ],
            'recordIds': ordered(bridge),
            'trackIds': [],
        },
    ]

    tracks = [
        {'id':'wetlab','label':'Wet-lab execution','recordIds':ordered(wetlab)},
        {'id':'compass','label':'COMPASS workflow','recordIds':ordered(compass)},
    ]

    for scene in scenes:
        for p in scene['primary']['phases'].values():
            unknown = [i for i in p['keyIds'] if i not in by]
            if unknown:
                raise ValueError(f"unknown phase ids in {scene['id']}: {unknown}")

    covered = set().union(*(set(s['recordIds']) for s in scenes), *(set(t['recordIds']) for t in tracks))
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
    return scenes, tracks, architecture


def write_data(scenes: list[dict], tracks: list[dict], architecture: dict) -> None:
    payload = {
        'nodes': NODES,
        'crossLinks': CROSSLINKS,
        'families': FAMILY_LABELS,
        'scenes': scenes,
        'tracks': tracks,
        'architecture': architecture,
        'meta': {
            'schema':'bridge-engineering-scenes-v8',
            'root':'enzymecage',
            'current':'bridge',
            'presentation':'progressive-engineering-scenes',
        },
    }
    DATA_JS.write_text('window.LINEAGE_DATA = ' + json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + ';\n', encoding='utf-8')


def write_doc(by: dict[str, dict], scenes: list[dict], tracks: list[dict], architecture: dict) -> None:
    lines = ['# BRIDGE Engineering Story','', 'The public Engineering page uses five progressive scenes. Each scene shows one system state and one primary DBTL loop; secondary loops remain local to that scene.','']
    for scene in scenes:
        lines += [f"## {scene['number']} · {scene['title']}",'',scene['lead'],'',f"### {scene['primary']['title']}",'']
        for key in ['design','build','test','learn']:
            p=scene['primary']['phases'][key]
            lines.append(f"- **{p['label']}:** {p['text']}")
        lines += ['',f"**Outcome:** {scene['primary']['outcome']}",f"**Why next:** {scene['primary']['whyNext']}",'']
        if scene['secondary']:
            lines += ['### Secondary loops','']
            for c in scene['secondary']:
                lines.append(f"- **{c['title']}** — {c['outcome']}")
            lines.append('')
        lines += ['### Full record','']
        for i in scene['recordIds']:
            n=by[i]
            lines.append(f"- **{n['label']}** `[{n['status'].upper()}]` — {n['result']} _{n['legacy']}_")
        lines.append('')
    lines += ['## BRIDGE today','',f"`{architecture['formula']}`",'']
    DOC.write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8')


def main() -> None:
    by, children = validate()
    scenes, tracks, architecture = build_scenes(by, children)
    write_data(scenes, tracks, architecture)
    write_doc(by, scenes, tracks, architecture)
    represented=set().union(*(set(s['recordIds']) for s in scenes),*(set(t['recordIds']) for t in tracks))
    print(json.dumps({'canonical_records':len(NODES),'represented_records':len(represented),'scenes':len(scenes),'schema':'bridge-engineering-scenes-v8'},indent=2))


if __name__ == '__main__':
    main()
