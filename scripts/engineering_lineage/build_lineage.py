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
    for node_id in by:
        seen = set()
        cur = node_id
        while cur is not None:
            if cur in seen:
                raise ValueError(f'cycle through {node_id}')
            seen.add(cur)
            cur = by[cur]['parent']
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


def group_records(
    roots: list[str],
    chapter_ids: set[str],
    children: dict[str, list[str]],
    *,
    exclude: list[str] | None = None,
    omit: list[str] | None = None,
) -> list[str]:
    selected: set[str] = set()
    for root in roots:
        selected |= subtree(root, children)
    for cut in exclude or []:
        selected -= subtree(cut, children)
    selected &= chapter_ids
    selected -= set(omit or [])
    return ordered(selected)


def build_story(by: dict[str, dict], children: dict[str, list[str]]) -> dict:
    chapter_sets = {
        'closed': before('enzymecage', ['open_problem'], children),
        'open': before('open_problem', ['broad', 'wetlab_program'], children),
        'broad': before('broad', ['bime', 'user_semantic_routing'], children),
        'bime': before('bime', ['fibre', 'return_broad'], children),
        'fibre': subtree('fibre', children),
        'bridge': subtree('return_broad', children),
    }
    track_sets = {
        'compass': subtree('user_semantic_routing', children),
        'wetlab': subtree('wetlab_program', children),
    }

    chapters = [
        {
            'id': 'closed',
            'number': '01',
            'eyebrow': 'The first constraint',
            'title': 'Structure ranking worked only after the right candidates were already present.',
            'problem': 'We began by trying to improve EnzymeCAGE for TPS-like retrieval. Pocket choice was worth checking, but the larger failure appeared when structural pair scores had to search an entire protein library.',
            'learn': 'Reaction-similarity transfer made the bounded task much better, yet every downstream ranker still inherited the candidate gate. On the full relation set, only about 43.98% of known enzyme records entered that gate.',
            'decision': 'Candidate coverage became the scientific bottleneck. The next system had to admit new reactions and proteins from molecular inputs rather than from membership in a precomputed relation table.',
            'pathIds': ['enzymecage', 'pocket_audit', 'full_library_structure', 'reaction_transfer', 'closed_pool', 'gate_coverage_ceiling', 'open_problem'],
            'groups': [
                {
                    'id': 'closed-pool-engineering',
                    'title': 'Inside the bounded candidate system',
                    'summary': 'We tried to improve recall, structural ranking and rescue without changing the closed-world assumption.',
                    'recordIds': group_records(['recall_union', 'cage_pool_rank', 'rf_hgb', 'rescue_slots'], chapter_sets['closed'], children),
                },
            ],
            'evidenceIds': ['pocket_audit', 'full_library_structure', 'gate_coverage_ceiling'],
        },
        {
            'id': 'open',
            'number': '02',
            'eyebrow': 'Change the retrieval problem',
            'title': 'Open-world retrieval replaced candidate gating with a full-space order.',
            'problem': 'Once unseen enzymes or reactions had to enter the system, candidate generation could no longer be a hidden preprocessing step. It had to become a scored retrieval problem of its own.',
            'learn': 'We explored candidate registries, protein/reaction representations, TPS-specific mechanism cues, uncertainty and graph propagation in parallel. The stable common denominator was a bidirectional molecular representation that could rank the full candidate universe even when optional evidence was absent.',
            'decision': 'Broad Retrieval became the default global order. Specialized evidence could still exist, but it could no longer define who was allowed into the search space.',
            'pathIds': ['open_problem', 'representation_program', 'dual_tower', 'broad'],
            'groups': [
                {
                    'id': 'candidate-space',
                    'title': 'Candidate universe',
                    'summary': 'Ways to admit new entities without silently losing positives.',
                    'recordIds': group_records(['candidate_program'], chapter_sets['open'], children, exclude=['wetlab_program']),
                },
                {
                    'id': 'broad-representation',
                    'title': 'Broad representations and training',
                    'summary': 'Learn a continuous reaction–enzyme order directly from molecular inputs.',
                    'recordIds': group_records(['representation_program'], chapter_sets['open'], children, exclude=['broad']),
                },
                {
                    'id': 'tps-mechanism',
                    'title': 'TPS mechanism as a stress case',
                    'summary': 'Biologically plausible local features were useful diagnostics, but most did not survive broad frozen evaluation.',
                    'recordIds': group_records(['tps_mech_program'], chapter_sets['open'], children),
                },
                {
                    'id': 'reliability',
                    'title': 'Reliability and applicability',
                    'summary': 'Estimate when a prediction or evidence source should be trusted rather than forcing every signal into rank.',
                    'recordIds': group_records(['evidence_program'], chapter_sets['open'], children),
                },
                {
                    'id': 'graph-alternatives',
                    'title': 'Graph and non-parametric alternatives',
                    'summary': 'Propagation and kernels were tested as alternate retrieval mechanisms and local rescue signals.',
                    'recordIds': group_records(['graph_program'], chapter_sets['open'], children),
                },
            ],
            'evidenceIds': ['uniprot_free', 'hard_model', 'topk_surrogate', 'horizyn_mlnce', 'dual_kernel_marts'],
        },
        {
            'id': 'broad',
            'number': '03',
            'eyebrow': 'Broad is strong, evidence is heterogeneous',
            'title': 'The next problem was no longer coverage. It was how to use extra evidence without damaging a strong base order.',
            'problem': 'Broad gave us a stable full-space ranking, but EnzGFM, structural models, reaction-center signals, context and TPS knowledge still contained complementary information. Direct continuation and fixed fusion repeatedly traded one regime for another.',
            'learn': 'The useful signals differed by direction, support and query. Routing and anchored learning-to-rank worked better than unconditional fusion. External and temporal tests also made clear that validation had to remain separate from model selection.',
            'decision': 'We moved from “one model should absorb everything” toward an explicit portfolio of experts around a protected broad retriever. That architecture became BiME-Rank.',
            'pathIds': ['broad', 'fusion_program', 'portfolio', 'r2e_lambdarank', 'bime'],
            'groups': [
                {
                    'id': 'generalization',
                    'title': 'Generalization and retention',
                    'summary': 'Attempts to expand Broad while preserving its existing ranking behavior.',
                    'recordIds': group_records(['budget_routing', 'generalization_program'], chapter_sets['broad'], children),
                },
                {
                    'id': 'expert-evidence',
                    'title': 'Candidate expert evidence',
                    'summary': 'Functional, structural and mechanistic signals tested as additions to Broad.',
                    'recordIds': group_records(['expert_program'], chapter_sets['broad'], children),
                },
                {
                    'id': 'fusion',
                    'title': 'From score fusion to learned routing',
                    'summary': 'How independent evidence was combined before BiME-Rank.',
                    'recordIds': group_records(['fusion_program'], chapter_sets['broad'], children, omit=['fusion_program', 'portfolio', 'r2e_lambdarank', 'bime']),
                },
            ],
            'evidenceGroup': {
                'title': 'Frozen evidence that constrained the design',
                'recordIds': group_records(['stress_program'], chapter_sets['broad'], children),
            },
            'evidenceIds': ['enzyme405', 'orphan335', 'rhea_transfer', 'temporal'],
        },
        {
            'id': 'bime',
            'number': '04',
            'eyebrow': 'BiME reveals the remaining abstraction error',
            'title': 'Expert admission was useful, but “globally admitted expert” was still too coarse.',
            'problem': 'BiME organized multiple capabilities, candidate union, cost and fallback. Its remaining weakness was conceptual: an expert could pass globally and still be harmful for a particular query or direction.',
            'learn': 'Anchored E2R rescue protected a strong baseline; generic CAGE degraded ranking while seed and structural context helped only under specific support. These results pointed toward conditional expert rights rather than one global admission decision.',
            'decision': 'Two routes were tested from here: unify the whole interaction into one relational model (FIBRE), or keep Broad authoritative and make expert rights increasingly local. The FIBRE detour was large enough to deserve its own chapter.',
            'pathIds': ['bime', 'fibre', 'return_broad'],
            'groups': [
                {
                    'id': 'directional-protection',
                    'title': 'Protect the strong direction-specific baseline',
                    'summary': 'E2R experiments showed why unconstrained expert fusion could not own the entire ranking.',
                    'recordIds': group_records(['e2r_branch'], chapter_sets['bime'], children),
                },
                {
                    'id': 'admission',
                    'title': 'Which experts deserve ranking authority?',
                    'summary': 'Formal admission, context evidence and structural experts exposed the difference between availability and usefulness.',
                    'recordIds': group_records(['admission'], chapter_sets['bime'], children),
                },
                {
                    'id': 'execution-policy',
                    'title': 'Execution and incumbent protection',
                    'summary': 'Cost-aware execution and strong-baseline absorption became explicit system policies.',
                    'recordIds': group_records(['cost_hierarchy', 'strong_base_policy'], chapter_sets['bime'], children),
                },
            ],
            'evidenceIds': ['e2r_four', 'e2r_anchor', 'generic_cage_expert'],
        },
        {
            'id': 'fibre',
            'number': '05',
            'eyebrow': 'The large detour',
            'title': 'FIBRE tested whether one relational geometry could replace the multi-module stack.',
            'problem': 'The expert stack felt inelegant, so we tried to represent biological relations, mechanisms and heterogeneous evidence inside a unified relational formulation.',
            'learn': 'The detour produced useful ideas—explicit scientific evidence, frozen-core gating, pluggable adapters and missing-neutral fallback—but repeated conditional-mode and relational-core experiments did not displace Broad under strict generalization tests.',
            'decision': 'The failed replacement clarified the final abstraction: preserve the global Broad order and let heterogeneous evidence act through explicit, query-conditioned interfaces.',
            'pathIds': ['fibre', 'scientific_evidence', 'query_mix_posthoc', 'relational_main', 'open_fallback'],
            'groups': [
                {
                    'id': 'relation-geometry',
                    'title': 'Relation geometry',
                    'summary': 'Ways to encode biological relation, catalytic context and interaction structure.',
                    'recordIds': group_records(['bio_relation', 'context_domain', 'tensor_field', 'catalytic_kernel'], chapter_sets['fibre'], children),
                },
                {
                    'id': 'conditional-modes',
                    'title': 'Conditional modes',
                    'summary': 'A dense series of attempts to reconcile asymmetric experts inside one probabilistic/conditional formulation.',
                    'recordIds': group_records(['conditional_modes'], chapter_sets['fibre'], children),
                },
                {
                    'id': 'explicit-evidence',
                    'title': 'Explicit scientific evidence',
                    'summary': 'Move structure, context and mechanism outside latent geometry and require evidence admission.',
                    'recordIds': group_records(['scientific_evidence'], chapter_sets['fibre'], children),
                },
                {
                    'id': 'adaptive-mixture',
                    'title': 'Adaptive expert mixtures',
                    'summary': 'Query-conditioned mixtures worked safely only when the strong core was protected.',
                    'recordIds': group_records(['hcm'], chapter_sets['fibre'], children),
                },
                {
                    'id': 'relational-replacement',
                    'title': 'Relational-core replacement',
                    'summary': 'A modern relational core was implemented and tested as a possible universal ranker.',
                    'recordIds': group_records(['eram'], chapter_sets['fibre'], children),
                },
                {
                    'id': 'plugins',
                    'title': 'Pluggable evidence',
                    'summary': 'Modularity and open-world fallback survived even though the unified core did not.',
                    'recordIds': group_records(['plugins'], chapter_sets['fibre'], children),
                },
            ],
            'evidenceIds': ['linear_expectation', 'query_mix_posthoc', 'temporal_relational', 'open_fallback'],
        },
        {
            'id': 'bridge',
            'number': '06',
            'eyebrow': 'Return to the right abstraction',
            'title': 'BRIDGE makes ranking authority explicit: Broad orders globally; experts earn bounded local correction rights.',
            'problem': 'The question after FIBRE was no longer how to create a more unified representation. It was how to allocate ranking authority without losing broad coverage or useful specialist evidence.',
            'learn': 'Query-conditioned routing, permission levels and family-specific validation showed that the same evidence can be valuable locally and harmful globally. Missing or inapplicable evidence must remain neutral.',
            'decision': 'BRIDGE keeps Broad as the universal base order and treats functional, structural, mechanistic, context and domain specialists as gated pair evidence with bounded influence.',
            'pathIds': ['return_broad', 'query_applicability', 'bridge'],
            'groups': [
                {
                    'id': 'reframing',
                    'title': 'Reframe experts around a protected base',
                    'summary': 'The final design principles that turned historical expert experiments into BRIDGE semantics.',
                    'recordIds': group_records(['pair_evidence', 'rebind_broad', 'score_evidence', 'expert_types', 'dynamic_v4'], chapter_sets['bridge'], children, exclude=['query_applicability']),
                },
                {
                    'id': 'specialist-validation',
                    'title': 'Validate specialists only in their own applicability domain',
                    'summary': 'Generic structural authority failed; family and TPS specialists returned under explicit gates.',
                    'recordIds': group_records(['cage_family', 'tps_correction', 'integrated_specialists'], chapter_sets['bridge'], children, omit=['bridge']),
                },
            ],
            'evidenceIds': ['p450_cage', 'phosphatase_cage', 'terpene_cage', 'layered_cage_eval'],
        },
    ]

    tracks = [
        {
            'id': 'compass',
            'atChapter': 'broad',
            'label': 'Parallel user-workflow track',
            'title': 'Real user requests turned retrieval into COMPASS.',
            'summary': 'Users changed scope, constraints and follow-up tasks conversationally. That pushed the project from semantic routing to a bounded agent and persistent research workspace.',
            'recordIds': ordered(track_sets['compass']),
        },
        {
            'id': 'wetlab',
            'atChapter': 'open',
            'label': 'Parallel experimental track',
            'title': 'Retrieval results had to become an executable wet-lab campaign.',
            'summary': 'Ranked candidates were translated into auditable panels, balanced plate assignment, randomized wells and a feedback contract for later learning.',
            'recordIds': ordered(track_sets['wetlab']),
        },
    ]

    architecture = {
        'title': 'BRIDGE today',
        'subtitle': 'Current composition, separated from the historical route that produced it.',
        'formula': 'S_BRIDGE(q,e) = S_Broad(q,e) + Σ_k g_k(q) Δ_k(q,e)',
        'base': {
            'title': 'Broad Retrieval',
            'body': 'Universal candidate generator and stable global ranker. It remains valid when every optional expert is silent.',
            'sourceIds': ['broad', 'rebind_broad', 'broad_calibration'],
        },
        'control': {
            'title': 'Applicability + permission layer',
            'body': 'For each query and direction, decide whether an expert is relevant, supported and allowed to modify rank; unavailable experts contribute exactly zero.',
            'sourceIds': ['dynamic_v4', 'dynamic_v6', 'query_applicability', 'evidence_admission'],
        },
        'experts': [
            {
                'id': 'functional',
                'title': 'Functional / evolutionary evidence',
                'members': ['EnzGFM with reaction features'],
                'body': 'General learned functional/evolutionary evidence, admitted only where it adds clean ranking information.',
                'sourceIds': ['enzgfm_rdkitplus'],
            },
            {
                'id': 'structural',
                'title': 'Structural evidence',
                'members': ['CLIPZyme', 'cached pocket / structure support'],
                'body': 'Structure is evidence with explicit support boundaries, not a universal replacement ranker.',
                'sourceIds': ['clip_fallback', 'clip_support', 'structure_evidence'],
            },
            {
                'id': 'mechanistic',
                'title': 'Mechanistic evidence',
                'members': ['bounded reaction-center correction', 'TPS specialist'],
                'body': 'Mechanistic signals can adjust a local shortlist only when their applicability tests pass.',
                'sourceIds': ['center_v3', 'tps_correction', 'tps_foundation'],
            },
            {
                'id': 'context',
                'title': 'Context evidence',
                'members': ['known-positive seed context', 'multi-seed context'],
                'body': 'Context is optional query evidence. When no seed is supplied, the channel is simply silent.',
                'sourceIds': ['seed_context', 'multi_seed_context', 'known_context'],
            },
            {
                'id': 'family',
                'title': 'Family-specific CAGE specialists',
                'members': ['P450 fine-tuned EnzymeCAGE', 'phosphatase fine-tuned EnzymeCAGE', 'terpene fine-tuned EnzymeCAGE'],
                'body': 'Activation uses candidate-independent generic→family EnzymeCAGE reaction-response features, a reaction-only applicability selector, and reaction-family / pair-level family agreement. The checkpoints remain specialists, never general CAGE ranking authority.',
                'sourceIds': ['cage_family', 'p450_cage', 'phosphatase_cage', 'terpene_cage'],
            },
        ],
        'correction': {
            'title': 'Bounded correction interface',
            'body': 'Experts act as pair evidence with direction-specific rights. Candidate scope, shortlist depth and correction magnitude constrain how much the base order can move.',
            'sourceIds': ['pair_evidence', 'score_evidence', 'integrated_specialists'],
        },
        'evidence': [
            {'label': 'Broad Core', 'value': 'MRR 0.2093 · Hit@10 36.87% · Hit@100 59.41%'},
            {'label': 'BRIDGE', 'value': 'MRR 0.2369 · Hit@10 41.54% · Hit@100 64.92%'},
            {'label': 'P450 specialist', 'value': 'MRR 0.0370 → 0.0705'},
            {'label': 'Phosphatase specialist', 'value': 'MRR 0.2522 → 0.3169'},
            {'label': 'Terpene specialist', 'value': 'MRR 0.0189 → 0.0387'},
        ],
        'policyNotes': [
            'Outside the reranked shortlist, preserve Broad order exactly.',
            'Family-finetuned CAGE features activate the matching specialist; they do not retune the general expert router.',
            'Missing pocket, structure or specialist support means absence of evidence, never negative evidence.',
        ],
    }

    chapter_union = set().union(*(chapter_sets[c['id']] for c in chapters))
    track_union = set().union(*track_sets.values())
    covered = chapter_union | track_union
    missing = set(by) - covered
    if missing:
        raise ValueError(f'story presentation lost records: {sorted(missing)}')

    relation_rows = []
    for left, right in zip(chapters, chapters[1:]):
        relation_rows.append({'type': 'led_to', 'source': left['id'], 'target': right['id']})
    for link in CROSSLINKS:
        relation_rows.append({'type': 'influenced', **link})
    for expert in architecture['experts']:
        for source in expert['sourceIds']:
            relation_rows.append({'type': 'part_of', 'source': source, 'target': 'bridge'})
    for source in architecture['control']['sourceIds'] + architecture['base']['sourceIds'] + architecture['correction']['sourceIds']:
        relation_rows.append({'type': 'part_of', 'source': source, 'target': 'bridge'})
    for chapter in chapters:
        for source in chapter.get('evidenceIds', []):
            relation_rows.append({'type': 'supports', 'source': source, 'target': chapter['id']})

    return {
        'chapters': chapters,
        'tracks': tracks,
        'architecture': architecture,
        'relations': relation_rows,
        'coverage': {'allCanonicalRecordsRepresented': True, 'recordIds': ordered(covered)},
    }


def write_data(story: dict) -> None:
    payload = {
        'nodes': NODES,
        'crossLinks': CROSSLINKS,
        'families': FAMILY_LABELS,
        'story': story,
        'meta': {
            'schema': 'bridge-engineering-story-v4',
            'root': 'enzymecage',
            'current': 'bridge',
            'presentation': 'decision-storyline',
            'semanticRoles': ['decision', 'thread', 'attempt', 'evidence', 'component', 'application'],
        },
    }
    DATA_JS.write_text(
        'window.LINEAGE_DATA = ' + json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + ';\n',
        encoding='utf-8',
    )


def write_doc(by: dict[str, dict], story: dict) -> None:
    lines = [
        '# BRIDGE Engineering Storyline',
        '',
        'This document separates **design evolution** from **current BRIDGE composition**.',
        'The historical route is organized as decisions. Concrete experiments remain attached to the problem they tested;',
        'evaluation records are treated as evidence; BRIDGE expert families are described again in the final architecture because they are current components rather than peer historical generations.',
        '',
        'Live Engineering page: `https://nju-igem.runnelzhang.com/engineering/`',
        '',
    ]
    for chapter in story['chapters']:
        lines += [
            f"## {chapter['number']} · {chapter['title']}",
            '',
            f"**Problem.** {chapter['problem']}",
            '',
            f"**What we learned.** {chapter['learn']}",
            '',
            f"**Decision.** {chapter['decision']}",
            '',
        ]
        for group in chapter['groups']:
            lines += [f"### {group['title']}", '', group['summary'], '']
            for node_id in group['recordIds']:
                node = by[node_id]
                lines.append(f"- **{node['label']}** `[{node['status'].upper()}]` — {node['result']} _{node['legacy']}_")
            lines.append('')
        if chapter.get('evidenceGroup'):
            group = chapter['evidenceGroup']
            lines += [f"### {group['title']}", '']
            for node_id in group['recordIds']:
                node = by[node_id]
                lines.append(f"- **{node['label']}** — {node['result']}")
            lines.append('')

    lines += ['## BRIDGE today', '', story['architecture']['subtitle'], '']
    architecture = story['architecture']
    lines += [f"- **Base:** {architecture['base']['title']} — {architecture['base']['body']}"]
    lines += [f"- **Control:** {architecture['control']['title']} — {architecture['control']['body']}"]
    for expert in architecture['experts']:
        lines.append(f"- **{expert['title']}:** {', '.join(expert['members'])}. {expert['body']}")
    lines += [f"- **Correction:** {architecture['correction']['title']} — {architecture['correction']['body']}", '']
    lines += ['Formula:', '', f"`{architecture['formula']}`", '']
    lines += ['## Parallel project tracks', '']
    for track in story['tracks']:
        lines += [f"### {track['title']}", '', track['summary'], '']
        for node_id in track['recordIds']:
            node = by[node_id]
            lines.append(f"- **{node['label']}** — {node['result']}")
        lines.append('')
    lines += [
        '## Source of truth',
        '',
        'The full experiment inventory remains in `scripts/engineering_lineage/lineage_data.py`.',
        'This generated presentation intentionally changes the **semantic role** of records without deleting the underlying history.',
    ]
    DOC.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    by, children = validate()
    story = build_story(by, children)
    write_data(story)
    write_doc(by, story)
    print(json.dumps({
        'canonical_records': len(NODES),
        'represented_records': len(story['coverage']['recordIds']),
        'chapters': len(story['chapters']),
        'parallel_tracks': len(story['tracks']),
        'schema': 'bridge-engineering-story-v4',
        'data_js': str(DATA_JS.relative_to(ROOT)),
        'doc': str(DOC.relative_to(ROOT)),
    }, indent=2))


if __name__ == '__main__':
    main()
