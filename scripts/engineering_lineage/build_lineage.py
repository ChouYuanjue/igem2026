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


def loop(
    loop_id: str,
    title: str,
    center: str,
    phases: dict[str, dict],
    outcome: str,
    why_next: str,
    record_ids: set[str],
    *,
    children: list[dict] | None = None,
) -> dict:
    return {
        'id': loop_id,
        'title': title,
        'center': center,
        'phases': phases,
        'outcome': outcome,
        'whyNext': why_next,
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

    # Scene 01 subloops
    structure_screen = subtree('pocket_audit', children) - subtree('reaction_transfer', children)
    candidate_rescue = subtree('reaction_transfer', children) | {'gate_coverage_ceiling', 'open_problem'}

    scene1_children = [
        loop(
            'structure-screen', 'Structure ranking audit', 'Structure limit',
            {
                'design': phase('Design','Check whether pocket choice explains ranking weakness.',['pocket_audit']),
                'build': phase('Build','Run the structural scorer across the full TPS library.',['full_library_structure']),
                'test': phase('Test','Compare pocket robustness with full-library ranking.',['pocket_audit','full_library_structure']),
                'learn': phase('Learn','Pair compatibility does not become useful library ranking by itself.',['full_library_structure']),
            },
            'Structure remained useful evidence, but not a complete full-library ranking solution.',
            'Try using reaction neighbourhoods to rescue candidate generation.',
            structure_screen,
        ),
        loop(
            'candidate-rescue', 'Candidate-pool rescue', 'Gate ceiling',
            {
                'design': phase('Design','Use reaction similarity to transfer candidates from nearby known reactions.',['reaction_transfer']),
                'build': phase('Build','Construct the rescued closed candidate pool.',['closed_pool']),
                'test': phase('Test','Measure how many known positives survive the gate.',['gate_coverage_ceiling']),
                'learn': phase('Learn','The gate still excludes positives before any upper ranker can see them.',['gate_coverage_ceiling','open_problem']),
            },
            'Candidate rescue improved ranks locally but preserved a hard recall ceiling.',
            'Remove gate membership from scoreability entirely.',
            candidate_rescue,
        ),
    ]

    # Scene 02 subloops
    broad_training = before('representation_program', ['broad'], children) | {'broad'}
    cage_upper_audit = {'broad','layered_cage_eval','clip_support','generic_cage_expert'}
    broad_training_children = [
        loop(
            'broad-representation', 'Open molecular representation', 'Open inputs',
            {
                'design': phase('Design','Represent proteins and reactions directly from molecular inputs.',['representation_program','esm_c','drfp']),
                'build': phase('Build','Train a multi-positive bidirectional dual tower.',['dual_tower']),
                'test': phase('Test','Check double-unseen retrieval rather than relation-table membership.',['dual_tower','topk_surrogate']),
                'learn': phase('Learn','Any new reaction or protein can now receive a score.',['broad']),
            },
            'The retrieval space is open rather than relation-gated.',
            'Improve the quality of the open-space order.',
            subtree('representation_program', children),
        ),
        loop(
            'broad-order-training', 'Make the open order useful', 'Useful order',
            {
                'design': phase('Design','Reduce false negatives and mine informative hard negatives.',['pu_mask','hard_negatives']),
                'build': phase('Build','Add hard-negative curricula, domain adaptation and ranking objectives.',['hard_negatives','marts_adapt','topk_surrogate']),
                'test': phase('Test','Stress the order across cold and domain-shift regimes.',['dual_kernel_marts','temporal']),
                'learn': phase('Learn','Broad is no longer only a recall mechanism; its ranking has standalone value.',['broad']),
            },
            'Broad develops a stable full-space ordering signal.',
            'Test whether CAGE can still own the upper-layer decision.',
            broad_training,
        ),
    ]

    retention_loop = loop(
        'retention-guardrails', 'Retention guardrails', 'Keep base',
        {
            'design': phase('Design','Add new domains without erasing the incumbent Broad behaviour.',['generalization_program','directional_cont']),
            'build': phase('Build','Try replay, blending, regularization and score retention.',['replay_anchor','checkpoint_blend','score_distill']),
            'test': phase('Test','Compare external and temporal retrieval after adaptation.',['enzyme405','orphan335','temporal','rhea_transfer']),
            'learn': phase('Learn','Some capabilities are safer to route around Broad than to force into one parameter state.',['posthoc_router']),
        },
        'Broad protection becomes a guardrail, not a separate system identity.',
        'Externalize complementary capabilities when appropriate.',
        retention,
    )

    cage_upper_loop = loop(
        'cage-upper-audit', 'CAGE as the ideal upper reranker', 'Upper limit',
        {
            'design': phase('Design','Keep CAGE above Broad and let it make the final structural judgement.',['broad','generic_cage_expert']),
            'build': phase('Build','Feed Broad candidates into generic CAGE where structural inputs are available.',['generic_cage_expert','clip_support']),
            'test': phase('Test','Compare candidate reach with upper-layer scoreability on the layered suite.',['layered_cage_eval']),
            'learn': phase('Learn','The upper layer has its own coverage and support ceiling.',['layered_cage_eval','generic_cage_expert']),
        },
        'Broad can reach much farther than a generic CAGE upper layer can reliably score.',
        'Let Broad keep more ranking responsibility and treat extra evidence conditionally.',
        cage_upper_audit,
    )

    # Scene 03 recursive expert loops
    enzgfm_features = subtree('enzgfm', children) | {'enzgfm_rdkitplus'}
    functional_loop = loop(
        'functional-evidence', 'Functional / evolutionary evidence', 'Optional signal',
        {
            'design': phase('Design','Test whether learned enzyme-function representations add complementary evidence.',['enzgfm']),
            'build': phase('Build','Add RDKit and reaction features around EnzGFM.',['enzgfm_rdkit','enzgfm_rdkitplus']),
            'test': phase('Test','Evaluate on frozen external retrieval tasks.',['enzyme405','orphan335']),
            'learn': phase('Learn','The signal helps as an expert but does not replace Broad globally.',['enzgfm_rdkitplus']),
        },
        'Functional evidence is complementary and conditional.',
        'Admit it as an optional expert.',
        enzgfm_features,
        children=[
            loop(
                'enzgfm-feature-loop','EnzGFM feature adaptation','Reaction-aware',
                {
                    'design': phase('Design','Start from the native EnzGFM signal.',['enzgfm']),
                    'build': phase('Build','Add RDKit descriptors and reaction features.',['enzgfm_rdkit','enzgfm_rdkitplus']),
                    'test': phase('Test','Compare the feature variants under the same frozen task.',['enzgfm_rdkitplus']),
                    'learn': phase('Learn','Reaction-aware features are the useful retained variant.',['enzgfm_rdkitplus']),
                },
                'The adapted feature view is retained as an expert input.',
                'Pass the surviving signal to expert admission.',
                enzgfm_features,
            )
        ],
    )

    reaction_center_loop = loop(
        'reaction-center-loop','Reaction-center correction','Bounded center',
        {
            'design': phase('Design','Inject local reaction-center mechanism into ranking.',['reaction_center']),
            'build': phase('Build','Try direct fusion, identity-preserving residual and bounded correction.',['center_v1','center_identity','center_v3']),
            'test': phase('Test','Check whether local gains survive without damaging the base order.',['center_identity','center_v3']),
            'learn': phase('Learn','The useful form is bounded and local, not globally authoritative.',['center_v3']),
        },
        'Reaction-center evidence survives only as a bounded local correction.',
        'Treat structural/mechanistic evidence as applicability-limited expertise.',
        subtree('reaction_center', children),
    )

    clipzyme_loop = loop(
        'clipzyme-loop','CLIPZyme structural branch','Support first',
        {
            'design': phase('Design','Test a learned structural expert alongside Broad.',['clipzyme','clip_native']),
            'build': phase('Build','Audit encoder substitution, support coverage and fallback behaviour.',['clip_encoder_sub','clip_support','clip_fallback']),
            'test': phase('Test','Measure gains only where author-supported inputs exist.',['clip_support']),
            'learn': phase('Learn','Structural expertise must expose support and fallback explicitly.',['clip_fallback']),
        },
        'CLIPZyme is usable only with explicit support semantics.',
        'Route structural evidence rather than treating it as universal.',
        subtree('clipzyme', children),
    )

    structural_loop = loop(
        'structural-evidence','Structure / mechanism evidence','Local evidence',
        {
            'design': phase('Design','Search for structural and mechanistic signals Broad does not encode directly.',['reaction_center','clipzyme']),
            'build': phase('Build','Develop reaction-center and CLIPZyme branches.',['center_v3','clip_fallback']),
            'test': phase('Test','Audit support, transfer and local ranking gains.',['clip_support','rhea_transfer']),
            'learn': phase('Learn','These signals are valuable only inside explicit applicability regions.',['center_v3','clip_fallback']),
        },
        'Structure and mechanism become local expert evidence.',
        'Combine experts only through protected fusion and routing.',
        subtree('reaction_center', children) | subtree('clipzyme', children) | subtree('top2000', children) | subtree('reactzyme', children),
        children=[reaction_center_loop, clipzyme_loop],
    )

    r2e_loop = loop(
        'r2e-fusion','R2E learned fusion','Directional gain',
        {
            'design': phase('Design','Learn a ranker over a selected expert portfolio.',['portfolio']),
            'build': phase('Build','Union expert candidates and train R2E LambdaRank.',['candidate_union','r2e_lambdarank']),
            'test': phase('Test','Compare learned fusion against fixed score rules.',['r2e_lambdarank','r2e_similarity_router']),
            'learn': phase('Learn','Learned fusion helps, but routing must respect direction and support.',['joint_reliability_gate']),
        },
        'R2E confirms that expert value can be learned directionally.',
        'Build an anchored E2R counterpart and formal admission.',
        subtree('portfolio', children),
    )

    e2r_loop = loop(
        'e2r-fusion','E2R anchored rescue','Anchor base',
        {
            'design': phase('Design','Extend expert fusion to E2R without losing the strong base.',['e2r_branch']),
            'build': phase('Build','Try four-expert fusion, then baseline-anchored rescue and learning-to-rank.',['e2r_four','e2r_rescue','e2r_anchor']),
            'test': phase('Test','Compare free fusion against the anchored form.',['e2r_four','e2r_anchor']),
            'learn': phase('Learn','Anchoring to Broad is essential when the base is already strong.',['e2r_anchor']),
        },
        'E2R establishes the protected-incumbent principle.',
        'Formalize which experts are allowed into the system.',
        subtree('e2r_branch', children),
    )

    admission_children = [
        loop(
            'clip-admission','CLIPZyme admission','Admit with support',
            {
                'design': phase('Design','Treat CLIPZyme as a candidate structural expert.',['bime_clip']),
                'build': phase('Build','Expose structural support and fallback semantics.',['bime_clip','clip_support']),
                'test': phase('Test','Confirm gains only where structure is supported.',['bime_clip']),
                'learn': phase('Learn','Structural expertise is admissible only with scoped support.',['bime_clip']),
            },
            'CLIPZyme is admitted conditionally.',
            'Apply the same discipline to other context experts.',
            {'bime_clip','clip_support'},
        ),
        loop(
            'seed-admission','Seed-context admission','Context only',
            {
                'design': phase('Design','Use known-positive context when it exists.',['seed_context']),
                'build': phase('Build','Add seed and multi-seed context features.',['seed_context','multi_seed_context']),
                'test': phase('Test','Evaluate only queries carrying valid context.',['seed_context']),
                'learn': phase('Learn','Context is useful but intrinsically optional.',['seed_context']),
            },
            'Seed context becomes a conditional expert.',
            'Keep missing context neutral.',
            subtree('seed_context', children) | ({'multi_seed_context'} if 'multi_seed_context' in by else set()),
        ),
        loop(
            'generic-cage-admission','Generic CAGE admission','Reject generic',
            {
                'design': phase('Design','Test whether CAGE can return as a generic expert above Broad.',['generic_cage_expert']),
                'build': phase('Build','Route generic CAGE through the same admission framework.',['generic_cage_expert']),
                'test': phase('Test','Run frozen OOF expert confirmation.',['generic_cage_expert']),
                'learn': phase('Learn','Generic CAGE does not earn universal expert authority.',['generic_cage_expert']),
            },
            'The generic CAGE expert is rejected.',
            'Later revisit CAGE at family scope rather than globally.',
            {'generic_cage_expert'},
        ),
    ]

    admission_loop = loop(
        'expert-admission','Formal expert admission','Admit selectively',
        {
            'design': phase('Design','Require every extra capability to justify entry.',['admission']),
            'build': phase('Build','Evaluate structural, context, homology, reciprocal and CAGE experts.',['bime_clip','seed_context','homology_context','reciprocal_expert','generic_cage_expert']),
            'test': phase('Test','Use frozen direction-specific expert confirmation.',['admission','generic_cage_expert']),
            'learn': phase('Learn','Expert usefulness is heterogeneous; admission cannot be one global yes/no forever.',['generic_cage_expert','bime']),
        },
        'BiME can organize multiple experts, but expert authority remains too coarse.',
        'Search for a cleaner relational formulation and, later, query-level authority.',
        subtree('admission', children),
        children=admission_children,
    )

    fusion_loop = loop(
        'fusion-routing','Fusion and routing','BiME',
        {
            'design': phase('Design','Move from fixed score addition to learned, protected expert fusion.',['fusion_program','portfolio']),
            'build': phase('Build','Develop R2E, anchored E2R and formal expert admission.',['r2e_lambdarank','e2r_anchor','admission']),
            'test': phase('Test','Compare expert value across direction, support and context.',['bime_clip','seed_context','generic_cage_expert']),
            'learn': phase('Learn','The system needs explicit expert routing around a Broad incumbent.',['bime']),
        },
        'BiME is the first stable expert system over Broad.',
        'Ask whether the experts can be unified under one relational object.',
        subtree('fusion_program', children) - fibre - bridge,
        children=[r2e_loop, e2r_loop, admission_loop],
    )

    # Scene 04 recursive FIBRE loops
    symmetry_ids = {'consistency_penalty','symmetric_potential','geomean_gate','symmetric_linear','dim_consistency'}
    probabilistic_ids = {'gibbs','kl_bary','variance_fallback','normalized_mix','logmeanexp','second_order'}
    directional_ids = {'molecular_comp','linear_expectation','rc_posterior'}

    conditional_children = [
        loop(
            'symmetric-modes','Symmetric formulations','Reject symmetry',
            {
                'design': phase('Design','Force both retrieval directions into a shared symmetric potential.',['symmetric_potential']),
                'build': phase('Build','Try consistency, geometric gates, symmetric linear mixtures and scaling fixes.',['consistency_penalty','geomean_gate','symmetric_linear','dim_consistency']),
                'test': phase('Test','Compare frozen directional retrieval under the symmetric forms.',['symmetric_linear','dim_consistency']),
                'learn': phase('Learn','Forced symmetry erases useful directional differences.',['symmetric_linear']),
            },
            'Symmetric formulations are rejected.',
            'Test probabilistic and directional alternatives.',
            symmetry_ids,
        ),
        loop(
            'probabilistic-modes','Probabilistic aggregations','No stable win',
            {
                'design': phase('Design','Aggregate experts with probabilistic or entropy-based rules.',['gibbs','kl_bary']),
                'build': phase('Build','Try variance fallback, normalized mixtures and second-order forms.',['variance_fallback','normalized_mix','second_order']),
                'test': phase('Test','Compare against the simpler directional baseline.',['logmeanexp','second_order']),
                'learn': phase('Learn','Extra aggregation complexity does not justify itself.',['second_order']),
            },
            'Probabilistic aggregations fail to become the main conditional rule.',
            'Prefer the simpler directional conditional expectation.',
            probabilistic_ids,
        ),
        loop(
            'directional-modes','Directional conditional expectation','Local winner',
            {
                'design': phase('Design','Allow expert value to differ by retrieval direction.',['linear_expectation']),
                'build': phase('Build','Use directional linear conditional expectation and test local compensations.',['molecular_comp','linear_expectation','rc_posterior']),
                'test': phase('Test','Compare against symmetric and higher-order alternatives.',['linear_expectation','rc_posterior']),
                'learn': phase('Learn','Directional conditioning is the stable local survivor.',['linear_expectation']),
            },
            'Directional conditional expectation is retained.',
            'Anchor the relational core in explicit scientific evidence.',
            directional_ids,
        ),
    ]

    conditional_loop = loop(
        'conditional-modes','Conditional-mode search','Directional modes',
        {
            'design': phase('Design','Find a common mathematical form for heterogeneous expert value.',['conditional_modes']),
            'build': phase('Build','Explore symmetric, probabilistic and directional conditional forms.',['symmetric_potential','gibbs','linear_expectation']),
            'test': phase('Test','Evaluate candidate forms under frozen directional ranking.',['linear_expectation','second_order']),
            'learn': phase('Learn','A directional conditional view survives; many elegant symmetric forms do not.',['linear_expectation']),
        },
        'The useful mathematics is directional and conditional, not universally symmetric.',
        'Move unsupported science into explicit evidence channels.',
        subtree('conditional_modes', children),
        children=conditional_children,
    )

    relation_loop = loop(
        'relational-formulation','Relational formulation','Relation object',
        {
            'design': phase('Design','Describe enzyme–reaction matching as biological relation rather than only rank score.',['bio_relation','context_domain']),
            'build': phase('Build','Try tensor/product geometry and catalytic kernels.',['tensor_field','catalytic_kernel']),
            'test': phase('Test','Check whether the geometry yields a stable conditional ranking object.',['interaction_atlas','atlas_gluing']),
            'learn': phase('Learn','Geometry helps organize the problem but does not solve ranking alone.',['atlas_gluing']),
        },
        'The relation abstraction is useful, but the ranking core still needs conditional evidence.',
        'Explore conditional expert modes.',
        subtree('bio_relation', children) | subtree('context_domain', children) | subtree('tensor_field', children) | subtree('catalytic_kernel', children) | subtree('conditional_modes', children),
        children=[conditional_loop],
    )

    evidence_loop = loop(
        'scientific-evidence','Scientific evidence admission','Admit evidence',
        {
            'design': phase('Design','Represent physical or biochemical cues as explicit evidence.',['scientific_evidence']),
            'build': phase('Build','Separate structure, known context and reaction-center evidence.',['structure_evidence','known_context','rc_evidence']),
            'test': phase('Test','Calibrate whether each evidence source is actually supported.',['evidence_admission']),
            'learn': phase('Learn','Missing or unsupported evidence should contribute neutrally.',['evidence_admission']),
        },
        'Evidence admission and missing-neutral semantics survive into BRIDGE.',
        'Use those interfaces around, rather than instead of, a global base order.',
        subtree('scientific_evidence', children),
        children=[
            loop('structure-evidence-loop','Structure evidence','Support-aware',{
                'design':phase('Design','Expose structure as an explicit scientific channel.',['structure_evidence']),
                'build':phase('Build','Materialize available structural evidence.',['structure_evidence']),
                'test':phase('Test','Check support coverage before use.',['structure_evidence']),
                'learn':phase('Learn','No structure means no structural correction.',['structure_evidence']),
            },'Structure stays optional and support-aware.','Keep unsupported structure neutral.',{'structure_evidence'}),
            loop('context-evidence-loop','Known-positive context','Context-aware',{
                'design':phase('Design','Use known-positive context when present.',['known_context']),
                'build':phase('Build','Encode context as its own evidence channel.',['known_context']),
                'test':phase('Test','Evaluate only where context exists.',['known_context']),
                'learn':phase('Learn','Context cannot be assumed globally.',['known_context']),
            },'Context remains a conditional evidence source.','Route it only when available.',{'known_context'}),
            loop('center-evidence-loop','Reaction-center evidence','Mechanism-aware',{
                'design':phase('Design','Expose reaction-center information explicitly.',['rc_evidence']),
                'build':phase('Build','Attach center evidence to pair scoring.',['rc_evidence']),
                'test':phase('Test','Validate center support before correction.',['rc_evidence']),
                'learn':phase('Learn','Mechanistic evidence is local and optional.',['rc_evidence']),
            },'Reaction-center evidence stays local.','Feed it through evidence admission.',{'rc_evidence','evidence_admission'}),
        ],
    )

    query_mix_loop = loop(
        'query-mix-loop','Query-adaptive mixture','Post-hoc gate',
        {
            'design': phase('Design','Let expert weights depend on the current query.',['query_mix']),
            'build': phase('Build','Try end-to-end, E2R-only and frozen-core post-hoc gates.',['query_mix','query_mix_e2r','query_mix_posthoc']),
            'test': phase('Test','Compare which routing scheme preserves the base while adding local gains.',['query_mix_posthoc']),
            'learn': phase('Learn','Frozen-core post-hoc gating is the stable form.',['query_mix_posthoc']),
        },
        'Query-adaptive routing works best around a frozen core.',
        'Test a modern relational main model without sacrificing fallback.',
        subtree('query_mix', children),
    )

    eram_loop = loop(
        'eram-loop','ERAM relational core','Fails replace',
        {
            'design': phase('Design','Test an ERAM-style relational core as the new main model.',['eram']),
            'build': phase('Build','Prepare UniMol features and train the relational main model.',['unimol','relational_main']),
            'test': phase('Test','Use strict temporal and double-cold evaluation.',['temporal_relational']),
            'learn': phase('Learn','The relational main model cannot justify replacing Broad.',['temporal_relational']),
        },
        'ERAM does not displace Broad under strict replacement tests.',
        'Keep plugins and open-world fallback around the incumbent instead.',
        subtree('eram', children),
    )

    plugin_loop = loop(
        'plugin-loop','Pluggable expert adapters','Plugin interface',
        {
            'design': phase('Design','Keep new evidence sources detachable from the core.',['plugins']),
            'build': phase('Build','Implement frozen-context plugin and open-world fallback.',['context_plugin','open_fallback']),
            'test': phase('Test','Verify the system remains valid when optional evidence is absent.',['open_fallback']),
            'learn': phase('Learn','Plugins plus neutral fallback are more robust than replacing the global core.',['open_fallback']),
        },
        'The plugin/fallback interface survives the FIBRE detour.',
        'Return to Broad as ordering authority.',
        subtree('plugins', children),
    )

    fibre_core_loop = loop(
        'fibre-core','Adaptive relational core','Return to Broad',
        {
            'design': phase('Design','Turn FIBRE principles into an adaptive relational main model.',['hcm','query_mix']),
            'build': phase('Build','Combine query-adaptive routing, ERAM and pluggable adapters.',['query_mix_posthoc','eram','plugins']),
            'test': phase('Test','Challenge the replacement under strict temporal and double-cold tests.',['temporal_relational']),
            'learn': phase('Learn','Keep routing, plugins and fallback; restore Broad as global order.',['open_fallback','return_broad']),
        },
        'The replacement hypothesis fails, while interfaces survive.',
        'Use these principles to allocate local authority over Broad.',
        subtree('hcm', children) | subtree('eram', children) | subtree('plugins', children),
        children=[query_mix_loop, eram_loop, plugin_loop],
    )

    # Scene 05 recursive BRIDGE loops
    rebind_loop = loop(
        'rebind-loop','Rebind evidence to Broad','Broad anchor',
        {
            'design': phase('Design','Treat expert outputs as optional pair evidence.',['pair_evidence','rebind_broad']),
            'build': phase('Build','Calibrate Broad before evidence fusion.',['broad_calibration']),
            'test': phase('Test','Check that missing evidence leaves the Broad score valid.',['broad_calibration']),
            'learn': phase('Learn','Broad must remain the valid default score everywhere.',['rebind_broad']),
        },
        'All expert evidence is defined relative to a valid Broad base.',
        'Decide when an expert is allowed to act.',
        {'pair_evidence','rebind_broad','broad_calibration'},
    )

    routing_loop = loop(
        'routing-loop','Query-conditioned routing','Permission',
        {
            'design': phase('Design','Separate expert availability from expert usefulness.',['dynamic_v4']),
            'build': phase('Build','Add expert types and explicit permission levels.',['expert_types','dynamic_v4','dynamic_v6']),
            'test': phase('Test','Evaluate whether the current query should activate each expert.',['query_applicability']),
            'learn': phase('Learn','Authority must be decided per query and direction.',['query_applicability']),
        },
        'Query applicability becomes the permission layer.',
        'Specialize experts inside their own applicability domains.',
        {'score_evidence','expert_types','dynamic_v4','dynamic_v6','query_applicability'},
    )

    family_children = [
        loop('p450-loop','P450 CAGE specialist','P450 local',{
            'design':phase('Design','Specialize CAGE for P450 queries.',['p450_cage']),
            'build':phase('Build','Fine-tune the P450 structural expert.',['p450_cage']),
            'test':phase('Test','Evaluate only inside the P450 applicability domain.',['p450_cage']),
            'learn':phase('Learn','Family-local CAGE improves where generic CAGE did not.',['p450_cage']),
        },'P450 earns local structural authority.','Keep the same rule for other responsive families.',{'p450_cage'}),
        loop('phosphatase-loop','Phosphatase CAGE specialist','Family local',{
            'design':phase('Design','Specialize CAGE for phosphatase queries.',['phosphatase_cage']),
            'build':phase('Build','Fine-tune the phosphatase structural expert.',['phosphatase_cage']),
            'test':phase('Test','Evaluate only inside the phosphatase domain.',['phosphatase_cage']),
            'learn':phase('Learn','The family-specific expert gives a strong local gain.',['phosphatase_cage']),
        },'Phosphatase earns local structural authority.','Retain family-scoped activation.',{'phosphatase_cage'}),
        loop('terpene-cage-loop','Terpene CAGE specialist','Terpene local',{
            'design':phase('Design','Specialize CAGE for terpene queries.',['terpene_cage']),
            'build':phase('Build','Fine-tune the terpene structural expert.',['terpene_cage']),
            'test':phase('Test','Evaluate only inside the terpene family.',['terpene_cage']),
            'learn':phase('Learn','Even the weaker family gain is safer as a local specialist than a global expert.',['terpene_cage']),
        },'Terpene CAGE remains family-scoped.','Merge only through query permission.',{'terpene_cage'}),
    ]

    family_loop = loop(
        'family-cage','Family-specific CAGE','CAGE returns',
        {
            'design': phase('Design','Revisit CAGE at family scope after generic admission failed.',['cage_family','generic_cage_expert']),
            'build': phase('Build','Train P450, phosphatase and terpene specialists.',['p450_cage','phosphatase_cage','terpene_cage']),
            'test': phase('Test','Evaluate each specialist only in its applicability domain.',['p450_cage','phosphatase_cage','terpene_cage']),
            'learn': phase('Learn','CAGE regains authority only as a family-specific specialist.',['cage_family']),
        },
        'CAGE completes its role transition: upper ranker → rejected generic expert → family specialist.',
        'Let query permission activate the right family expert.',
        subtree('cage_family', children),
        children=family_children,
    )

    tps_loop = loop(
        'tps-specialist','TPS specialist','TPS local',
        {
            'design': phase('Design','Reuse TPS-specific mechanism after the task became broad.',['tps_correction','tps_foundation']),
            'build': phase('Build','Gate a bounded TPS correction.',['tps_correction']),
            'test': phase('Test','Activate only on matched TPS queries.',['layered_cage_eval']),
            'learn': phase('Learn','TPS is now one biochemical specialist rather than the whole task.',['tps_correction']),
        },
        'TPS returns as a local mechanistic expert.',
        'Combine it with other admitted specialists through the same bounded interface.',
        subtree('tps_correction', children) | {'tps_foundation'},
    )

    integration_loop = loop(
        'bounded-integration','Bounded integration','Bounded delta',
        {
            'design': phase('Design','Combine only evidence that has passed query-level permission.',['integrated_specialists']),
            'build': phase('Build','Apply bounded corrections while leaving Broad valid outside the local scope.',['integrated_specialists']),
            'test': phase('Test','Run the layered full-suite comparison.',['layered_cage_eval']),
            'learn': phase('Learn','Broad stays global; specialists contribute bounded local gains.',['bridge']),
        },
        'The final interface is a Broad base plus query-gated bounded corrections.',
        'This closes the engineering loop in BRIDGE.',
        subtree('integrated_specialists', children),
    )

    permission_loop = loop(
        'permission-model','Permission model','Who may act?',
        {
            'design': phase('Design','Reframe expert outputs as evidence relative to Broad.',['pair_evidence','rebind_broad']),
            'build': phase('Build','Add directional score evidence, expert types and routing.',['score_evidence','expert_types','dynamic_v4']),
            'test': phase('Test','Separate availability, usefulness and query applicability.',['dynamic_v6','query_applicability']),
            'learn': phase('Learn','Expert authority is query- and direction-specific.',['query_applicability']),
        },
        'Permission becomes the control plane over optional evidence.',
        'Attach domain specialists behind this control plane.',
        {'pair_evidence','rebind_broad','broad_calibration','score_evidence','expert_types','dynamic_v4','dynamic_v6','query_applicability'},
        children=[rebind_loop, routing_loop],
    )

    scenes = [
        {
            'id':'candidate-ceiling','number':'01','eyebrow':'Candidate eligibility',
            'title':'The first failure was upstream of ranking.',
            'lead':'EnzymeCAGE could judge pairs inside a bounded candidate system, but a positive excluded by the gate could never be recovered downstream.',
            'stack':{'upper':{'title':'EnzymeCAGE / meta-ranker','note':'owns in-pool ranking'},'lower':{'title':'Similarity candidate gate','note':'decides who can be scored'}},
            'cageRole':'primary upper ranker','broadRole':None,
            'primary':loop('candidate-system','Can better ranking rescue a bounded candidate system?','Recall ceiling',{
                'design':phase('Design','Audit structural ranking inside the existing candidate system.',['enzymecage','pocket_audit']),
                'build':phase('Build','Run full-library structural scoring and candidate rescue.',['full_library_structure','reaction_transfer','closed_pool']),
                'test':phase('Test','Measure ranking and candidate-gate coverage.',['full_library_structure','gate_coverage_ceiling']),
                'learn':phase('Learn','A missed positive never reaches the upper ranker.',['gate_coverage_ceiling','open_problem']),
            },'Candidate eligibility becomes the dominant bottleneck.','Replace fixed gate membership with retrieval from molecular inputs.',early,children=scene1_children),
            'recordIds':ordered(early),'trackIds':[],
        },
        {
            'id':'broad-under-cage','number':'02','eyebrow':'Recall layer changes first',
            'title':'Broad entered below CAGE before it became the base order.',
            'lead':'The initial goal was conservative: let Broad open the candidate universe while CAGE remained the ideal upper reranker. That architecture exposed a second ceiling in the upper layer itself.',
            'stack':{'upper':{'title':'CAGE upper reranker','note':'still expected to make the final judgement'},'lower':{'title':'Broad Retrieval','note':'opens the candidate universe'}},
            'cageRole':'intended upper reranker','broadRole':'broad recall layer → meaningful order',
            'primary':loop('broad-cage-stack','Can Broad handle recall while CAGE keeps ranking authority?','Rank moves down',{
                'design':phase('Design','Replace the hard gate while keeping CAGE above it.',['open_problem','candidate_program','representation_program']),
                'build':phase('Build','Train open Broad retrieval and feed its candidates upward.',['dual_tower','hard_negatives','broad']),
                'test':phase('Test','Compare Broad candidate reach with generic CAGE scoreability.',['layered_cage_eval']),
                'learn':phase('Learn','Broad reaches farther than the generic CAGE upper layer can support.',['layered_cage_eval','generic_cage_expert']),
            },'Ranking responsibility begins moving into Broad itself.','Keep Broad\'s order and make extra evidence conditional.',open_world|retention,children=[loop('broad-training','Build a useful Broad order','Broad order',{
                'design':phase('Design','Score arbitrary reactions and proteins from molecular inputs.',['representation_program']),
                'build':phase('Build','Train representations and hard-negative-aware retrieval.',['dual_tower','hard_negatives','marts_adapt']),
                'test':phase('Test','Stress the ranking under cold and shifted regimes.',['topk_surrogate','dual_kernel_marts']),
                'learn':phase('Learn','Broad develops standalone ordering value.',['broad']),
            },'Broad is more than a recall-only candidate generator.','Audit whether CAGE can still remain the upper authority.',broad_training,children=broad_training_children),cage_upper_loop,retention_loop]),
            'evidence':[{'label':'Broad Top-1000 positive-query coverage','value':'≈77.91%'},{'label':'generic CAGE scoreable-positive coverage','value':'≈25.08%'}],
            'recordIds':ordered(open_world|retention),'trackIds':['wetlab','compass'],
        },
        {
            'id':'experts-bime','number':'03','eyebrow':'Ranking responsibility is shared',
            'title':'Broad became the base order; other capabilities became experts.',
            'lead':'Once Broad itself had a meaningful order, the engineering problem changed from replacing the ranker to deciding when additional evidence deserves to modify that order.',
            'stack':{'upper':{'title':'BiME expert layer','note':'admission, routing, context, cost'},'lower':{'title':'Broad base order','note':'protected incumbent ranking'}},
            'cageRole':'generic expert candidate','broadRole':'global base order',
            'primary':loop('expert-delegation','How should heterogeneous evidence modify a strong Broad order?','BiME',{
                'design':phase('Design','Externalize complementary capabilities around Broad.',['fusion_program','portfolio']),
                'build':phase('Build','Develop functional, structural and learned fusion experts.',['enzgfm','reaction_center','r2e_lambdarank','e2r_anchor']),
                'test':phase('Test','Confirm expert value by direction, support and context.',['admission','generic_cage_expert','seed_context']),
                'learn':phase('Learn','Expert usefulness is conditional rather than universal.',['bime','generic_cage_expert']),
            },'BiME organizes experts around Broad, but global admission remains too coarse.','Ask whether one relational object can explain the expert stack.',experts,children=[functional_loop,structural_loop,fusion_loop]),
            'recordIds':ordered(experts),'trackIds':[],
        },
        {
            'id':'fibre-detour','number':'04','eyebrow':'Side experiment',
            'title':'FIBRE tested a cleaner abstraction without replacing the main line.',
            'lead':'BiME made the stack work, but left a scientific question: what common relation are all these experts estimating? FIBRE explored that question as a side branch, then returned useful principles to the Broad-centered design.',
            'stack':{'upper':{'title':'BiME experts','note':'main line remains intact'},'lower':{'title':'Broad base order','note':'replacement target, not discarded'},'side':{'title':'FIBRE relational core','note':'detour / replacement hypothesis'}},
            'cageRole':'one evidence family among many','broadRole':'incumbent challenged by FIBRE',
            'primary':loop('fibre-replacement','Can one relational core replace the expert stack and Broad order?','No replacement',{
                'design':phase('Design','Model enzyme–reaction matching as one relational problem.',['fibre','bio_relation','context_domain']),
                'build':phase('Build','Explore conditional modes, explicit evidence and adaptive relational cores.',['conditional_modes','scientific_evidence','eram']),
                'test':phase('Test','Use strict temporal and double-cold replacement tests.',['relational_main','temporal_relational']),
                'learn':phase('Learn','Keep evidence admission, plugins and fallback; restore Broad authority.',['open_fallback','return_broad']),
            },'The replacement fails, while several interfaces survive.','Return to Broad and allocate expert authority locally.',fibre,children=[relation_loop,evidence_loop,fibre_core_loop]),
            'recordIds':ordered(fibre),'trackIds':[],'detour':True,
        },
        {
            'id':'bridge-authority','number':'05','eyebrow':'Final authority model',
            'title':'BRIDGE decides who may change Broad\'s order, where, and by how much.',
            'lead':'The final design keeps Broad globally valid and gives specialists bounded ranking rights only when the current query and direction support them.',
            'stack':{'upper':{'title':'Gated specialist evidence','note':'family CAGE, TPS, context, functional, structural'},'middle':{'title':'Applicability + permission','note':'query- and direction-specific authority'},'lower':{'title':'Broad base order','note':'global default remains valid'}},
            'cageRole':'family-specific specialist','broadRole':'global default order',
            'primary':loop('local-authority','Who may alter Broad\'s order for this query?','Local authority',{
                'design':phase('Design','Rebind every expert to Broad as optional pair evidence.',['pair_evidence','rebind_broad']),
                'build':phase('Build','Add query-conditioned permission and domain specialists.',['dynamic_v4','dynamic_v6','query_applicability','cage_family','tps_correction']),
                'test':phase('Test','Validate specialists only inside their applicability domains.',['p450_cage','phosphatase_cage','terpene_cage','layered_cage_eval']),
                'learn':phase('Learn','Broad stays global; local experts earn bounded correction rights.',['integrated_specialists','bridge']),
            },'BRIDGE is a global Broad order plus query-gated local authority.','The engineering story closes in the current architecture.',bridge,children=[permission_loop,family_loop,tps_loop,integration_loop]),
            'recordIds':ordered(bridge),'trackIds':[],
        },
    ]

    tracks=[{'id':'wetlab','label':'Wet-lab execution','recordIds':ordered(wetlab)},{'id':'compass','label':'COMPASS workflow','recordIds':ordered(compass)}]

    def walk_loops(item: dict):
        yield item
        for child in item.get('children', []):
            yield from walk_loops(child)

    for scene in scenes:
        for item in walk_loops(scene['primary']):
            for p in item['phases'].values():
                unknown=[i for i in p['keyIds'] if i not in by]
                if unknown:
                    raise ValueError(f"unknown phase ids in {item['id']}: {unknown}")

    covered=set().union(*(set(s['recordIds']) for s in scenes),*(set(t['recordIds']) for t in tracks))
    missing=set(by)-covered
    if missing:
        raise ValueError(f'presentation lost canonical records: {sorted(missing)}')

    architecture={
        'formula':'S_BRIDGE(q,e) = S_Broad(q,e) + Σ_k g_k(q) Δ_k(q,e)',
        'base':'Broad Retrieval','control':'Query applicability + permission',
        'experts':['Functional / evolutionary','Structural','Mechanistic / TPS','Context','Family-specific CAGE'],
        'correction':'Bounded pair-evidence correction',
    }
    return scenes,tracks,architecture


def write_data(scenes:list[dict],tracks:list[dict],architecture:dict)->None:
    payload={'nodes':NODES,'crossLinks':CROSSLINKS,'families':FAMILY_LABELS,'scenes':scenes,'tracks':tracks,'architecture':architecture,'meta':{'schema':'bridge-engineering-scenes-v9','root':'enzymecage','current':'bridge','presentation':'recursive-engineering-scenes'}}
    DATA_JS.write_text('window.LINEAGE_DATA = '+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')


def write_doc(by:dict[str,dict],scenes:list[dict],tracks:list[dict],architecture:dict)->None:
    lines=['# BRIDGE Engineering Story','','The public Engineering page reveals one primary DBTL loop at a time. Each loop may contain recursively nested subloops; raw experiment records are evidence at the leaves rather than the primary navigation.','']
    def emit(item:dict,level:int=3):
        lines.extend([f"{'#'*level} {item['title']}",'',f"**Outcome:** {item['outcome']}",f"**Why next:** {item['whyNext']}",''])
        for key in ['design','build','test','learn']:
            p=item['phases'][key];lines.append(f"- **{p['label']}:** {p['text']}")
        lines.append('')
        for child in item.get('children',[]):emit(child,min(level+1,6))
    for scene in scenes:
        lines.extend([f"## {scene['number']} · {scene['title']}",'',scene['lead'],''])
        emit(scene['primary'])
    lines.extend(['## BRIDGE today','',f"`{architecture['formula']}`",''])
    DOC.write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8')


def main()->None:
    by,children=validate();scenes,tracks,architecture=build_scenes(by,children);write_data(scenes,tracks,architecture);write_doc(by,scenes,tracks,architecture)
    represented=set().union(*(set(s['recordIds']) for s in scenes),*(set(t['recordIds']) for t in tracks))
    def depth(item:dict)->int:
        return 1+max((depth(c) for c in item.get('children',[])),default=0)
    print(json.dumps({'canonical_records':len(NODES),'represented_records':len(represented),'scenes':len(scenes),'max_loop_depth':max(depth(s['primary']) for s in scenes),'schema':'bridge-engineering-scenes-v9'},indent=2))


if __name__=='__main__':main()
