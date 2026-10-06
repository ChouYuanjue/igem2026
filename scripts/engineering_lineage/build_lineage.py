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


def build_atlas(scenes:list[dict]) -> dict:
    """Build three coupled Atlas Engineering storylines.

    EDGE, BRIDGE and COMPASS are the only primary trunks. Cross-system work is
    embedded as junction loops inside those trunks; it never becomes a fourth
    public tree. The frontend renders the three trunks in parallel and uses
    phase-level handoffs to show where one system's Learn changes another
    system's Design.
    """

    def p(design:str,build:str,test:str,learn:str)->dict:
        return {
            'design':{'label':'Design','text':design},
            'build':{'label':'Build','text':build},
            'test':{'label':'Test','text':test},
            'learn':{'label':'Learn','text':learn},
        }

    def a(
        loop_id:str,
        title:str,
        systems:list[str],
        phases:dict,
        outcome:str,
        *,
        children:list[dict]|None=None,
        eyebrow:str='DBTL loop',
        evidence:list[str]|None=None,
        position:dict|None=None,
    )->dict:
        return {
            'id':loop_id,
            'title':title,
            'systems':systems,
            'eyebrow':eyebrow,
            'phases':phases,
            'outcome':outcome,
            'children':children or [],
            'evidence':evidence or [],
            'position':position,
        }

    # Preserve the established BRIDGE engineering spine and all of its recursive
    # subloops. BRIDGE remains the visual and methodological main trunk.
    bridge_children=[]
    for scene in scenes:
        item=json.loads(json.dumps(scene['primary']))
        item['systems']=['bridge']
        item['eyebrow']=scene['eyebrow']
        item['outcome']=item.get('outcome') or scene['lead']
        bridge_children.append(item)

    edge_foundation_children=[
        a('edge-schema-etl','Turn heterogeneous files into persistent biochemical entities',['edge'],p(
            'Give enzymes, reactions, compounds, evidence and their relations stable database identities.',
            'Build the SQL schema and ETL pipeline that materialize the source files into linked records.',
            'Rebuild the database and verify that linked entities survive ingestion as coherent objects.',
            'A usable biochemical database needs persistent entity identity before any graph experience can be trusted.'
        ),'EDGE acquires a stable biochemical data model.',
          evidence=['SQL schema','ETL for enzymes / reactions / compounds / edges']),
        a('edge-graph-api','Make relations first-class instead of table joins',['edge'],p(
            'Expose the biochemical relations users actually reason about, not just database rows.',
            'Build read and graph services over enzyme–reaction–compound relations.',
            'Traverse neighborhoods and inspect whether graph responses recover the intended biochemical context.',
            'The relation graph, not the individual table, is the natural unit of scientific exploration.'
        ),'EDGE becomes a relation graph.'),
        a('edge-search-pathways','Let users enter the graph through search and pathways',['edge'],p(
            'Make the graph reachable without knowing internal identifiers or SQL structure.',
            'Add parsed search, pathway traversal and channel/search services on top of the graph model.',
            'Test free-form lookup, entity navigation and pathway-level exploration.',
            'A graph only becomes useful when users can reliably find the right entry point and continue through connected context.'
        ),'EDGE becomes an explorable scientific portal.'),
    ]

    edge_scale_children=[
        a('edge-source-segmentation','Expand to Swiss-Prot and TrEMBL without flattening provenance',['edge'],p(
            'Increase coverage while preserving the difference between reviewed and unreviewed evidence.',
            'Split acquisition and ETL by source, add source/review state, and merge under explicit precedence rules.',
            'Compare source-segmented imports, duplicates, upgrades/downgrades and missing-source failure cases.',
            'Coverage is inseparable from provenance; source identity must survive every layer of the database.'
        ),'EDGE scales from a curated slice to a provenance-aware corpus.',
          evidence=['1,535 Swiss-Prot enzymes','94,334 TrEMBL enzymes','source precedence registry']),
        a('edge-persistent-identities','Keep entity identity stable while sources change',['edge'],p(
            'Prevent refreshes and source churn from silently changing internal entity identity.',
            'Introduce persistent enzyme ID maps, alias continuity and source-scoped replacement semantics.',
            'Refresh segmented data and verify that surviving entities retain stable identifiers.',
            'A growing database needs identity continuity across ingestion cycles, not merely correct rows in one snapshot.'
        ),'EDGE can evolve without breaking references.'),
        a('edge-search-scope','Make search scale with the graph rather than with raw row count',['edge'],p(
            'Keep interactive discovery useful after the graph grows by two orders of magnitude.',
            'Introduce search sets, streamed indexing, scoped BLAST/search and removal of silent result caps.',
            'Stress exact, prefix and broad scans over millions of search-index rows.',
            'At scale, search scope is part of query meaning and must be explicit rather than an accidental performance shortcut.'
        ),'EDGE learns to expose a large graph without changing the semantics of retrieval.'),
    ]

    edge_service_children=[
        a('edge-entity-bundle','Give one canonical answer for an entity and its relations',['edge'],p(
            'Stop rebuilding enzyme, compound and pathway cards independently in each route.',
            'Create shared entity services and bundle endpoints for canonical entity-plus-relation payloads.',
            'Compare bundle, detail, graph and pathway responses for identity and field consistency.',
            'Downstream systems need one canonical evidence object rather than several page-specific interpretations.'
        ),'EDGE becomes machine-readable as well as human-readable.'),
        a('edge-smiles-identity','Accept molecular structure without depending on the browser',['edge'],p(
            'Let API clients resolve compounds from SMILES while preserving chemical identity.',
            'Move SMILES→InChIKey conversion to the server using the same chemistry semantics as the database.',
            'Compare server resolution against stored structures and browser-derived keys, including invalid and generic structures.',
            'Molecular identity belongs in the evidence service, not in one frontend widget.'
        ),'EDGE gains a backend-native molecular identity entrance.'),
        a('edge-reproducible-data','Make the evidence graph reproducible from the repository',['edge'],p(
            'Ensure a clean checkout contains the segmented source material required by the declared ETL contract.',
            'Track the source-partitioned tables and formalize the dependency/rebuild workflow.',
            'Rebuild from a clean repository and reject missing-source partial imports.',
            'A database release is incomplete if its graph cannot be reconstructed without hidden local artifacts.'
        ),'EDGE becomes a reproducible evidence substrate.'),
    ]

    evaluation_children=[
        a('eval-fixed-edge','Test missing relations inside a fixed graph',['edge','bridge'],p(
            'Ask whether a model can recover held-out known relations while graph entities stay fixed.',
            'Construct relation-unseen missing-edge tasks over the observed graph.',
            'Rank hidden positives in both retrieval directions.',
            'This tests graph completion, but a deployed biochemical database also grows through new entities and later curation.'
        ),'Evaluation starts from graph completion and immediately exposes its static-world assumption.'),
        a('eval-cold-splits','Introduce unseen endpoints',['edge','bridge'],p(
            'Represent new proteins and reactions explicitly instead of hiding only edges.',
            'Create protein-cold, reaction-cold and double-cold relation groups.',
            'Compare directional retrieval across endpoint-exposure groups.',
            'Nominal coldness is not a monotone difficulty scale; it is entangled with degree, graph structure and candidate count.'
        ),'Endpoint exposure becomes visible, but its meaning is still confounded.'),
        a('eval-arrival-events','Treat database growth as arrival events',['edge','bridge'],p(
            'Move from independent future edges to groups of relations associated with a newly arriving entity.',
            'Build event-level future slices and aggregate retrieval by arrival event.',
            'Compare edge-level and event-level conclusions.',
            'Entity timestamps cannot recover later curation between already-existing endpoints, and the event set is too narrow.'
        ),'Growth must be reconstructed from relation history itself.'),
        a('eval-rhea-history','Replay official Rhea graph history',['edge','bridge'],p(
            'Observe how the biochemical relation graph actually changed instead of inferring history from present metadata.',
            'Reconstruct historical Rhea releases on a fixed deployment universe and take release-to-release graph differences.',
            'Separate within-graph completion from expansion through newly exposed endpoints and compare model/candidate-system coverage.',
            'Real database growth contains both completion and expansion, and closed candidate systems fail structurally on expansion.'
        ),'Evaluation becomes grounded in real graph history.',
          evidence=['Rhea historical releases','5,409 strict future relations','completion vs expansion']),
        a('eval-continuous-growth','Recover continuous curation events',['edge','bridge'],p(
            'Avoid reducing years of graph evolution to one before/after split.',
            'Replay releases 116–142 and group insertions by release and query event.',
            'Report edge metrics together with event-macro metrics.',
            'Database growth is clustered curation; large update events must not dominate conclusions simply because they contain more edges.'
        ),'The benchmark becomes a temporal graph-growth process.'),
        a('eval-directed-exposure','Separate query exposure from candidate exposure',['edge','bridge'],p(
            'Describe what was known at the immediately previous release in the direction the model is actually queried.',
            'Classify query and positive candidate independently as warm/cold for R2E and E2R.',
            'Compare all exposure quadrants in both directions.',
            'The same biological event changes interpretation when retrieval direction reverses; warm/cold describes information exposure, not scalar difficulty.'
        ),'Exposure becomes a directional explanatory variable.'),
        a('eval-candidate-cardinality','Separate knowledge exposure from search-space size',['edge','bridge'],p(
            'Prevent a smaller candidate set from masquerading as better generalization.',
            'Keep the complete deployment universe as the headline task and build a matched-size counterfactual by uniform negative subsampling.',
            'Compare exposure groups under full and matched candidate counts without model-generated prefiltering.',
            'Candidate cardinality can dominate apparent cold-start performance, so exposure labels must never silently change the ranking universe.'
        ),'The final graph-growth protocol fixes the deployment universe and treats candidate size only as a diagnostic.',
          evidence=['185,918 protein candidates','11,081 reaction candidates','matched-size counterfactual']),
    ]

    edge_compass_children=[
        a('edge-compass-direct','Serve canonical evidence directly to COMPASS',['edge','compass'],p(
            'Let the agent consume the same canonical entities and relations users see in EDGE.',
            'Connect COMPASS to EDGE over the server-local backend path and expose entity/evidence bundles.',
            'Compare identity, relation and literature payloads across both applications.',
            'The agent should reuse the database interpretation instead of maintaining a parallel evidence ontology.'
        ),'EDGE becomes the canonical evidence source for COMPASS.'),
        a('edge-compass-molecules','Carry molecular structures into research context only when needed',['edge','compass'],p(
            'Make structural context available without turning the agent interface into a database dashboard.',
            'Attach molecular structure previews lazily inside existing folded evidence surfaces.',
            'Check that hidden candidates remain visually compact and structures appear only on demand.',
            'Evidence richness and interface minimalism can coexist when molecular context is progressively disclosed.'
        ),'Molecular evidence becomes available without overwhelming the research workspace.'),
        a('edge-compass-provenance','Preserve database provenance inside later reasoning',['edge','compass'],p(
            'Keep source/review/literature status attached when EDGE evidence moves into COMPASS.',
            'Store canonical evidence as provenance-bearing workspace context rather than flattened prose.',
            'Reuse the same evidence in follow-up operations without re-resolving identity.',
            'Database facts become durable scientific state only when their provenance survives later reasoning.'
        ),'EDGE evidence becomes reusable research state.'),
    ]

    compass_portal_children=[
        a('compass-intent-router','Translate a scientific request into a model operation',['compass'],p(
            'Remove the need for users to memorize which model route, direction or parameter set answers a scientific question.',
            'Map natural-language requests onto explicit R2E/E2R retrieval operations and route parameters.',
            'Ask equivalent questions in different wording and verify they reach the same deterministic operation.',
            'The first useful role of the agent is routing convenience: the user should state the scientific goal, not the backend command.'
        ),'COMPASS begins as a portal over model routes.'),
        a('compass-user-scope','Let users express scope in scientific language',['compass'],p(
            'Accept organism, direction, known-association and depth preferences conversationally.',
            'Convert those preferences into explicit retrieval-plan fields before model execution.',
            'Vary scope language while checking deterministic backend constraints.',
            'Convenience only works if natural language becomes inspectable parameters rather than hidden prompt behavior.'
        ),'The portal evolves from command selection to explicit research-intent planning.'),
    ]

    compass_bounded_children=[
        a('compass-typed-actions','Give the agent typed tools instead of free-form execution',['compass'],p(
            'Let language understanding choose actions without giving it direct control over scientific state.',
            'Build typed tool contracts, a registry and explicit capability descriptions.',
            'Stress malformed actions, missing arguments and unsupported requests.',
            'The model can propose actions; deterministic code must own execution semantics.'
        ),'Agent planning gains a hard scientific boundary.'),
        a('compass-context-isolation','Keep conversations and context layers isolated',['compass'],p(
            'Prevent one session or one evidence layer from leaking into another scientific task.',
            'Separate conversation state, target context and tool-derived context in the session store.',
            'Run parallel and sequential sessions with overlapping entities and verify no cross-session contamination.',
            'A scientific agent needs explicit context layers, not one undifferentiated chat history.'
        ),'Conversation context becomes a controlled data structure.'),
        a('compass-action-recovery','Recover from invalid model actions without corrupting state',['compass'],p(
            'Make tool-planning failures survivable rather than fatal or silently wrong.',
            'Add controller action validation, repair and deterministic fallback behavior.',
            'Inject malformed actions and tool failures and inspect resulting state.',
            'Robustness means preserving scientific state through planner failure, not merely returning an answer.'
        ),'The agent can fail safely.'),
    ]

    compass_workspace_children=[
        a('compass-reusable-objects','Turn results into reusable workspace objects',['compass'],p(
            'Allow an enzyme, reaction, ranking or route produced in one step to become a first-class input to the next.',
            'Persist typed workspace objects and expose them back to the agent/tool layer.',
            'Reuse prior objects across follow-up requests without re-parsing prose.',
            'Scientific conversations are evolving object graphs, not isolated question-answer pairs.'
        ),'COMPASS becomes a persistent research workspace.'),
        a('compass-local-route-patching','Modify only the part of a route the user actually changes',['compass'],p(
            'Avoid regenerating an entire pathway when the user edits one local decision.',
            'Support AI-native local route patching against the existing verified route object.',
            'Change one segment and verify unaffected segments remain identical.',
            'Local scientific edits should preserve validated context instead of restarting the workflow.'
        ),'Route editing becomes incremental.'),
        a('compass-route-lineage','Remember where derived routes came from',['compass'],p(
            'Keep a modified route connected to the version from which it was derived.',
            'Store parent/derived route lineage inside workspace state.',
            'Create chained edits and verify ancestry remains recoverable.',
            'A reusable scientific object also needs provenance over its own transformations.'
        ),'Workspace objects acquire history, not just identity.'),
    ]

    compass_identity_children=[
        a('compass-stereochemistry','Normalize stereochemistry before applying constraints',['compass'],p(
            'Prevent superficially similar compound strings from changing route constraints incorrectly.',
            'Normalize compound stereochemistry in route design and constraint matching.',
            'Test stereochemical variants against route constraints.',
            'Natural-language sameness is weaker than chemical identity.'
        ),'Chemical identity becomes an explicit precondition for route reasoning.'),
        a('compass-verified-compounds','Bind route design to verified compound references',['compass'],p(
            'Stop pathway design from relying on unverified compound names extracted from conversation.',
            'Resolve route compounds to verified identities before using them as constraints or anchors.',
            'Compare ambiguous names, explicit structures and resolved references.',
            'The agent may interpret user language, but route state must attach to verified entities.'
        ),'Verified identity becomes part of scientific state.'),
        a('compass-pathway-ambiguity','Resolve pathway ambiguity with verified references',['compass'],p(
            'Handle requests where several biochemical pathways or compounds match the same conversational description.',
            'Use verified references and structured pathway context to disambiguate before modification.',
            'Test ambiguous pathway requests and follow-up edits.',
            'Ambiguity should be resolved by evidence-bearing references rather than by confident prose.'
        ),'Verified references become the basis of multi-step pathway reasoning.'),
    ]

    compass_evidence_children=[
        a('compass-evidence-counts','Preserve how much evidence actually supports a claim',['compass'],p(
            'Prevent bundled summaries from erasing whether a claim has zero, one or many supporting records.',
            'Carry evidence counts and source provenance through retrieval and presentation.',
            'Compare repeated answers and zero-evidence cases.',
            '“No recorded evidence” and “negative evidence” are different scientific states.'
        ),'Evidence amount becomes explicit state.'),
        a('compass-fact-model-separation','Keep database facts separate from model candidates',['compass','edge'],p(
            'Prevent a high model score from being presented as a known biochemical relation and prevent database absence from being treated as a model negative.',
            'Represent canonical facts, literature evidence and inferred candidates as different evidence layers.',
            'Inspect candidate answers with and without matching database relations.',
            'Scientific orchestration requires epistemic separation, not one blended confidence score.'
        ),'COMPASS gains an explicit evidence model.'),
    ]

    compass_observation_children=[
        a('compass-budget-planning','Plan evidence acquisition under a budget',['compass'],p(
            'Choose how much evidence to acquire based on user intent, cost and expected value.',
            'Add budget-aware planning modes and observation acquisition choices.',
            'Compare fast, deep and reproduction-oriented plans for the same target.',
            'Scientific planning is partly a resource-allocation problem, not only a routing problem.'
        ),'COMPASS begins managing the cost of knowing more.'),
        a('compass-verified-observations','Reuse observations once they have been verified',['compass'],p(
            'Avoid paying again—computationally or cognitively—for evidence already established in the workspace.',
            'Cache verified observations with target identity and provenance and expose them to later plans.',
            'Repeat related tasks and check that valid observations are reused while incompatible ones are rejected.',
            'A scientific agent needs memory of verified observations, not just memory of conversation text.'
        ),'Observation state becomes reusable across research steps.'),
        a('compass-source-bound','Keep extracted claims bound to their source context',['compass'],p(
            'Prevent an extracted publication fact from drifting beyond the source passage that justified it.',
            'Store source-bound publication context and fact provenance with observations.',
            'Re-use extracted facts in later reasoning and verify the original source binding remains visible.',
            'Reusable observations need source scope as well as a value.'
        ),'Observation reuse becomes provenance-safe.'),
    ]

    compass_eval_children=[
        a('compass-history-eval','Evaluate long-horizon agent behavior',['compass'],p(
            'Test whether multi-step scientific state survives a realistic sequence of user operations.',
            'Replay agent histories and inspect tool choices, state transitions and resulting objects.',
            'Compare expected scientific invariants across complete histories rather than one answer.',
            'Agent quality lives in state evolution, not sentence similarity.'
        ),'COMPASS acquires a workflow-level test harness.'),
        a('compass-metamorphic-eval','Perturb context and wording while holding the science fixed',['compass'],p(
            'Check whether equivalent scientific situations remain equivalent when prompts, ordering or nonessential context change.',
            'Build metamorphic evaluation over targets, context and tool-failure conditions.',
            'Apply controlled perturbations and assert invariant scientific constraints.',
            'A robust agent should preserve scientific invariants under interaction variability.'
        ),'The agent gains a self-correcting engineering feedback loop.'),
    ]

    compass_bridge_children=[
        a('compass-bridge-scope','Keep semantic scope independent of candidate cardinality',['compass','bridge'],p(
            'Let COMPASS interpret an application-domain request without quietly making BRIDGE easier by shrinking its search space.',
            'Separate semantic scope/evidence depth from the full BRIDGE deployment candidate universe.',
            'Compare planned route, actual route and candidate counts for application and broad requests.',
            'Intent may change the question, but candidate cardinality is an independent experimental axis.'
        ),'The agent/model boundary inherits the final evaluation principle.'),
        a('compass-bridge-authority','Give ranking authority back to the complete BRIDGE runtime',['compass','bridge'],p(
            'Use one canonical inference system rather than historical partial routes selected by the agent.',
            'Bind registered queries to the final Broad + gated-expert runtime in both directions.',
            'Verify full-universe R2E/E2R routing, model provenance and expert activation.',
            'COMPASS should decide what research operation to request; BRIDGE should decide how evidence may alter ranking.'
        ),'Orchestration and inference authority become cleanly separated.'),
        a('compass-bridge-feedback','Let verified research state design the next model call',['compass','bridge'],p(
            'Use confirmed entities, masks, taxonomy and evidence state to formulate a subsequent retrieval without changing base model semantics.',
            'Carry structured workspace context into later BRIDGE requests.',
            'Run chained R2E/E2R searches and context perturbations.',
            'BRIDGE becomes a repeatable operation inside a larger scientific feedback loop.'
        ),'The model participates in an iterative research workflow rather than a one-shot portal.'),
    ]

    edge_children=[
        a('edge-foundation','Build a biochemical graph people can actually enter',['edge'],p(
            'Turn heterogeneous biochemical records into a persistent graph that supports real exploration.',
            'Develop the data model, ETL, graph APIs, search and pathway entry points.',
            'Rebuild, query and traverse the graph from several user-facing entrances.',
            'Identity, relations and discoverability have to be engineered together.'
        ),'EDGE grows from stored records into an explorable relation graph.',
          children=edge_foundation_children,eyebrow='EDGE trunk'),
        a('edge-scale','Scale the graph without erasing provenance',['edge'],p(
            'Expand coverage by orders of magnitude while keeping source meaning, identity continuity and search semantics intact.',
            'Introduce source segmentation, persistent IDs and scope-aware search over Swiss-Prot and TrEMBL.',
            'Refresh and query the expanded graph under mixed-source conditions.',
            'A larger graph is only useful if provenance and retrieval meaning survive scale.'
        ),'EDGE becomes a source-aware large graph.',
          children=edge_scale_children,eyebrow='EDGE trunk'),
        a('edge-service','Turn the graph into a canonical evidence service',['edge'],p(
            'Make canonical entities and relations consumable by software as well as by the database frontend.',
            'Unify entity assembly, bundle endpoints, molecular identity resolution and reproducible data assets.',
            'Compare human-facing and machine-facing representations of the same entities.',
            'The database should have one evidence interpretation no matter who consumes it.'
        ),'EDGE becomes Atlas’s canonical evidence substrate.',
          children=edge_service_children,eyebrow='EDGE trunk'),
        a('edge-evaluation','Evaluate a graph that keeps growing',['edge','bridge'],p(
            'Make retrieval evaluation follow the way biochemical knowledge actually accumulates.',
            'Progress from fixed missing-edge tests to official Rhea history, event growth, directional exposure and candidate-cardinality decomposition.',
            'Replay historical graph growth while BRIDGE ranks against the fixed deployment universe.',
            'Evaluation belongs to the knowledge graph story: the benchmark must model how the known graph changes while inference searches beyond it.'
        ),'EDGE defines the moving knowledge boundary; BRIDGE is tested against that boundary.',
          children=evaluation_children,eyebrow='EDGE trunk · BRIDGE junction'),
        a('edge-compass-junction','Make graph evidence reusable inside scientific reasoning',['edge','compass'],p(
            'Let canonical database evidence enter a persistent research workspace without losing identity or provenance.',
            'Connect EDGE bundles and molecular context directly into COMPASS and preserve them as structured evidence.',
            'Reuse the same evidence through follow-up scientific operations.',
            'The graph becomes more valuable when its facts persist as verified research state.'
        ),'EDGE’s evidence layer joins the COMPASS research loop.',
          children=edge_compass_children,eyebrow='EDGE trunk · COMPASS junction'),
    ]

    compass_children=[
        a('compass-portal','Hide model-routing complexity from the user',['compass'],p(
            'Let a scientist ask for a result without memorizing the model’s routes, directions and control vocabulary.',
            'Build a conversational portal that translates scientific intent into explicit retrieval operations.',
            'Ask equivalent questions in different language and inspect the resolved operation.',
            'The first reason for the agent to exist is convenience: scientific intent should be enough to enter the model.'
        ),'The model gains a usable portal.',
          children=compass_portal_children,eyebrow='COMPASS trunk'),
        a('compass-bounded','Turn the portal into a bounded scientific agent',['compass'],p(
            'Expand beyond route selection without giving a language model uncontrolled authority over scientific state.',
            'Introduce typed tools, layered context, session isolation and action recovery.',
            'Stress malformed plans, concurrent contexts and tool failures.',
            'Useful agency requires hard boundaries around identity, execution and state mutation.'
        ),'COMPASS becomes an agent that can fail safely.',
          children=compass_bounded_children,eyebrow='COMPASS trunk'),
        a('compass-workspace','Turn conversations into a reusable research workspace',['compass'],p(
            'Make results persist as objects that can be edited, reused and traced across subsequent tasks.',
            'Add reusable workspace objects, local route patching and derived-route lineage.',
            'Perform chained edits while checking that verified unaffected state survives.',
            'Research progresses by transforming persistent objects, not by repeatedly regenerating prose.'
        ),'COMPASS acquires persistent research objects.',
          children=compass_workspace_children,eyebrow='COMPASS trunk'),
        a('compass-identity','Require verified identity before multi-step reasoning',['compass','edge'],p(
            'Prevent compound names and pathway references from drifting as the workspace becomes more stateful.',
            'Normalize stereochemistry, bind route design to verified compounds and resolve pathway ambiguity with structured references.',
            'Test ambiguous names, stereochemical variants and derived route edits.',
            'Persistent state is only useful when the entities inside it are chemically and biologically well defined.'
        ),'Verified identity becomes the anchor of the workspace.',
          children=compass_identity_children,eyebrow='COMPASS trunk · EDGE junction'),
        a('compass-evidence','Represent what is known, inferred and unsupported separately',['compass','edge'],p(
            'Keep database facts, literature support and model candidates from collapsing into one answer confidence.',
            'Preserve evidence counts, source provenance and explicit evidence layers.',
            'Compare zero-evidence, known-relation and model-only candidate cases.',
            'A scientific agent must track epistemic state, not just content.'
        ),'COMPASS acquires an explicit evidence state model.',
          children=compass_evidence_children,eyebrow='COMPASS trunk · EDGE junction'),
        a('compass-observation','Manage the lifecycle and cost of scientific observations',['compass'],p(
            'Decide which evidence is worth acquiring and keep useful observations alive across later work.',
            'Add budget-aware planning, verified observation reuse and source-bound extraction.',
            'Repeat related tasks under different budgets and contexts.',
            'Scientific memory is a set of verified observations with cost and provenance, not a transcript.'
        ),'COMPASS starts managing research state rather than chat state.',
          children=compass_observation_children,eyebrow='COMPASS trunk'),
        a('compass-evaluation','Evaluate whether the agent preserves scientific invariants',['compass'],p(
            'Test the agent as a stateful research process instead of scoring one generated answer.',
            'Build history-level and metamorphic evaluation with failure-safe execution.',
            'Perturb wording, context and tool outcomes while holding scientific constraints fixed.',
            'The agent needs its own DBTL loop: failures in invariant behavior must feed back into its design.'
        ),'COMPASS becomes self-evaluating.',
          children=compass_eval_children,eyebrow='COMPASS trunk'),
        a('compass-bridge-junction','Bind scientific orchestration to the complete BRIDGE model',['compass','bridge'],p(
            'Connect the mature research workspace to one stable inference authority without reintroducing hidden candidate filtering.',
            'Separate semantic scope from candidate size and route registered entities through the final BRIDGE runtime.',
            'Verify bidirectional route identity, full candidate counts and model provenance.',
            'COMPASS may decide what to ask and what context to carry; BRIDGE owns ranking authority.'
        ),'The original portal closes into a full research loop around BRIDGE.',
          children=compass_bridge_children,eyebrow='COMPASS trunk · BRIDGE junction'),
    ]

    edge=a('edge-program','Atlas EDGE',['edge'],p(
        'Build the system that defines and exposes the current known biochemical graph.',
        'Evolve storage, graph access, scale semantics, evidence services and growth-aware evaluation.',
        'Stress the graph through users, model evaluation and downstream evidence consumers.',
        'EDGE is the moving knowledge boundary of Atlas.'
    ),'Known graph and evidence boundary.',children=edge_children,eyebrow='Primary storyline',
      position={'row':1,'column':1,'span':1})

    bridge=a('bridge-program','BRIDGE',['bridge'],p(
        'Search beyond the known graph without surrendering a stable broad ranking foundation.',
        'Develop Broad retrieval and progressively condition specialist authority.',
        'Stress the system under open-world, family-specific and temporal graph-growth evaluation.',
        'A broad order should remain stable while query-relevant experts earn bounded correction rights.'
    ),'Main inference trunk.',children=bridge_children,eyebrow='Primary storyline · main trunk',
      position={'row':1,'column':2,'span':1})

    compass=a('compass-program','Atlas COMPASS',['compass'],p(
        'Remove model-routing burden from users, then grow that portal into a stateful scientific research agent.',
        'Develop bounded planning, reusable objects, verified identity, evidence state, observation reuse and agent evaluation.',
        'Stress the workflow through repeated scientific tasks and direct integration with EDGE and BRIDGE.',
        'COMPASS is the research-state layer around the model and database.'
    ),'Scientific orchestration trunk.',children=compass_children,eyebrow='Primary storyline',
      position={'row':1,'column':3,'span':1})

    root=a('atlas-root','Atlas Engineering',['edge','bridge','compass'],p(
        'Engineer one coupled system with three persistent lines of authority: known graph, inference and scientific orchestration.',
        'Grow EDGE, BRIDGE and COMPASS in parallel and connect them only where one system’s Learn changes another system’s next Design.',
        'Evaluate each trunk locally and test the junctions without creating a fourth “shared” subsystem.',
        'The coherent story is three evolving trunks whose feedback loops repeatedly cross and rejoin.'
    ),'Three coupled engineering storylines.',children=[edge,bridge,compass],eyebrow='Atlas system')

    handoffs=[
        # EDGE trunk.
        {'from':{'loop':'edge-schema-etl','phase':'learn'},'to':{'loop':'edge-graph-api','phase':'design'},'label':'stable entities make graph relations meaningful'},
        {'from':{'loop':'edge-graph-api','phase':'learn'},'to':{'loop':'edge-search-pathways','phase':'design'},'label':'a relation graph needs usable entry points'},
        {'from':{'loop':'edge-foundation','phase':'learn'},'to':{'loop':'edge-scale','phase':'design'},'label':'a useful graph immediately exposes the cost of scale'},
        {'from':{'loop':'edge-source-segmentation','phase':'learn'},'to':{'loop':'edge-persistent-identities','phase':'design'},'label':'source churn makes identity continuity necessary'},
        {'from':{'loop':'edge-persistent-identities','phase':'learn'},'to':{'loop':'edge-search-scope','phase':'design'},'label':'large stable graphs need explicit search scope'},
        {'from':{'loop':'edge-scale','phase':'learn'},'to':{'loop':'edge-service','phase':'design'},'label':'a large graph needs one canonical machine-readable interpretation'},
        {'from':{'loop':'edge-entity-bundle','phase':'learn'},'to':{'loop':'edge-smiles-identity','phase':'design'},'label':'canonical entities need a backend-native molecular entrance'},
        {'from':{'loop':'edge-smiles-identity','phase':'learn'},'to':{'loop':'edge-reproducible-data','phase':'design'},'label':'machine use makes reproducibility part of the interface contract'},
        {'from':{'loop':'edge-service','phase':'learn'},'to':{'loop':'edge-evaluation','phase':'design'},'label':'once the graph is explicit, its growth can become an evaluation object'},
        {'from':{'loop':'eval-fixed-edge','phase':'learn'},'to':{'loop':'eval-cold-splits','phase':'design'},'label':'completion is not graph growth'},
        {'from':{'loop':'eval-cold-splits','phase':'learn'},'to':{'loop':'eval-arrival-events','phase':'design'},'label':'coldness mixes several kinds of novelty'},
        {'from':{'loop':'eval-arrival-events','phase':'learn'},'to':{'loop':'eval-rhea-history','phase':'design'},'label':'relation history is more faithful than entity timestamps'},
        {'from':{'loop':'eval-rhea-history','phase':'learn'},'to':{'loop':'eval-continuous-growth','phase':'design'},'label':'one before/after split hides curation events'},
        {'from':{'loop':'eval-continuous-growth','phase':'learn'},'to':{'loop':'eval-directed-exposure','phase':'design'},'label':'event growth exposes query/candidate asymmetry'},
        {'from':{'loop':'eval-directed-exposure','phase':'learn'},'to':{'loop':'eval-candidate-cardinality','phase':'design'},'label':'exposure and search-space size are different axes'},
        {'from':{'loop':'edge-evaluation','phase':'learn'},'to':{'loop':'edge-compass-junction','phase':'design'},'label':'a moving knowledge boundary needs reusable canonical evidence'},
        {'from':{'loop':'edge-compass-direct','phase':'learn'},'to':{'loop':'edge-compass-molecules','phase':'design'},'label':'canonical entities make richer molecular context safe'},
        {'from':{'loop':'edge-compass-molecules','phase':'learn'},'to':{'loop':'edge-compass-provenance','phase':'design'},'label':'rich context must remain provenance-bearing'},

        # BRIDGE trunk.
        {'from':{'loop':'candidate-system','phase':'learn'},'to':{'loop':'broad-cage-stack','phase':'design'},'label':'candidate recall ceiling forces open retrieval'},
        {'from':{'loop':'broad-cage-stack','phase':'learn'},'to':{'loop':'expert-delegation','phase':'design'},'label':'ranking responsibility moves into Broad'},
        {'from':{'loop':'expert-delegation','phase':'learn'},'to':{'loop':'fibre-replacement','phase':'design'},'label':'conditional experts motivate a relational abstraction'},
        {'from':{'loop':'fibre-replacement','phase':'learn'},'to':{'loop':'local-authority','phase':'design'},'label':'failed replacement returns authority to Broad'},

        # COMPASS trunk.
        {'from':{'loop':'compass-intent-router','phase':'learn'},'to':{'loop':'compass-user-scope','phase':'design'},'label':'routing convenience must become explicit scientific intent'},
        {'from':{'loop':'compass-portal','phase':'learn'},'to':{'loop':'compass-bounded','phase':'design'},'label':'a useful portal invites broader agency and therefore stronger boundaries'},
        {'from':{'loop':'compass-typed-actions','phase':'learn'},'to':{'loop':'compass-context-isolation','phase':'design'},'label':'typed actions still need clean context ownership'},
        {'from':{'loop':'compass-context-isolation','phase':'learn'},'to':{'loop':'compass-action-recovery','phase':'design'},'label':'stateful tools must survive planner failure'},
        {'from':{'loop':'compass-bounded','phase':'learn'},'to':{'loop':'compass-workspace','phase':'design'},'label':'bounded actions become more useful when outputs persist'},
        {'from':{'loop':'compass-reusable-objects','phase':'learn'},'to':{'loop':'compass-local-route-patching','phase':'design'},'label':'persistent routes make local editing possible'},
        {'from':{'loop':'compass-local-route-patching','phase':'learn'},'to':{'loop':'compass-route-lineage','phase':'design'},'label':'derived routes need transformation history'},
        {'from':{'loop':'compass-workspace','phase':'learn'},'to':{'loop':'compass-identity','phase':'design'},'label':'persistent objects make identity errors persistent too'},
        {'from':{'loop':'compass-stereochemistry','phase':'learn'},'to':{'loop':'compass-verified-compounds','phase':'design'},'label':'normalized strings still need verified entity binding'},
        {'from':{'loop':'compass-verified-compounds','phase':'learn'},'to':{'loop':'compass-pathway-ambiguity','phase':'design'},'label':'verified compounds enable evidence-based pathway disambiguation'},
        {'from':{'loop':'compass-identity','phase':'learn'},'to':{'loop':'compass-evidence','phase':'design'},'label':'verified entities reveal the need to separate evidence states'},
        {'from':{'loop':'compass-evidence-counts','phase':'learn'},'to':{'loop':'compass-fact-model-separation','phase':'design'},'label':'amount of evidence is meaningless if fact and inference are conflated'},
        {'from':{'loop':'compass-evidence','phase':'learn'},'to':{'loop':'compass-observation','phase':'design'},'label':'explicit evidence state turns acquisition into a planning problem'},
        {'from':{'loop':'compass-budget-planning','phase':'learn'},'to':{'loop':'compass-verified-observations','phase':'design'},'label':'costly observations should be reused once acquired'},
        {'from':{'loop':'compass-verified-observations','phase':'learn'},'to':{'loop':'compass-source-bound','phase':'design'},'label':'reused observations need durable source scope'},
        {'from':{'loop':'compass-observation','phase':'learn'},'to':{'loop':'compass-evaluation','phase':'design'},'label':'stateful scientific memory needs invariant evaluation'},
        {'from':{'loop':'compass-history-eval','phase':'learn'},'to':{'loop':'compass-metamorphic-eval','phase':'design'},'label':'history replay still needs controlled perturbations'},
        {'from':{'loop':'compass-evaluation','phase':'learn'},'to':{'loop':'compass-bridge-junction','phase':'design'},'label':'a mature agent needs one stable inference contract'},
        {'from':{'loop':'compass-bridge-scope','phase':'learn'},'to':{'loop':'compass-bridge-authority','phase':'design'},'label':'full-universe semantics require one canonical model authority'},
        {'from':{'loop':'compass-bridge-authority','phase':'learn'},'to':{'loop':'compass-bridge-feedback','phase':'design'},'label':'stable model authority makes iterative research context safe'},

        # Cross-trunk handoffs: these are junctions, never separate trees.
        {'from':{'loop':'local-authority','phase':'learn'},'to':{'loop':'edge-evaluation','phase':'design'},'label':'open ranking needs a deployment-realistic moving graph benchmark'},
        {'from':{'loop':'eval-candidate-cardinality','phase':'learn'},'to':{'loop':'compass-bridge-scope','phase':'design'},'label':'semantic intent must not silently shrink the model task'},
        {'from':{'loop':'compass-evidence','phase':'learn'},'to':{'loop':'edge-compass-junction','phase':'design'},'label':'explicit evidence state needs a canonical database substrate'},
        {'from':{'loop':'edge-compass-provenance','phase':'learn'},'to':{'loop':'compass-observation','phase':'design'},'label':'canonical database facts can become reusable observations'},
        {'from':{'loop':'local-authority','phase':'learn'},'to':{'loop':'compass-bridge-authority','phase':'design'},'label':'query-gated expert authority defines what the agent may request but not override'},
    ]

    return {
        'root':root,
        'handoffs':handoffs,
        'systems':{
            'edge':{'label':'EDGE','role':'Known graph'},
            'bridge':{'label':'BRIDGE','role':'Main inference trunk'},
            'compass':{'label':'COMPASS','role':'Scientific orchestration'},
        },
        'storylines':[
            {'id':'edge','root':'edge-program','label':'EDGE','role':'Known graph','lane':'upper','weight':1.0},
            {'id':'bridge','root':'bridge-program','label':'BRIDGE','role':'Main inference trunk','lane':'middle','weight':1.65},
            {'id':'compass','root':'compass-program','label':'COMPASS','role':'Scientific orchestration','lane':'lower','weight':1.0},
        ],
        'presentation':'three-storylines-fixed-focus',
    }


def validate_atlas(atlas:dict)->None:
    by:dict[str,dict]={}
    parent:dict[str,str|None]={}

    def walk(item:dict,parent_id:str|None=None)->None:
        loop_id=str(item.get('id') or '')
        if not loop_id:
            raise ValueError('Atlas loop missing id')
        if loop_id in by:
            raise ValueError(f'duplicate Atlas loop id: {loop_id}')
        by[loop_id]=item
        parent[loop_id]=parent_id
        phases=item.get('phases') or {}
        missing={'design','build','test','learn'}-set(phases)
        if missing:
            raise ValueError(f'Atlas loop {loop_id} missing DBTL phases: {sorted(missing)}')
        for phase_name in ('design','build','test','learn'):
            if not str((phases.get(phase_name) or {}).get('text') or '').strip():
                raise ValueError(f'Atlas loop {loop_id} has empty {phase_name} phase')
        for child in item.get('children') or []:
            walk(child,loop_id)

    walk(atlas['root'])

    allowed_phases={'design','build','test','learn'}
    for handoff in atlas.get('handoffs') or []:
        source=handoff.get('from') or {}
        target=handoff.get('to') or {}
        if source.get('loop') not in by:
            raise ValueError(f"Atlas handoff has unknown source loop: {source}")
        if target.get('loop') not in by:
            raise ValueError(f"Atlas handoff has unknown target loop: {target}")
        if source.get('phase') not in allowed_phases:
            raise ValueError(f"Atlas handoff has invalid source phase: {source}")
        if target.get('phase') not in allowed_phases:
            raise ValueError(f"Atlas handoff has invalid target phase: {target}")
        if not str(handoff.get('label') or '').strip():
            raise ValueError(f'Atlas handoff missing causal label: {handoff}')

    top=atlas['root'].get('children') or []
    expected_top=['edge-program','bridge-program','compass-program']
    actual_top=[item['id'] for item in top]
    if actual_top != expected_top:
        raise ValueError(
            f'Atlas public root must contain only EDGE/BRIDGE/COMPASS in that order: {actual_top}'
        )
    missing_positions=[item['id'] for item in top if not item.get('position')]
    if missing_positions:
        raise ValueError(f'Atlas top-level loops need fixed positions: {missing_positions}')

    storylines=atlas.get('storylines') or []
    expected_storylines=['edge','bridge','compass']
    actual_storylines=[item.get('id') for item in storylines]
    if actual_storylines != expected_storylines:
        raise ValueError(
            f'Atlas storylines must be EDGE/BRIDGE/COMPASS only: {actual_storylines}'
        )
    bridge_line=next(item for item in storylines if item.get('id') == 'bridge')
    if float(bridge_line.get('weight') or 0) <= 1:
        raise ValueError('BRIDGE storyline must remain visually dominant')


def write_data(scenes:list[dict],tracks:list[dict],architecture:dict,atlas:dict)->None:
    payload={'nodes':NODES,'crossLinks':CROSSLINKS,'families':FAMILY_LABELS,'scenes':scenes,'tracks':tracks,'architecture':architecture,'atlas':atlas,'meta':{'schema':'atlas-engineering-storylines-v11','root':'atlas-root','current':'bridge-program','presentation':'three-storylines-fixed-focus'}}
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
    by,children=validate();scenes,tracks,architecture=build_scenes(by,children);atlas=build_atlas(scenes);validate_atlas(atlas);write_data(scenes,tracks,architecture,atlas);write_doc(by,scenes,tracks,architecture)
    represented=set().union(*(set(s['recordIds']) for s in scenes),*(set(t['recordIds']) for t in tracks))
    def depth(item:dict)->int:
        return 1+max((depth(c) for c in item.get('children',[])),default=0)
    print(json.dumps({'canonical_records':len(NODES),'represented_records':len(represented),'scenes':len(scenes),'max_loop_depth':max(depth(s['primary']) for s in scenes),'schema':'atlas-engineering-storylines-v11','atlas_top_level_loops':len(atlas['root']['children']),'atlas_handoffs':len(atlas['handoffs'])},indent=2))


if __name__=='__main__':main()
