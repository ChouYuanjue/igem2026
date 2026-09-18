from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
LANGUAGE=(ROOT/'scripts/starase_navigator/routing/language.py').read_text()
TOOLS=(ROOT/'scripts/starase_navigator/agent_harness/tool_registry.py').read_text()
FRONT_JS=(ROOT/'frontend/starase_navigator/app.js').read_text()
FRONT_HTML=(ROOT/'frontend/starase_navigator/index.html').read_text()

def test_semantic_route_prompts_do_not_expose_backend_architecture():
    route_region=LANGUAGE[LANGUAGE.index('def select_e2r_route'):LANGUAGE.index('def parse', LANGUAGE.index('def select_e2r_route')) if 'def parse' in LANGUAGE[LANGUAGE.index('def select_e2r_route'):] else len(LANGUAGE)]
    for internal in ('candidate_universe', 'marts_correspondence', 'tps_specialized', 'MARTS correspondence', 'special atlas'):
        assert internal not in route_region
    assert 'retrieval_scope in broad or application_domain' in route_region
    assert 'analysis_depth in standard or deep' in route_region

def test_candidate_search_is_semantic_not_trigger_word_gated():
    assert 'Infer exploratory/predictive intent from the meaning of the request, not from literal trigger words.' in TOOLS
    assert 'do not require, search for, or pattern-match any particular trigger words' in LANGUAGE
    assert 'Use only for explicit possible/potential/new/unrecorded/model-ranked candidate requests.' not in TOOLS

def test_browser_has_no_architecture_or_manual_depth_selector():
    combined=FRONT_JS+FRONT_HTML
    assert 'observationDepth' not in combined
    for internal in ('marts_correspondence', 'tps_specialized', 'MARTS correspondence', 'Candidate universe'):
        assert internal not in combined
    assert 'Search scope and evidence depth are chosen automatically from your question.' in FRONT_HTML
    assert 'Route catalog' not in FRONT_HTML+FRONT_JS
    assert 'routeCatalog' not in FRONT_JS
    assert '/api/routes' not in FRONT_JS
    assert 'AI search plan' in FRONT_HTML
    assert 'Only the plan actually used for this request is shown here; there is no workflow to choose in advance.' in FRONT_HTML


def test_application_domain_selection_is_based_on_verified_context_not_magic_user_phrase():
    from projects.active.fibre.core.candidate_universes import MARTS_CORRESPONDENCE_UNIVERSE
    from scripts.starase_navigator.routing.reaction_to_enzyme import RoutePlanner
    from scripts.starase_navigator.routing.enzyme_to_reaction import E2RRoutePlanner

    # The user never names a model, atlas, special scope, or "focused" mode.
    r2e=RoutePlanner(
        proposal_fn=lambda *_a,**_k:{
            '_semantic_source':'deepseek','top_k':10,'retrieval_scope':'application_domain',
            'analysis_depth':'standard','seed_mode':'none','known_association_policy':'separate_known',
            'reason':'the verified chemistry is inside the supported terpene-catalysis domain',
        },
        protein_ids={'P1'},
    )
    rplan=r2e.plan(
        user_text='帮我找这个反应最可能的候选酶。',
        reaction_equation='geranylgeranyl diphosphate -> diterpene cyclization product',
        route_mode='intelligent',is_current=False,orientation='forward',
    )
    assert rplan['retrieval_scope']=='application_domain'
    assert rplan['candidate_universe']==MARTS_CORRESPONDENCE_UNIVERSE

    e2r=E2RRoutePlanner(proposal_fn=lambda *_a,**_k:{
        '_semantic_source':'deepseek','top_k':10,'retrieval_scope':'application_domain',
        'analysis_depth':'standard','seed_mode':'catalog_known','known_association_policy':'separate_known',
        'reason':'the verified protein and activities identify the supported terpene-catalysis domain',
    })
    eplan=e2r.plan(
        user_text='这个酶还可能催化哪些反应？',route_mode='intelligent',is_current=True,
        catalog_known_reactions=['RHEA:54512'],
        target_context={'protein':{'name':'terpene synthase'},'recorded_reactions':[{'reaction_id':'RHEA:54512','name':'terpene cyclization'}]},
    )
    assert eplan['retrieval_scope']=='application_domain'
    assert eplan['candidate_universe']==MARTS_CORRESPONDENCE_UNIVERSE
