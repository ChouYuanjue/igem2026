from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
FRONT=ROOT/'frontend/starase_navigator'

def test_frontend_leaves_search_scope_and_evidence_depth_to_ai():
    html=(FRONT/'index.html').read_text()
    js=(FRONT/'app.js').read_text()
    assert 'id="observationDepth"' not in html
    assert 'observationDepth' not in js
    assert 'payload.body.observation_mode' not in js
    assert 'Search scope and evidence depth are chosen automatically from your question.' in html
    assert '检索范围和证据深度会根据你的问题自动选择。' in html
    assert 'AI-selected search scope' in js
    assert 'AI-selected evidence depth' in js
    assert 'Application-focused discovery' in js
    assert 'Broad discovery' in js

def test_frontend_does_not_expose_internal_retrieval_architecture_as_user_choices():
    html=(FRONT/'index.html').read_text()
    js=(FRONT/'app.js').read_text()
    combined=html+js
    for internal in ('marts_correspondence','tps_specialized','MARTS correspondence','Candidate universe'):
        assert internal not in combined
    assert '100 × [0.35·rank priority' not in js
    assert 'How this search was planned' in js
    assert 'Search scope' in js
    assert 'Scientific scoring' in js

def test_frontend_renders_observation_plan_as_executed_evidence_not_model_choices():
    js=(FRONT/'app.js').read_text()
    assert 'renderObservationPlan(result.observation_plan)' in js
    assert 'Available now' in js
    assert 'Planned for this request' in js
    assert 'Computed for this query' in js
    assert 'Additional evidence available if needed' in js
    assert 'Mechanistic evidence only; does not alter canonical ranking.' in js
    assert 'AI selected a deeper evidence pass' in js
    assert 'AI selected a focused evidence pass' in js


def test_candidate_side_cached_evidence_is_visible_without_architecture_terms():
    js=(FRONT/'app.js').read_text()
    assert 'Cached evidence across ranked candidates' in js
    assert '候选分子已缓存证据' in js
    assert 'Candidate-side cached measurements are reused wherever available' in js


def test_public_ranking_http_does_not_accept_architecture_routing_overrides():
    transport=(ROOT/'scripts/starase_navigator/http_transport.py').read_text()
    rank_start=transport.index('parsed.path == "/api/rank"')
    rank_end=transport.index('parsed.path == "/api/rank-family-reactions"', rank_start)
    block=transport[rank_start:rank_end]
    assert 'payload.get("route_mode")' not in block
    assert 'payload.get("observation_mode")' not in block
    assert 'route_mode="intelligent"' in block
    # Actual evidence depth remains semantic-planner-owned; this fallback input is
    # never allowed to bypass it from the public HTTP contract.
    assert 'observation_mode="standard"' in block
