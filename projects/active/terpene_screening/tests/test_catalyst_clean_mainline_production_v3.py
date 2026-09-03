import json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[4]

def test_production_v3_is_confirmed_learned_fusion_default():
    p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_CLEAN_MAINLINE_PRODUCTION_V3.json').read_text())
    r=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_CLEAN_MAINLINE_PRODUCTION_V3_RESULT.json').read_text())
    assert p['status']=='promoted_confirmed_lambdarank_r2e_route'
    assert p['route_version']=='terpene-production-routes-v4'
    assert p['model_bundle_version']=='catalyst-r2e-lambdarank-fusion-v1'
    assert p['frozen_config']['config_id']=='cfg_07_392fe119'
    assert p['frozen_config']['pool_k']==p['frozen_config']['prefix_k']==100
    assert p['ranker_sha256']=='86b6fc7ff43fe1c59916dc6692cb38f513c877e1beed2c88902f00909cb7bb6e'
    assert p['cage_rescue_inside_confirmed_scope'] is False
    assert r['status']=='production_v3_ready'
    assert r['all_runtime_checks_pass'] is True and r['all_confirmation_checks_pass'] is True
    assert r['confirmation_delta_vs_v3_router']['mrr'] > .019
    assert r['confirmation_delta_vs_v3_router']['map'] > .020
    assert r['confirmation_delta_vs_v3_router']['hit_at_50'] > .066
    assert r['runtime_median_ratio_vs_v3'] < 1.3 and r['runtime_p95_ratio_vs_v3'] < 1.3

def test_live_manifest_is_v5_and_historical_v3_is_preserved():
    live=yaml.safe_load((ROOT/'configs/production_routes/terpene_v1.yaml').read_text())
    old=yaml.safe_load((ROOT/'configs/production_routes/terpene_similarity_router_v3.yaml').read_text())
    assert live['route_version']=='terpene-production-routes-v5'
    assert old['route_version']=='terpene-production-routes-v3'
    spec=live['routes']['reaction_to_enzyme']['external']['top10']
    assert spec['model_bundle_version']=='catalyst-r2e-lambdarank-fusion-v1'
    assert spec['lambdarank_fusion']['config_id']=='cfg_07_392fe119'
    assert spec['similarity_model_router']['threshold']==.9
