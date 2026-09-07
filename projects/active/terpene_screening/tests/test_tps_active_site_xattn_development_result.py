import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]

def test_frozen_xattn_development_result_allows_only_confirmation():
    x=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_TPS_ACTIVE_SITE_XATTN_V1_DEVELOPMENT_RESULT.json').read_text())
    assert x['status']=='passed_frozen_development_selection_confirmation_allowed'
    assert x['trials_completed']==18
    assert x['selection_folds']==[0,1,2]
    assert x['confirmation_folds_unread']==[3,4]
    assert x['confirmation_metrics_read'] is False
    assert x['selected_trial']==8
    assert x['development_gate']['passed'] is True
    assert x['development_deltas']['hit_at_10_pp'] >= 3.0
    assert x['selected_xattn']['mrr'] >= x['baseline']['mrr']
    assert x['selected_xattn']['ndcg_at_10'] >= x['baseline']['ndcg_at_10']
    assert any('immutable' in s for s in x['claim_boundary'])
