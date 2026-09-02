import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_corrected_xattn_family_is_closed_before_marts():
    x=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_TPS_ACTIVE_SITE_XATTN_V1R1_DEVELOPMENT_RESULT.json').read_text())
    assert x['status']=='rejected_corrected_development_gate_marts_one_shot_forbidden'
    assert x['trials_completed']==18 and x['selection_folds']==[0,1,2]
    assert x['selected_trial']==6
    assert x['development_deltas']['hit_at_10_pp'] < 3.0
    assert x['development_deltas']['mrr'] < 0
    assert x['development_deltas']['ndcg_at_10'] < 0
    assert x['development_gate']['passed'] is False
    assert x['forbidden_evidence']['marts_v2_1_family_specific_score_read'] is False
    assert x['forbidden_evidence']['marts_v2_1_one_shot_allowed'] is False
    assert x['no_post_hoc_retuning'] is True
