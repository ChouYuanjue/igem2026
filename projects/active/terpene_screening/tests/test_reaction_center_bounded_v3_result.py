import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_v3_result_selects_preregistered_cap_and_confirmation():
 r=json.loads((ROOT/'projects/active/terpene_screening/CLEANROOM_R2E_REACTION_CENTER_BOUNDED_RESIDUAL_V3_RESULT.json').read_text())
 assert r['status']=='passed_internal_development_gate' and r['selected_max_residual_ratio']==0.1 and r['passing_candidate_count']==3
 assert r['revealed_outer_metrics_used_for_selection'] is False and r['confirmation_dev_fold']==3
def test_confirmation_is_pre_performance_frozen():
 p=json.loads((ROOT/'projects/active/terpene_screening/CLEANROOM_R2E_REACTION_CENTER_BOUNDED_RESIDUAL_V3_CONFIRMATION_V1.json').read_text())
 assert p['status']=='frozen_before_confirmation_performance_materialization' and p['selected_max_residual_ratio']==0.1
 assert p['confirmation_split']['split_salt']=='r2e_center_bounded_v3_confirm_20260831' and p['confirmation_split']['dev_fold']==3
