import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_v3_candidates_and_fresh_splits_are_frozen_before_performance():
    p=json.loads((ROOT/'projects/active/terpene_screening/CLEANROOM_R2E_REACTION_CENTER_BOUNDED_RESIDUAL_V3.json').read_text())
    assert p['status']=='frozen_before_any_v3_split_performance_materialization'
    assert p['candidate_family']['max_residual_ratio_candidates']==[0.075,0.1,0.16]
    assert p['development_split']['split_salt']=='r2e_center_bounded_v3_dev_20260831' and p['development_split']['development_folds']==[0,1,2]
    assert p['future_confirmation_split']['split_salt']=='r2e_center_bounded_v3_confirm_20260831' and p['future_confirmation_split']['dev_fold']==3
    assert 'Rhea128->141 V2' in p['prohibited_evidence_for_selection']
    assert p['selection']['external_evaluation_after_internal_confirmation']=='not authorized by this protocol'
def test_geometry_candidate_derivation_is_train_only():
    g=json.loads((ROOT/'projects/active/terpene_screening/R2E_REACTION_CENTER_RESIDUAL_V2_TRAIN_GEOMETRY.json').read_text())
    assert g['target_outer_labels_used'] is False and g['source_model_frozen_before_rhea128_to141_reveal'] is True and g['n_train_reactions']==9750
    assert 0.07 < g['residual_to_base_hidden_norm_ratio']['p25'] < .075
    assert .09 < g['residual_to_base_hidden_norm_ratio']['p50'] < .1
    assert .15 < g['residual_to_base_hidden_norm_ratio']['p90'] < .16
