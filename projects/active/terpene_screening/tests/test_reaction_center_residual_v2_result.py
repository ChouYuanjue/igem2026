import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
RESULT=ROOT/'projects/active/terpene_screening/CLEANROOM_R2E_REACTION_CENTER_RESIDUAL_V2_RESULT.json'
CONFIRM=ROOT/'projects/active/terpene_screening/CLEANROOM_R2E_REACTION_CENTER_RESIDUAL_V2_CONFIRMATION_V1.json'

def test_v2_passes_every_frozen_development_check():
    r=json.loads(RESULT.read_text())
    assert r['status']=='passed_internal_development_gate'
    assert r['pass'] is True and all(r['checks'].values())
    assert r['fold_stability']['hard_both_improve_folds']==3
    assert r['identity_audit']['all_three_folds_max_abs_diff']==0.0
    assert r['identity_audit']['all_three_folds_trainable_parameter_names']==['aux_to_hidden.weight']
    assert r['pooled_lt0p3']['candidate']['mrr'] > r['pooled_lt0p3']['baseline']['mrr']
    assert r['pooled_lt0p3']['candidate']['hit_at_10'] >= r['pooled_lt0p3']['baseline']['hit_at_10']

def test_confirmation_is_fresh_fixed_and_not_an_outer_reuse():
    p=json.loads(CONFIRM.read_text())
    s=p['confirmation_split']
    assert p['status']=='frozen_before_confirmation_performance'
    assert s['split_salt']=='r2e_center_residual_v2_confirm_20260831'
    assert s['folds']==5 and s['dev_fold']==4
    assert s['salt_and_fold_frozen_before_materializing_confirmation_performance'] is True
    assert p['residual_recipe']['only_trainable_parameter']=='aux_to_hidden.weight'
    assert p['residual_recipe']['loss_candidate_scope']=='training_entities'
    assert p['confirmation_pass_rule']['no_alternative_salt_fold_or_hyperparameter_after_reveal'] is True
    assert 'genuinely fresh frozen external benchmark' in p['if_pass']
