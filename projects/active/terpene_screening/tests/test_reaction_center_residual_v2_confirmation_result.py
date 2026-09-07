import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
PATH=ROOT/'projects/active/terpene_screening/CLEANROOM_R2E_REACTION_CENTER_RESIDUAL_V2_CONFIRMATION_RESULT.json'

def test_confirmation_result_passes_all_frozen_checks_and_disables_retuning():
    r=json.loads(PATH.read_text())
    assert r['status']=='passed_fresh_internal_salted_double_cold_confirmation'
    assert r['pass'] is True and all(r['checks'].values())
    assert r['model_selection_allowed_after_confirmation'] is False
    assert r['split_audit']['exact_train_test_pair_overlap']==0
    assert r['split_audit']['test_protein_seen_fraction']==0.0
    assert r['split_audit']['test_reaction_seen_fraction']==0.0
    assert r['residual_training']['identity_audit']['max_abs_diff']==0.0
    assert r['residual_training']['trainable_parameter_names']==['aux_to_hidden.weight']

def test_confirmation_result_improves_primary_hard_slice_without_hiding_external_scope():
    r=json.loads(PATH.read_text())
    h=r['lt0p3']; a=r['all']
    assert h['candidate']['mrr']>h['baseline']['mrr']
    assert h['candidate']['map']>h['baseline']['map']
    assert h['candidate']['ndcg_at_10']>h['baseline']['ndcg_at_10']
    assert h['candidate']['hit_at_10']>h['baseline']['hit_at_10']
    assert h['candidate']['hit_at_50']>=h['baseline']['hit_at_50']
    assert a['candidate']['hit_at_50']>a['baseline']['hit_at_50']
    assert 'not a fresh external benchmark claim' in r['claim_scope']
    assert 'Enzyme-405' in r['external_policy']['previously_revealed_cells_forbidden_as_unbiased_confirmation']
