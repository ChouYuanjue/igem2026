import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
PATH=ROOT/'projects/active/terpene_screening/CLEANROOM_R2E_REACTION_CENTER_RESIDUAL_V2.json'

def test_residual_v2_freezes_source_towers_and_reuses_preexisting_recipe():
    p=json.loads(PATH.read_text())
    assert p['status']=='frozen_before_performance_execution'
    assert p['model']['freeze_base_protein'] is True
    assert p['model']['freeze_base_reaction'] is True
    assert p['model']['only_trainable_parameter']=='aux_to_hidden.weight'
    assert p['training']['epochs']==2 and p['training']['learning_rate']==0.00003
    assert p['training']['loss_candidate_scope']=='training_entities'
    assert p['outer_labels_used'] is False and p['target_benchmark_labels_used'] is False

def test_residual_v2_keeps_center_v1_gate_and_forbids_sweep():
    p=json.loads(PATH.read_text())
    assert p['screening_rule']['same_gate_as_center_v1'] is True
    assert p['screening_rule']['candidate_count']==1
    assert p['screening_rule']['no_hyperparameter_sweep'] is True
    assert 'brand-new salted strict double-cold' in p['screening_rule']['if_pass']
