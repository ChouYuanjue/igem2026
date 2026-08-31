import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_fixed_model_asset_audit_locks_models_and_input_matrices_before_target():
 r=json.loads((ROOT/'projects/active/terpene_screening/RHEA128_TO141_V2_FIXED_MODEL_ASSETS.json').read_text())
 assert r['status']=='frozen_before_release141_association_extraction'
 assert r['target_release141_associations_read'] is False and r['target_model_scores_read'] is False
 assert r['base_audit']['n_train_pairs']==218537 and r['base_audit']['dev_fold']==-1
 assert r['base_audit']['reaction_input_dim']==3139
 assert r['residual_audit']['identity_audit']['max_abs_diff']==0.0
 assert r['residual_audit']['trainable_parameter_names']==['aux_to_hidden.weight']
 assert r['residual_audit']['reaction_input_dim']==4419 and r['residual_audit']['aux_input_dim']==1280
 for item in r['assets'].values():
  assert len(item['sha256'])==64 and item['bytes']>0
