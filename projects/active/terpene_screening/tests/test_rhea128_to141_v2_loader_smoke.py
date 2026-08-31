import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_fixed_external_models_were_loader_smoked_before_target():
 r=json.loads((ROOT/'projects/active/terpene_screening/RHEA128_TO141_V2_FIXED_MODEL_LOADER_SMOKE.json').read_text())
 assert r['status']=='pre_target_reveal_pass' and r['target_release141_associations_read'] is False
 assert r['protein_candidates']==185918 and r['reaction_candidates']==11081
 assert r['protein_embedding_max_abs_diff']==0.0
 assert r['reaction_embedding_mean_abs_diff_after_residual_training']>0.0
 assert r['base_embedding_dim']==r['candidate_embedding_dim']==320
