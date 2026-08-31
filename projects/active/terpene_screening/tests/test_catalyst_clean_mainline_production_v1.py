import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_mainline_production_freezes_confirmed_v3_without_fake_fusion():
 p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_CLEAN_MAINLINE_PRODUCTION_V1.json').read_text())
 assert p['status']=='frozen_before_mainline_production_packaging' and p['outer_labels_used_for_production_training'] is False
 r=p['r2e_center_mainline']; assert r['selected_max_residual_ratio']==0.1 and r['n_training_pairs']==218537 and r['training']['dev_fold']==-1
 assert 'No claim' in p['other_confirmed_experts_are_referenced_not_fused']['policy']
