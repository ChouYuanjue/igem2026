import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_v2_is_frozen_before_new_salt_and_structurally_safe():
 d=json.loads((ROOT/'projects/active/terpene_screening/UNIFIED_SAFE_SYSTEM_E2R_BASELINE_RESCUE_V2.json').read_text()); assert d['status']=='frozen_before_any_v2_salted_development_performance'; assert d['development_split']['split_salt']=='e2r_baseline_rescue_v2_dev_20260901_a'; assert d['future_confirmation_split']['split_salt']=='e2r_baseline_rescue_v2_confirm_20260901_a'; assert d['rescue_mechanism']['single_mutable_slot']==10; assert 'ranks 1 through 9' in d['rescue_mechanism']['immutable_prefix']; assert d['slot_ranker']['no_threshold_or_hyperparameter_sweep'] is True; assert d['development_gate']['material_breakthrough'].endswith('>= 0.05')
def test_expert_runner_propagates_new_salt_and_old_recipe():
 s=(ROOT/'projects/active/terpene_screening/run_unified_safe_system_e2r_rescue_v2_experts.py').read_text(); assert "DEV_SALT='e2r_baseline_rescue_v2_dev_20260901_a'" in s; assert "'--split-salt',DEV_SALT" in s; assert "'--epochs','8'" in s and "'--hard-negatives','80'" in s and "'--temperature','0.035'" in s; assert "RDKIT if name=='rdkitplus' else BASE_RXN" in s
def test_v2_evaluator_crossfits_and_only_promotes_slot10():
 s=(ROOT/'projects/active/terpene_screening/run_unified_safe_system_e2r_baseline_rescue_v2.py').read_text(); assert 'if g==f: continue' in s; assert 'elif p==chosen: nr=10' in s; assert 'elif 10<=br<cr: nr=br+1' in s; assert "'rank:pairwise',80,2,.05,5.,10." in s; assert "d['hit_at_10']>=.05" in s; assert "'same_dev_retuning_allowed':False" in s
def test_rank_transform_is_single_promotion():
 import numpy as np
 ns={}; exec((ROOT/'projects/active/terpene_screening/run_unified_safe_system_e2r_baseline_rescue_v2.py').read_text().split('def evaluate(f):')[0],ns); order=np.arange(20); ranks=ns['transform'](order,14,{0,8,9,10,14,15}); assert ranks.tolist()==[1,9,10,11,12,16]
