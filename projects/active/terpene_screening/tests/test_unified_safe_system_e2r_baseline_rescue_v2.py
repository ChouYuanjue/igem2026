import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_v2_is_frozen_before_new_salt_and_structurally_safe():
 d=json.loads((ROOT/'projects/active/terpene_screening/UNIFIED_SAFE_SYSTEM_E2R_BASELINE_RESCUE_V2.json').read_text()); assert d['status']=='frozen_before_any_v2_salted_development_performance'; assert d['development_split']['split_salt']=='e2r_baseline_rescue_v2_dev_20260901_a'; assert d['future_confirmation_split']['split_salt']=='e2r_baseline_rescue_v2_confirm_20260901_a'; assert d['rescue_mechanism']['single_mutable_slot']==10; assert 'ranks 1 through 9' in d['rescue_mechanism']['immutable_prefix']; assert d['slot_ranker']['no_threshold_or_hyperparameter_sweep'] is True; assert d['development_gate']['material_breakthrough'].endswith('>= 0.05')
def test_expert_runner_propagates_new_salt_and_old_recipe():
 s=(ROOT/'projects/active/terpene_screening/run_unified_safe_system_e2r_rescue_v2_experts.py').read_text(); assert "DEV_SALT='e2r_baseline_rescue_v2_dev_20260901_a'" in s; assert "'--split-salt',DEV_SALT" in s; assert "'--epochs','8'" in s and "'--hard-negatives','80'" in s and "'--temperature','0.035'" in s; assert "RDKIT if name=='rdkitplus' else BASE_RXN" in s
