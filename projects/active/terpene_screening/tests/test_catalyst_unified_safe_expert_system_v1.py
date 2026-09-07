import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]

def load():
 d=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_UNIFIED_SAFE_EXPERT_SYSTEM_V1.json').read_text())
 return d

def test_unified_goal_is_baseline_guarded_and_benchmark_agnostic():
 d=load()
 assert d['nonnegotiable_rules']['baseline_is_executable_expert'] is True
 assert d['nonnegotiable_rules']['dataset_name_or_benchmark_id_may_enter_router'] is False
 assert d['nonnegotiable_rules']['exactly_one_authoritative_external_baseline_per_capability'] is True
 assert d['nonnegotiable_rules']['internal_safety_floor_may_substitute_for_external_baseline'] is False
 assert d['nonnegotiable_rules']['metrics_may_be_cherry_picked_across_baselines'] is False
 assert d['promotion_gate']['required_material_breakthrough_contracts']>=2
 assert d['promotion_gate']['hard_slices_must_not_be_hidden_by_aggregate'] is True
 assert d['baseline_contract_registry'].endswith('CATALYST_CAPABILITY_BASELINE_CONTRACT_V1.json')
 assert set(d['baseline_contracts']) >= {'r2e_structure_available','r2e_sequence_reaction','e2r_sequence_reaction','r2e_native_molecule_bag'}
 for cid,c in d['baseline_contracts'].items():
  assert c['registry_contract']==cid
  assert '/' not in c['authoritative_external_baseline']

def test_small_regression_needs_material_compensation():
 d=load()
 assert d['nonnegotiable_rules']['small_regression_allowed_only_when_compensated_by_clear_material_gain_in_another_predeclared_contract'] is True
 assert d['nonnegotiable_rules']['isolated_new_input_coverage_without_score_gain_does_not_count_as_material_gain'] is True
