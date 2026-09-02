import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]

def test_open_world_temporal_contract_is_frozen_and_target_unread():
 p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_OPEN_WORLD_TEMPORAL_EXTERNAL_V1.json').read_text())
 assert p['status'].startswith('frozen_before_target_2026_02')
 assert p['target']['target_labels_unread_at_freeze'] is True
 assert p['authoritative_baseline']['name']=='historical sequence_x_reaction nearest-neighbor transfer'
 assert p['selection_allowed'] is False
 assert p['minimum_support']['strict_double_cold_queries']==30

def test_tps_temporal_contract_uses_official_versions_and_separate_general_tps_axes():
 p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_TPS_TEMPORAL_R2E_V1.json').read_text())
 assert p['status'].startswith('frozen_before_marts_v2_1')
 assert 'v1.5' in p['source']['train_snapshot'] and 'v2.1' in p['source']['target_snapshot']
 assert p['promotion_target']['material_gain_pp_vs_strongest_valid_baseline']==5.0
 assert p['cells']['legacy_frozen_double_cold']
 assert p['baselines']['general_model_reference'] != p['baselines']['tps_internal_reference']
 assert p['minimum_support']['temporal_strict_double_cold_queries']==20

def test_tps_foundation_r2e_search_is_small_and_development_only():
 p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_TPS_FOUNDATION_R2E_V1.json').read_text())
 assert p['status']=='frozen_before_new_foundation_feature_development_scores'
 assert len(p['candidates'])==7
 assert p['feature_policy']['clean2023_retrieval_weights_allowed'] is False
 assert p['feature_policy']['v2_1_labels_or_performance_allowed'] is False
 assert p['split_semantics']['selection_may_read_legacy_frozen'] is False
 assert p['development_evaluation']['minimum_hit10_absolute_gain_to_open_legacy_frozen']==0.02
 assert p['legacy_frozen_policy']['at_most_one_new_candidate'] is True

def test_enzymarc_open_world_contract_is_external_and_nonselecting():
 p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_OPEN_WORLD_ENZYMARC_V1.json').read_text())
 assert p['status']=='frozen_before_enzymarc_rows_are_scored_by_catalyst'
 assert p['promotion_or_selection']['this_benchmark_may_select_new_model'] is False
 assert p['promotion_or_selection']['this_benchmark_may_select_new_threshold'] is False
 assert p['reporting']['minimum_mapped_parents_for_claim']==50
 assert 'historical association transfer' in p['baselines']['classical_same_task']

def test_tps_active_site_xattn_protocol_has_fresh_split_and_automated_hpo():
 p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_TPS_ACTIVE_SITE_XATTN_V1.json').read_text())
 assert p['status']=='frozen_before_fresh_salted_protein_cold_model_scores'
 assert p['fresh_internal_split']['hpo_folds']==[0,1,2]
 assert p['fresh_internal_split']['internal_confirmation_folds']==[3,4]
 assert p['fresh_internal_split']['internal_confirmation_unread_until_hpo_config_frozen'] is True
 assert p['automated_hpo']['method'].startswith('Optuna TPE')
 assert p['automated_hpo']['trials']==18
 assert p['two_stage_retrieval']['per_representation_topk']==160
 assert p['reaction_tokens']['mapped_coverage_pre_score']=='453/453 TPS reactions'
 assert p['internal_confirmation_gate']['required_hit_at_10_gain_pp']==3.0
 assert p['automated_hpo']['no_manual_trials_outside_space_after_scores'] is True

def test_invalidated_rhea_temporal_v1_was_not_scored():
 p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_OPEN_WORLD_TEMPORAL_EXTERNAL_V1_INVALIDATION.json').read_text())
 assert p['status']=='invalidated_before_target_materialization_or_scoring'
 assert p['target_rows_scored_under_invalid_protocol'] is False
 assert p['current_official_rhea_association_snapshot_audit']['same_as_already_revealed_release141'] is True

def test_xattn_transfer_baseline_and_negative_pool_are_frozen():
 p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_TPS_ACTIVE_SITE_XATTN_V1.json').read_text())
 two=p['two_stage_retrieval']; inter=p['interaction_model']
 assert 'max over training association pairs' in two['transfer_score_per_representation']
 assert 'equal arithmetic mean' in two['baseline_fusion']
 assert 'top160' in two['shortlist_order']
 assert 'fixed pool size' in inter['hard_negative_pool_semantics']
 assert 'deterministically cycles one negative' in inter['hard_negative_pool_semantics']

def test_xattn_final_prefix_and_bounded_residual_are_frozen():
 p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_TPS_ACTIVE_SITE_XATTN_V1.json').read_text())
 assert 'complete leading prefix' in p['two_stage_retrieval']['final_order']
 assert 'same candidate support' in p['two_stage_retrieval']['baseline_comparator_order']
 assert '2.0*tanh' in p['interaction_model']['residual_combination']
 assert 'excluded from HPO' in p['interaction_model']['residual_combination']
