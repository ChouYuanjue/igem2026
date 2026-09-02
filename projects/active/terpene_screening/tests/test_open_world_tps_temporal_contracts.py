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
