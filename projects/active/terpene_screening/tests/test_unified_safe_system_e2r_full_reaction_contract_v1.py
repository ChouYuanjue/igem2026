import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_e2r_full_contract_is_strong_baseline_and_full_support():
 d=json.loads((ROOT/'projects/active/terpene_screening/UNIFIED_SAFE_SYSTEM_E2R_FULL_REACTION_CONTRACT_V1.json').read_text())
 assert d['status']=='frozen_before_full_universe_performance'
 assert d['baseline']['name'].startswith('EnzGFM-only')
 assert d['support']['expected_candidate_count']==11081
 assert d['support']['no_query_or_candidate_subsampling'] is True
 assert d['external_test_labels_used'] is False
 assert d['material_breakthrough']['primary'].endswith('>= 0.05')
