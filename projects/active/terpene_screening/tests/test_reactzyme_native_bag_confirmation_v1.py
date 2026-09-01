import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_confirmation_is_fresh_and_frozen():
 d=json.loads((ROOT/'projects/active/terpene_screening/REACTZYME_NATIVE_BAG_ADAPTER_V1_CONFIRMATION.json').read_text())
 assert d['status']=='frozen_before_confirmation_materialization'
 assert d['split']['salt']=='reactzyme_native_bag_confirm_v1_20260901_a'
 assert d['split']['fold_count']==7 and d['split']['confirmation_fold']==6
 assert d['adapter']['confirmation_reaction_ids_used_for_training'] is False
 assert d['confirmation_gate']['mrr_retention_min']==0.90
 assert d['confirmation_gate']['hit_at_10_retention_min']==0.88
 assert d['confirmation_gate']['median_best_positive_rank_ratio_max']==1.50
def test_builder_matches_frozen_split_identity():
 s=(ROOT/'projects/active/terpene_screening/build_reactzyme_native_bag_confirmation_v1.py').read_text()
 assert "SALT='reactzyme_native_bag_confirm_v1_20260901_a'" in s
 assert 'FOLDS=7; DEV=6' in s
 assert 'split_double_cold' in s
