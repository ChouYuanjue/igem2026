import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_lambdarank_protocol_is_single_config_crossfit():
 d=json.loads((ROOT/'projects/active/terpene_screening/UNIFIED_SAFE_SYSTEM_E2R_LAMBDARANK_STACK_V1.json').read_text()); assert d['status']=='frozen_before_stacker_oof_performance'; assert d['training']['objective']=='rank:ndcg'; assert d['training']['rounds']==80; assert d['training']['hard_negatives_per_query']==64; assert d['no_hyperparameter_sweep_v1'] is True; assert d['material_breakthrough'].endswith('>= 0.05')
def test_lambdarank_is_bounded_union_shortlist_with_baseline_tail():
 d=json.loads((ROOT/'projects/active/terpene_screening/UNIFIED_SAFE_SYSTEM_E2R_LAMBDARANK_STACK_V1.json').read_text()); s=d['inference_scope']; assert s['shortlist_top_per_expert']==50; assert s['full_candidate_output_count']==11081; assert 'EnzGFM baseline' in s['tail']
def test_lambdarank_implementation_uses_crossfit_and_baseline_tail():
 s=(ROOT/'projects/active/terpene_screening/run_unified_safe_system_e2r_lambdarank_stack_v1.py').read_text(); assert "if g==f: continue" in s; assert "train_ranker" in s and "'rank:ndcg',80,3,.05" in s; assert "b_order=np.argsort(-S[0],kind='stable')" in s; assert "top_rows(S[e],50)" in s; assert "external_test_metrics_used" in s
