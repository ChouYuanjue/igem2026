import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_router_protocol_forbids_benchmark_identity_and_freezes_candidates():
 d=json.loads((ROOT/'projects/active/terpene_screening/UNIFIED_SAFE_SYSTEM_E2R_ROUTER_V1.json').read_text()); assert d['status']=='frozen_before_router_oof_performance'; assert d['candidate_thresholds']==[.5,.6,.7,.8,.9]; assert 'dataset_name' in d['forbidden_features']; assert d['material_breakthrough'].endswith('>= 0.05')
def test_router_is_crossfit_baseline_fallback():
 s=(ROOT/'projects/active/terpene_screening/select_unified_safe_system_e2r_router_v1.py').read_text(); assert "tr=allz.fold.ne(f); te=allz.fold.eq(f)" in s; assert 'np.where(choose' in s; assert "for family in ['logistic','hist_gb']" in s
