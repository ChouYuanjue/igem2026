import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_lambdarank_protocol_is_single_config_crossfit():
 d=json.loads((ROOT/'projects/active/terpene_screening/UNIFIED_SAFE_SYSTEM_E2R_LAMBDARANK_STACK_V1.json').read_text()); assert d['status']=='frozen_before_stacker_oof_performance'; assert d['training']['objective']=='rank:ndcg'; assert d['training']['rounds']==80; assert d['training']['hard_negatives_per_query']==64; assert d['no_hyperparameter_sweep_v1'] is True; assert d['material_breakthrough'].endswith('>= 0.05')
