import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_four_expert_portfolio_is_frozen_and_no_training():
 d=json.loads((ROOT/'projects/active/terpene_screening/UNIFIED_SAFE_SYSTEM_E2R_FOUR_EXPERT_PORTFOLIO_V1.json').read_text()); assert d['status']=='frozen_before_extra_expert_performance'; assert d['no_new_model_training'] is True; assert len(d['experts'])==4; assert d['baseline']=='EnzGFM-only'
