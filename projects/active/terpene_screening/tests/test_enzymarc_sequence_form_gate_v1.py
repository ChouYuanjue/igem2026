import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]

def test_sequence_form_gate_is_frozen_before_alignment_stats():
    p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_OPEN_WORLD_ENZYMARC_V1.json').read_text())
    g=p['sequence_form_gate']; assert g['thresholds_frozen_before_alignment_statistics'] is True; assert '>=0.80' in g['eligibility']; assert g['performance_scores_used'] is False
