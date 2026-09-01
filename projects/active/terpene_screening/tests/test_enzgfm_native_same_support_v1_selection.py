import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_selection_was_frozen_before_outer_reveal():
 d=json.loads((ROOT/'projects/active/terpene_screening/ENZGFM_NATIVE_SAME_SUPPORT_CATALYST_V1_SELECTION.json').read_text())
 assert d['status']=='frozen_before_native_test_reveal'
 assert d['selected_candidate']=='dual_tower'
 assert d['native_test_metrics_read'] is False
 assert d['native_test_metrics_used_for_selection'] is False
 assert d['alternative_candidate_outer_scoring_allowed'] is False
 assert {x['candidate'] for x in d['candidate_development_evidence']}=={'dual_tower','author_pairwise'}
