import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_sequence_form_gate_keeps_formal_support_before_scores():
    x=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_OPEN_WORLD_ENZYMARC_SEQUENCE_FORM_GATE_V1_RESULT.json').read_text())
    assert x['status']=='passed_frozen_sequence_form_support_gate_model_scores_unread'
    assert x['eligible_parents']==62630 and x['excluded_parents']==5
    assert x['eligible_decoys']==236448 and x['eligible_relations']==92148
    assert x['minimum_claim_support_passed'] is True
    assert x['model_scores_read'] is False and x['selection_allowed'] is False
