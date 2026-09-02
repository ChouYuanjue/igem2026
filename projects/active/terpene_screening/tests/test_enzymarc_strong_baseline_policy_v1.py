import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_strong_baseline_is_absorbed_without_reusing_revealed_outer_labels():
    p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_STRONG_BASELINE_ABSORPTION_POLICY_V1.json').read_text())
    assert p['fairness_boundary']['revealed_external_or_outer_labels_may_select_new_model'] is False
    assert p['enzymarc_v1_pre_score_absorption_trigger']['enzymarc_reuse_for_that_family'] is False
    assert 'safety floor' in p['enzymarc_v1_pre_score_absorption_trigger']['mandatory_next_family_if_triggered']
def test_mmseqs_baseline_is_same_task_and_frozen_before_scores():
    p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_OPEN_WORLD_ENZYMARC_MMSEQS_BASELINE_V1.json').read_text())
    assert p['status']=='frozen_before_any_same_task_baseline_or_catalyst_score'
    assert p['mmseqs2']['max-seqs' if 'max-seqs' in p['mmseqs2'] else 'command']
    assert p['selection_allowed'] is False and p['threshold_tuning_allowed'] is False
    assert 'maximum MMseqs2' in p['score']
