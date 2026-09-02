import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]

def test_v1_performance_is_invalidated_and_old_confirmation_forbidden():
    x=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_TPS_ACTIVE_SITE_XATTN_V1_INVALIDATION.json').read_text())
    assert x['status']=='invalidated_due_to_candidate_id_alignment_bug'
    assert x['scientific_use_of_all_v1_performance_metrics']=='forbidden'
    assert all(v[1]==0 and v[0]>0 for v in x['nonfinite_shortlist_baseline_entries_wrong_vs_correct_index'].values())
    assert x['leakage_boundary_after_discovery']['folds_3_4'].startswith('irrevocably revealed')

def test_v1r1_changes_only_alignment_and_uses_external_one_shot():
    x=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_TPS_ACTIVE_SITE_XATTN_V1R1.json').read_text())
    assert x['method_changes_from_v1'] is False
    assert x['model_training_changes_from_v1'] is False
    assert x['hpo_space_changes_from_v1'] is False
    assert x['development']['folds']==[0,1,2]
    assert x['forbidden_internal_confirmation']['folds']==[3,4]
    assert x['replacement_one_shot_confirmation']['benchmark']=='MARTS v2.1 temporal protein-cold'
