import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]

def test_enzymarc_full_support_is_frozen_before_scores():
    p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_OPEN_WORLD_ENZYMARC_V1.json').read_text())
    r=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_OPEN_WORLD_ENZYMARC_V1_SUPPORT_RESULT.json').read_text())
    assert p['status']=='frozen_full_support_before_any_enzymarc_model_score'
    assert 'temporary candidates' in p['task_adapter']['lambdarank_boundary']
    assert 'r2e_center_bounded_cap0p1' in p['task_adapter']['production_source_score']
    assert p['task_adapter']['cohort'].startswith('all mapped support rows')
    assert r['mapped_parent_count']==62635
    assert r['mapped_decoy_count']==236468
    assert r['mapped_relation_count']==92157
    assert r['model_scores_read'] is False
    assert r['selection_allowed'] is False
    assert set(r['mapped_decoys_by_category'])=={'catalytic_residue','radius_5A','radius_10A','radius_15A'}
