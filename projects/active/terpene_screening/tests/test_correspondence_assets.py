from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[4]
def test_canonical_correspondence_result_contract():
    d=ROOT/'results/terpene_product_correspondence_dev_v1'
    s=json.loads((d/'summary.json').read_text())
    m=pd.read_csv(d/'metrics.csv')
    sp=pd.read_csv(d/'split_summary.csv')
    assert s['version']=='terpene-product-correspondence-dev-v1'
    assert s['negatives_used'] is False
    assert s['all_train_test_entity_overlaps_zero'] is True
    assert len(sp)==9
    assert (sp.protein_overlap==0).all()
    assert (sp.reaction_overlap==0).all()
    assert set(m.direction)=={'enzyme_to_reaction','reaction_to_enzyme'}
def test_real_seed_update_parity_is_exact():
    s=json.loads((ROOT/'results/terpene_correspondence_seed_update_audit_v1/summary.json').read_text())
    assert s['parity_max_abs_joint']==0.0
    assert s['parity_max_abs_defect']==0.0
def test_finite_temperature_diagnostic_is_not_canonical():
    s=json.loads((ROOT/'results/terpene_free_energy_correspondence_dev_v1/summary.json').read_text())
    assert s['parameter_selection'].startswith('none')
    main=(ROOT/'projects/active/terpene_screening/docs/method.md').read_text()
    assert 'rejected diagnostic' in main
