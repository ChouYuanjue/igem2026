import numpy as np,pandas as pd
from projects.active.terpene_screening.evaluate_enzymarc_open_world_v1 import metrics,bootstrap

def test_integrity_metrics_strict_pair_semantics():
    x=pd.DataFrame({'parent_accession':['a','b'],'parent_score':[.8,.2],'decoy_score':[.3,.2]})
    m=metrics(x); assert m['paired_parent_win_rate']==.5; assert np.isclose(m['paired_score_delta_mean'],.25); assert m['parent_vs_decoy_AUROC']>=.5
    b=bootstrap(x,n=20,seed=1); assert b['replicates']==20

def test_feature_and_evaluator_require_sequence_form_gate():
    from pathlib import Path
    root=Path(__file__).resolve().parents[4]
    f=(root/'projects/active/terpene_screening/prepare_enzymarc_esmc_features_v1.py').read_text()
    e=(root/'projects/active/terpene_screening/evaluate_enzymarc_open_world_v1.py').read_text()
    assert "sequence-form gate is not finalized" in f
    assert "sequence-form gate not finalized" in e
    assert "eligible_parents.csv" in f and "eligible_parents.csv" in e
