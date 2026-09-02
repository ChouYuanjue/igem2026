import numpy as np,pandas as pd
from projects.active.terpene_screening.evaluate_enzymarc_open_world_v1 import metrics,bootstrap

def test_integrity_metrics_strict_pair_semantics():
    x=pd.DataFrame({'parent_accession':['a','b'],'parent_score':[.8,.2],'decoy_score':[.3,.2]})
    m=metrics(x); assert m['paired_parent_win_rate']==.5; assert np.isclose(m['paired_score_delta_mean'],.25); assert m['parent_vs_decoy_AUROC']>=.5
    b=bootstrap(x,n=20,seed=1); assert b['replicates']==20
