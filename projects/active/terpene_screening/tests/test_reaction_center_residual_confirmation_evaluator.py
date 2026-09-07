import pandas as pd
from projects.active.terpene_screening.evaluate_reaction_center_residual_confirmation import evaluate


def frame(values):
    rows=[]
    for q,mrr,ap,auc,ndcg,h10,h20,h50,rank in values:
        rows.append({'direction':'reaction_to_enzyme','query_id':q,'candidate_count':185918,'reciprocal_rank':mrr,'average_precision':ap,'roc_auc':auc,'ndcg_at_10':ndcg,'hit_at_10':h10,'hit_at_20':h20,'hit_at_50':h50,'best_positive_rank':rank,'best_positive_rank_fraction':rank/185918})
    return pd.DataFrame(rows)


def test_frozen_confirmation_gate_passes_only_when_all_required_checks_hold():
    difficulty=pd.DataFrame({'reaction_id':['A','B','C'],'reaction_similarity_bucket':['lt0p3','lt0p3','ge0p9'],'max_train_drfp_tanimoto':[0.1,0.2,0.95]})
    base=frame([('A',.10,.08,.80,.10,0,0,0,100),('B',.20,.15,.90,.20,1,1,1,5),('C',.30,.25,.95,.30,1,1,1,3)])
    cand=frame([('A',.12,.10,.82,.12,0,0,0,80),('B',.22,.17,.91,.21,1,1,1,4),('C',.31,.26,.96,.31,1,1,1,2)])
    r=evaluate(base,cand,difficulty)
    assert r['pass'] is True and all(r['checks'].values())


def test_frozen_confirmation_gate_rejects_hard_hit10_regression_even_if_mrr_improves():
    difficulty=pd.DataFrame({'reaction_id':['A','B','C'],'reaction_similarity_bucket':['lt0p3','lt0p3','ge0p9'],'max_train_drfp_tanimoto':[0.1,0.2,0.95]})
    base=frame([('A',.10,.08,.80,.10,1,1,1,10),('B',.10,.08,.80,.10,1,1,1,10),('C',.10,.08,.80,.10,1,1,1,10)])
    cand=frame([('A',.20,.18,.90,.20,0,1,1,5),('B',.20,.18,.90,.20,1,1,1,5),('C',.20,.18,.90,.20,1,1,1,5)])
    r=evaluate(base,cand,difficulty)
    assert r['lt0p3']['candidate']['mrr'] > r['lt0p3']['baseline']['mrr']
    assert r['checks']['lt0p3_hit10_no_regress'] is False
    assert r['pass'] is False
