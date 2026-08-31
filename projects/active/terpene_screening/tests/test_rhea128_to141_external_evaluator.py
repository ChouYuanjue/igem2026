import pandas as pd
from projects.active.terpene_screening.evaluate_rhea128_to141_external import evaluate


def metrics(rows):
    out=[]
    for q,mrr,ap,auc,ndcg,h10,h20,h50,rank in rows:
        out.append({"direction":"reaction_to_enzyme","query_id":q,"candidate_count":185918,"reciprocal_rank":mrr,
                    "average_precision":ap,"roc_auc":auc,"ndcg_at_10":ndcg,"hit_at_10":h10,"hit_at_20":h20,
                    "hit_at_50":h50,"best_positive_rank":rank,"best_positive_rank_fraction":rank/185918})
    return pd.DataFrame(out)


def manifest(nq=2,npairs=4):
    return {"audit":{"test_query_reactions":nq,"test_pairs":npairs,"exact_train_test_pair_overlap":0,
                     "train_test_protein_overlap":0,"train_test_reaction_overlap":0}}


def test_external_gate_passes_when_powered_and_all_seven_checks_hold():
    b=metrics([("A",.1,.08,.8,.1,0,0,0,100),("B",.2,.15,.9,.2,1,1,1,5)])
    c=metrics([("A",.12,.10,.82,.12,0,0,0,80),("B",.22,.17,.91,.21,1,1,1,4)])
    r=evaluate(manifest(),b,c,min_queries=2,min_pairs=4,candidate_count=185918)
    assert r["minimum_support"]["met"] is True
    assert r["pass"] is True and all(r["checks"].values())
    assert r["status"]=="passed_fresh_external_snapshot"


def test_external_gate_cannot_promote_underpowered_result_even_if_metrics_win():
    b=metrics([("A",.1,.08,.8,.1,0,0,0,100),("B",.2,.15,.9,.2,1,1,1,5)])
    c=metrics([("A",.12,.10,.82,.12,0,0,0,80),("B",.22,.17,.91,.21,1,1,1,4)])
    r=evaluate(manifest(),b,c,min_queries=50,min_pairs=200,candidate_count=185918)
    assert all(r["checks"].values())
    assert r["minimum_support"]["met"] is False
    assert r["pass"] is False
    assert r["status"]=="underpowered_external_descriptive"


def test_external_gate_fails_any_required_metric_regression_without_retuning():
    b=metrics([("A",.1,.08,.8,.1,1,1,1,10),("B",.1,.08,.8,.1,1,1,1,10)])
    c=metrics([("A",.2,.18,.9,.2,0,1,1,5),("B",.2,.18,.9,.2,1,1,1,5)])
    r=evaluate(manifest(),b,c,min_queries=2,min_pairs=4,candidate_count=185918)
    assert r["checks"]["hit10_no_regress"] is False
    assert r["pass"] is False
    assert r["status"]=="failed_fresh_external_snapshot_no_retuning"
