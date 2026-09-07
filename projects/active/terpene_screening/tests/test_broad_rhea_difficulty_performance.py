import pandas as pd
from projects.active.terpene_screening.summarize_broad_rhea_difficulty_performance import aggregate, parse_source


def test_parse_source_contract():
    name, path = parse_source('base=/tmp/q.csv')
    assert name == 'base'
    assert path.name == 'q.csv'


def test_difficulty_aggregate_preserves_core_ranking_metrics():
    frame = pd.DataFrame({
        'candidate_count':[100,100], 'positive_count':[1,2],
        'reciprocal_rank':[1.0,0.5], 'average_precision':[1.0,0.5],
        'roc_auc':[0.9,0.7], 'best_positive_rank':[1,2],
        'best_positive_rank_fraction':[0.01,0.02], 'mean_positive_rank':[1.0,3.0],
        'mean_positive_reciprocal_rank':[1.0,0.4], 'hit_at_10':[1,1],
        'precision_at_10':[0.1,0.2], 'positive_recall_at_10':[1.0,0.5], 'ndcg_at_10':[1.0,0.6],
    })
    out=aggregate(frame)
    assert out['n_queries']==2
    assert out['mrr']==0.75
    assert out['map']==0.75
    assert out['hit_at_10']==1.0
    assert out['positive_recall_at_10']==0.75
