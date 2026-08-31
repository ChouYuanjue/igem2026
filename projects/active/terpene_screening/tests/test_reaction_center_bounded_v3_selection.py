import pandas as pd
from projects.active.terpene_screening.evaluate_reaction_center_bounded_v3_selection import paired_summary,candidate_gate

def frame(mrr=.1,mapv=.09,auc=.9,ndcg=.08,h10=.2,h20=.3,h50=.4,rank=100.):
 return pd.DataFrame([{'direction':'reaction_to_enzyme','query_id':'R1','candidate_count':185918,'reciprocal_rank':mrr,'average_precision':mapv,'roc_auc':auc,'ndcg_at_10':ndcg,'hit_at_10':h10,'hit_at_20':h20,'hit_at_50':h50,'best_positive_rank':rank,'best_positive_rank_fraction':rank/185918}])
def diff(): return pd.DataFrame([{'reaction_id':'R1','reaction_similarity_bucket':'lt0p3'}])
def test_paired_summary_uses_exact_registered_support():
 r=paired_summary(frame(),frame(mrr=.11,mapv=.10,auc=.91,ndcg=.09,h10=.21,h20=.31,h50=.41,rank=90),diff(),candidate_count=185918)
 assert r['lt0p3']['delta']['mrr']>0 and r['all']['delta']['hit_at_20']>0 and r['lt0p3']['delta']['median_best_positive_rank']<0
def test_strict_v3_gate_requires_all_query_nonregression():
 good=paired_summary(frame(),frame(mrr=.11,mapv=.10,auc=.91,ndcg=.09,h10=.21,h20=.31,h50=.41,rank=90),diff(),candidate_count=185918)
 gate=candidate_gate([good,good,good],good); assert gate['pass'] is True
 bad=paired_summary(frame(),frame(mrr=.11,mapv=.10,auc=.91,ndcg=.09,h10=.19,h20=.31,h50=.41,rank=90),diff(),candidate_count=185918)
 gate=candidate_gate([good,good,good],bad); assert gate['checks']['all_hit10_no_regress'] is False and gate['pass'] is False
def test_fold_stability_rejects_two_metric_regressions():
 good=paired_summary(frame(),frame(mrr=.11,mapv=.10,auc=.91,ndcg=.09,h10=.21,h20=.31,h50=.41,rank=90),diff(),candidate_count=185918)
 weak=paired_summary(frame(),frame(mrr=.094,mapv=.084,auc=.91,ndcg=.09,h10=.21,h20=.31,h50=.41,rank=90),diff(),candidate_count=185918)
 gate=candidate_gate([good,weak,weak],good); assert gate['checks']['fold_stability_both_mrr_map_improve_2of3'] is False and gate['pass'] is False
