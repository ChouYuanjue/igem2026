from __future__ import annotations
import json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'results/unified_safe_system_v1/e2r_full_reaction_contract_v1'
def agg(q): return {'n_queries':len(q),'mrr':q.reciprocal_rank.mean(),'map':q.average_precision.mean(),'macro_roc_auc':q.roc_auc.mean(),'ndcg_at_10':q.ndcg_at_10.mean(),'hit_at_1':q.hit_at_1.mean(),'hit_at_3':q.hit_at_3.mean(),'hit_at_5':q.hit_at_5.mean(),'hit_at_10':q.hit_at_10.mean(),'hit_at_20':q.hit_at_20.mean(),'hit_at_50':q.hit_at_50.mean(),'median_best_positive_rank':q.best_positive_rank.median()}
def main():
 b=[];c=[]
 for f in range(3):
  b.append(pd.read_csv(OUT/f'fold{f}/baseline_query_metrics.csv')); c.append(pd.read_csv(OUT/f'fold{f}/candidate_query_metrics.csv'))
 b=pd.concat(b,ignore_index=True); c=pd.concat(c,ignore_index=True); assert b.query_id.tolist()==c.query_id.tolist(); bm,cm=agg(b),agg(c); keys=['mrr','map','macro_roc_auc','ndcg_at_10','hit_at_1','hit_at_3','hit_at_5','hit_at_10','hit_at_20','hit_at_50']; delta={k:float(cm[k]-bm[k]) for k in keys}; checks={'mrr_nonregress':delta['mrr']>=0,'map_nonregress':delta['map']>=0,'auc_nonregress':delta['macro_roc_auc']>=0,'ndcg10_nonregress':delta['ndcg_at_10']>=0,'hit10_floor':delta['hit_at_10']>=-0.005,'hit20_floor':delta['hit_at_20']>=-0.005,'hit50_floor':delta['hit_at_50']>=-0.005}; result={'status':'pass_floor' if all(checks.values()) else 'fail_floor','candidate_count':11081,'baseline':bm,'candidate':cm,'delta':delta,'checks':checks,'material_breakthrough_hit10':delta['hit_at_10']>=0.05,'external_test_labels_used':False,'retuning_from_this_result_allowed':False}; (OUT/'pooled_result.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
