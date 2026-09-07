from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3]; B=ROOT/'results/unified_safe_system_v1/e2r_full_reaction_contract_v1'; X=ROOT/'results/unified_safe_system_v1/e2r_four_expert_portfolio_v1'
def load(name):
 frames=[]
 for f in range(3):
  p=B/f'fold{f}/baseline_query_metrics.csv' if name=='enzgfm' else B/f'fold{f}/candidate_query_metrics.csv' if name=='rdkitplus' else X/name/f'fold{f}/query_metrics.csv'; q=pd.read_csv(p); q['fold']=f; frames.append(q)
 return pd.concat(frames,ignore_index=True)
def agg(q): return {k:float(q[c].mean()) for k,c in {'mrr':'reciprocal_rank','map':'average_precision','auc':'roc_auc','ndcg10':'ndcg_at_10','hit1':'hit_at_1','hit3':'hit_at_3','hit5':'hit_at_5','hit10':'hit_at_10','hit20':'hit_at_20','hit50':'hit_at_50'}.items()}
def main():
 names=['enzgfm','esmc','equalblock','rdkitplus']; qs={n:load(n) for n in names}; ids=qs['enzgfm'].query_id.tolist(); assert all(q.query_id.tolist()==ids for q in qs.values()); metrics={n:agg(q) for n,q in qs.items()}; base=metrics['enzgfm']; deltas={n:{k:v-base[k] for k,v in m.items()} for n,m in metrics.items() if n!='enzgfm'}; oracle={}
 for k in [1,3,5,10,20,50]:
  arr=np.stack([qs[n][f'hit_at_{k}'].to_numpy() for n in names]); oracle[f'hit{k}']=float(arr.max(axis=0).mean()); oracle[f'hit{k}_delta_vs_baseline']=oracle[f'hit{k}']-base[f'hit{k}']
 ranks=np.stack([qs[n].best_positive_rank.to_numpy() for n in names]); oracle['mrr']=float(np.mean(1/ranks.min(axis=0))); oracle['median_rank']=float(np.median(ranks.min(axis=0)))
 out={'metrics':metrics,'delta_vs_enzgfm':deltas,'oracle':oracle,'material_single_experts':[n for n,d in deltas.items() if d['hit10']>=.05],'oracle_only_not_router':True,'external_test_metrics_used':False}; (X/'pooled_result.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
