from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[3]; FROOT=ROOT/'results/unified_safe_system_v1/e2r_router_v1'; MROOT=ROOT/'results/unified_safe_system_v1/e2r_full_reaction_contract_v1'
FEATURES=['baseline_top1_score','baseline_top1_top2_margin','baseline_top1_top10_margin','candidate_top1_score','candidate_top1_top2_margin','candidate_top1_top10_margin','top1_agreement','top10_jaccard','protein_identity','protein_no_hit']; TH=[.5,.6,.7,.8,.9]
def agg(q): return {k:float(q[c].mean()) for k,c in {'mrr':'reciprocal_rank','map':'average_precision','auc':'roc_auc','ndcg10':'ndcg_at_10','hit10':'hit_at_10','hit20':'hit_at_20','hit50':'hit_at_50'}.items()}
def main():
 frames=[]
 for f in range(3):
  feat=pd.read_csv(FROOT/f'fold{f}/router_features.csv'); b=pd.read_csv(MROOT/f'fold{f}/baseline_query_metrics.csv').add_prefix('b_'); c=pd.read_csv(MROOT/f'fold{f}/candidate_query_metrics.csv').add_prefix('c_'); assert feat.query_id.tolist()==b.b_query_id.tolist()==c.c_query_id.tolist(); z=pd.concat([feat.reset_index(drop=True),b.drop(columns='b_query_id'),c.drop(columns='c_query_id')],axis=1); z['target']=(z.c_best_positive_rank<z.b_best_positive_rank).astype(int); z['weight']=1+np.abs(np.log1p(z.b_best_positive_rank)-np.log1p(z.c_best_positive_rank)); frames.append(z)
 allz=pd.concat(frames,ignore_index=True); configs=[]
 for family in ['logistic','hist_gb']:
  probs=np.zeros(len(allz));
  for f in range(3):
   tr=allz.fold.ne(f); te=allz.fold.eq(f); X=allz.loc[tr,FEATURES]; y=allz.loc[tr,'target']; w=allz.loc[tr,'weight'];
   if family=='logistic': model=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,penalty='l2',class_weight='balanced',max_iter=1000,random_state=20260901)); model.fit(X,y,logisticregression__sample_weight=w)
   else: model=HistGradientBoostingClassifier(max_depth=2,max_iter=100,learning_rate=.05,l2_regularization=1.0,min_samples_leaf=50,random_state=20260901); model.fit(X,y,sample_weight=w)
   probs[te]=model.predict_proba(allz.loc[te,FEATURES])[:,1]
  for t in TH:
   choose=probs>=t; routed=pd.DataFrame({c:np.where(choose,allz['c_'+c],allz['b_'+c]) for c in ['reciprocal_rank','average_precision','roc_auc','ndcg_at_10','hit_at_10','hit_at_20','hit_at_50']}); rm=agg(routed); bm=agg(pd.DataFrame({c:allz['b_'+c] for c in routed.columns})); d={k:rm[k]-bm[k] for k in rm}; feasible=d['mrr']>=0 and d['map']>=0 and d['auc']>=0 and d['ndcg10']>=0 and d['hit20']>=-.005 and d['hit50']>=-.005; configs.append({'family':family,'threshold':t,'candidate_fraction':float(choose.mean()),'feasible':bool(feasible),**{'routed_'+k:v for k,v in rm.items()},**{'baseline_'+k:v for k,v in bm.items()},**{'delta_'+k:v for k,v in d.items()}})
 out=pd.DataFrame(configs); out.to_csv(FROOT/'crossfit_configs.csv',index=False); feas=out[out.feasible].copy();
 if feas.empty: result={'status':'reject_no_feasible_router','external_metrics_used':False}
 else:
  famorder={'logistic':0,'hist_gb':1}; feas['family_order']=feas.family.map(famorder); best=feas.sort_values(['delta_hit10','delta_mrr','threshold','family_order'],ascending=[False,False,False,True]).iloc[0]; result={'status':'selected' if best.delta_hit10>=.05 else 'pass_floor_but_no_material_breakthrough','selected':best.drop(labels='family_order').to_dict(),'material_breakthrough_hit10':bool(best.delta_hit10>=.05),'external_metrics_used':False,'next':'fresh_salted_confirmation_required_before_promotion'}
 (FROOT/'selection.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
