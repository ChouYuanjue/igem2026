from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import load_npz
from scipy.sparse.linalg import expm_multiply
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.geometric_product_field import product_characteristic_scale
from projects.active.terpene_screening.geometric_pair_measure import anisotropic_factor_sum_heat_pushforward
from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics
from projects.active.terpene_screening.evaluate_multiresolution_product_heat_marts_v3 import symmetric_heat_generator,normalized_positive_measure
C=ROOT/'data/terpene_marts_adaptation'; PG=ROOT/'data/terpene_multiresolution_protein_geometry_v4'; RG=ROOT/'data/terpene_multiresolution_reaction_geometry_v2'; OUT=ROOT/'results/terpene_anisotropic_product_measure_dev_v1'; B=(3,10,20)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1<<20),b''): h.update(x)
 return h.hexdigest()
def agg(q):
 rows=[]
 for (method,direction),g in q.groupby(['method','direction']):
  x={'method':method,'direction':direction,'n_queries':len(g),'mrr':float(g.reciprocal_rank.mean()),'median_rank':float(g.best_positive_rank.median())}
  for k in B: x[f'hit{k}']=float(g[f'hit_at_{k}'].mean()); x[f'recall{k}']=float(g[f'positive_recall_at_{k}'].mean())
  rows.append(x)
 return pd.DataFrame(rows)
def main():
 p=pd.read_csv(C/'protein_entities.csv',dtype=str).fillna(''); r=pd.read_csv(C/'reaction_entities.csv',dtype=str).fillna(''); z=pd.read_csv(C/'marts_pair_folds.csv',dtype=str).fillna('')
 z[['protein_fold','reaction_fold']]=z[['protein_fold','reaction_fold']].astype(int); z['protein_seen']=z.protein_seen.str.lower().eq('true'); z['reaction_seen']=z.reaction_seen.str.lower().eq('true')
 pids=p.protein_id.tolist();rids=r.reaction_id.tolist();pi={x:i for i,x in enumerate(pids)};ri={x:i for i,x in enumerate(rids)}
 rg=load_npz(RG/'diffusion_conformal_affinity.npz').tocsr();pg=load_npz(PG/'diffusion_conformal_affinity.npz').tocsr();qR=symmetric_heat_generator(rg);qE=symmetric_heat_generator(pg)
 rows=[]; splits=[]
 for pf in range(5):
  for rf in range(5):
   if not(pf==4 or rf==4):continue
   sid=f'p{pf}_r{rf}'; train=z[z.protein_fold.ne(pf)&z.reaction_fold.ne(rf)].drop_duplicates(['rhea_id','Entry']); test=z[z.protein_fold.eq(pf)&z.reaction_fold.eq(rf)&~z.protein_seen&~z.reaction_seen].drop_duplicates(['rhea_id','Entry'])
   A=np.zeros((len(rids),len(pids)),float)
   for x in train.itertuples(index=False):A[ri[x.rhea_id],pi[x.Entry]]=1
   mu=normalized_positive_measure(A); J=expm_multiply(qE,expm_multiply(qR,mu).T).T;J=np.maximum(J,0);J/=J.sum();prior=np.sqrt(J);scale=product_characteristic_scale(prior,rg,[pg])
   t=time.time(); adaptive=anisotropic_factor_sum_heat_pushforward(prior,mu,rg,pg,scale=scale);runtime=time.time()-t
   splits.append({'split_id':sid,'train_pairs':len(train),'test_pairs':len(test),'test_reactions':test.rhea_id.nunique(),'test_proteins':test.Entry.nunique(),'scale':scale,'adaptive_runtime_s':runtime,'adaptive_mass':float(adaptive.sum())})
   for method,F in [('linear_heat',J),('anisotropic_factor_sum',adaptive)]:
    for rid,g in test.groupby('rhea_id'):
     rows.append({'split_id':sid,'method':method,'direction':'reaction_to_enzyme','query_id':rid,**rank_metrics(F[ri[rid]],pids,set(g.Entry),set(),B)})
    for pid,g in test.groupby('Entry'):
     rows.append({'split_id':sid,'method':method,'direction':'enzyme_to_reaction','query_id':pid,**rank_metrics(F[:,pi[pid]],rids,set(g.rhea_id),set(),B)})
   print(json.dumps(splits[-1]),flush=True)
 q=pd.DataFrame(rows); s=pd.DataFrame(splits); m=agg(q); OUT.mkdir(parents=True,exist_ok=True);q.to_csv(OUT/'query_metrics.csv',index=False);s.to_csv(OUT/'split_summary.csv',index=False);m.to_csv(OUT/'metrics.csv',index=False)
 # paired query deltas
 a=q.pivot_table(index=['split_id','direction','query_id'],columns='method',values=['reciprocal_rank','best_positive_rank','hit_at_10','hit_at_20'],aggfunc='first')
 paired=[]
 for direction in ['reaction_to_enzyme','enzyme_to_reaction']:
  x=a.xs(direction,level='direction')
  paired.append({'direction':direction,'queries':len(x),'rr_improved':int((x['reciprocal_rank']['anisotropic_factor_sum']>x['reciprocal_rank']['linear_heat']).sum()),'rr_tied':int((x['reciprocal_rank']['anisotropic_factor_sum']==x['reciprocal_rank']['linear_heat']).sum()),'rr_worsened':int((x['reciprocal_rank']['anisotropic_factor_sum']<x['reciprocal_rank']['linear_heat']).sum()),'median_rank_delta_linear_minus_aniso':float(np.median(x['best_positive_rank']['linear_heat']-x['best_positive_rank']['anisotropic_factor_sum']))})
 summary={'version':'terpene-anisotropic-product-measure-dev-v1','partition':'development_only','operator':'unit-time factor-sum Charbonnier heat exp(L_R+L_E) applied to degree-normalized positive empirical measure; linear normalized heat Hellinger amplitude supplies the fixed conductance reference field only','parameter_selection':'none','frozen_cells_evaluated':False,'metrics':m.to_dict(orient='records'),'paired':paired,'input_sha256':{'split':sha(C/'marts_pair_folds.csv'),'protein_geometry':sha(PG/'manifest.json'),'reaction_geometry':sha(RG/'manifest.json')}}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('\n'+m.to_string(index=False));print(json.dumps(paired,indent=2))
if __name__=='__main__':main()
