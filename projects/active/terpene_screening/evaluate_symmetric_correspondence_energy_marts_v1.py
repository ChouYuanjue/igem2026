from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import load_npz,csr_matrix
from scipy.sparse.csgraph import shortest_path,connected_components
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.product_correspondence_field import correspondence_state
from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics
C=ROOT/'data/terpene_marts_adaptation';PG=ROOT/'data/terpene_multiresolution_protein_geometry_v4';RG=ROOT/'data/terpene_multiresolution_reaction_geometry_v1';OUT=ROOT/'results/terpene_symmetric_correspondence_energy_dev_v1';BUD=(3,10,20)

def b(s):return s.astype(str).str.lower().isin({'1','true','yes'})
def geo(w):
 w=csr_matrix(w,dtype=float).maximum(csr_matrix(w,dtype=float).T).tocsr();w.setdiag(0);w.eliminate_zeros();coo=w.tocoo();u=coo.row<coo.col;raw=np.sqrt(np.maximum(-np.log(np.clip(coo.data[u],1e-300,1.0)),1e-12));ell=float(np.median(raw));edge=np.sqrt(np.maximum(-np.log(np.clip(w.data,1e-300,1.0)),1e-12))/ell;g=csr_matrix((edge,w.indices,w.indptr),shape=w.shape);nc,_=connected_components(g,directed=False);assert nc==1;return np.asarray(shortest_path(g,directed=False,unweighted=False)),ell
def agg(q):
 rows=[]
 for d,g in q.groupby('direction'):
  x={'direction':d,'n_queries':len(g),'mrr':g.reciprocal_rank.mean(),'median_rank':g.best_positive_rank.median()}
  for k in BUD:x[f'hit{k}']=g[f'hit_at_{k}'].mean();x[f'recall{k}']=g[f'positive_recall_at_{k}'].mean()
  rows.append(x)
 return pd.DataFrame(rows)
def main():
 p=pd.read_csv(C/'protein_entities.csv',dtype=str).fillna('');r=pd.read_csv(C/'reaction_entities.csv',dtype=str).fillna('');pairs=pd.read_csv(C/'marts_pair_folds.csv',dtype=str).fillna('');pairs[['protein_fold','reaction_fold']]=pairs[['protein_fold','reaction_fold']].astype(int);pairs['protein_seen']=b(pairs.protein_seen);pairs['reaction_seen']=b(pairs.reaction_seen)
 pids=p.protein_id.tolist();rids=r.reaction_id.tolist();pi={x:i for i,x in enumerate(pids)};ri={x:i for i,x in enumerate(rids)}
 Dr,lr=geo(load_npz(RG/'partial_pullback_affinity.npz'));De,le=geo(load_npz(PG/'partial_pullback_affinity.npz'));Dr2=Dr*Dr;De2=De*De
 rows=[];sp=[]
 for pf in range(5):
  for rf in range(5):
   if not(pf==4 or rf==4):continue
   sid=f'p{pf}_r{rf}';train=pairs[pairs.protein_fold.ne(pf)&pairs.reaction_fold.ne(rf)].drop_duplicates(['rhea_id','Entry']);test=pairs[pairs.protein_fold.eq(pf)&pairs.reaction_fold.eq(rf)&~pairs.protein_seen&~pairs.reaction_seen].drop_duplicates(['rhea_id','Entry']);arr=np.asarray([(ri[x.rhea_id],pi[x.Entry]) for x in train.itertuples()],int);st=correspondence_state(Dr2,De2,arr)
   Gamma=st.joint_sq-0.5*st.reaction_marginal_sq[:,None]-0.5*st.protein_marginal_sq[None,:];F=-Gamma
   sp.append({'split_id':sid,'protein_overlap':len(set(train.Entry)&set(test.Entry)),'reaction_overlap':len(set(train.rhea_id)&set(test.rhea_id))})
   for rid,g in test.groupby('rhea_id',sort=True):rows.append({'split_id':sid,'direction':'reaction_to_enzyme','query_id':rid,**rank_metrics(F[ri[rid]],pids,set(g.Entry.astype(str)),set(),BUD)})
   for pid,g in test.groupby('Entry',sort=True):rows.append({'split_id':sid,'direction':'enzyme_to_reaction','query_id':pid,**rank_metrics(F[:,pi[pid]],rids,set(g.rhea_id.astype(str)),set(),BUD)})
 q=pd.DataFrame(rows);sp=pd.DataFrame(sp);met=agg(q);OUT.mkdir(parents=True,exist_ok=True);q.to_csv(OUT/'query_metrics.csv',index=False);met.to_csv(OUT/'metrics.csv',index=False)
 summary={'version':'terpene-symmetric-correspondence-energy-dev-v1','partition':'development_only','field':'F=-Gamma_Omega, Gamma_Omega=J_Omega-(m_R+m_E)/2','derivation':'zero-temperature symmetric normalization of the joint precedent kernel by the geometric mean of its reaction and protein marginal kernels; the 1/2 coefficient is fixed by symmetric normalization, not selected','parameter_selection':'none','characteristic_lengths':{'reaction':lr,'protein':le},'all_overlaps_zero':bool(((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()),'metrics':met.to_dict('records')};(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(met.to_string(index=False));print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2))
if __name__=='__main__':main()
