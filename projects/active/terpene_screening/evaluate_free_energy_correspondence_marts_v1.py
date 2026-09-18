from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import load_npz,csr_matrix
from scipy.sparse.csgraph import shortest_path,connected_components
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics
CACHE=ROOT/'data/terpene_marts_adaptation';PG=ROOT/'data/terpene_multiresolution_protein_geometry_v4';RG=ROOT/'data/terpene_multiresolution_reaction_geometry_v1';OUT=ROOT/'results/terpene_free_energy_correspondence_dev_v1';BUD=(3,10,20)

def bools(s):return s.astype(str).str.lower().isin({'1','true','yes'})
def geo(w):
 w=csr_matrix(w,dtype=np.float64).maximum(csr_matrix(w,dtype=np.float64).T).tocsr();w.setdiag(0);w.eliminate_zeros()
 coo=w.tocoo();u=coo.row<coo.col;raw=np.sqrt(np.maximum(-np.log(np.clip(coo.data[u],1e-300,1.0)),1e-12));ell=float(np.median(raw))
 edge=np.sqrt(np.maximum(-np.log(np.clip(w.data,1e-300,1.0)),1e-12))/ell;g=csr_matrix((edge,w.indices,w.indptr),shape=w.shape)
 nc,_=connected_components(g,directed=False)
 if nc!=1:raise RuntimeError(f'disconnected factor {nc}')
 return np.asarray(shortest_path(g,directed=False,unweighted=False),float),ell
def aggregate(q):
 out=[]
 for direction,g in q.groupby('direction'):
  r={'direction':direction,'n_queries':len(g),'mrr':g.reciprocal_rank.mean(),'median_rank':g.best_positive_rank.median()}
  for k in BUD:r[f'hit{k}']=g[f'hit_at_{k}'].mean();r[f'recall{k}']=g[f'positive_recall_at_{k}'].mean()
  out.append(r)
 return pd.DataFrame(out)
def field(train,Dr2,De2,ri,pi):
 nr,ne=Dr2.shape[0],De2.shape[0]
 A=np.zeros((nr,ne),dtype=np.float64)
 for x in train.itertuples(index=False):A[ri[str(x.rhea_id)],pi[str(x.Entry)]]=1.0
 rs=np.flatnonzero(A.sum(1)>0);es=np.flatnonzero(A.sum(0)>0)
 mr=np.min(Dr2[:,rs],axis=1);me=np.min(De2[:,es],axis=1)
 # Marginal-distance subtraction is performed before exponentiation.  At least
 # one support point per row then has exponent zero, preventing underflow.
 KR=np.exp(-(Dr2[:,rs]-mr[:,None]))
 KE=np.exp(-(De2[:,es]-me[:,None]))
 As=A[np.ix_(rs,es)]
 ZR=KR.sum(1);ZE=KE.sum(1);ZJ=KR@As@KE.T
 if np.any(ZJ<=0):raise RuntimeError('free-energy joint partition vanished')
 defect=np.log(ZR[:,None])+np.log(ZE[None,:])-np.log(ZJ)
 if np.min(defect)<-1e-10:raise RuntimeError(f'free-energy defect should be nonnegative, min={defect.min()}')
 return -np.maximum(defect,0.0)
def main():
 proteins=pd.read_csv(CACHE/'protein_entities.csv',dtype=str).fillna('');reactions=pd.read_csv(CACHE/'reaction_entities.csv',dtype=str).fillna('');pairs=pd.read_csv(CACHE/'marts_pair_folds.csv',dtype=str).fillna('')
 pairs[['protein_fold','reaction_fold']]=pairs[['protein_fold','reaction_fold']].astype(int);pairs['protein_seen']=bools(pairs.protein_seen);pairs['reaction_seen']=bools(pairs.reaction_seen)
 pids=proteins.protein_id.astype(str).tolist();rids=reactions.reaction_id.astype(str).tolist();pi={x:i for i,x in enumerate(pids)};ri={x:i for i,x in enumerate(rids)}
 Dr,lr=geo(load_npz(RG/'partial_pullback_affinity.npz'));De,le=geo(load_npz(PG/'partial_pullback_affinity.npz'));Dr2=Dr*Dr;De2=De*De
 rows=[];spl=[]
 for pf in range(5):
  for rf in range(5):
   if not(pf==4 or rf==4):continue
   sid=f'p{pf}_r{rf}';train=pairs[pairs.protein_fold.ne(pf)&pairs.reaction_fold.ne(rf)].drop_duplicates(['rhea_id','Entry']);test=pairs[pairs.protein_fold.eq(pf)&pairs.reaction_fold.eq(rf)&~pairs.protein_seen&~pairs.reaction_seen].drop_duplicates(['rhea_id','Entry'])
   F=field(train,Dr2,De2,ri,pi);spl.append({'split_id':sid,'train_pairs':len(train),'test_pairs':len(test),'protein_overlap':len(set(train.Entry)&set(test.Entry)),'reaction_overlap':len(set(train.rhea_id)&set(test.rhea_id))})
   for rid,g in test.groupby('rhea_id',sort=True):rows.append({'split_id':sid,'direction':'reaction_to_enzyme','query_id':rid,**rank_metrics(F[ri[rid]],pids,set(g.Entry.astype(str)),set(),BUD)})
   for pid,g in test.groupby('Entry',sort=True):rows.append({'split_id':sid,'direction':'enzyme_to_reaction','query_id':pid,**rank_metrics(F[:,pi[pid]],rids,set(g.rhea_id.astype(str)),set(),BUD)})
 q=pd.DataFrame(rows);sp=pd.DataFrame(spl);met=aggregate(q);OUT.mkdir(parents=True,exist_ok=True);q.to_csv(OUT/'query_metrics.csv',index=False);sp.to_csv(OUT/'split_summary.csv',index=False);met.to_csv(OUT/'metrics.csv',index=False)
 summary={'version':'terpene-free-energy-correspondence-dev-v1','partition':'development_only','field':'F=-Delta_1, Delta_tau=tau log(Z_R Z_E / Z_Omega), evaluated only at intrinsic unit temperature tau=1 after factor characteristic-length normalization','zero_temperature_limit':'tau -> 0 gives the canonical hard product-geodesic correspondence defect','parameter_selection':'none; tau=1 is the intrinsic unit after factor normalization and was not swept','characteristic_lengths':{'reaction':lr,'protein':le},'all_overlaps_zero':bool(((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()),'metrics':met.to_dict('records')}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(met.to_string(index=False));print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2))
if __name__=='__main__':main()
