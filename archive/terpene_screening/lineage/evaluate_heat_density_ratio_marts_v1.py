from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import load_npz,csr_matrix,diags
from scipy.sparse.linalg import expm_multiply
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.runtime.base_model import rank_metrics
C=ROOT/'data/terpene_marts_adaptation';PG=ROOT/'data/terpene_multiresolution_protein_geometry_v4';RG=ROOT/'data/terpene_multiresolution_reaction_geometry_v1';OUT=ROOT/'results/terpene_heat_density_ratio_dev_v1';BUD=(3,10,20)

def b(s):return s.astype(str).str.lower().isin({'1','true','yes'})
def sym_generator(w):
 w=csr_matrix(w,dtype=np.float64).maximum(csr_matrix(w,dtype=np.float64).T).tocsr();w.setdiag(0);w.eliminate_zeros()
 d=np.asarray(w.sum(1)).ravel();inv=np.zeros_like(d);inv[d>0]=d[d>0]**-.5
 S=(diags(inv)@w@diags(inv)).tocsr()
 return S-eye_like(S)
def eye_like(x):
 from scipy.sparse import eye
 return eye(x.shape[0],format='csr',dtype=np.float64)
def normalized_positive_measure(train,ri,pi,nr,ne):
 A=np.zeros((nr,ne),dtype=np.float64)
 for x in train.itertuples(index=False):A[ri[str(x.rhea_id)],pi[str(x.Entry)]]=1.0
 dr=A.sum(1);de=A.sum(0);sr=np.zeros(nr);se=np.zeros(ne);sr[dr>0]=dr[dr>0]**-.5;se[de>0]=de[de>0]**-.5
 M=sr[:,None]*A*se[None,:];z=M.sum()
 if z<=0:raise RuntimeError('empty positive measure')
 return M/z
def diffuse(M,Qr,Qe):
 # Symmetric intrinsic heat on the factor normalized-Laplacian geometry.
 P=expm_multiply(Qr,M)
 P=expm_multiply(Qe,P.T).T
 P=np.maximum(np.asarray(P,dtype=np.float64),0.0)
 return P
def density_ratio(P):
 pr=P.sum(1);pe=P.sum(0)
 den=pr[:,None]*pe[None,:]
 C=np.divide(P,den,out=np.zeros_like(P),where=den>0)
 if not np.all(np.isfinite(C)):raise RuntimeError('nonfinite density ratio')
 return C,pr,pe
def agg(q):
 rows=[]
 for d,g in q.groupby('direction'):
  x={'direction':d,'n_queries':len(g),'mrr':g.reciprocal_rank.mean(),'median_rank':g.best_positive_rank.median()}
  for k in BUD:x[f'hit{k}']=g[f'hit_at_{k}'].mean();x[f'recall{k}']=g[f'positive_recall_at_{k}'].mean()
  rows.append(x)
 return pd.DataFrame(rows)
def main():
 p=pd.read_csv(C/'protein_entities.csv',dtype=str).fillna('');r=pd.read_csv(C/'reaction_entities.csv',dtype=str).fillna('');pairs=pd.read_csv(C/'marts_pair_folds.csv',dtype=str).fillna('')
 pairs[['protein_fold','reaction_fold']]=pairs[['protein_fold','reaction_fold']].astype(int);pairs['protein_seen']=b(pairs.protein_seen);pairs['reaction_seen']=b(pairs.reaction_seen)
 pids=p.protein_id.tolist();rids=r.reaction_id.tolist();pi={x:i for i,x in enumerate(pids)};ri={x:i for i,x in enumerate(rids)}
 Qr=sym_generator(load_npz(RG/'partial_pullback_affinity.npz'));Qe=sym_generator(load_npz(PG/'partial_pullback_affinity.npz'))
 rows=[];sp=[];diag=[]
 for pf in range(5):
  for rf in range(5):
   if not(pf==4 or rf==4):continue
   sid=f'p{pf}_r{rf}';train=pairs[pairs.protein_fold.ne(pf)&pairs.reaction_fold.ne(rf)].drop_duplicates(['rhea_id','Entry']);test=pairs[pairs.protein_fold.eq(pf)&pairs.reaction_fold.eq(rf)&~pairs.protein_seen&~pairs.reaction_seen].drop_duplicates(['rhea_id','Entry'])
   M=normalized_positive_measure(train,ri,pi,len(rids),len(pids));P=diffuse(M,Qr,Qe);F,pr,pe=density_ratio(P)
   sp.append({'split_id':sid,'protein_overlap':len(set(train.Entry)&set(test.Entry)),'reaction_overlap':len(set(train.rhea_id)&set(test.rhea_id))})
   diag.append({'split_id':sid,'diffused_mass':float(P.sum()),'positive_ratio_fraction':float(np.mean(F>0)),'max_ratio':float(F.max()),'median_positive_ratio':float(np.median(F[F>0]))})
   for rid,g in test.groupby('rhea_id',sort=True):rows.append({'split_id':sid,'direction':'reaction_to_enzyme','query_id':rid,**rank_metrics(F[ri[rid]],pids,set(g.Entry.astype(str)),set(),BUD)})
   for pid,g in test.groupby('Entry',sort=True):rows.append({'split_id':sid,'direction':'enzyme_to_reaction','query_id':pid,**rank_metrics(F[:,pi[pid]],rids,set(g.rhea_id.astype(str)),set(),BUD)})
 q=pd.DataFrame(rows);sp=pd.DataFrame(sp);dg=pd.DataFrame(diag);met=agg(q);OUT.mkdir(parents=True,exist_ok=True);q.to_csv(OUT/'query_metrics.csv',index=False);met.to_csv(OUT/'metrics.csv',index=False);dg.to_csv(OUT/'diagnostics.csv',index=False)
 summary={'version':'terpene-heat-density-ratio-dev-v1','partition':'development_only','field':'C(r,e)=p_1(r,e)/(p_R,1(r) p_E,1(e)), where p_1=exp(-L_R) mu exp(-L_E) under the factor symmetric normalized-Laplacian heat semigroups','interpretation':'Radon-Nikodym density ratio of the smoothed positive correspondence against the product of its own smoothed marginals; log C is the pointwise mutual-information field','source':'degree-normalized positive empirical pair measure, unit mass','parameter_selection':'none; intrinsic unit heat time only, no sweep, negatives, score mixture, source strength, or direction-specific rule','all_overlaps_zero':bool(((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()),'metrics':met.to_dict('records'),'diagnostics':dg.to_dict('records')}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(met.to_string(index=False));print(json.dumps({k:v for k,v in summary.items() if k not in {'metrics','diagnostics'}},indent=2))
if __name__=='__main__':main()
