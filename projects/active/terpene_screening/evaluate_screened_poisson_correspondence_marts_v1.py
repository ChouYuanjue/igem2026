from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import load_npz,csr_matrix,diags,eye
from scipy.linalg import eigh
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics
C=ROOT/'data/terpene_marts_adaptation';PG=ROOT/'data/terpene_multiresolution_protein_geometry_v4';RG=ROOT/'data/terpene_multiresolution_reaction_geometry_v1';OUT=ROOT/'results/terpene_screened_poisson_correspondence_dev_v1';BUD=(3,10,20)

def b(s):return s.astype(str).str.lower().isin({'1','true','yes'})
def lap_eig(w):
 w=csr_matrix(w,dtype=np.float64).maximum(csr_matrix(w,dtype=np.float64).T).tocsr();w.setdiag(0);w.eliminate_zeros();d=np.asarray(w.sum(1)).ravel();inv=np.zeros_like(d);inv[d>0]=d[d>0]**-.5;S=(diags(inv)@w@diags(inv)).toarray();L=np.eye(len(d))-S;lam,U=eigh(L,overwrite_a=True,check_finite=False);return lam,U
def norm_measure(train,ri,pi,nr,ne):
 A=np.zeros((nr,ne),dtype=np.float64)
 for x in train.itertuples(index=False):A[ri[str(x.rhea_id)],pi[str(x.Entry)]]=1
 dr=A.sum(1);de=A.sum(0);sr=np.zeros(nr);se=np.zeros(ne);sr[dr>0]=dr[dr>0]**-.5;se[de>0]=de[de>0]**-.5;M=sr[:,None]*A*se[None,:];z=M.sum();return M/z if z>0 else M
def solve(M,lr,Ur,le,Ue):
 B=Ur.T@M@Ue
 B/=1.0+lr[:,None]+le[None,:]
 return Ur@B@Ue.T
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
 t=time.time();lr,Ur=lap_eig(load_npz(RG/'partial_pullback_affinity.npz'));le,Ue=lap_eig(load_npz(PG/'partial_pullback_affinity.npz'));eig_s=time.time()-t
 rows=[];sp=[]
 for pf in range(5):
  for rf in range(5):
   if not(pf==4 or rf==4):continue
   sid=f'p{pf}_r{rf}';train=pairs[pairs.protein_fold.ne(pf)&pairs.reaction_fold.ne(rf)].drop_duplicates(['rhea_id','Entry']);test=pairs[pairs.protein_fold.eq(pf)&pairs.reaction_fold.eq(rf)&~pairs.protein_seen&~pairs.reaction_seen].drop_duplicates(['rhea_id','Entry']);M=norm_measure(train,ri,pi,len(rids),len(pids));F=solve(M,lr,Ur,le,Ue)
   sp.append({'split_id':sid,'protein_overlap':len(set(train.Entry)&set(test.Entry)),'reaction_overlap':len(set(train.rhea_id)&set(test.rhea_id))})
   for rid,g in test.groupby('rhea_id',sort=True):rows.append({'split_id':sid,'direction':'reaction_to_enzyme','query_id':rid,**rank_metrics(F[ri[rid]],pids,set(g.Entry.astype(str)),set(),BUD)})
   for pid,g in test.groupby('Entry',sort=True):rows.append({'split_id':sid,'direction':'enzyme_to_reaction','query_id':pid,**rank_metrics(F[:,pi[pid]],rids,set(g.rhea_id.astype(str)),set(),BUD)})
 q=pd.DataFrame(rows);sp=pd.DataFrame(sp);met=agg(q);OUT.mkdir(parents=True,exist_ok=True);q.to_csv(OUT/'query_metrics.csv',index=False);met.to_csv(OUT/'metrics.csv',index=False)
 summary={'version':'terpene-screened-poisson-correspondence-dev-v1','partition':'development_only','field':'(I + L_R tensor-sum L_E) F = mu, solved exactly in the factor Laplacian eigenbasis','source':'positive empirical pair measure with symmetric pair-degree normalization A/sqrt(d_R^pair d_E^pair), normalized to unit mass','geometry':'global reaction chemistry x multiresolution protein geometry','parameter_selection':'none; unit screening after normalized-Laplacian geometry, no diffusion time, temperature, negatives, source strength, or direction-specific rule','eigendecomposition_seconds':eig_s,'reaction_spectrum':[float(lr.min()),float(lr.max())],'protein_spectrum':[float(le.min()),float(le.max())],'all_overlaps_zero':bool(((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()),'metrics':met.to_dict('records')};(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(met.to_string(index=False));print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2))
if __name__=='__main__':main()
