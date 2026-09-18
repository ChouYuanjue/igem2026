from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix,load_npz
from scipy.sparse.csgraph import shortest_path,connected_components
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.multiscale_geometry import partial_observation_pullback_affinity
from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics
CACHE=ROOT/'data/terpene_marts_adaptation';PG=ROOT/'data/terpene_multiresolution_protein_geometry_v4';RGMIN=ROOT/'data/terpene_multiresolution_reaction_geometry_v2';RGGLOBAL=ROOT/'data/terpene_multiresolution_reaction_geometry_v1';RGREF=ROOT/'data/terpene_reaction_atlas_refined_geometry_v2';OUT=ROOT/'results/terpene_normalized_product_geodesic_dev_v3';BUD=(3,10,20)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bools(s):return s.astype(str).str.lower().isin({'1','true','yes'})
def one_view(d,a):
 g,_=partial_observation_pullback_affinity([np.asarray(d)],[np.asarray(a,bool)]);return g.tocsr()
def normalized_geodesic(w):
 w=csr_matrix(w,dtype=np.float64).maximum(csr_matrix(w,dtype=np.float64).T).tocsr();w.setdiag(0);w.eliminate_zeros();coo=w.tocoo();mask=coo.row<coo.col;edge=np.sqrt(np.maximum(-np.log(np.clip(coo.data[mask],1e-300,1.0)),1e-12));ell=float(np.median(edge));
 if not np.isfinite(ell) or ell<=0:raise RuntimeError('invalid characteristic edge length')
 data=np.sqrt(np.maximum(-np.log(np.clip(w.data,1e-300,1.0)),1e-12))/ell;g=csr_matrix((data,w.indices,w.indptr),shape=w.shape);ncomp,_=connected_components(g,directed=False);D=shortest_path(g,directed=False,unweighted=False);return np.asarray(D,dtype=np.float64),ell,int(ncomp)
def score_r2e(qr,train,Dr2,De2,ri,pi,nE):
 best=np.full(nE,np.inf)
 for rid,g in train.groupby('rhea_id',sort=False):
  er=np.asarray([pi[x] for x in g.Entry.astype(str).unique()],int);best=np.minimum(best,Dr2[qr,ri[str(rid)]]+np.min(De2[:,er],axis=1))
 return -best
def score_e2r(qe,train,Dr2,De2,ri,pi,nR):
 best=np.full(nR,np.inf)
 for pid,g in train.groupby('Entry',sort=False):
  rr=np.asarray([ri[x] for x in g.rhea_id.astype(str).unique()],int);best=np.minimum(best,De2[qe,pi[str(pid)]]+np.min(Dr2[:,rr],axis=1))
 return -best
def aggregate(q):
 out=[]
 for (m,d),g in q.groupby(['method','direction']):
  r={'method':m,'direction':d,'n_queries':len(g),'mrr':g.reciprocal_rank.mean(),'median_rank':g.best_positive_rank.median()}
  for k in BUD:r[f'hit{k}']=g[f'hit_at_{k}'].mean();r[f'recall{k}']=g[f'positive_recall_at_{k}'].mean()
  out.append(r)
 return pd.DataFrame(out)
def main():
 p=pd.read_csv(CACHE/'protein_entities.csv',dtype=str).fillna('');r=pd.read_csv(CACHE/'reaction_entities.csv',dtype=str).fillna('');pairs=pd.read_csv(CACHE/'marts_pair_folds.csv',dtype=str).fillna('');pairs[['protein_fold','reaction_fold']]=pairs[['protein_fold','reaction_fold']].astype(int);pairs['protein_seen']=bools(pairs.protein_seen);pairs['reaction_seen']=bools(pairs.reaction_seen)
 pids=p.protein_id.astype(str).tolist();rids=r.reaction_id.astype(str).tolist();pi={x:i for i,x in enumerate(pids)};ri={x:i for i,x in enumerate(rids)}
 pg0=one_view(np.load(PG/'global_esmc_chordal_distance.npy'),np.load(PG/'global_esmc_available.npy'));pgM=load_npz(PG/'partial_pullback_affinity.npz');rg0=one_view(np.load(RGMIN/'drfp_chordal_distance.npy'),np.load(RGMIN/'drfp_available.npy'));rgG=load_npz(RGGLOBAL/'partial_pullback_affinity.npz');rgR=load_npz(RGREF/'atlas_refined_affinity.npz')
 Dp0,lp0,cp0=normalized_geodesic(pg0);DpM,lpM,cpM=normalized_geodesic(pgM);Dr0,lr0,cr0=normalized_geodesic(rg0);DrG,lrG,crG=normalized_geodesic(rgG);DrR,lrR,crR=normalized_geodesic(rgR)
 methods={'minimal_global':(Dr0,Dp0),'protein_multires_v4':(Dr0,DpM),'global_chemistry_plus_protein_v4':(DrG,DpM),'all_information_hierarchical':(DrR,DpM),'reaction_global_only':(DrG,Dp0),'reaction_refined_only':(DrR,Dp0)}
 rows=[];spl=[]
 for pf in range(5):
  for rf in range(5):
   if not(pf==4 or rf==4):continue
   sid=f'p{pf}_r{rf}';train=pairs[pairs.protein_fold.ne(pf)&pairs.reaction_fold.ne(rf)].drop_duplicates(['rhea_id','Entry']);test=pairs[pairs.protein_fold.eq(pf)&pairs.reaction_fold.eq(rf)&~pairs.protein_seen&~pairs.reaction_seen].drop_duplicates(['rhea_id','Entry']);spl.append({'split_id':sid,'train_pairs':len(train),'test_pairs':len(test),'protein_overlap':len(set(train.Entry)&set(test.Entry)),'reaction_overlap':len(set(train.rhea_id)&set(test.rhea_id))})
   for method,(Dr,Dp) in methods.items():
    Dr2=Dr*Dr;Dp2=Dp*Dp
    for rid,g in test.groupby('rhea_id',sort=True):rows.append({'split_id':sid,'method':method,'direction':'reaction_to_enzyme','query_id':rid,**rank_metrics(score_r2e(ri[rid],train,Dr2,Dp2,ri,pi,len(pids)),pids,set(g.Entry.astype(str)),set(),BUD)})
    for pid,g in test.groupby('Entry',sort=True):rows.append({'split_id':sid,'method':method,'direction':'enzyme_to_reaction','query_id':pid,**rank_metrics(score_e2r(pi[pid],train,Dr2,Dp2,ri,pi,len(rids)),rids,set(g.rhea_id.astype(str)),set(),BUD)})
 q=pd.DataFrame(rows);sp=pd.DataFrame(spl);met=aggregate(q);OUT.mkdir(parents=True,exist_ok=True);q.to_csv(OUT/'query_metrics.csv',index=False);sp.to_csv(OUT/'split_summary.csv',index=False);met.to_csv(OUT/'metrics.csv',index=False)
 summary={'version':'terpene-normalized-product-geodesic-dev-v3','partition':'development_only','field':'F=-D_Omega^2 on the exact Cartesian product of factor geodesics','factor_normalization':'each factor geodesic is divided by the median undirected local edge length of its own self-tuned pullback graph before forming d_R^2+d_E^2','parameter_selection':'none','characteristic_lengths':{'protein_global':lp0,'protein_multires':lpM,'reaction_drfp':lr0,'reaction_global_chemistry':lrG,'reaction_hierarchical_refined':lrR},'components':{'protein_global':cp0,'protein_multires':cpM,'reaction_drfp':cr0,'reaction_global':crG,'reaction_refined':crR},'all_overlaps_zero':bool(((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()),'input_sha256':{'split':sha(CACHE/'marts_pair_folds.csv'),'protein_geometry':sha(PG/'manifest.json'),'reaction_global':sha(RGGLOBAL/'manifest.json'),'reaction_refined':sha(RGREF/'manifest.json')},'metrics':met.to_dict('records')};(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(met.to_string(index=False));print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2))
if __name__=='__main__':main()
