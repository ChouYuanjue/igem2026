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
CACHE=ROOT/'data/terpene_marts_adaptation';PG=ROOT/'data/terpene_multiresolution_protein_geometry_v4';RGGLOBAL=ROOT/'data/terpene_multiresolution_reaction_geometry_v1';RGREF=ROOT/'data/terpene_reaction_atlas_refined_geometry_v2';OUT=ROOT/'results/terpene_product_coupling_defect_dev_v4b';BUD=(3,10,20)
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
 data=np.sqrt(np.maximum(-np.log(np.clip(w.data,1e-300,1.0)),1e-12))/ell;g=csr_matrix((data,w.indices,w.indptr),shape=w.shape);ncomp,_=connected_components(g,directed=False)
 if ncomp!=1:raise RuntimeError(f'coupling defect requires connected factor geometry, got {ncomp} components')
 D=shortest_path(g,directed=False,unweighted=False);return np.asarray(D,dtype=np.float64),ell
def train_groups(train,ri,pi):
 by_r={};by_e={}; r_support=set();e_support=set()
 for x in train.itertuples(index=False):
  rr=ri[str(x.rhea_id)];ee=pi[str(x.Entry)];r_support.add(rr);e_support.add(ee);by_r.setdefault(rr,set()).add(ee);by_e.setdefault(ee,set()).add(rr)
 return {k:np.asarray(sorted(v),dtype=int) for k,v in by_r.items()},{k:np.asarray(sorted(v),dtype=int) for k,v in by_e.items()},np.asarray(sorted(r_support),dtype=int),np.asarray(sorted(e_support),dtype=int)
def defect_r2e(q,Dr2,Dp2,by_r,r_support,e_support):
 marg_r=float(np.min(Dr2[q,r_support]));marg_e=np.min(Dp2[:,e_support],axis=1);joint=np.full(Dp2.shape[0],np.inf)
 for rr,es in by_r.items():joint=np.minimum(joint,float(Dr2[q,rr])+np.min(Dp2[:,es],axis=1))
 defect=joint-marg_r-marg_e
 if not np.all(np.isfinite(defect)):raise RuntimeError('nonfinite R2E coupling defect on connected geometry')
 return np.maximum(defect,0.0)
def defect_e2r(q,Dr2,Dp2,by_e,r_support,e_support):
 marg_e=float(np.min(Dp2[q,e_support]));marg_r=np.min(Dr2[:,r_support],axis=1);joint=np.full(Dr2.shape[0],np.inf)
 for ee,rs in by_e.items():joint=np.minimum(joint,float(Dp2[q,ee])+np.min(Dr2[:,rs],axis=1))
 defect=joint-marg_e-marg_r
 if not np.all(np.isfinite(defect)):raise RuntimeError('nonfinite E2R coupling defect on connected geometry')
 return np.maximum(defect,0.0)
def aggregate(q):
 out=[]
 for (m,d),g in q.groupby(['method','direction']):
  r={'method':m,'direction':d,'n_queries':len(g),'mrr':g.reciprocal_rank.mean(),'median_rank':g.best_positive_rank.median()}
  for k in BUD:r[f'hit{k}']=g[f'hit_at_{k}'].mean();r[f'recall{k}']=g[f'positive_recall_at_{k}'].mean()
  out.append(r)
 return pd.DataFrame(out)
def main():
 proteins=pd.read_csv(CACHE/'protein_entities.csv',dtype=str).fillna('');reactions=pd.read_csv(CACHE/'reaction_entities.csv',dtype=str).fillna('');pairs=pd.read_csv(CACHE/'marts_pair_folds.csv',dtype=str).fillna('');pairs[['protein_fold','reaction_fold']]=pairs[['protein_fold','reaction_fold']].astype(int);pairs['protein_seen']=bools(pairs.protein_seen);pairs['reaction_seen']=bools(pairs.reaction_seen)
 pids=proteins.protein_id.astype(str).tolist();rids=reactions.reaction_id.astype(str).tolist();pi={x:i for i,x in enumerate(pids)};ri={x:i for i,x in enumerate(rids)}
 pg0=one_view(np.load(PG/'global_esmc_chordal_distance.npy'),np.load(PG/'global_esmc_available.npy'));pgM=load_npz(PG/'partial_pullback_affinity.npz');rgG=load_npz(RGGLOBAL/'partial_pullback_affinity.npz');rgR=load_npz(RGREF/'atlas_refined_affinity.npz')
 Dp0,lp0=normalized_geodesic(pg0);DpM,lpM=normalized_geodesic(pgM);DrG,lrG=normalized_geodesic(rgG);DrR,lrR=normalized_geodesic(rgR)
 methods={'global_geometry':(DrG,Dp0),'protein_multires':(DrG,DpM),'reaction_refined':(DrR,Dp0),'all_information_hierarchical':(DrR,DpM)}
 rows=[];spl=[]
 for pf in range(5):
  for rf in range(5):
   if not(pf==4 or rf==4):continue
   sid=f'p{pf}_r{rf}';train=pairs[pairs.protein_fold.ne(pf)&pairs.reaction_fold.ne(rf)].drop_duplicates(['rhea_id','Entry']);test=pairs[pairs.protein_fold.eq(pf)&pairs.reaction_fold.eq(rf)&~pairs.protein_seen&~pairs.reaction_seen].drop_duplicates(['rhea_id','Entry']);spl.append({'split_id':sid,'train_pairs':len(train),'test_pairs':len(test),'protein_overlap':len(set(train.Entry)&set(test.Entry)),'reaction_overlap':len(set(train.rhea_id)&set(test.rhea_id))});by_r,by_e,rs,es=train_groups(train,ri,pi)
   for method,(Dr,Dp) in methods.items():
    Dr2=Dr*Dr;Dp2=Dp*Dp
    for rid,g in test.groupby('rhea_id',sort=True):
     defect=defect_r2e(ri[rid],Dr2,Dp2,by_r,rs,es);rows.append({'split_id':sid,'method':method,'direction':'reaction_to_enzyme','query_id':rid,'defect_min':float(defect.min()),'defect_median':float(np.median(defect)),**rank_metrics(-defect,pids,set(g.Entry.astype(str)),set(),BUD)})
    for pid,g in test.groupby('Entry',sort=True):
     defect=defect_e2r(pi[pid],Dr2,Dp2,by_e,rs,es);rows.append({'split_id':sid,'method':method,'direction':'enzyme_to_reaction','query_id':pid,'defect_min':float(defect.min()),'defect_median':float(np.median(defect)),**rank_metrics(-defect,rids,set(g.rhea_id.astype(str)),set(),BUD)})
 q=pd.DataFrame(rows);sp=pd.DataFrame(spl);met=aggregate(q);OUT.mkdir(parents=True,exist_ok=True);q.to_csv(OUT/'query_metrics.csv',index=False);sp.to_csv(OUT/'split_summary.csv',index=False);met.to_csv(OUT/'metrics.csv',index=False)
 summary={'version':'terpene-product-coupling-defect-dev-v4b','partition':'development_only','field':'F(r,e)=-Delta_Omega(r,e), Delta_Omega=min_(ri,ei in Omega)[dR(r,ri)^2+dE(e,ei)^2]-min_(ri in Omega_R)dR(r,ri)^2-min_(ei in Omega_E)dE(e,ei)^2','meaning':'excess product-geodesic cost of requiring reaction and enzyme to be explained by the same known biochemical precedent rather than by independent marginal familiarity','factor_normalization':'each connected factor geodesic divided by its median local edge length before Cartesian product','parameter_selection':'none; no negative labels, score fusion, diffusion time, temperature, source strength, or direction-specific rule','characteristic_lengths':{'protein_global':lp0,'protein_multires':lpM,'reaction_global':lrG,'reaction_refined':lrR},'all_overlaps_zero':bool(((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()),'input_sha256':{'split':sha(CACHE/'marts_pair_folds.csv'),'protein_geometry':sha(PG/'manifest.json'),'reaction_global':sha(RGGLOBAL/'manifest.json'),'reaction_refined':sha(RGREF/'manifest.json')},'metrics':met.to_dict('records')};(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(met.to_string(index=False));print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2))
if __name__=='__main__':main()
