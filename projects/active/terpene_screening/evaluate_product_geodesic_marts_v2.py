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
CACHE=ROOT/'data/terpene_marts_adaptation'; PG=ROOT/'data/terpene_multiresolution_protein_geometry_v4'; RG=ROOT/'data/terpene_multiresolution_reaction_geometry_v2'; HG=ROOT/'data/terpene_hierarchical_factor_geometry_v1'; OUT=ROOT/'results/terpene_product_geodesic_dev_v2'; BUD=(3,10,20)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bools(s):return s.astype(str).str.lower().isin({'1','true','yes'})
def energy_geodesic(w):
 w=csr_matrix(w,dtype=np.float64).maximum(csr_matrix(w,dtype=np.float64).T).tocsr(); w.setdiag(0);w.eliminate_zeros()
 data=np.sqrt(np.maximum(-np.log(np.clip(w.data,1e-300,1.0)),1e-12)); g=csr_matrix((data,w.indices,w.indptr),shape=w.shape)
 ncomp,_=connected_components(g,directed=False); d=shortest_path(g,directed=False,unweighted=False)
 return np.asarray(d,dtype=np.float64),int(ncomp)
def one_view(d,a):
 g,_=partial_observation_pullback_affinity([np.asarray(d)],[np.asarray(a,bool)]);return g.tocsr()
def score_r2e(qr,train,Dr2,De2,ri,pi,nE):
 best=np.full(nE,np.inf,dtype=np.float64)
 for rid,g in train.groupby('rhea_id',sort=False):
  erows=np.asarray([pi[x] for x in g.Entry.astype(str).unique()],dtype=int)
  best=np.minimum(best,Dr2[qr,ri[str(rid)]]+np.min(De2[:,erows],axis=1))
 return -best
def score_e2r(qe,train,Dr2,De2,ri,pi,nR):
 best=np.full(nR,np.inf,dtype=np.float64)
 for pid,g in train.groupby('Entry',sort=False):
  rrows=np.asarray([ri[x] for x in g.rhea_id.astype(str).unique()],dtype=int)
  best=np.minimum(best,De2[qe,pi[str(pid)]]+np.min(Dr2[:,rrows],axis=1))
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
 pg0=one_view(np.load(PG/'global_esmc_chordal_distance.npy'),np.load(PG/'global_esmc_available.npy'));rg0=one_view(np.load(RG/'drfp_chordal_distance.npy'),np.load(RG/'drfp_available.npy'));pgH=load_npz(HG/'protein_affinity.npz');rgH=load_npz(HG/'reaction_affinity.npz')
 Dp0,cp0=energy_geodesic(pg0);Dr0,cr0=energy_geodesic(rg0);DpH,cpH=energy_geodesic(pgH);DrH,crH=energy_geodesic(rgH)
 methods={'global_only':(Dr0,Dp0),'protein_hierarchical':(Dr0,DpH),'reaction_hierarchical':(DrH,Dp0),'all_hierarchical':(DrH,DpH)}
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
 summary={'version':'terpene-product-geodesic-dev-v2','partition':'development_only','field':'F(r,e)=-D_Omega(r,e)^2 where D_Omega is exact Cartesian-product geodesic distance to the fold-training positive set and D_product^2=d_R^2+d_E^2','edge_length':'sqrt(-log(self-tuned pullback affinity)); numerical floor only at 1e-12 energy','parameter_selection':'none; no diffusion time, temperature, source strength, score fusion, negative pairs, or frozen labels','components':{'protein_global':cp0,'reaction_global':cr0,'protein_hierarchical':cpH,'reaction_hierarchical':crH},'all_overlaps_zero':bool(((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()),'input_sha256':{'split':sha(CACHE/'marts_pair_folds.csv'),'protein_geometry':sha(PG/'manifest.json'),'reaction_geometry':sha(RG/'manifest.json'),'hierarchical_geometry':sha(HG/'manifest.json')},'metrics':met.to_dict('records')};(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(met.to_string(index=False));print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2))
if __name__=='__main__':main()
