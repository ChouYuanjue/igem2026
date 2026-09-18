from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix,load_npz
from scipy.sparse.csgraph import shortest_path,connected_components
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.geometry.multiscale import partial_observation_pullback_affinity
from projects.active.terpene_screening.runtime.base_model import rank_metrics
CACHE=ROOT/'data/terpene_marts_adaptation';PG=ROOT/'data/terpene_multiresolution_protein_geometry_v4';RG=ROOT/'data/terpene_multiresolution_reaction_geometry_v2';HG=ROOT/'data/terpene_hierarchical_factor_geometry_v1';OUT=ROOT/'results/terpene_product_geodesic_potential_dev_v1';BUD=(3,10,20)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bools(s):return s.astype(str).str.lower().isin({'1','true','yes'})
def one(d,a):
 g,_=partial_observation_pullback_affinity([np.load(d)],[np.load(a).astype(bool)]);return g.tocsr()
def intrinsic_kernel(g):
 w=g.maximum(g.T).tocsr(); data=np.sqrt(np.maximum(-np.log(np.clip(w.data,1e-300,1)),1e-12)); L=csr_matrix((data,w.indices,w.indptr),shape=w.shape); ncomp,_=connected_components(L,directed=False);D=np.asarray(shortest_path(L,directed=False),dtype=np.float64);off=np.isfinite(D)&(~np.eye(len(D),dtype=bool));ell=float(np.sqrt(np.mean(D[off]**2))); Z=D/ell; K=np.zeros_like(Z); finite=np.isfinite(Z);K[finite]=np.exp(-Z[finite]**2);return K,int(ncomp),ell
def normalized_measure(A):
 A=np.asarray(A,dtype=np.float64);dr=A.sum(1);de=A.sum(0);sr=np.zeros_like(dr);se=np.zeros_like(de);sr[dr>0]=dr[dr>0]**-.5;se[de>0]=de[de>0]**-.5;M=sr[:,None]*A*se[None,:];z=M.sum();return M/z if z>0 else M
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
 pg0=one(PG/'global_esmc_chordal_distance.npy',PG/'global_esmc_available.npy');rg0=one(RG/'drfp_chordal_distance.npy',RG/'drfp_available.npy');pgH=load_npz(HG/'protein_affinity.npz');rgH=load_npz(HG/'reaction_affinity.npz')
 KE0,cp0,ep0=intrinsic_kernel(pg0);KR0,cr0,er0=intrinsic_kernel(rg0);KEH,cpH,epH=intrinsic_kernel(pgH);KRH,crH,erH=intrinsic_kernel(rgH)
 methods={'global_only':(KR0,KE0),'protein_hierarchical':(KR0,KEH),'reaction_hierarchical':(KRH,KE0),'all_hierarchical':(KRH,KEH)}
 rows=[];spl=[]
 for pf in range(5):
  for rf in range(5):
   if not(pf==4 or rf==4):continue
   sid=f'p{pf}_r{rf}';train=pairs[pairs.protein_fold.ne(pf)&pairs.reaction_fold.ne(rf)].drop_duplicates(['rhea_id','Entry']);test=pairs[pairs.protein_fold.eq(pf)&pairs.reaction_fold.eq(rf)&~pairs.protein_seen&~pairs.reaction_seen].drop_duplicates(['rhea_id','Entry']);spl.append({'split_id':sid,'train_pairs':len(train),'test_pairs':len(test),'protein_overlap':len(set(train.Entry)&set(test.Entry)),'reaction_overlap':len(set(train.rhea_id)&set(test.rhea_id))})
   A=np.zeros((len(rids),len(pids)),dtype=np.float64)
   for x in train.itertuples(index=False):A[ri[str(x.rhea_id)],pi[str(x.Entry)]]=1.0
   M=normalized_measure(A); Ms=csr_matrix(M)
   for method,(KR,KE) in methods.items():
    # Product radial potential of the whole empirical positive measure:
    # F = K_R M K_E^T.  Sparse M keeps the fold solve cheap and exact.
    F=KR @ (Ms @ KE.T)
    for rid,g in test.groupby('rhea_id',sort=True):rows.append({'split_id':sid,'method':method,'direction':'reaction_to_enzyme','query_id':rid,**rank_metrics(F[ri[rid]],pids,set(g.Entry.astype(str)),set(),BUD)})
    for pid,g in test.groupby('Entry',sort=True):rows.append({'split_id':sid,'method':method,'direction':'enzyme_to_reaction','query_id':pid,**rank_metrics(F[:,pi[pid]],rids,set(g.rhea_id.astype(str)),set(),BUD)})
 q=pd.DataFrame(rows);sp=pd.DataFrame(spl);met=aggregate(q);OUT.mkdir(parents=True,exist_ok=True);q.to_csv(OUT/'query_metrics.csv',index=False);sp.to_csv(OUT/'split_summary.csv',index=False);met.to_csv(OUT/'metrics.csv',index=False)
 summary={'version':'terpene-product-geodesic-potential-dev-v1','partition':'development_only','field':'F(x)=integral exp(-D_product(x,y)^2) dmu(y), with unit-RMS intrinsic factor geodesics and D_product^2=dR^2+dE^2; separably evaluated as K_R mu K_E^T','positive_measure':'A(r,e)/sqrt(deg_R(r)deg_E(e)), normalized to unit total mass; positives only, no negatives','parameter_selection':'none; radial scale is fixed by unit second moment of each factor geometry; no diffusion time, temperature, top-k, source strength, learned weights, or frozen labels','components':{'protein_global':cp0,'reaction_global':cr0,'protein_hierarchical':cpH,'reaction_hierarchical':crH},'factor_characteristic_lengths':{'protein_global':ep0,'reaction_global':er0,'protein_hierarchical':epH,'reaction_hierarchical':erH},'all_overlaps_zero':bool(((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()),'input_sha256':{'split':sha(CACHE/'marts_pair_folds.csv'),'hierarchical_geometry':sha(HG/'manifest.json')},'metrics':met.to_dict('records')};(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(met.to_string(index=False));print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2))
if __name__=='__main__':main()
