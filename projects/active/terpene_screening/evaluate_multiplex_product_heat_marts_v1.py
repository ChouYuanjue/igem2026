from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix,diags
from scipy.sparse.linalg import expm_multiply
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.multiscale_geometry import partial_observation_pullback_affinity,diffusion_conformal_affinity
from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics
CACHE=ROOT/'data/terpene_marts_adaptation'; PG=ROOT/'data/terpene_multiresolution_protein_geometry_v4'; RG=ROOT/'data/terpene_multiresolution_reaction_geometry_v2'; MOT=ROOT/'data/terpene_family_aware_motif_coordinates_v1'; OUT=ROOT/'results/terpene_multiplex_product_heat_dev_v1'; BUD=(3,10,20)
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def one_graph(d,a):
 g,_=partial_observation_pullback_affinity([np.asarray(d)],[np.asarray(a,bool)]); g,_=diffusion_conformal_affinity(g); return g.tocsr()
def chordal(x,a):
 x=np.asarray(x,dtype=np.float64); a=np.asarray(a,bool); n=np.linalg.norm(x,axis=1,keepdims=True); y=np.zeros_like(x); ok=a&(n[:,0]>1e-12); y[ok]=x[ok]/n[ok]; s=np.clip(y@y.T,-1,1); d=np.sqrt(np.maximum(2-2*s,0)); d[~np.outer(ok,ok)]=np.inf; np.fill_diagonal(d,0); return d,ok
def layer_generator(w):
 w=csr_matrix(w,dtype=np.float64).maximum(csr_matrix(w,dtype=np.float64).T).tocsr(); w.setdiag(0); w.eliminate_zeros(); deg=np.asarray(w.sum(1)).ravel(); inv=np.zeros_like(deg); ok=deg>0; inv[ok]=1/deg[ok]; p=(diags(inv)@w).tocsr(); q=p.copy().tolil(); idx=np.flatnonzero(ok); q[idx,idx]=-1.0; return q.tocsr(),ok
def multiplex_generator(graphs):
 qs=[]; active=[]
 for g in graphs:
  q,a=layer_generator(g); qs.append(q); active.append(a.astype(np.float64))
 count=np.sum(active,axis=0); inv=np.zeros_like(count); inv[count>0]=1/count[count>0]
 # Row-wise equal probability over the layers actually observed at that point.
 q=sum(qs[1:],qs[0].copy()) if len(qs)>1 else qs[0].copy()
 return (diags(inv)@q).tocsr(),count
def normalize_measure(a):
 a=np.asarray(a,dtype=np.float64); dr=a.sum(1); de=a.sum(0); sr=np.zeros_like(dr); se=np.zeros_like(de); sr[dr>0]=dr[dr>0]**-.5; se[de>0]=de[de>0]**-.5; z=sr[:,None]*a*se[None,:]; return z/z.sum() if z.sum()>0 else z
def bools(s): return s.astype(str).str.lower().isin({'1','true','yes'})
def aggregate(q):
 rows=[]
 for (m,d),g in q.groupby(['method','direction']):
  r={'method':m,'direction':d,'n_queries':len(g),'mrr':g.reciprocal_rank.mean(),'median_rank':g.best_positive_rank.median()}
  for k in BUD:r[f'hit{k}']=g[f'hit_at_{k}'].mean();r[f'recall{k}']=g[f'positive_recall_at_{k}'].mean()
  rows.append(r)
 return pd.DataFrame(rows)
def main():
 proteins=pd.read_csv(CACHE/'protein_entities.csv',dtype=str).fillna(''); reactions=pd.read_csv(CACHE/'reaction_entities.csv',dtype=str).fillna(''); pairs=pd.read_csv(CACHE/'marts_pair_folds.csv',dtype=str).fillna(''); pairs[['protein_fold','reaction_fold']]=pairs[['protein_fold','reaction_fold']].astype(int); pairs['protein_seen']=bools(pairs.protein_seen); pairs['reaction_seen']=bools(pairs.reaction_seen)
 pids=proteins.protein_id.astype(str).tolist(); rids=reactions.reaction_id.astype(str).tolist(); pi={x:i for i,x in enumerate(pids)}; ri={x:i for i,x in enumerate(rids)}
 # Reaction layers: DRFP, center W2, center token Jaccard.
 r_specs=[('drfp_chordal_distance.npy','drfp_available.npy'),('center_transition_wasserstein_distance.npy','center_transition_wasserstein_available.npy'),('center_token_jaccard_distance.npy','center_token_jaccard_available.npy')]
 r_layers=[one_graph(np.load(RG/d),np.load(RG/a)) for d,a in r_specs]
 # Protein layers: global, pocket-local, three structure views, four family-aware catalytic motif coordinates.
 p_specs=[('global_esmc_chordal_distance.npy','global_esmc_available.npy'),('pocket_local_esmc_chordal_distance.npy','pocket_local_esmc_available.npy'),('whole_3di_diffusion_distance.npy','whole_3di_relational_available.npy'),('pocket_3di_diffusion_distance.npy','pocket_3di_relational_available.npy'),('pocket_ot_diffusion_distance.npy','pocket_ot_relational_available.npy')]
 p_layers=[one_graph(np.load(PG/d),np.load(PG/a)) for d,a in p_specs]
 for nm in ['typeI_aspartate','nse_dte','dxdd','qw']:
  x=np.load(MOT/f'{nm}_embeddings.npy',mmap_mode='r'); a=np.load(MOT/f'{nm}_available.npy'); d,aa=chordal(x,a); p_layers.append(one_graph(d,aa))
 # Controls reuse exactly one base layer; multires variants use factor-specific layer families.
 qr0,_=multiplex_generator([r_layers[0]]); qe0,_=multiplex_generator([p_layers[0]]); qrM,rc=multiplex_generator(r_layers); qeM,pc=multiplex_generator(p_layers)
 methods={'global_only':(qr0,qe0),'protein_multiplex':(qr0,qeM),'reaction_multiplex':(qrM,qe0),'all_multiplex':(qrM,qeM)}
 rows=[]; splits=[]
 for pf in range(5):
  for rf in range(5):
   if not (pf==4 or rf==4):continue
   sid=f'p{pf}_r{rf}'; train=pairs[pairs.protein_fold.ne(pf)&pairs.reaction_fold.ne(rf)].drop_duplicates(['rhea_id','Entry']); test=pairs[pairs.protein_fold.eq(pf)&pairs.reaction_fold.eq(rf)&~pairs.protein_seen&~pairs.reaction_seen].drop_duplicates(['rhea_id','Entry']); splits.append({'split_id':sid,'train_pairs':len(train),'test_pairs':len(test),'protein_overlap':len(set(train.Entry)&set(test.Entry)),'reaction_overlap':len(set(train.rhea_id)&set(test.rhea_id))})
   A=np.zeros((len(rids),len(pids)),dtype=np.float64)
   for x in train.itertuples(index=False):A[ri[str(x.rhea_id)],pi[str(x.Entry)]]=1
   mu=normalize_measure(A)
   for method,(qr,qe) in methods.items():
    J=expm_multiply(qe,expm_multiply(qr,mu).T).T; J=np.maximum(np.asarray(J),0)
    for rid,g in test.groupby('rhea_id',sort=True): rows.append({'split_id':sid,'method':method,'direction':'reaction_to_enzyme','query_id':rid,**rank_metrics(J[ri[rid]],pids,set(g.Entry.astype(str)),set(),BUD)})
    for pid,g in test.groupby('Entry',sort=True): rows.append({'split_id':sid,'method':method,'direction':'enzyme_to_reaction','query_id':pid,**rank_metrics(J[:,pi[pid]],rids,set(g.rhea_id.astype(str)),set(),BUD)})
 q=pd.DataFrame(rows); sp=pd.DataFrame(splits); met=aggregate(q); OUT.mkdir(parents=True,exist_ok=True); q.to_csv(OUT/'query_metrics.csv',index=False); sp.to_csv(OUT/'split_summary.csv',index=False); met.to_csv(OUT/'metrics.csv',index=False)
 summary={'version':'terpene-multiplex-product-heat-dev-v1','partition':'development_only','operator':'each molecular measurement map retains its own self-tuned graph; at each factor point the continuous-time generator is the equal mixture of only locally active layer generators; product generator is Q_R+Q_E; degree-normalized positive empirical measure; unit intrinsic time','parameter_selection':'none','reaction_layers':3,'protein_layers':9,'reaction_active_layer_count_distribution':{str(int(k)):int(v) for k,v in pd.Series(rc).value_counts().sort_index().items()},'protein_active_layer_count_distribution':{str(int(k)):int(v) for k,v in pd.Series(pc).value_counts().sort_index().items()},'all_overlaps_zero':bool(((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()),'input_sha256':{'split':sha(CACHE/'marts_pair_folds.csv'),'protein_geometry':sha(PG/'manifest.json'),'reaction_geometry':sha(RG/'manifest.json'),'motif_manifest':sha(MOT/'manifest.json')},'metrics':met.to_dict('records')}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(met.to_string(index=False)); print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2))
if __name__=='__main__':main()
