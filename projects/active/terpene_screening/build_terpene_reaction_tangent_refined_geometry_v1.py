from __future__ import annotations
import hashlib,json,math,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix,save_npz
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.multiscale_geometry import _self_tuning_scale,diffusion_conformal_affinity
SRC=ROOT/'data/terpene_multiresolution_reaction_geometry_v2';OUT=ROOT/'data/terpene_reaction_tangent_refined_geometry_v1'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ids=pd.read_csv(SRC/'reaction_ids.csv',dtype=str).fillna(''); n=len(ids); k=int(math.ceil(math.sqrt(n)))
 specs=[('drfp','drfp_chordal_distance.npy','drfp_available.npy'),('center_transition_wasserstein','center_transition_wasserstein_distance.npy','center_transition_wasserstein_available.npy'),('center_token_jaccard','center_token_jaccard_distance.npy','center_token_jaccard_available.npy')]
 ds=[];av=[];sig=[]
 for _,df,af in specs:
  d=np.load(SRC/df).astype(np.float64);a=np.load(SRC/af).astype(bool);ds.append(d);av.append(a);sig.append(_self_tuning_scale(d,a,epsilon=1e-8))
 # The atlas topology is defined only by global DRFP chemistry.
 base_energy=np.full((n,n),np.inf,dtype=np.float64); den=sig[0][:,None]*sig[0][None,:]; valid=av[0][:,None]&av[0][None,:]&np.isfinite(den)&(den>1e-8);base_energy[valid]=ds[0][valid]**2/den[valid];np.fill_diagonal(base_energy,np.inf)
 rows=[];cols=[]
 for i in range(n):
  finite=np.flatnonzero(np.isfinite(base_energy[i])); kk=min(k,len(finite)); pick=np.argpartition(base_energy[i,finite],kk-1)[:kk]; sel=finite[pick];rows.extend([i]*len(sel));cols.extend(sel.tolist())
 topology=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n)).maximum(csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n)).T).tocsr();topology.setdiag(0);topology.eliminate_zeros()
 # On each globally local edge, refine tangent energy with all jointly observed local measurements.
 rr,cc=topology.nonzero(); keep=rr<cc; rr=rr[keep];cc=cc[keep]; vals=[]; counts=[]
 for i,j in zip(rr,cc):
  es=[]
  for d,a,s in zip(ds,av,sig):
   if not(a[i] and a[j]):continue
   denom=s[i]*s[j]
   if np.isfinite(denom) and denom>1e-8:es.append(float(d[i,j]**2/denom))
  if not es: continue
  vals.append(float(np.exp(-np.mean(es))));counts.append(len(es))
 r=np.concatenate([rr,cc]);c=np.concatenate([cc,rr]);v=np.asarray(vals+vals,dtype=np.float32);g=csr_matrix((v,(r,c)),shape=(n,n));g.setdiag(0);g.eliminate_zeros();gc,info=diffusion_conformal_affinity(g)
 OUT.mkdir(parents=True,exist_ok=True);save_npz(OUT/'tangent_pullback_affinity.npz',g);save_npz(OUT/'diffusion_conformal_affinity.npz',gc);ids.to_csv(OUT/'reaction_ids.csv',index=False)
 deg=np.asarray(gc.getnnz(axis=1)); manifest={'version':'terpene-reaction-tangent-refined-geometry-v1','reaction_count':n,'atlas_topology':'sqrt(N) self-tuned DRFP neighbourhood only','tangent_metric':'on fixed global-atlas edges, arithmetic mean of dimensionless pullback energies from DRFP, exact changed-atom W2, and explicit transition-token Jaccard when jointly observed','interpretation':'reaction-center observations refine the local metric tensor but cannot create edges between globally distant reactions','graph_k':k,'topology_edges':int(topology.nnz//2),'refined_edges':int(g.nnz//2),'observed_view_count_distribution':{str(int(x)):int(np.sum(np.asarray(counts)==x)) for x in sorted(set(counts))},'isolated_after_conformal':int(np.sum(deg==0)),'diffusion_conformal_info':info,'labels_used':False,'input_sha256':{'source_manifest':sha(SRC/'manifest.json')}};(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
