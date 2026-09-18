from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import load_npz,csr_matrix,save_npz
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.multiscale_geometry import _self_tuning_scale,diffusion_conformal_affinity
GLOBAL=ROOT/'data/terpene_multiresolution_reaction_geometry_v1'; CENTER=ROOT/'data/terpene_multiresolution_reaction_geometry_v2'; OUT=ROOT/'data/terpene_reaction_atlas_refined_geometry_v2'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ids=pd.read_csv(GLOBAL/'reaction_ids.csv',dtype=str).fillna(''); topology=load_npz(GLOBAL/'partial_pullback_affinity.npz').tocsr(); n=len(ids)
 specs=[(GLOBAL/'drfp_chordal_distance.npy',GLOBAL/'drfp_available.npy','drfp_global'),(GLOBAL/'reactant_diffusion_distance.npy',GLOBAL/'reactant_available.npy','reactant_global'),(GLOBAL/'product_diffusion_distance.npy',GLOBAL/'product_available.npy','product_global'),(CENTER/'center_transition_wasserstein_distance.npy',CENTER/'center_transition_wasserstein_available.npy','center_transition_w2'),(CENTER/'center_token_jaccard_distance.npy',CENTER/'center_token_jaccard_available.npy','center_transition_token')]
 ds=[]; av=[]; sig=[]
 for df,af,_ in specs:
  d=np.load(df).astype(np.float64);a=np.load(af).astype(bool);ds.append(d);av.append(a);sig.append(_self_tuning_scale(d,a,epsilon=1e-8))
 rr,cc=topology.nonzero(); keep=rr<cc;rr=rr[keep];cc=cc[keep]; vals=[];counts=[]
 for i,j in zip(rr,cc):
  es=[]
  for d,a,s in zip(ds,av,sig):
   if not(a[i] and a[j]):continue
   den=s[i]*s[j]
   if np.isfinite(den) and den>1e-8: es.append(float(d[i,j]**2/den))
  if not es: raise RuntimeError('global atlas edge without any observable tangent coordinate')
  vals.append(float(np.exp(-np.mean(es))));counts.append(len(es))
 r=np.concatenate([rr,cc]);c=np.concatenate([cc,rr]);v=np.asarray(vals+vals,dtype=np.float32);g=csr_matrix((v,(r,c)),shape=(n,n));g.setdiag(0);g.eliminate_zeros();gc,info=diffusion_conformal_affinity(g)
 OUT.mkdir(parents=True,exist_ok=True);save_npz(OUT/'atlas_refined_affinity.npz',g);save_npz(OUT/'diffusion_conformal_affinity.npz',gc);ids.to_csv(OUT/'reaction_ids.csv',index=False)
 deg=np.asarray(gc.getnnz(axis=1)); manifest={'version':'terpene-reaction-atlas-refined-geometry-v2','reaction_count':n,'atlas_topology':'fixed by global-scale DRFP + reactant-neighbourhood + product-neighbourhood pullback geometry only','tangent_measurements':[x[2] for x in specs],'tangent_metric':'on fixed global-atlas edges, mean of jointly observed self-tuned dimensionless pullback energies across the five molecular measurements','interpretation':'global reaction chemistry defines local chart membership; catalytic-center observations refine tangent edge lengths but cannot create long-range shortcuts','topology_edges':int(topology.nnz//2),'refined_edges':int(g.nnz//2),'observed_view_count_distribution':{str(int(x)):int(np.sum(np.asarray(counts)==x)) for x in sorted(set(counts))},'isolated_after_conformal':int(np.sum(deg==0)),'diffusion_conformal_info':info,'labels_used':False,'input_sha256':{'global_manifest':sha(GLOBAL/'manifest.json'),'center_manifest':sha(CENTER/'manifest.json')}};(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
