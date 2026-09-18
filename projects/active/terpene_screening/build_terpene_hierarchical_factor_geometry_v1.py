from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import save_npz
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.multiscale_geometry import hierarchical_pullback_refinement_affinity
PG=ROOT/'data/terpene_multiresolution_protein_geometry_v4'; RG=ROOT/'data/terpene_multiresolution_reaction_geometry_v2'; MOT=ROOT/'data/terpene_family_aware_motif_coordinates_v1'; OUT=ROOT/'data/terpene_hierarchical_factor_geometry_v1'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def chordal(x,a):
 x=np.asarray(x,dtype=np.float64);a=np.asarray(a,bool);norm=np.linalg.norm(x,axis=1,keepdims=True);ok=a&(norm[:,0]>1e-12);y=np.zeros_like(x);y[ok]=x[ok]/norm[ok];s=np.clip(y@y.T,-1,1);d=np.sqrt(np.maximum(2-2*s,0));d[~np.outer(ok,ok)]=np.inf;np.fill_diagonal(d,0);return d,ok
# protein: global ESM-C topology, all other observations refine only those local edges
pbase=np.load(PG/'global_esmc_chordal_distance.npy'); pa=np.load(PG/'global_esmc_available.npy').astype(bool)
prefs=[]; pmasks=[]; pnames=[]
for nm,df,af in [
 ('pocket_local_esmc','pocket_local_esmc_chordal_distance.npy','pocket_local_esmc_available.npy'),
 ('whole_3di','whole_3di_diffusion_distance.npy','whole_3di_relational_available.npy'),
 ('pocket_3di','pocket_3di_diffusion_distance.npy','pocket_3di_relational_available.npy'),
 ('pocket_ot','pocket_ot_diffusion_distance.npy','pocket_ot_relational_available.npy')]:
 prefs.append(np.load(PG/df));pmasks.append(np.load(PG/af).astype(bool));pnames.append(nm)
for nm in ['typeI_aspartate','nse_dte','dxdd','qw']:
 d,a=chordal(np.load(MOT/f'{nm}_embeddings.npy',mmap_mode='r'),np.load(MOT/f'{nm}_available.npy'));prefs.append(d);pmasks.append(a);pnames.append(nm)
pg,pinfo=hierarchical_pullback_refinement_affinity(pbase,pa,prefs,pmasks)
# reaction: DRFP topology; center W2 + explicit transition set refine edges. One missing DRFP node gets only a W2 fallback atlas.
rbase=np.load(RG/'drfp_chordal_distance.npy');ra=np.load(RG/'drfp_available.npy').astype(bool)
rw=np.load(RG/'center_transition_wasserstein_distance.npy'); rwa=np.load(RG/'center_transition_wasserstein_available.npy').astype(bool)
rj=np.load(RG/'center_token_jaccard_distance.npy'); rja=np.load(RG/'center_token_jaccard_available.npy').astype(bool)
rg,rinfo=hierarchical_pullback_refinement_affinity(rbase,ra,[rw,rj],[rwa,rja],fallback_distance=rw,fallback_available=rwa)
OUT.mkdir(parents=True,exist_ok=True);save_npz(OUT/'protein_affinity.npz',pg);save_npz(OUT/'reaction_affinity.npz',rg)
pd.read_csv(PG/'protein_ids.csv').to_csv(OUT/'protein_ids.csv',index=False);pd.read_csv(RG/'reaction_ids.csv').to_csv(OUT/'reaction_ids.csv',index=False)
manifest={'version':'terpene-hierarchical-factor-geometry-v1','definition':'global sequence / global reaction-difference coordinates define the base atlas topology; jointly observed structure, pocket, catalytic-motif, and reaction-center coordinates contribute normalized pullback energy only on existing base edges; missing refinements are neutral; one reaction lacking DRFP uses changed-atom Wasserstein only to instantiate its fallback local atlas','protein_refinements':pnames,'reaction_refinements':['center_transition_wasserstein','center_token_jaccard'],'protein_info':pinfo,'reaction_info':rinfo,'labels_used':False,'input_sha256':{'protein_manifest':sha(PG/'manifest.json'),'reaction_manifest':sha(RG/'manifest.json'),'motif_manifest':sha(MOT/'manifest.json')}}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
