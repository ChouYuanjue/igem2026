from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from projects.active.terpene_screening.evaluate_zero_shot_retrieval_cold import (
    reaction_features as molecule_side_features,
    best_match_similarity,
)
from projects.active.terpene_screening.multiscale_geometry import (
    one_step_diffusion_distance,
    partial_observation_pullback_affinity,
    diffusion_conformal_affinity,
)

DEFAULT_CACHE=ROOT/'data/terpene_marts_adaptation'
DEFAULT_OUTPUT=ROOT/'data/terpene_multiresolution_reaction_geometry_v1'


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def chordal_distance(x: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    x=np.asarray(x,dtype=np.float64)
    norms=np.linalg.norm(x,axis=1,keepdims=True)
    available=norms[:,0]>1e-12
    xn=np.zeros_like(x); xn[available]=x[available]/norms[available]
    sim=np.clip(xn@xn.T,-1.0,1.0)
    d=np.sqrt(np.maximum(2.0-2.0*sim,0.0))
    d[~(available[:,None]&available[None,:])]=np.inf
    np.fill_diagonal(d,0.0)
    return d,available


def side_similarity(features: list[dict[str,object]], key: str) -> tuple[np.ndarray,np.ndarray]:
    n=len(features); sim=np.zeros((n,n),dtype=np.float64)
    available=np.asarray([len(f[key])>0 for f in features],dtype=bool)
    for i in range(n):
        if available[i]: sim[i,i]=1.0
        for j in range(i):
            if not (available[i] and available[j]): continue
            # Symmetrise set-to-set best-match evidence before it defines an observation graph.
            a=float(best_match_similarity(features[i][key],features[j][key]))
            b=float(best_match_similarity(features[j][key],features[i][key]))
            value=0.5*(a+b)
            sim[i,j]=sim[j,i]=value
    return sim,available


def diffusion_distance_from_similarity(sim: np.ndarray, available: np.ndarray) -> tuple[np.ndarray,np.ndarray,dict[str,object]]:
    x=np.where(np.outer(available,available),np.maximum(sim,0.0),0.0)
    np.fill_diagonal(x,0.0)
    graph=csr_matrix(x)
    degree=np.asarray((graph>0).sum(axis=1)).reshape(-1)
    relational=available&(degree>0)
    d=one_step_diffusion_distance(graph)
    d[~(relational[:,None]&relational[None,:])]=np.inf
    np.fill_diagonal(d,0.0)
    return d,relational,{
      'raw_available':int(available.sum()),'relational_available':int(relational.sum()),
      'isolated_observed_nodes':int(np.sum(available&~relational)),
      'positive_undirected_edges':int(graph.nnz//2),
    }


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,default=DEFAULT_CACHE); ap.add_argument('--output',type=Path,default=DEFAULT_OUTPUT); a=ap.parse_args()
    reactions=pd.read_csv(a.cache/'reaction_entities.csv',dtype=str).fillna('')
    drfp_cat=np.load(a.cache/'reaction_features.npy',mmap_mode='r').astype(np.float32)
    if drfp_cat.shape[0]!=len(reactions) or drfp_cat.shape[1]<2048: raise RuntimeError('reaction feature matrix mismatch')
    # Only the DRFP transformation block enters this view. Historical categorical one-hot bonuses remain annotations.
    drfp=np.asarray(drfp_cat[:,:2048],dtype=np.float32)
    drfp_d,drfp_avail=chordal_distance(drfp)
    chemistry=[molecule_side_features(s) for s in reactions.reaction_smiles.astype(str)]
    react_sim,react_avail=side_similarity(chemistry,'reactant_fps')
    prod_sim,prod_avail=side_similarity(chemistry,'product_fps')
    react_d,react_rel,react_info=diffusion_distance_from_similarity(react_sim,react_avail)
    prod_d,prod_rel,prod_info=diffusion_distance_from_similarity(prod_sim,prod_avail)
    affinity,pullback=partial_observation_pullback_affinity([drfp_d,react_d,prod_d],[drfp_avail,react_rel,prod_rel])
    conformal,conformal_info=diffusion_conformal_affinity(affinity)
    out=a.output; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'row':np.arange(len(reactions)),'reaction_id':reactions.reaction_id.astype(str)}).to_csv(out/'reaction_ids.csv',index=False)
    np.save(out/'drfp_chordal_distance.npy',drfp_d.astype(np.float32)); np.save(out/'drfp_available.npy',drfp_avail)
    np.save(out/'reactant_diffusion_distance.npy',react_d.astype(np.float32)); np.save(out/'reactant_available.npy',react_rel)
    np.save(out/'product_diffusion_distance.npy',prod_d.astype(np.float32)); np.save(out/'product_available.npy',prod_rel)
    save_npz(out/'partial_pullback_affinity.npz',affinity); save_npz(out/'diffusion_conformal_affinity.npz',conformal)
    count=drfp_avail.astype(int)+react_rel.astype(int)+prod_rel.astype(int)
    pd.DataFrame({'reaction_id':reactions.reaction_id.astype(str),'observed_view_count':count,'drfp':drfp_avail,'reactant':react_rel,'product':prod_rel}).to_csv(out/'coverage.csv',index=False)
    manifest={
      'version':'terpene-multiresolution-reaction-geometry-v1','reaction_count':int(len(reactions)),
      'views':['drfp_chordal','reactant_diffusion','product_diffusion'],
      'view_policy':'DRFP transformation, reactant molecular neighbourhood, and product molecular neighbourhood are separate self-tuned measurement maps; no historical 0.4/0.4/0.1/0.1 score mixture is used.',
      'categorical_policy':'precursor_class and product_skeleton_class remain annotations and do not contribute a hard equality bonus to g_R.',
      'reactant_info':react_info,'product_info':prod_info,'pullback_info':pullback,'diffusion_conformal_info':conformal_info,
      'view_available_counts':[int(drfp_avail.sum()),int(react_rel.sum()),int(prod_rel.sum())],
      'view_count_distribution':{str(int(k)):int(v) for k,v in pd.Series(count).value_counts().sort_index().items()},
      'isolated_after_conformal':int(np.sum(np.asarray(conformal.sum(axis=1)).reshape(-1)<=0)),
      'labels_used':False,
      'input_sha256':{'reaction_entities':sha(a.cache/'reaction_entities.csv'),'reaction_features':sha(a.cache/'reaction_features.npy')}
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
