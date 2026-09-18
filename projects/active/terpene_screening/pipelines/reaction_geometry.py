from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ot
from scipy.spatial.distance import cdist
from scipy.sparse import save_npz

ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from projects.active.terpene_screening.geometry.multiscale import (
    partial_observation_pullback_affinity,
    diffusion_conformal_affinity,
)

CACHE=ROOT/'data/terpene_marts_adaptation'
CENTER=ROOT/'data/terpene_reaction_center_observations_v1'
OUT=ROOT/'data/terpene_multiresolution_reaction_geometry_v2'


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def chordal_distance(x: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    x=np.asarray(x,dtype=np.float64); norm=np.linalg.norm(x,axis=1,keepdims=True); a=norm[:,0]>1e-12
    xn=np.zeros_like(x); xn[a]=x[a]/norm[a]
    s=np.clip(xn@xn.T,-1,1); d=np.sqrt(np.maximum(2-2*s,0.0)); d[~(a[:,None]&a[None,:])]=np.inf; np.fill_diagonal(d,0)
    return d,a


def load_token_sets(path: Path, expected_ids: list[str]) -> list[set[str]]:
    out=[]; ids=[]
    for line in path.read_text().splitlines():
        x=json.loads(line); ids.append(str(x['reaction_id'])); out.append(set(map(str,x['tokens'])))
    if ids!=expected_ids: raise RuntimeError('transition token order mismatch')
    return out


def jaccard_distance(sets: list[set[str]]) -> np.ndarray:
    n=len(sets); d=np.zeros((n,n),dtype=np.float64)
    for i in range(n):
        for j in range(i):
            u=sets[i]|sets[j]; inter=sets[i]&sets[j]
            value=1.0-(len(inter)/len(u) if u else 1.0)
            d[i,j]=d[j,i]=value
    return d


def wasserstein_transition_distance(states: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n=len(states); d=np.zeros((n,n),dtype=np.float64)
    # Each coordinate in the atom-state construction is already dimensionless/bounded.
    # Equal masses express an empirical measure over changed atom-map IDs; no atom-type weights are introduced.
    for i in range(n):
        x=np.asarray(states[i,mask[i]],dtype=np.float64)
        a=np.full(len(x),1.0/len(x),dtype=np.float64)
        for j in range(i):
            y=np.asarray(states[j,mask[j]],dtype=np.float64)
            b=np.full(len(y),1.0/len(y),dtype=np.float64)
            cost=cdist(x,y,metric='sqeuclidean')
            value=float(np.sqrt(max(float(ot.emd2(a,b,cost)),0.0)))
            d[i,j]=d[j,i]=value
    return d


def main() -> None:
    rx=pd.read_csv(CACHE/'reaction_entities.csv',dtype=str).fillna(''); ids=rx.reaction_id.astype(str).tolist(); n=len(ids)
    ce=pd.read_csv(CENTER/'reaction_ids.csv',dtype=str).fillna(''); ce['row']=pd.to_numeric(ce.row).astype(int); ce=ce.sort_values('row')
    if ce.reaction_id.tolist()!=ids: raise RuntimeError('center/entity order mismatch')
    base=np.load(CACHE/'reaction_features.npy',mmap_mode='r')
    drfp_d,drfp_a=chordal_distance(np.asarray(base[:,:2048],dtype=np.float32))
    states=np.load(CENTER/'transition_atom_states.npy').astype(np.float32); mask=np.load(CENTER/'transition_atom_mask.npy').astype(bool)
    if states.shape[0]!=n or mask.shape[:2]!=states.shape[:2]: raise RuntimeError('center measure shape mismatch')
    wd=wasserstein_transition_distance(states,mask)
    tokens=load_token_sets(CENTER/'transition_tokens.jsonl',ids); jd=jaccard_distance(tokens)
    all_a=np.ones(n,dtype=bool)
    graph,pull=partial_observation_pullback_affinity([drfp_d,wd,jd],[drfp_a,all_a,all_a])
    conformal,conf=diffusion_conformal_affinity(graph)
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'row':np.arange(n),'reaction_id':ids}).to_csv(OUT/'reaction_ids.csv',index=False)
    np.save(OUT/'drfp_chordal_distance.npy',drfp_d.astype(np.float32)); np.save(OUT/'drfp_available.npy',drfp_a)
    np.save(OUT/'center_transition_wasserstein_distance.npy',wd.astype(np.float32)); np.save(OUT/'center_transition_wasserstein_available.npy',all_a)
    np.save(OUT/'center_token_jaccard_distance.npy',jd.astype(np.float32)); np.save(OUT/'center_token_jaccard_available.npy',all_a)
    save_npz(OUT/'partial_pullback_affinity.npz',graph); save_npz(OUT/'diffusion_conformal_affinity.npz',conformal)
    count=drfp_a.astype(int)+2
    pd.DataFrame({'reaction_id':ids,'observed_view_count':count,'drfp':drfp_a,'center_transition_wasserstein':True,'center_token_jaccard':True}).to_csv(OUT/'coverage.csv',index=False)
    off=~np.eye(n,dtype=bool)
    manifest={
      'version':'terpene-multiresolution-reaction-geometry-v2','reaction_count':n,
      'views':['drfp_chordal','center_transition_wasserstein','center_token_jaccard'],
      'view_policy':'global reaction-difference and two catalytic-local reaction-center measurements enter the same self-tuned partial pullback; no inter-view coefficients are learned or manually chosen',
      'wasserstein':'exact balanced W2 over equal-mass empirical changed-atom transition states using squared Euclidean ground cost on already dimensionless bounded atom-state coordinates',
      'token_geometry':'Jaccard distance on unhashed explicit changed-bond and changed-atom transition-token sets',
      'drfp_available':int(drfp_a.sum()),'center_available':n,'isolated_after_conformal':int(np.sum(np.asarray(conformal.sum(axis=1)).reshape(-1)<=0)),
      'pullback_info':pull,'diffusion_conformal_info':conf,
      'distance_diagnostics':{
        'wasserstein_median':float(np.median(wd[off])),'wasserstein_p90':float(np.quantile(wd[off],.9)),'wasserstein_max':float(np.max(wd[off])),
        'jaccard_median':float(np.median(jd[off])),'jaccard_p90':float(np.quantile(jd[off],.9)),'jaccard_max':float(np.max(jd[off])),
      },
      'labels_used':False,
      'input_sha256':{'reaction_entities':sha(CACHE/'reaction_entities.csv'),'reaction_features':sha(CACHE/'reaction_features.npy'),'center_manifest':sha(CENTER/'manifest.json')}
    }
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
