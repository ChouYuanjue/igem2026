from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, eye, load_npz
from scipy.sparse.linalg import expm_multiply

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from projects.active.terpene_screening.multiscale_geometry import (
    partial_observation_pullback_affinity,
    diffusion_conformal_affinity,
)
from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics

CACHE=ROOT/'data/terpene_marts_adaptation'
PROTEIN_GEOM=ROOT/'data/terpene_multiresolution_protein_geometry_v4'
REACTION_GEOM=ROOT/'data/terpene_multiresolution_reaction_geometry_v2'
OUT=ROOT/'results/terpene_multiresolution_product_heat_dev_v3'
BUDGETS=(3,10,20)


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def boolean(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({'1','true','yes'})


def symmetric_heat_generator(w: csr_matrix) -> csr_matrix:
    w=csr_matrix(w,dtype=np.float64).maximum(csr_matrix(w,dtype=np.float64).T).tocsr()
    w.setdiag(0.0); w.eliminate_zeros()
    degree=np.asarray(w.sum(axis=1)).reshape(-1)
    invsqrt=np.zeros_like(degree); positive=degree>0; invsqrt[positive]=degree[positive]**-0.5
    s=(diags(invsqrt)@w@diags(invsqrt)).tocsr()
    q=s.tolil(); idx=np.flatnonzero(positive); q[idx,idx]=-1.0
    return q.tocsr()


def normalized_positive_measure(relation: np.ndarray) -> np.ndarray:
    a=np.asarray(relation,dtype=np.float64)
    dr=a.sum(axis=1); de=a.sum(axis=0)
    sr=np.zeros_like(dr); se=np.zeros_like(de)
    sr[dr>0]=dr[dr>0]**-0.5; se[de>0]=de[de>0]**-0.5
    out=sr[:,None]*a*se[None,:]
    total=float(out.sum())
    return out/total if total>0 else out


def heat_joint(relation: np.ndarray, reaction_graph: csr_matrix, protein_graph: csr_matrix) -> np.ndarray:
    qr=symmetric_heat_generator(reaction_graph)
    qe=symmetric_heat_generator(protein_graph)
    measure=normalized_positive_measure(relation)
    left=expm_multiply(qr,measure)
    joint=expm_multiply(qe,left.T).T
    joint=np.asarray(joint,dtype=np.float64)
    joint[np.abs(joint)<1e-15]=0.0
    if np.min(joint)<-1e-10: raise RuntimeError(f'heat extension produced negative mass {joint.min()}')
    return np.maximum(joint,0.0)


def one_view_graph(distance_path: Path, available_path: Path) -> csr_matrix:
    d=np.load(distance_path).astype(np.float64)
    a=np.load(available_path).astype(bool)
    g,_=partial_observation_pullback_affinity([d],[a])
    g,_=diffusion_conformal_affinity(g)
    return g


def aggregate(q: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (method,direction),g in q.groupby(['method','direction'],sort=True):
        row={'method':method,'direction':direction,'n_queries':len(g),'mrr':float(g.reciprocal_rank.mean()),'median_rank':float(g.best_positive_rank.median())}
        for k in BUDGETS:
            row[f'hit{k}']=float(g[f'hit_at_{k}'].mean()); row[f'recall{k}']=float(g[f'positive_recall_at_{k}'].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    proteins=pd.read_csv(CACHE/'protein_entities.csv',dtype=str).fillna('')
    reactions=pd.read_csv(CACHE/'reaction_entities.csv',dtype=str).fillna('')
    pairs=pd.read_csv(CACHE/'marts_pair_folds.csv',dtype=str).fillna('')
    pairs[['protein_fold','reaction_fold']]=pairs[['protein_fold','reaction_fold']].astype(int)
    pairs['protein_seen']=boolean(pairs.protein_seen); pairs['reaction_seen']=boolean(pairs.reaction_seen)
    pids=proteins.protein_id.astype(str).tolist(); rids=reactions.reaction_id.astype(str).tolist()
    p2i={x:i for i,x in enumerate(pids)}; r2i={x:i for i,x in enumerate(rids)}
    # Freeze ID alignment before any label is read into a relation matrix.
    pgids=pd.read_csv(PROTEIN_GEOM/'protein_ids.csv',dtype=str).fillna(''); pgids['row']=pd.to_numeric(pgids.row).astype(int); pgids=pgids.sort_values('row')
    rgids=pd.read_csv(REACTION_GEOM/'reaction_ids.csv',dtype=str).fillna(''); rgids['row']=pd.to_numeric(rgids.row).astype(int); rgids=rgids.sort_values('row')
    if pgids.protein_id.tolist()!=pids or rgids.reaction_id.tolist()!=rids: raise RuntimeError('geometry/entity order mismatch')

    protein_global=one_view_graph(PROTEIN_GEOM/'global_esmc_chordal_distance.npy',PROTEIN_GEOM/'global_esmc_available.npy')
    reaction_drfp=one_view_graph(REACTION_GEOM/'drfp_chordal_distance.npy',REACTION_GEOM/'drfp_available.npy')
    protein_v4=load_npz(PROTEIN_GEOM/'diffusion_conformal_affinity.npz').tocsr()
    reaction_v1=load_npz(REACTION_GEOM/'diffusion_conformal_affinity.npz').tocsr()
    methods={
      'global_only':(reaction_drfp,protein_global),
      'protein_multires_v4':(reaction_drfp,protein_v4),
      'reaction_center_v2':(reaction_v1,protein_global),
      'all_information_v2':(reaction_v1,protein_v4),
    }
    query_rows=[]; split_rows=[]
    for pf in range(5):
      for rf in range(5):
        if not (pf==4 or rf==4): continue
        sid=f'p{pf}_r{rf}'
        train=pairs[pairs.protein_fold.ne(pf)&pairs.reaction_fold.ne(rf)].drop_duplicates(['rhea_id','Entry'])
        test=pairs[pairs.protein_fold.eq(pf)&pairs.reaction_fold.eq(rf)&~pairs.protein_seen&~pairs.reaction_seen].drop_duplicates(['rhea_id','Entry'])
        # Strong leakage checks on cluster identities and exact entities.
        train_p=set(train.Entry); train_r=set(train.rhea_id); test_p=set(test.Entry); test_r=set(test.rhea_id)
        split_rows.append({'split_id':sid,'train_pairs':len(train),'test_pairs':len(test),'test_proteins':len(test_p),'test_reactions':len(test_r),
                           'protein_entity_overlap':len(train_p&test_p),'reaction_entity_overlap':len(train_r&test_r)})
        if test.empty: continue
        relation=np.zeros((len(rids),len(pids)),dtype=np.float64)
        for x in train.itertuples(index=False): relation[r2i[str(x.rhea_id)],p2i[str(x.Entry)]]=1.0
        for method,(rg,pg) in methods.items():
            joint=heat_joint(relation,rg,pg)
            for rid,g in test.groupby('rhea_id',sort=True):
                positives=set(g.Entry.astype(str)); metrics=rank_metrics(joint[r2i[str(rid)]],pids,positives,set(),BUDGETS)
                query_rows.append({'split_id':sid,'method':method,'direction':'reaction_to_enzyme','query_id':str(rid),**metrics})
            for pid,g in test.groupby('Entry',sort=True):
                positives=set(g.rhea_id.astype(str)); metrics=rank_metrics(joint[:,p2i[str(pid)]],rids,positives,set(),BUDGETS)
                query_rows.append({'split_id':sid,'method':method,'direction':'enzyme_to_reaction','query_id':str(pid),**metrics})
    q=pd.DataFrame(query_rows); splits=pd.DataFrame(split_rows); metrics=aggregate(q)
    OUT.mkdir(parents=True,exist_ok=True); q.to_csv(OUT/'query_metrics.csv',index=False); splits.to_csv(OUT/'split_summary.csv',index=False); metrics.to_csv(OUT/'metrics.csv',index=False)
    summary={
      'version':'terpene-multiresolution-product-heat-dev-v3','partition':'development_only','development_rule':'protein_fold==4 or reaction_fold==4',
      'test_rule':'same cell and protein_seen==false and reaction_seen==false','methods':list(methods),
      'operator':'J = exp(S_R-I) mu_norm exp(S_E-I) at unit continuous-time diffusion, S=D^-1/2 W D^-1/2 and mu_norm=A/sqrt(d_r d_e)',
      'parameter_selection':'none; 2x2 factor ablation of DRFP vs reaction-center-v2 and global-sequence vs protein-v4 under the identical symmetric normalized-Laplacian heat operator; no grid search, learned score fusion, negatives, or frozen partition labels',
      'completed_cells':int(len(splits)),'nonempty_cells':int((splits.test_pairs>0).sum()),
      'all_train_test_entity_overlaps_zero':bool(((splits.protein_entity_overlap==0)&(splits.reaction_entity_overlap==0)).all()),
      'input_sha256':{'split':sha(CACHE/'marts_pair_folds.csv'),'protein_geometry':sha(PROTEIN_GEOM/'manifest.json'),'reaction_geometry':sha(REACTION_GEOM/'manifest.json')},
      'metrics':metrics.to_dict(orient='records'),
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(metrics.to_string(index=False)); print('\n',json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2))

if __name__=='__main__': main()
