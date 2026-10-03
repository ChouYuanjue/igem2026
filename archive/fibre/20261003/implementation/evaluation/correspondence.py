from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import load_npz, csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.fibre.geometry.correspondence import correspondence_state
from projects.active.fibre.runtime.base_model import rank_metrics

CACHE=ROOT/'data/terpene_marts_adaptation'
PG=ROOT/'data/terpene_multiresolution_protein_geometry_v4'
RG=ROOT/'data/terpene_multiresolution_reaction_geometry_v1'
OUT=ROOT/'results/terpene_product_correspondence_dev_v1'
BUD=(3,10,20)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def bools(s): return s.astype(str).str.lower().isin({'1','true','yes'})

def intrinsic_geodesic(w):
    w=csr_matrix(w,dtype=np.float64).maximum(csr_matrix(w,dtype=np.float64).T).tocsr()
    w.setdiag(0); w.eliminate_zeros()
    coo=w.tocoo(); upper=coo.row<coo.col
    raw_edge=np.sqrt(np.maximum(-np.log(np.clip(coo.data[upper],1e-300,1.0)),1e-12))
    ell=float(np.median(raw_edge))
    if not np.isfinite(ell) or ell<=0: raise RuntimeError('invalid characteristic length')
    edge=np.sqrt(np.maximum(-np.log(np.clip(w.data,1e-300,1.0)),1e-12))/ell
    g=csr_matrix((edge,w.indices,w.indptr),shape=w.shape)
    ncomp,_=connected_components(g,directed=False)
    if ncomp!=1: raise RuntimeError(f'factor geometry must be connected, got {ncomp} components')
    return np.asarray(shortest_path(g,directed=False,unweighted=False),dtype=np.float64),ell

def pair_array(df,ri,pi):
    return np.asarray([(ri[str(x.rhea_id)],pi[str(x.Entry)]) for x in df.itertuples(index=False)],dtype=np.int64)

def aggregate(q):
    rows=[]
    for direction,g in q.groupby('direction'):
        r={'direction':direction,'n_queries':len(g),'mrr':g.reciprocal_rank.mean(),'median_rank':g.best_positive_rank.median()}
        for k in BUD:
            r[f'hit{k}']=g[f'hit_at_{k}'].mean()
            r[f'recall{k}']=g[f'positive_recall_at_{k}'].mean()
        rows.append(r)
    return pd.DataFrame(rows)

def main():
    proteins=pd.read_csv(CACHE/'protein_entities.csv',dtype=str).fillna('')
    reactions=pd.read_csv(CACHE/'reaction_entities.csv',dtype=str).fillna('')
    pairs=pd.read_csv(CACHE/'marts_pair_folds.csv',dtype=str).fillna('')
    pairs[['protein_fold','reaction_fold']]=pairs[['protein_fold','reaction_fold']].astype(int)
    pairs['protein_seen']=bools(pairs.protein_seen); pairs['reaction_seen']=bools(pairs.reaction_seen)
    pids=proteins.protein_id.astype(str).tolist(); rids=reactions.reaction_id.astype(str).tolist()
    pi={x:i for i,x in enumerate(pids)}; ri={x:i for i,x in enumerate(rids)}

    Dr,lr=intrinsic_geodesic(load_npz(RG/'partial_pullback_affinity.npz'))
    De,le=intrinsic_geodesic(load_npz(PG/'partial_pullback_affinity.npz'))
    Dr2=Dr*Dr; De2=De*De

    rows=[]; splits=[]
    for pf in range(5):
      for rf in range(5):
        if not(pf==4 or rf==4): continue
        sid=f'p{pf}_r{rf}'
        train=pairs[pairs.protein_fold.ne(pf)&pairs.reaction_fold.ne(rf)].drop_duplicates(['rhea_id','Entry'])
        test=pairs[pairs.protein_fold.eq(pf)&pairs.reaction_fold.eq(rf)&~pairs.protein_seen&~pairs.reaction_seen].drop_duplicates(['rhea_id','Entry'])
        splits.append({'split_id':sid,'train_pairs':len(train),'test_pairs':len(test),
                       'protein_overlap':len(set(train.Entry)&set(test.Entry)),
                       'reaction_overlap':len(set(train.rhea_id)&set(test.rhea_id))})
        state=correspondence_state(Dr2,De2,pair_array(train,ri,pi)); F=state.field
        for rid,g in test.groupby('rhea_id',sort=True):
            rows.append({'split_id':sid,'direction':'reaction_to_enzyme','query_id':rid,
                         **rank_metrics(F[ri[rid]],pids,set(g.Entry.astype(str)),set(),BUD)})
        for pid,g in test.groupby('Entry',sort=True):
            rows.append({'split_id':sid,'direction':'enzyme_to_reaction','query_id':pid,
                         **rank_metrics(F[:,pi[pid]],rids,set(g.rhea_id.astype(str)),set(),BUD)})
    q=pd.DataFrame(rows); sp=pd.DataFrame(splits); met=aggregate(q)
    OUT.mkdir(parents=True,exist_ok=True)
    q.to_csv(OUT/'query_metrics.csv',index=False); sp.to_csv(OUT/'split_summary.csv',index=False); met.to_csv(OUT/'metrics.csv',index=False)
    summary={
      'version':'terpene-product-correspondence-dev-v1',
      'partition':'development_only',
      'method_status':'legacy_geometry_experiment_not_current_fibre',
      'one_sentence':'Historical product-manifold correspondence experiment retained for reproducibility; it is not the current Factorized Interaction Basis for Reaction-Enzyme method.',
      'field':'F=-Delta_Omega, Delta_Omega=D_Omega^2-d_R(.,Omega_R)^2-d_E(.,Omega_E)^2',
      'reaction_geometry':'global molecular reaction manifold: DRFP transformation + reactant molecular neighbourhood + product molecular neighbourhood; no label-dependent reaction metric',
      'protein_geometry':'partially observed multiresolution molecular-state manifold v4: global sequence for 1421/1421, plus available pocket-local sequence, whole/pocket structure, pocket OT, and family-applicable catalytic motif coordinates',
      'factor_scale':'each factor geodesic divided by its own median local edge length before forming the Cartesian product',
      'r2e_e2r':'rows and columns of the same scalar field; no direction-specific model',
      'negatives_used':False,
      'parameter_selection':'none in the correspondence operator; no score mixture, diffusion time, temperature, source strength, negative sampling, or direction-specific rule',
      'characteristic_lengths':{'reaction':lr,'protein':le},
      'all_train_test_entity_overlaps_zero':bool(((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()),
      'input_sha256':{'split':sha(CACHE/'marts_pair_folds.csv'),'reaction_geometry':sha(RG/'manifest.json'),'protein_geometry':sha(PG/'manifest.json')},
      'metrics':met.to_dict('records')
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(met.to_string(index=False)); print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2))
if __name__=='__main__': main()
