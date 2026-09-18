from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import load_npz
from scipy.sparse.csgraph import shortest_path, connected_components
ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.fibre.geometry.correspondence import correspondence_state, add_positive_seed
from projects.active.fibre.runtime.base_model import rank_metrics

CACHE=ROOT/'data/terpene_marts_adaptation'
PG=ROOT/'data/terpene_multiresolution_protein_geometry_v4'
RG=ROOT/'data/terpene_multiresolution_reaction_geometry_v1'
OUT=ROOT/'results/terpene_correspondence_seed_update_audit_v1'
BUD=(3,10,20)

def bools(s): return s.astype(str).str.lower().isin({'1','true','yes'})

def normalized_geodesic(w):
    w=w.maximum(w.T).tocsr(); w.setdiag(0); w.eliminate_zeros()
    coo=w.tocoo(); mask=coo.row<coo.col
    edge=np.sqrt(np.maximum(-np.log(np.clip(coo.data[mask],1e-300,1.0)),1e-12))
    ell=float(np.median(edge))
    data=np.sqrt(np.maximum(-np.log(np.clip(w.data,1e-300,1.0)),1e-12))/ell
    from scipy.sparse import csr_matrix
    g=csr_matrix((data,w.indices,w.indptr),shape=w.shape)
    ncomp,_=connected_components(g,directed=False)
    if ncomp!=1: raise RuntimeError(f'factor graph disconnected: {ncomp}')
    return np.asarray(shortest_path(g,directed=False,unweighted=False),dtype=np.float64)

def pair_array(df,ri,pi):
    return np.asarray([(ri[str(x.rhea_id)],pi[str(x.Entry)]) for x in df.itertuples(index=False)],dtype=np.int64)

def query_metrics_for_pairs(F,test,rids,pids,ri,pi):
    rows=[]
    for rid,g in test.groupby('rhea_id',sort=True):
        rows.append({'direction':'reaction_to_enzyme','query_id':rid,
                     **rank_metrics(F[ri[rid]],pids,set(g.Entry.astype(str)),set(),BUD)})
    for pid,g in test.groupby('Entry',sort=True):
        rows.append({'direction':'enzyme_to_reaction','query_id':pid,
                     **rank_metrics(F[:,pi[pid]],rids,set(g.rhea_id.astype(str)),set(),BUD)})
    return pd.DataFrame(rows)

def main():
    proteins=pd.read_csv(CACHE/'protein_entities.csv',dtype=str).fillna('')
    reactions=pd.read_csv(CACHE/'reaction_entities.csv',dtype=str).fillna('')
    pairs=pd.read_csv(CACHE/'marts_pair_folds.csv',dtype=str).fillna('')
    pairs[['protein_fold','reaction_fold']]=pairs[['protein_fold','reaction_fold']].astype(int)
    pairs['protein_seen']=bools(pairs.protein_seen);pairs['reaction_seen']=bools(pairs.reaction_seen)
    pids=proteins.protein_id.astype(str).tolist();rids=reactions.reaction_id.astype(str).tolist()
    pi={x:i for i,x in enumerate(pids)};ri={x:i for i,x in enumerate(rids)}
    Dr=normalized_geodesic(load_npz(RG/'partial_pullback_affinity.npz')); De=normalized_geodesic(load_npz(PG/'partial_pullback_affinity.npz'))
    Dr2=Dr*Dr; De2=De*De
    parity=[]; audits=[]
    for pf in range(5):
      for rf in range(5):
        if not(pf==4 or rf==4): continue
        sid=f'p{pf}_r{rf}'
        train=pairs[pairs.protein_fold.ne(pf)&pairs.reaction_fold.ne(rf)].drop_duplicates(['rhea_id','Entry'])
        test=pairs[pairs.protein_fold.eq(pf)&pairs.reaction_fold.eq(rf)&~pairs.protein_seen&~pairs.reaction_seen].drop_duplicates(['rhea_id','Entry']).sort_values(['rhea_id','Entry'])
        if test.empty: continue
        base=correspondence_state(Dr2,De2,pair_array(train,ri,pi))
        # Real-data exact parity check uses the lexicographically first seed only; this is not a selection rule.
        first=test.iloc[0]
        seed=(ri[str(first.rhea_id)],pi[str(first.Entry)])
        inc=add_positive_seed(base,Dr2,De2,*seed)
        full=correspondence_state(Dr2,De2,np.vstack([pair_array(train,ri,pi),np.asarray(seed,dtype=int)]))
        parity.append({'split_id':sid,'seed_reaction':first.rhea_id,'seed_protein':first.Entry,
                       'max_abs_joint':float(np.max(np.abs(inc.joint_sq-full.joint_sq))),
                       'max_abs_defect':float(np.max(np.abs(inc.defect-full.defect)))})
        # Symmetric leave-one-seed audit: every held-out positive is used once as the sole new seed,
        # and evaluation excludes that seed pair itself.
        for row in test.itertuples(index=False):
            seed_pair=(ri[str(row.rhea_id)],pi[str(row.Entry)])
            remaining=test[~((test.rhea_id==row.rhea_id)&(test.Entry==row.Entry))]
            if remaining.empty: continue
            upd=add_positive_seed(base,Dr2,De2,*seed_pair)
            before=query_metrics_for_pairs(base.field,remaining,rids,pids,ri,pi)
            after=query_metrics_for_pairs(upd.field,remaining,rids,pids,ri,pi)
            key=['direction','query_id']
            j=before.merge(after,on=key,suffixes=('_before','_after'))
            for x in j.itertuples(index=False):
                audits.append({'split_id':sid,'seed_reaction':row.rhea_id,'seed_protein':row.Entry,
                               'direction':x.direction,'query_id':x.query_id,
                               'rr_delta':x.reciprocal_rank_after-x.reciprocal_rank_before,
                               'rank_delta':x.best_positive_rank_before-x.best_positive_rank_after,
                               'hit10_delta':x.hit_at_10_after-x.hit_at_10_before,
                               'hit20_delta':x.hit_at_20_after-x.hit_at_20_before})
    par=pd.DataFrame(parity); aud=pd.DataFrame(audits)
    OUT.mkdir(parents=True,exist_ok=True); par.to_csv(OUT/'incremental_parity.csv',index=False);aud.to_csv(OUT/'leave_one_seed_query_deltas.csv',index=False)
    summary={'version':'terpene-correspondence-seed-update-audit-v1',
             'geometry':'global reaction chemistry x multiresolution protein geometry',
             'incremental_update':'exact pointwise-min update of joint and marginal distance transforms; no retraining',
             'parity_max_abs_joint':float(par.max_abs_joint.max()),'parity_max_abs_defect':float(par.max_abs_defect.max()),
             'leave_one_seed':{}}
    for direction,g in aud.groupby('direction'):
        summary['leave_one_seed'][direction]={
            'comparisons':len(g),'mean_rr_delta':float(g.rr_delta.mean()),'median_rank_delta':float(g.rank_delta.median()),
            'rr_improve_tie_worse':[int((g.rr_delta>0).sum()),int((g.rr_delta==0).sum()),int((g.rr_delta<0).sum())],
            'mean_hit10_delta':float(g.hit10_delta.mean()),'mean_hit20_delta':float(g.hit20_delta.mean())
        }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
