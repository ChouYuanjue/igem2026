from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from projects.active.fibre.evaluation.correspondence import (
    CACHE, PG, RG, bools, intrinsic_geodesic, pair_array,
)
from projects.active.fibre.geometry.correspondence import correspondence_state
from projects.active.fibre.geometry.levelset import numerical_level_tolerance
from projects.active.fibre.geometry.partial_relation import (
    partial_correspondence_relation,
)

ROOT=Path(__file__).resolve().parents[4]
LOCAL=ROOT/'data/terpene_catalytic_consensus_geometry_v1'
OUT=ROOT/'results/fibre_partial_relation_dev_v1'
COORDS=('pocket_local_esmc','pocket_3di','pocket_ot')
ALL_COORDS=('global_correspondence',)+COORDS


def query_relation_metrics(
    relation,
    global_defect: np.ndarray,
    ids: list[str],
    positives: set[str],
) -> dict:
    pos=np.asarray([i for i,x in enumerate(ids) if x in positives],dtype=int)
    baseline=np.argsort(global_defect,kind='stable')
    global_rank={int(i):rank+1 for rank,i in enumerate(baseline)}
    best_global=min((global_rank[int(i)] for i in pos),default=float('inf'))
    complete_pos=np.asarray([i for i in pos if relation.complete[int(i)]],dtype=int)
    complete_count=relation.complete_count
    front0_size=int(np.sum(relation.front==0))
    dominance_pairs=int(np.sum(relation.dominance))
    possible_pairs=max(1,complete_count*(complete_count-1))

    out={
        'global_best_positive_rank':float(best_global),
        'candidate_count':int(len(ids)),
        'complete_candidate_count':int(complete_count),
        'complete_candidate_fraction':float(complete_count/len(ids)),
        'positive_count':int(len(pos)),
        'complete_positive_count':int(len(complete_pos)),
        'front_count':int(relation.front_count),
        'front0_size':front0_size,
        'front0_fraction_of_complete':(
            float(front0_size/complete_count) if complete_count else float('nan')
        ),
        'dominance_density':float(dominance_pairs/possible_pairs),
        'best_positive_front':float('inf'),
        'positive_front0':False,
        'min_positive_dominator_count':float('inf'),
        'max_positive_dominated_count':0,
        'global_better_complete_count':0,
        'global_better_still_dominates':0,
        'global_better_tradeoff_incomparable':0,
        'global_better_blocked_fraction':float('nan'),
    }
    if len(complete_pos)==0:
        return out

    fronts=relation.front[complete_pos]
    out['best_positive_front']=float(np.min(fronts))
    out['positive_front0']=bool(np.any(fronts==0))
    out['min_positive_dominator_count']=int(
        min(np.sum(relation.dominance[:,int(i)]) for i in complete_pos)
    )
    out['max_positive_dominated_count']=int(
        max(np.sum(relation.dominance[int(i),:]) for i in complete_pos)
    )

    # Use the globally best complete positive as a conservative anchor. Any
    # complete candidate strictly better on the global coordinate used to sit
    # ahead of it in the old total order. The multi-coordinate relation asks
    # how many of those comparisons remain certain after catalytic coordinates.
    anchor=min(complete_pos,key=lambda i:global_rank[int(i)])
    tol=numerical_level_tolerance(np.asarray(global_defect,dtype=np.float64))
    better=[
        int(i) for i in np.flatnonzero(relation.complete)
        if global_defect[int(i)] < global_defect[int(anchor)]-tol
        and int(i) not in set(int(x) for x in pos)
    ]
    still=0; tradeoff=0
    for i in better:
        status=relation.relation(i,int(anchor))
        if status=='dominates':
            still+=1
        elif status=='incomparable_tradeoff':
            tradeoff+=1
    out['global_better_complete_count']=len(better)
    out['global_better_still_dominates']=still
    out['global_better_tradeoff_incomparable']=tradeoff
    if better:
        out['global_better_blocked_fraction']=float(tradeoff/len(better))
    return out


def summarize(q: pd.DataFrame) -> list[dict]:
    rows=[]
    for direction,g in q.groupby('direction'):
        informative=g[g.complete_positive_count.gt(0)].copy()
        blocked=informative[
            informative.global_better_complete_count.gt(0)
        ]
        rows.append({
            'direction':direction,
            'queries':int(len(g)),
            'informative_queries':int(len(informative)),
            'informative_fraction':float(len(informative)/len(g)),
            'mean_complete_candidate_fraction':float(g.complete_candidate_fraction.mean()),
            'positive_front0_fraction_among_informative':(
                float(informative.positive_front0.mean()) if len(informative) else float('nan')
            ),
            'median_best_positive_front':(
                float(informative.best_positive_front.median()) if len(informative) else float('nan')
            ),
            'median_front0_fraction_of_complete':(
                float(informative.front0_fraction_of_complete.median()) if len(informative) else float('nan')
            ),
            'median_min_positive_dominator_count':(
                float(informative.min_positive_dominator_count.median()) if len(informative) else float('nan')
            ),
            'mean_dominance_density':float(g.dominance_density.mean()),
            'queries_with_global_better_complete_candidate':int(len(blocked)),
            'mean_global_better_blocked_fraction':(
                float(blocked.global_better_blocked_fraction.mean()) if len(blocked) else float('nan')
            ),
            'queries_where_any_global_better_becomes_tradeoff':(
                int(blocked.global_better_tradeoff_incomparable.gt(0).sum())
            ),
        })
    return rows


def main() -> None:
    proteins=pd.read_csv(CACHE/'protein_entities.csv',dtype=str).fillna('')
    reactions=pd.read_csv(CACHE/'reaction_entities.csv',dtype=str).fillna('')
    pairs=pd.read_csv(CACHE/'marts_pair_folds.csv',dtype=str).fillna('')
    pairs[['protein_fold','reaction_fold']]=pairs[[
        'protein_fold','reaction_fold'
    ]].astype(int)
    pairs['protein_seen']=bools(pairs.protein_seen)
    pairs['reaction_seen']=bools(pairs.reaction_seen)

    pids=proteins.protein_id.astype(str).tolist()
    rids=reactions.reaction_id.astype(str).tolist()
    pi={x:i for i,x in enumerate(pids)}
    ri={x:i for i,x in enumerate(rids)}

    Dr,_=intrinsic_geodesic(load_npz(RG/'partial_pullback_affinity.npz'))
    De,_=intrinsic_geodesic(load_npz(PG/'partial_pullback_affinity.npz'))
    Dr2=Dr*Dr; De2=De*De

    local_rids=pd.read_csv(LOCAL/'reaction_ids.csv',dtype=str).reaction_id.astype(str).tolist()
    if local_rids != rids:
        raise RuntimeError('local reaction geometry misaligned')
    global_rows=np.load(LOCAL/'protein_global_rows.npy').astype(int)
    local_pids=pd.read_csv(LOCAL/'protein_ids.csv',dtype=str).protein_id.astype(str).tolist()
    if [pids[int(i)] for i in global_rows] != local_pids:
        raise RuntimeError('local protein mapping mismatch')
    local_pos={int(g):i for i,g in enumerate(global_rows)}
    common=np.zeros(len(pids),dtype=bool); common[global_rows]=True

    DrL,_=intrinsic_geodesic(load_npz(LOCAL/'reaction_local_affinity.npz'))
    DrL2=DrL*DrL
    protein_dist={}
    for coord in COORDS:
        d,_=intrinsic_geodesic(load_npz(LOCAL/f'protein_{coord}_affinity.npz'))
        protein_dist[coord]=d*d

    rows=[]; split_rows=[]
    for pf in range(5):
        for rf in range(5):
            if not (pf==4 or rf==4):
                continue
            split_id=f'p{pf}_r{rf}'
            train=pairs[
                pairs.protein_fold.ne(pf)&pairs.reaction_fold.ne(rf)
            ].drop_duplicates(['rhea_id','Entry'])
            test=pairs[
                pairs.protein_fold.eq(pf)&pairs.reaction_fold.eq(rf)
                & ~pairs.protein_seen & ~pairs.reaction_seen
            ].drop_duplicates(['rhea_id','Entry'])

            coarse=correspondence_state(Dr2,De2,pair_array(train,ri,pi))
            local_train=train[
                common[train.Entry.map(pi).to_numpy(dtype=int)]
            ].copy()
            lpairs=np.asarray([
                (ri[x.rhea_id],local_pos[pi[x.Entry]])
                for x in local_train.itertuples(index=False)
            ],dtype=int)
            local_states={
                coord:correspondence_state(DrL2,protein_dist[coord],lpairs)
                for coord in COORDS
            }

            split_rows.append({
                'split_id':split_id,
                'train_pairs':int(len(train)),
                'local_train_pairs':int(len(local_train)),
                'test_pairs':int(len(test)),
                'protein_overlap':len(set(train.Entry)&set(test.Entry)),
                'reaction_overlap':len(set(train.rhea_id)&set(test.rhea_id)),
            })

            for rid,g in test.groupby('rhea_id',sort=True):
                r=ri[rid]
                global_defect=np.asarray(coarse.defect[r],dtype=np.float64)
                local=np.full((len(COORDS),len(pids)),np.nan,dtype=np.float64)
                for c,coord in enumerate(COORDS):
                    local[c,global_rows]=local_states[coord].defect[r]
                defects=np.vstack([global_defect[None,:],local])
                avail=np.vstack([
                    np.ones((1,len(pids)),dtype=bool),
                    np.repeat(common[None,:],len(COORDS),axis=0),
                ])
                relation=partial_correspondence_relation(
                    defects,avail,coordinate_names=ALL_COORDS
                )
                metrics=query_relation_metrics(
                    relation,global_defect,pids,set(g.Entry.astype(str))
                )
                rows.append({
                    'split_id':split_id,
                    'direction':'reaction_to_enzyme',
                    'query_id':rid,
                    **metrics,
                })

            for pid,g in test.groupby('Entry',sort=True):
                e=pi[pid]
                global_defect=np.asarray(coarse.defect[:,e],dtype=np.float64)
                if common[e]:
                    le=local_pos[e]
                    local=np.vstack([
                        local_states[coord].defect[:,le]
                        for coord in COORDS
                    ])
                    defects=np.vstack([global_defect[None,:],local])
                    avail=np.ones_like(defects,dtype=bool)
                else:
                    defects=np.vstack([
                        global_defect[None,:],
                        np.full((len(COORDS),len(rids)),np.nan,dtype=np.float64),
                    ])
                    avail=np.vstack([
                        np.ones((1,len(rids)),dtype=bool),
                        np.zeros((len(COORDS),len(rids)),dtype=bool),
                    ])
                relation=partial_correspondence_relation(
                    defects,avail,coordinate_names=ALL_COORDS
                )
                metrics=query_relation_metrics(
                    relation,global_defect,rids,set(g.rhea_id.astype(str))
                )
                rows.append({
                    'split_id':split_id,
                    'direction':'enzyme_to_reaction',
                    'query_id':pid,
                    **metrics,
                })

    q=pd.DataFrame(rows); sp=pd.DataFrame(split_rows)
    summary_rows=summarize(q)
    OUT.mkdir(parents=True,exist_ok=True)
    q.to_csv(OUT/'query_metrics.csv',index=False)
    sp.to_csv(OUT/'split_summary.csv',index=False)
    pd.DataFrame(summary_rows).to_csv(OUT/'summary_metrics.csv',index=False)
    summary={
        'version':'fibre-partial-relation-dev-v1',
        'partition':'development_only',
        'coordinate_family':list(ALL_COORDS),
        'relation':(
            'fixed-domain Pareto minimization across global and three catalytic-pocket '
            'FIBRE defects; no weights and no lexicographic priority'
        ),
        'missing_policy':(
            'candidate missing any declared coordinate is unordered in this relation; '
            'missingness is never mapped to a defect value'
        ),
        'front_semantics':(
            'Pareto fronts are diagnostics of the partial order, not a promoted total ranking'
        ),
        'all_train_test_entity_overlaps_zero':bool(
            ((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()
        ),
        'metrics':summary_rows,
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
