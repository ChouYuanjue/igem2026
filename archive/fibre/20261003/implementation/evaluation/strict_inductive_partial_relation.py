from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from projects.active.fibre.evaluation.partial_correspondence_relation import (
    ALL_COORDS, query_relation_metrics, summarize,
)
from projects.active.fibre.evaluation.strict_inductive import (
    CACHE, bools, build_protein_factor_inputs, build_reaction_factor_inputs,
    materialize_views, factor_all_to_reference, score_r2e, score_e2r,
)
from projects.active.fibre.evaluation.strict_inductive_consensus import (
    COORDS, build_pocket_raw, build_coordinate_atlases,
)
from projects.active.fibre.evaluation.strict_inductive_stratified import (
    build_local_reaction_inputs, materialize_named_views,
    factor_all_to_reference_information,
)
from projects.active.fibre.geometry.correspondence import (
    reaction_to_protein_section_from_support_distances,
    protein_to_reaction_section_from_support_distances,
)
from projects.active.fibre.geometry.partial_relation import (
    partial_correspondence_relation,
)

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'results/fibre_partial_relation_strict_inductive_v1'


def main() -> None:
    t0=time.time()
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
    pfold={str(k):int(v) for k,v in pairs[[
        'Entry','protein_fold'
    ]].drop_duplicates().itertuples(index=False)}
    rfold={str(k):int(v) for k,v in pairs[[
        'rhea_id','reaction_fold'
    ]].drop_duplicates().itertuples(index=False)}

    coarse_p_raw=build_protein_factor_inputs()
    coarse_r_raw=build_reaction_factor_inputs(reactions)
    pocket_raw=build_pocket_raw()
    local_r_raw=build_local_reaction_inputs()
    pocket_common=np.logical_and.reduce([
        np.asarray(v[3],dtype=bool) for v in pocket_raw
    ])

    coarse_p_cache={}; pocket_cache={}
    for pf in range(5):
        pref=np.asarray([
            i for i,pid in enumerate(pids) if pfold.get(pid)!=pf
        ],dtype=int)
        coarse_p_cache[pf]=(
            pref,
            factor_all_to_reference(pref,materialize_views(coarse_p_raw,pref)),
        )
        initial=np.asarray([
            i for i,pid in enumerate(pids)
            if pfold.get(pid)!=pf and pocket_common[i]
        ],dtype=int)
        pocket_cache[pf]=build_coordinate_atlases(pocket_raw,initial)
        print('protein fold',pf,'coarse',len(pref),'pocket',len(pocket_cache[pf][0]),flush=True)

    coarse_r_cache={}; local_r_cache={}
    for rf in range(5):
        rref=np.asarray([
            i for i,rid in enumerate(rids) if rfold.get(rid)!=rf
        ],dtype=int)
        coarse_r_cache[rf]=(
            rref,
            factor_all_to_reference(rref,materialize_views(coarse_r_raw,rref)),
        )
        lr=factor_all_to_reference_information(
            rref,materialize_named_views(local_r_raw,rref)
        )[0]
        local_r_cache[rf]=(rref,lr)
        print('reaction fold',rf,'reference',len(rref),flush=True)

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

        pref,(Dp,pinfo)=coarse_p_cache[pf]
        rref,(Dr,rinfo)=coarse_r_cache[rf]
        pmap={int(g):i for i,g in enumerate(pref)}
        rmap={int(g):i for i,g in enumerate(rref)}
        Dp2=np.asarray(Dp,dtype=np.float64)**2
        Dr2=np.asarray(Dr,dtype=np.float64)**2

        lpref,coord_dist,coord_qavail=pocket_cache[pf]
        lpmap={int(g):i for i,g in enumerate(lpref)}
        lrref,lr=local_r_cache[rf]
        lrmap={int(g):i for i,g in enumerate(lrref)}

        local_train=train[
            train.Entry.map(pi).map(lambda g:int(g) in lpmap)
        ].copy()
        if local_train.empty:
            raise RuntimeError(f'{split_id}: no local positive support')

        rs_global=sorted({ri[str(x)] for x in local_train.rhea_id})
        es_global=sorted({pi[str(x)] for x in local_train.Entry})
        rs_pos={g:i for i,g in enumerate(rs_global)}
        es_pos={g:i for i,g in enumerate(es_global)}
        support_pairs=np.asarray([
            (rs_pos[ri[str(x.rhea_id)]],es_pos[pi[str(x.Entry)]])
            for x in local_train.itertuples(index=False)
        ],dtype=int)
        rs_ref=np.asarray([lrmap[g] for g in rs_global],dtype=int)
        es_ref=np.asarray([lpmap[g] for g in es_global],dtype=int)

        reaction_available=np.all(np.isfinite(lr[:,rs_ref]),axis=1)
        protein_available=np.logical_and.reduce([
            np.asarray(coord_qavail[c],dtype=bool)
            & np.all(np.isfinite(coord_dist[c][:,es_ref]),axis=1)
            for c in COORDS
        ])

        for rid,g in test.groupby('rhea_id',sort=True):
            qg=ri[rid]
            coarse_score=score_r2e(
                qg,train,Dr2,Dp2,rmap,pmap,ri,pi,len(pids)
            )
            global_defect=-np.asarray(coarse_score,dtype=np.float64)
            local=np.full((len(COORDS),len(pids)),np.nan,dtype=np.float64)
            local_avail=np.zeros((len(COORDS),len(pids)),dtype=bool)
            if reaction_available[qg]:
                cand=np.flatnonzero(protein_available)
                for c,coord in enumerate(COORDS):
                    sec=reaction_to_protein_section_from_support_distances(
                        np.square(lr[qg,rs_ref]),
                        np.square(coord_dist[coord][np.ix_(cand,es_ref)]),
                        support_pairs,query_index=qg,candidate_indices=cand,
                    )
                    local[c,cand]=sec.defect
                local_avail[:,cand]=True
            defects=np.vstack([global_defect[None,:],local])
            avail=np.vstack([
                np.ones((1,len(pids)),dtype=bool),local_avail
            ])
            relation=partial_correspondence_relation(
                defects,avail,coordinate_names=ALL_COORDS
            )
            rows.append({
                'split_id':split_id,
                'direction':'reaction_to_enzyme',
                'query_id':rid,
                **query_relation_metrics(
                    relation,global_defect,pids,set(g.Entry.astype(str))
                ),
            })

        for pid,g in test.groupby('Entry',sort=True):
            qg=pi[pid]
            coarse_score=score_e2r(
                qg,train,Dr2,Dp2,rmap,pmap,ri,pi,len(rids)
            )
            global_defect=-np.asarray(coarse_score,dtype=np.float64)
            local=np.full((len(COORDS),len(rids)),np.nan,dtype=np.float64)
            local_avail=np.zeros((len(COORDS),len(rids)),dtype=bool)
            if protein_available[qg]:
                cand=np.flatnonzero(reaction_available)
                for c,coord in enumerate(COORDS):
                    sec=protein_to_reaction_section_from_support_distances(
                        np.square(coord_dist[coord][qg,es_ref]),
                        np.square(lr[np.ix_(cand,rs_ref)]),
                        support_pairs,query_index=qg,candidate_indices=cand,
                    )
                    local[c,cand]=sec.defect
                local_avail[:,cand]=True
            defects=np.vstack([global_defect[None,:],local])
            avail=np.vstack([
                np.ones((1,len(rids)),dtype=bool),local_avail
            ])
            relation=partial_correspondence_relation(
                defects,avail,coordinate_names=ALL_COORDS
            )
            rows.append({
                'split_id':split_id,
                'direction':'enzyme_to_reaction',
                'query_id':pid,
                **query_relation_metrics(
                    relation,global_defect,rids,set(g.rhea_id.astype(str))
                ),
            })

        split_rows.append({
            'split_id':split_id,
            'train_pairs':int(len(train)),
            'local_train_pairs':int(len(local_train)),
            'test_pairs':int(len(test)),
            'protein_overlap':len(set(train.Entry)&set(test.Entry)),
            'reaction_overlap':len(set(train.rhea_id)&set(test.rhea_id)),
            'pocket_reference_count':int(len(lpref)),
            'reaction_local_reference_count':int(len(lrref)),
        })
        print(split_id,'done',flush=True)

    q=pd.DataFrame(rows); sp=pd.DataFrame(split_rows)
    summary_rows=summarize(q)
    OUT.mkdir(parents=True,exist_ok=True)
    q.to_csv(OUT/'query_metrics.csv',index=False)
    sp.to_csv(OUT/'split_summary.csv',index=False)
    pd.DataFrame(summary_rows).to_csv(OUT/'summary_metrics.csv',index=False)
    summary={
        'version':'fibre-partial-relation-strict-inductive-v1',
        'protocol':(
            'rebuild global, reaction-center and each pocket coordinate atlas '
            'without held-out factor entities; attach held-out entities only '
            'through query-to-reference observations; then evaluate the fixed-domain '
            'Pareto relation without total-order promotion'
        ),
        'coordinate_family':list(ALL_COORDS),
        'missing_policy':(
            'missing any coordinate removes that candidate from this relation; '
            'it is not penalized or assigned a surrogate value'
        ),
        'all_train_test_entity_overlaps_zero':bool(
            ((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()
        ),
        'metrics':summary_rows,
        'elapsed_seconds':time.time()-t0,
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
