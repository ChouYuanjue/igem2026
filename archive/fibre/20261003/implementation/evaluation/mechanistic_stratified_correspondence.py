from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from projects.active.fibre.evaluation.correspondence import (
    CACHE,
    PG,
    RG,
    bools,
    intrinsic_geodesic,
    pair_array,
)
from projects.active.fibre.geometry.correspondence import correspondence_state
from projects.active.fibre.geometry.stratified import (
    consensus_stratified_resolution,
    mechanistic_chart_resolution,
)

ROOT=Path(__file__).resolve().parents[4]
CATALYTIC=ROOT/"data/terpene_catalytic_consensus_geometry_v1"
MECH=ROOT/"data/terpene_mechanistic_chart_geometry_v1"
OUT=ROOT/"results/fibre_mechanistic_stratified_dev_v1"
POCKET_COORDS=("pocket_local_esmc","pocket_3di","pocket_ot")
MECH_COORDS=("typeI_aspartate","nse_dte","dxdd","qw")


def main() -> None:
    proteins=pd.read_csv(CACHE/"protein_entities.csv",dtype=str).fillna("")
    reactions=pd.read_csv(CACHE/"reaction_entities.csv",dtype=str).fillna("")
    pairs=pd.read_csv(CACHE/"marts_pair_folds.csv",dtype=str).fillna("")
    pairs[["protein_fold","reaction_fold"]]=pairs[
        ["protein_fold","reaction_fold"]
    ].astype(int)
    pairs["protein_seen"]=bools(pairs.protein_seen)
    pairs["reaction_seen"]=bools(pairs.reaction_seen)

    pids=proteins.protein_id.astype(str).tolist()
    rids=reactions.reaction_id.astype(str).tolist()
    pi={x:i for i,x in enumerate(pids)}
    ri={x:i for i,x in enumerate(rids)}

    Dr,_=intrinsic_geodesic(load_npz(RG/"partial_pullback_affinity.npz"))
    De,_=intrinsic_geodesic(load_npz(PG/"partial_pullback_affinity.npz"))
    Dr2=Dr*Dr
    De2=De*De

    local_rids=pd.read_csv(CATALYTIC/"reaction_ids.csv",dtype=str).reaction_id.astype(str).tolist()
    if local_rids!=rids:
        raise RuntimeError("catalytic reaction order mismatch")
    DrL=np.load(CATALYTIC/"reaction_local_geodesic.npy",mmap_mode="r").astype(np.float64)
    DrL2=DrL*DrL

    pocket_rows=np.load(CATALYTIC/"protein_global_rows.npy").astype(int)
    pocket_pids=pd.read_csv(CATALYTIC/"protein_ids.csv",dtype=str).protein_id.astype(str).tolist()
    if [pids[int(i)] for i in pocket_rows]!=pocket_pids:
        raise RuntimeError("catalytic protein mapping mismatch")
    pocket_pos={int(g):i for i,g in enumerate(pocket_rows)}
    pocket_common=np.zeros(len(pids),dtype=bool)
    pocket_common[pocket_rows]=True
    pocket_dist={}
    for coord in POCKET_COORDS:
        d=np.load(CATALYTIC/f"protein_{coord}_geodesic.npy",mmap_mode="r").astype(np.float64)
        pocket_dist[coord]=d*d

    mech_ids=pd.read_csv(MECH/"protein_ids.csv",dtype=str).protein_id.astype(str).tolist()
    if mech_ids!=pids:
        raise RuntimeError("mechanistic protein order mismatch")
    mech_rows={}
    mech_pos={}
    mech_dist={}
    mech_applicable=[]
    mech_available=[]
    for coord in MECH_COORDS:
        rows=np.load(MECH/f"protein_{coord}_global_rows.npy").astype(int)
        geo=np.load(MECH/f"protein_{coord}_geodesic.npy",mmap_mode="r").astype(np.float64)
        if geo.shape!=(len(rows),len(rows)):
            raise RuntimeError(f"{coord}: mechanistic geodesic shape mismatch")
        mech_rows[coord]=rows
        mech_pos[coord]={int(g):i for i,g in enumerate(rows)}
        mech_dist[coord]=geo*geo
        mech_applicable.append(np.load(MECH/f"protein_{coord}_applicable.npy").astype(bool))
        mech_available.append(np.load(MECH/f"protein_{coord}_available.npy").astype(bool))
    mech_applicable=np.vstack(mech_applicable)
    mech_available=np.vstack(mech_available)

    rows_out=[]
    split_out=[]
    chart_counter=Counter()

    for pf in range(5):
        for rf in range(5):
            if not (pf==4 or rf==4):
                continue
            sid=f"p{pf}_r{rf}"
            train=pairs[
                pairs.protein_fold.ne(pf)&pairs.reaction_fold.ne(rf)
            ].drop_duplicates(["rhea_id","Entry"])
            test=pairs[
                pairs.protein_fold.eq(pf)&pairs.reaction_fold.eq(rf)
                & ~pairs.protein_seen & ~pairs.reaction_seen
            ].drop_duplicates(["rhea_id","Entry"])

            coarse=correspondence_state(Dr2,De2,pair_array(train,ri,pi))

            pocket_train=train[
                pocket_common[train.Entry.map(pi).to_numpy(dtype=int)]
            ].copy()
            pocket_pairs=np.asarray([
                (ri[x.rhea_id],pocket_pos[pi[x.Entry]])
                for x in pocket_train.itertuples(index=False)
            ],dtype=int)
            pocket_states={
                coord:correspondence_state(DrL2,pocket_dist[coord],pocket_pairs)
                for coord in POCKET_COORDS
            }

            mech_states={}
            for coord in MECH_COORDS:
                pos=mech_pos[coord]
                local_train=train[
                    train.Entry.map(pi).map(lambda g:int(g) in pos)
                ].copy()
                if local_train.empty:
                    mech_states[coord]=None
                    continue
                lpairs=np.asarray([
                    (ri[x.rhea_id],pos[pi[x.Entry]])
                    for x in local_train.itertuples(index=False)
                ],dtype=int)
                mech_states[coord]=correspondence_state(
                    DrL2,mech_dist[coord],lpairs
                )

            split_out.append({
                "split_id":sid,
                "train_pairs":int(len(train)),
                "test_pairs":int(len(test)),
                "protein_overlap":len(set(train.Entry)&set(test.Entry)),
                "reaction_overlap":len(set(train.rhea_id)&set(test.rhea_id)),
                **{
                    f"{coord}_train_pair_count":int(
                        train.Entry.map(pi).map(
                            lambda g:int(g) in mech_pos[coord]
                        ).sum()
                    )
                    for coord in MECH_COORDS
                },
            })

            for rid,g in test.groupby("rhea_id",sort=True):
                r=ri[rid]
                coarse_defect=coarse.defect[r]
                pocket_local=np.full(
                    (len(POCKET_COORDS),len(pids)),np.nan,dtype=np.float64
                )
                for c,coord in enumerate(POCKET_COORDS):
                    pocket_local[c,pocket_rows]=pocket_states[coord].defect[r]
                pocket_avail=np.repeat(
                    pocket_common[None,:],len(POCKET_COORDS),axis=0
                )
                catalytic=consensus_stratified_resolution(
                    coarse_defect,pocket_local,pocket_avail
                )

                mdef=np.full(
                    (len(MECH_COORDS),len(pids)),np.nan,dtype=np.float64
                )
                mavail=mech_available.copy()
                for c,coord in enumerate(MECH_COORDS):
                    state=mech_states[coord]
                    if state is None:
                        mavail[c]=False
                        continue
                    mdef[c,mech_rows[coord]]=state.defect[r]
                mech=mechanistic_chart_resolution(
                    catalytic.coarse_level,
                    catalytic.catalytic_stratum,
                    catalytic.observed_coarse_levels,
                    mdef,
                    mech_applicable,
                    mavail,
                )
                positives=np.asarray([pi[x] for x in g.Entry.astype(str)],dtype=int)
                positive_resolved=int(np.sum(mech.mechanistic_stratum[positives]>=0))
                positive_front0=int(np.sum(mech.mechanistic_stratum[positives]==0))
                for mask in mech.mechanistic_chart[mech.mechanistic_chart>0]:
                    chart_counter[("reaction_to_enzyme",int(mask))]+=1
                rows_out.append({
                    "split_id":sid,
                    "direction":"reaction_to_enzyme",
                    "query_id":rid,
                    "positive_count":int(len(positives)),
                    "catalytic_observed_level_count":int(len(catalytic.observed_coarse_levels)),
                    "catalytic_refined_level_count":int(len(catalytic.refined_coarse_levels)),
                    "mechanistic_observed_parent_chart_count":int(len(mech.observed_parent_charts)),
                    "mechanistic_refined_parent_chart_count":int(len(mech.refined_parent_charts)),
                    "mechanistic_refined_candidate_count":int(mech.refined_candidate_count),
                    "positive_mechanistic_resolved_count":positive_resolved,
                    "positive_mechanistic_front0_count":positive_front0,
                })

            for pid,g in test.groupby("Entry",sort=True):
                e=pi[pid]
                coarse_defect=coarse.defect[:,e]
                if pocket_common[e]:
                    le=pocket_pos[e]
                    pocket_local=np.vstack([
                        pocket_states[coord].defect[:,le]
                        for coord in POCKET_COORDS
                    ])
                    pocket_avail=np.ones_like(pocket_local,dtype=bool)
                    catalytic=consensus_stratified_resolution(
                        coarse_defect,pocket_local,pocket_avail
                    )
                else:
                    # The intermediate catalytic layer was not observed, so the
                    # third layer must not bypass it.
                    catalytic=consensus_stratified_resolution(
                        coarse_defect,
                        np.full((len(POCKET_COORDS),len(rids)),np.nan),
                        np.zeros((len(POCKET_COORDS),len(rids)),dtype=bool),
                    )

                mdef=np.full(
                    (len(MECH_COORDS),len(rids)),np.nan,dtype=np.float64
                )
                q_app=mech_applicable[:,e]
                q_avail=mech_available[:,e]
                app=np.repeat(q_app[:,None],len(rids),axis=1)
                avail=np.repeat(q_avail[:,None],len(rids),axis=1)
                for c,coord in enumerate(MECH_COORDS):
                    state=mech_states[coord]
                    pos=mech_pos[coord]
                    if state is None or e not in pos:
                        avail[c]=False
                        continue
                    mdef[c]=state.defect[:,pos[e]]
                mech=mechanistic_chart_resolution(
                    catalytic.coarse_level,
                    catalytic.catalytic_stratum,
                    catalytic.observed_coarse_levels,
                    mdef,app,avail,
                )
                positives=np.asarray([ri[x] for x in g.rhea_id.astype(str)],dtype=int)
                positive_resolved=int(np.sum(mech.mechanistic_stratum[positives]>=0))
                positive_front0=int(np.sum(mech.mechanistic_stratum[positives]==0))
                for mask in mech.mechanistic_chart[mech.mechanistic_chart>0]:
                    chart_counter[("enzyme_to_reaction",int(mask))]+=1
                rows_out.append({
                    "split_id":sid,
                    "direction":"enzyme_to_reaction",
                    "query_id":pid,
                    "positive_count":int(len(positives)),
                    "catalytic_observed_level_count":int(len(catalytic.observed_coarse_levels)),
                    "catalytic_refined_level_count":int(len(catalytic.refined_coarse_levels)),
                    "mechanistic_observed_parent_chart_count":int(len(mech.observed_parent_charts)),
                    "mechanistic_refined_parent_chart_count":int(len(mech.refined_parent_charts)),
                    "mechanistic_refined_candidate_count":int(mech.refined_candidate_count),
                    "positive_mechanistic_resolved_count":positive_resolved,
                    "positive_mechanistic_front0_count":positive_front0,
                })

    q=pd.DataFrame(rows_out)
    sp=pd.DataFrame(split_out)
    aggregate=[]
    for direction,g in q.groupby("direction"):
        aggregate.append({
            "direction":direction,
            "queries":int(len(g)),
            "query_fraction_with_observed_mechanistic_chart":float(
                (g.mechanistic_observed_parent_chart_count>0).mean()
            ),
            "query_fraction_with_refined_mechanistic_chart":float(
                (g.mechanistic_refined_parent_chart_count>0).mean()
            ),
            "mean_observed_parent_charts":float(
                g.mechanistic_observed_parent_chart_count.mean()
            ),
            "mean_refined_parent_charts":float(
                g.mechanistic_refined_parent_chart_count.mean()
            ),
            "mean_refined_candidate_count":float(
                g.mechanistic_refined_candidate_count.mean()
            ),
            "positive_mechanistic_resolution_fraction":float(
                g.positive_mechanistic_resolved_count.sum()
                / max(g.positive_count.sum(),1)
            ),
            "positive_front0_fraction_among_resolved":float(
                g.positive_mechanistic_front0_count.sum()
                / max(g.positive_mechanistic_resolved_count.sum(),1)
            ),
        })
    met=pd.DataFrame(aggregate)

    chart_rows=[
        {"direction":d,"chart_mask":mask,"candidate_occurrences":count}
        for (d,mask),count in sorted(chart_counter.items())
    ]
    OUT.mkdir(parents=True,exist_ok=True)
    q.to_csv(OUT/"query_resolution.csv",index=False)
    sp.to_csv(OUT/"split_summary.csv",index=False)
    met.to_csv(OUT/"metrics.csv",index=False)
    pd.DataFrame(chart_rows).to_csv(OUT/"chart_counts.csv",index=False)
    summary={
        "version":"fibre-mechanistic-stratified-dev-v1",
        "partition":"development_only",
        "canonical_total_rank_changed":False,
        "coarse_operator":"zero-temperature FIBRE correspondence defect",
        "catalytic_operator":"same defect independently on reaction-center x each pocket coordinate; Pareto relation",
        "mechanistic_operator":"same defect independently on reaction-center x each family-aware motif coordinate; Pareto relation inside observed catalytic parent and mechanism chart",
        "mechanistic_coordinates":list(MECH_COORDS),
        "chart_definition":"family-applicability bit mask; distinct charts are incomparable",
        "missing_policy":"applicable but unobserved motif leaves comparison unresolved; non-applicable motif is outside the chart",
        "labels_used_to_build_geometry":False,
        "labels_used_only_for_resolution_audit":True,
        "all_train_test_entity_overlaps_zero":bool(
            ((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()
        ),
        "metrics":aggregate,
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(met.to_string(index=False))
    print(json.dumps({k:v for k,v in summary.items() if k!="metrics"},indent=2))


if __name__=="__main__":
    main()
