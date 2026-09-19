from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.fibre.evaluation.strict_inductive import (
    CACHE,
    MOTIF,
    bools,
    chordal_full,
    build_protein_factor_inputs,
    build_reaction_factor_inputs,
    materialize_views,
    factor_all_to_reference,
    score_r2e,
    score_e2r,
)
from projects.active.fibre.evaluation.strict_inductive_stratified import (
    build_local_reaction_inputs,
    materialize_named_views,
    factor_all_to_reference_information,
)
from projects.active.fibre.evaluation.strict_inductive_consensus import (
    COORDS as POCKET_COORDS,
    build_pocket_raw,
    build_coordinate_atlases,
)
from projects.active.fibre.geometry.correspondence import (
    reaction_to_protein_section_from_support_distances,
    protein_to_reaction_section_from_support_distances,
)
from projects.active.fibre.geometry.stratified import (
    consensus_stratified_resolution,
    mechanistic_chart_resolution,
)

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/"results/fibre_mechanistic_stratified_strict_inductive_v1"
DEV=ROOT/"results/fibre_mechanistic_stratified_dev_v1/query_resolution.csv"
MECH_COORDS=("typeI_aspartate","nse_dte","dxdd","qw")


def as_bool(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.lower().isin({"1","true","yes"}).to_numpy(bool)


def build_mechanistic_raw():
    audit=pd.read_csv(MOTIF/"audit.csv",dtype=str).fillna("")
    ids=pd.read_csv(MOTIF/"protein_ids.csv",dtype=str).fillna("")
    if ids.protein_id.astype(str).tolist()!=audit.protein_id.astype(str).tolist():
        audit=audit.set_index("protein_id").loc[
            ids.protein_id.astype(str)
        ].reset_index()
    raw={}
    applicable={}
    for name in MECH_COORDS:
        emb=np.load(MOTIF/f"{name}_embeddings.npy",mmap_mode="r")
        available=np.load(MOTIF/f"{name}_available.npy").astype(bool)
        app=as_bool(audit[f"{name}_applicable"])
        if np.any(available & ~app):
            raise RuntimeError(f"{name}: observed motif outside applicable family chart")
        raw[name]=(name,"coordinate",chordal_full(emb,available),available)
        applicable[name]=app
    return raw,applicable


def support_contract(
    train: pd.DataFrame,
    *,
    pi: dict[str,int],
    ri: dict[str,int],
    protein_ref: np.ndarray,
    reaction_ref: np.ndarray,
    protein_distance: np.ndarray,
    protein_query_available: np.ndarray,
    reaction_distance: np.ndarray,
):
    pmap={int(g):i for i,g in enumerate(protein_ref)}
    rmap={int(g):i for i,g in enumerate(reaction_ref)}
    local_train=train[
        train.Entry.map(pi).map(lambda g:int(g) in pmap)
    ].copy()
    if local_train.empty:
        return None

    rs_global=sorted({ri[str(x)] for x in local_train.rhea_id})
    es_global=sorted({pi[str(x)] for x in local_train.Entry})
    rs_pos={g:i for i,g in enumerate(rs_global)}
    es_pos={g:i for i,g in enumerate(es_global)}
    pairs=np.asarray([
        (rs_pos[ri[str(x.rhea_id)]],es_pos[pi[str(x.Entry)]])
        for x in local_train.itertuples(index=False)
    ],dtype=int)
    rs_ref=np.asarray([rmap[g] for g in rs_global],dtype=int)
    es_ref=np.asarray([pmap[g] for g in es_global],dtype=int)
    reaction_available=np.all(
        np.isfinite(reaction_distance[:,rs_ref]),axis=1
    )
    protein_available=(
        np.asarray(protein_query_available,dtype=bool)
        & np.all(np.isfinite(protein_distance[:,es_ref]),axis=1)
    )
    return {
        "train_pairs":int(len(local_train)),
        "support_pairs":pairs,
        "reaction_support_ref":rs_ref,
        "protein_support_ref":es_ref,
        "reaction_available":reaction_available,
        "protein_available":protein_available,
    }


def empty_catalytic(coarse_defect: np.ndarray, n_local: int):
    return consensus_stratified_resolution(
        coarse_defect,
        np.full((n_local,len(coarse_defect)),np.nan,dtype=np.float64),
        np.zeros((n_local,len(coarse_defect)),dtype=bool),
    )


def stability_summary(strict: pd.DataFrame) -> dict[str,dict[str,float|int]]:
    if not DEV.is_file():
        return {}
    dev=pd.read_csv(DEV,dtype={"query_id":str})
    keys=["split_id","direction","query_id"]
    a=dev[keys+["mechanistic_refined_parent_chart_count"]].rename(
        columns={"mechanistic_refined_parent_chart_count":"dev_count"}
    )
    b=strict[keys+["mechanistic_refined_parent_chart_count"]].rename(
        columns={"mechanistic_refined_parent_chart_count":"strict_count"}
    )
    z=a.merge(b,on=keys,validate="one_to_one")
    out={}
    for direction,g in z.groupby("direction"):
        d=g.dev_count.to_numpy(int)>0
        s=g.strict_count.to_numpy(int)>0
        both=int(np.sum(d&s))
        union=int(np.sum(d|s))
        out[direction]={
            "queries":int(len(g)),
            "dev_refined_queries":int(d.sum()),
            "strict_refined_queries":int(s.sum()),
            "refined_in_both":both,
            "dev_only":int(np.sum(d&~s)),
            "strict_only":int(np.sum(~d&s)),
            "neither":int(np.sum(~d&~s)),
            "refined_query_jaccard":float(both/union) if union else 1.0,
        }
    return out


def main() -> None:
    t0=time.time()
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
    pfold={str(k):int(v) for k,v in pairs[
        ["Entry","protein_fold"]
    ].drop_duplicates().itertuples(index=False)}
    rfold={str(k):int(v) for k,v in pairs[
        ["rhea_id","reaction_fold"]
    ].drop_duplicates().itertuples(index=False)}

    print("materializing raw coarse, pocket, reaction-local and motif views",flush=True)
    coarse_p_raw=build_protein_factor_inputs()
    coarse_r_raw=build_reaction_factor_inputs(reactions)
    pocket_raw=build_pocket_raw()
    local_r_raw=build_local_reaction_inputs()
    mech_raw,mech_app=build_mechanistic_raw()

    pocket_common=np.logical_and.reduce([
        np.asarray(v[3],dtype=bool) for v in pocket_raw
    ])

    protein_cache={}
    for pf in range(5):
        pref=np.asarray([
            i for i,pid in enumerate(pids) if pfold.get(pid)!=pf
        ],dtype=int)
        coarse=factor_all_to_reference(
            pref,materialize_views(coarse_p_raw,pref)
        )
        initial=np.asarray([
            i for i,pid in enumerate(pids)
            if pfold.get(pid)!=pf and pocket_common[i]
        ],dtype=int)
        pocket=build_coordinate_atlases(pocket_raw,initial)

        mech={}
        for coord in MECH_COORDS:
            available=np.asarray(mech_raw[coord][3],dtype=bool)
            mref=np.asarray([
                i for i,pid in enumerate(pids)
                if pfold.get(pid)!=pf and available[i]
            ],dtype=int)
            if len(mref)<2:
                mech[coord]=None
                continue
            try:
                ref,atlas,qavail=build_coordinate_atlases(
                    [mech_raw[coord]],mref
                )
            except (ValueError,RuntimeError):
                mech[coord]=None
                continue
            mech[coord]=(ref,atlas[coord],qavail[coord])
        protein_cache[pf]=(pref,coarse,pocket,mech)
        print(
            "protein fold",pf,
            "coarse",len(pref),
            "pocket",len(pocket[0]),
            "motif",{k:(0 if v is None else len(v[0])) for k,v in mech.items()},
            flush=True,
        )

    reaction_cache={}
    for rf in range(5):
        rref=np.asarray([
            i for i,rid in enumerate(rids) if rfold.get(rid)!=rf
        ],dtype=int)
        coarse=factor_all_to_reference(
            rref,materialize_views(coarse_r_raw,rref)
        )
        local=factor_all_to_reference_information(
            rref,materialize_named_views(local_r_raw,rref)
        )[0]
        reaction_cache[rf]=(rref,coarse,local)
        print("reaction fold",rf,"reference",len(rref),flush=True)

    rows=[]
    splits=[]
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

        pref,(Dp,pinfo),pocket,mech=protein_cache[pf]
        rref,(Dr,rinfo),lr=reaction_cache[rf]
        pmap={int(g):i for i,g in enumerate(pref)}
        rmap={int(g):i for i,g in enumerate(rref)}
        Dp2=np.asarray(Dp,dtype=np.float64)**2
        Dr2=np.asarray(Dr,dtype=np.float64)**2

        lpref,pocket_dist,pocket_qavail=pocket
        pocket_support=support_contract(
            train,pi=pi,ri=ri,
            protein_ref=lpref,
            reaction_ref=rref,
            protein_distance=pocket_dist[POCKET_COORDS[0]],
            protein_query_available=np.logical_and.reduce([
                np.asarray(pocket_qavail[c],dtype=bool)
                for c in POCKET_COORDS
            ]),
            reaction_distance=lr,
        )
        if pocket_support is None:
            raise RuntimeError(f"{sid}: no strict catalytic support")
        # All pocket coordinates share the same complete reference support.
        pocket_protein_available=np.logical_and.reduce([
            np.asarray(pocket_qavail[c],dtype=bool)
            & np.all(
                np.isfinite(
                    pocket_dist[c][
                        :,pocket_support["protein_support_ref"]
                    ]
                ),
                axis=1,
            )
            for c in POCKET_COORDS
        ])
        pocket_reaction_available=np.asarray(
            pocket_support["reaction_available"],dtype=bool
        )

        mech_support={}
        for coord in MECH_COORDS:
            cached=mech[coord]
            if cached is None:
                mech_support[coord]=None
                continue
            mref,mdist,mqavail=cached
            mech_support[coord]=support_contract(
                train,pi=pi,ri=ri,
                protein_ref=mref,
                reaction_ref=rref,
                protein_distance=mdist,
                protein_query_available=mqavail,
                reaction_distance=lr,
            )

        split_row={
            "split_id":sid,
            "train_pairs":int(len(train)),
            "test_pairs":int(len(test)),
            "protein_overlap":len(set(train.Entry)&set(test.Entry)),
            "reaction_overlap":len(set(train.rhea_id)&set(test.rhea_id)),
            "coarse_protein_reference_count":int(pinfo["reference_count"]),
            "coarse_reaction_reference_count":int(rinfo["reference_count"]),
            "pocket_reference_count":int(len(lpref)),
        }
        for coord in MECH_COORDS:
            cached=mech[coord]
            contract=mech_support[coord]
            split_row[f"{coord}_reference_count"]=(
                0 if cached is None else int(len(cached[0]))
            )
            split_row[f"{coord}_support_pair_count"]=(
                0 if contract is None else int(contract["train_pairs"])
            )
        splits.append(split_row)

        for rid,g in test.groupby("rhea_id",sort=True):
            qg=ri[rid]
            coarse_score=score_r2e(
                qg,train,Dr2,Dp2,rmap,pmap,ri,pi,len(pids)
            )
            coarse_defect=-np.asarray(coarse_score,dtype=np.float64)

            pocket_local=np.full(
                (len(POCKET_COORDS),len(pids)),np.nan,dtype=np.float64
            )
            if pocket_reaction_available[qg]:
                cand=np.flatnonzero(pocket_protein_available)
                for c,coord in enumerate(POCKET_COORDS):
                    sec=reaction_to_protein_section_from_support_distances(
                        np.square(
                            lr[
                                qg,
                                pocket_support["reaction_support_ref"],
                            ]
                        ),
                        np.square(
                            pocket_dist[coord][
                                np.ix_(
                                    cand,
                                    pocket_support["protein_support_ref"],
                                )
                            ]
                        ),
                        pocket_support["support_pairs"],
                        query_index=qg,
                        candidate_indices=cand,
                    )
                    pocket_local[c,cand]=sec.defect
                pocket_avail=np.repeat(
                    pocket_protein_available[None,:],
                    len(POCKET_COORDS),
                    axis=0,
                )
                catalytic=consensus_stratified_resolution(
                    coarse_defect,pocket_local,pocket_avail
                )
            else:
                catalytic=empty_catalytic(
                    coarse_defect,len(POCKET_COORDS)
                )

            mdef=np.full(
                (len(MECH_COORDS),len(pids)),np.nan,dtype=np.float64
            )
            mavail=np.zeros_like(mdef,dtype=bool)
            mapp=np.vstack([
                np.asarray(mech_app[c],dtype=bool) for c in MECH_COORDS
            ])
            for c,coord in enumerate(MECH_COORDS):
                contract=mech_support[coord]
                cached=mech[coord]
                if contract is None or cached is None:
                    continue
                _mref,mdist,_mqavail=cached
                candidates=np.flatnonzero(
                    contract["protein_available"]
                )
                mavail[c]=contract["protein_available"]
                if (
                    not contract["reaction_available"][qg]
                    or not len(candidates)
                ):
                    continue
                sec=reaction_to_protein_section_from_support_distances(
                    np.square(
                        lr[
                            qg,
                            contract["reaction_support_ref"],
                        ]
                    ),
                    np.square(
                        mdist[
                            np.ix_(
                                candidates,
                                contract["protein_support_ref"],
                            )
                        ]
                    ),
                    contract["support_pairs"],
                    query_index=qg,
                    candidate_indices=candidates,
                )
                mdef[c,candidates]=sec.defect

            mechres=mechanistic_chart_resolution(
                catalytic.coarse_level,
                catalytic.catalytic_stratum,
                catalytic.observed_coarse_levels,
                mdef,mapp,mavail,
            )
            positives=np.asarray([
                pi[x] for x in g.Entry.astype(str)
            ],dtype=int)
            rows.append({
                "split_id":sid,
                "direction":"reaction_to_enzyme",
                "query_id":rid,
                "positive_count":int(len(positives)),
                "catalytic_observed_level_count":int(
                    len(catalytic.observed_coarse_levels)
                ),
                "catalytic_refined_level_count":int(
                    len(catalytic.refined_coarse_levels)
                ),
                "mechanistic_observed_parent_chart_count":int(
                    len(mechres.observed_parent_charts)
                ),
                "mechanistic_refined_parent_chart_count":int(
                    len(mechres.refined_parent_charts)
                ),
                "mechanistic_refined_candidate_count":int(
                    mechres.refined_candidate_count
                ),
                "positive_mechanistic_resolved_count":int(
                    np.sum(mechres.mechanistic_stratum[positives]>=0)
                ),
                "positive_mechanistic_front0_count":int(
                    np.sum(mechres.mechanistic_stratum[positives]==0)
                ),
            })

        for pid,g in test.groupby("Entry",sort=True):
            qg=pi[pid]
            coarse_score=score_e2r(
                qg,train,Dr2,Dp2,rmap,pmap,ri,pi,len(rids)
            )
            coarse_defect=-np.asarray(coarse_score,dtype=np.float64)

            if pocket_protein_available[qg]:
                cand=np.flatnonzero(pocket_reaction_available)
                pocket_local=np.full(
                    (len(POCKET_COORDS),len(rids)),np.nan,dtype=np.float64
                )
                for c,coord in enumerate(POCKET_COORDS):
                    sec=protein_to_reaction_section_from_support_distances(
                        np.square(
                            pocket_dist[coord][
                                qg,
                                pocket_support["protein_support_ref"],
                            ]
                        ),
                        np.square(
                            lr[
                                np.ix_(
                                    cand,
                                    pocket_support["reaction_support_ref"],
                                )
                            ]
                        ),
                        pocket_support["support_pairs"],
                        query_index=qg,
                        candidate_indices=cand,
                    )
                    pocket_local[c,cand]=sec.defect
                pocket_avail=np.repeat(
                    pocket_reaction_available[None,:],
                    len(POCKET_COORDS),
                    axis=0,
                )
                catalytic=consensus_stratified_resolution(
                    coarse_defect,pocket_local,pocket_avail
                )
            else:
                catalytic=empty_catalytic(
                    coarse_defect,len(POCKET_COORDS)
                )

            mdef=np.full(
                (len(MECH_COORDS),len(rids)),np.nan,dtype=np.float64
            )
            mapp=np.zeros_like(mdef,dtype=bool)
            mavail=np.zeros_like(mdef,dtype=bool)
            for c,coord in enumerate(MECH_COORDS):
                app=bool(mech_app[coord][qg])
                mapp[c,:]=app
                if not app:
                    continue
                contract=mech_support[coord]
                cached=mech[coord]
                if contract is None or cached is None:
                    continue
                _mref,mdist,_mqavail=cached
                query_ok=bool(contract["protein_available"][qg])
                if not query_ok:
                    continue
                candidate_ok=np.asarray(
                    contract["reaction_available"],dtype=bool
                )
                mavail[c,:]=candidate_ok
                cand=np.flatnonzero(candidate_ok)
                if not len(cand):
                    continue
                sec=protein_to_reaction_section_from_support_distances(
                    np.square(
                        mdist[
                            qg,
                            contract["protein_support_ref"],
                        ]
                    ),
                    np.square(
                        lr[
                            np.ix_(
                                cand,
                                contract["reaction_support_ref"],
                            )
                        ]
                    ),
                    contract["support_pairs"],
                    query_index=qg,
                    candidate_indices=cand,
                )
                mdef[c,cand]=sec.defect

            mechres=mechanistic_chart_resolution(
                catalytic.coarse_level,
                catalytic.catalytic_stratum,
                catalytic.observed_coarse_levels,
                mdef,mapp,mavail,
            )
            positives=np.asarray([
                ri[x] for x in g.rhea_id.astype(str)
            ],dtype=int)
            rows.append({
                "split_id":sid,
                "direction":"enzyme_to_reaction",
                "query_id":pid,
                "positive_count":int(len(positives)),
                "catalytic_observed_level_count":int(
                    len(catalytic.observed_coarse_levels)
                ),
                "catalytic_refined_level_count":int(
                    len(catalytic.refined_coarse_levels)
                ),
                "mechanistic_observed_parent_chart_count":int(
                    len(mechres.observed_parent_charts)
                ),
                "mechanistic_refined_parent_chart_count":int(
                    len(mechres.refined_parent_charts)
                ),
                "mechanistic_refined_candidate_count":int(
                    mechres.refined_candidate_count
                ),
                "positive_mechanistic_resolved_count":int(
                    np.sum(mechres.mechanistic_stratum[positives]>=0)
                ),
                "positive_mechanistic_front0_count":int(
                    np.sum(mechres.mechanistic_stratum[positives]==0)
                ),
            })

        print(sid,"done",flush=True)

    q=pd.DataFrame(rows)
    sp=pd.DataFrame(splits)
    metrics=[]
    for direction,g in q.groupby("direction"):
        metrics.append({
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
    stability=stability_summary(q)

    OUT.mkdir(parents=True,exist_ok=True)
    q.to_csv(OUT/"query_resolution.csv",index=False)
    sp.to_csv(OUT/"split_summary.csv",index=False)
    pd.DataFrame(metrics).to_csv(OUT/"metrics.csv",index=False)
    summary={
        "version":"fibre-mechanistic-stratified-strict-inductive-v1",
        "protocol":"for every held-out factor fold, rebuild coarse, catalytic-pocket, reaction-center, and each motif-coordinate reference atlas without held-out entities; attach excluded entities only from their query-to-reference molecular observations; define mechanism charts only inside observed catalytic parents",
        "canonical_total_rank_changed":False,
        "mechanistic_coordinates":list(MECH_COORDS),
        "chart_definition":"family-applicability bit mask; different mechanism charts are incomparable",
        "missing_policy":"applicable but unobserved motif leaves the mechanism comparison unresolved; non-applicable motif is outside that chart",
        "labels_used_to_build_geometry":False,
        "labels_used_only_for_resolution_audit":True,
        "external_retention_used":False,
        "all_train_test_entity_overlaps_zero":bool(
            ((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()
        ),
        "metrics":metrics,
        "development_vs_strict_refined_query_stability":stability,
        "elapsed_seconds":time.time()-t0,
    }
    (OUT/"summary.json").write_text(
        json.dumps(summary,indent=2)+"\n"
    )
    print(pd.DataFrame(metrics).to_string(index=False))
    print(json.dumps({
        "development_vs_strict_refined_query_stability":stability,
        "all_train_test_entity_overlaps_zero":summary[
            "all_train_test_entity_overlaps_zero"
        ],
        "elapsed_seconds":summary["elapsed_seconds"],
    },indent=2),flush=True)


if __name__=="__main__":
    main()
