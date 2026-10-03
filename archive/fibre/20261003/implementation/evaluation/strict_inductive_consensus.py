from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.fibre.evaluation.strict_inductive import (
    CACHE, POCKET, STRUCT, bools, chordal_full,
    build_protein_factor_inputs, build_reaction_factor_inputs,
    materialize_views, factor_all_to_reference,
    coordinate_reference_and_cross, diffusion_reference_and_cross,
    score_r2e, score_e2r,
)
from projects.active.fibre.evaluation.strict_inductive_stratified import (
    build_local_reaction_inputs,
    materialize_named_views,
    factor_all_to_reference_information,
)
from projects.active.fibre.geometry.correspondence import (
    reaction_to_protein_section_from_support_distances,
    protein_to_reaction_section_from_support_distances,
)
from projects.active.fibre.geometry.stratified import consensus_stratified_resolution

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/"results/fibre_consensus_stratified_strict_inductive_v1"
COORDS=("pocket_local_esmc","pocket_3di","pocket_ot")
BOOTSTRAP_REPLICATES=20_000
BOOTSTRAP_SEED=20260919


def build_pocket_raw():
    p=np.load(POCKET/"embeddings.npy",mmap_mode="r")
    pa=np.load(POCKET/"available.npy").astype(bool)
    views=[("pocket_local_esmc","coordinate",chordal_full(p,pa),pa)]
    for name in ("pocket_3di","pocket_ot"):
        s=np.load(STRUCT/name/"similarity.npy",mmap_mode="r")
        a=np.load(STRUCT/name/"available.npy").astype(bool)
        views.append((name,"diffusion",s,a))
    return views


def stable_common_reference(raw_views, initial_ref):
    ref=np.asarray(initial_ref,dtype=int)
    for _ in range(8):
        named=materialize_named_views(raw_views,ref)
        keep=np.ones(len(ref),dtype=bool)
        for _name,_dref,ravail,_cross,_qall in named:
            keep &= np.asarray(ravail,dtype=bool)
        new=ref[keep]
        if np.array_equal(new,ref):
            return ref,named
        if len(new)<2:
            raise RuntimeError("pocket consensus reference collapsed")
        ref=new
    raise RuntimeError("pocket consensus reference did not stabilize")


def build_coordinate_atlases(raw_views, initial_ref):
    ref,named=stable_common_reference(raw_views,initial_ref)
    atlases={}
    qavail={}
    for item in named:
        name=item[0]
        dist,info=factor_all_to_reference_information(ref,[item])
        atlases[name]=dist
        qavail[name]=np.asarray(
            info["query_available_by_view"][name],dtype=bool
        ) & np.all(np.isfinite(dist),axis=1)
    return ref,atlases,qavail


def metrics_from_order(order, ids, positives):
    ordered=[ids[int(i)] for i in order]
    ranks=[i+1 for i,x in enumerate(ordered) if x in positives]
    if not ranks:
        return {"best_positive_rank":float("inf"),"reciprocal_rank":0.0}
    best=min(ranks)
    return {"best_positive_rank":float(best),"reciprocal_rank":1.0/float(best)}


def paired_summary(q):
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    out={}
    for direction,g in q.groupby("direction"):
        a=g[g.method.eq("coarse")][
            ["split_id","query_id","reciprocal_rank"]
        ].rename(columns={"reciprocal_rank":"rr_base"})
        b=g[g.method.eq("consensus")][
            ["split_id","query_id","reciprocal_rank"]
        ].rename(columns={"reciprocal_rank":"rr_refined"})
        z=a.merge(b,on=["split_id","query_id"],validate="one_to_one")
        delta=(z.rr_refined-z.rr_base).to_numpy(float)
        samples=np.empty(BOOTSTRAP_REPLICATES,dtype=np.float64)
        n=len(delta)
        for i in range(BOOTSTRAP_REPLICATES):
            samples[i]=float(np.mean(delta[rng.integers(0,n,size=n)]))
        out[direction]={
            "queries":int(n),
            "mean_rr_delta":float(np.mean(delta)),
            "rr_improve_tie_worse":[
                int(np.sum(delta>0)),int(np.sum(delta==0)),int(np.sum(delta<0))
            ],
            "bootstrap_95ci":[
                float(np.quantile(samples,.025)),float(np.quantile(samples,.975))
            ],
            "bootstrap_p_delta_gt_0":float(np.mean(samples>0)),
        }
    return out


def main():
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

    coarse_p_raw=build_protein_factor_inputs()
    coarse_r_raw=build_reaction_factor_inputs(reactions)
    pocket_raw=build_pocket_raw()
    local_r_raw=build_local_reaction_inputs()
    pocket_raw_common=np.logical_and.reduce([
        np.asarray(v[3],dtype=bool) for v in pocket_raw
    ])

    coarse_p_cache={}
    pocket_cache={}
    for pf in range(5):
        pref=np.asarray([
            i for i,pid in enumerate(pids) if pfold.get(pid)!=pf
        ],dtype=int)
        coarse_p_cache[pf]=factor_all_to_reference(
            pref,materialize_views(coarse_p_raw,pref)
        )
        initial=np.asarray([
            i for i,pid in enumerate(pids)
            if pfold.get(pid)!=pf and pocket_raw_common[i]
        ],dtype=int)
        pocket_cache[pf]=build_coordinate_atlases(pocket_raw,initial)
        print("protein fold",pf,"coarse",len(pref),"pocket consensus",len(pocket_cache[pf][0]),flush=True)

    coarse_r_cache={}
    local_r_cache={}
    for rf in range(5):
        rref=np.asarray([
            i for i,rid in enumerate(rids) if rfold.get(rid)!=rf
        ],dtype=int)
        coarse_r_cache[rf]=factor_all_to_reference(
            rref,materialize_views(coarse_r_raw,rref)
        )
        local_r_cache[rf]=(
            rref,
            factor_all_to_reference_information(
                rref,materialize_named_views(local_r_raw,rref)
            )[0],
        )
        print("reaction fold",rf,"reference",len(rref),flush=True)

    rows=[];split_rows=[]
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

        Dp,pinfo=coarse_p_cache[pf]
        Dr,rinfo=coarse_r_cache[rf]
        pref=np.asarray([
            i for i,pid in enumerate(pids) if pfold.get(pid)!=pf
        ],dtype=int)
        rref=np.asarray([
            i for i,rid in enumerate(rids) if rfold.get(rid)!=rf
        ],dtype=int)
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
            raise RuntimeError(f"{sid}: no local positive support")

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

        for rid,g in test.groupby("rhea_id",sort=True):
            qg=ri[rid]
            coarse_score=score_r2e(
                qg,train,Dr2,Dp2,rmap,pmap,ri,pi,len(pids)
            )
            coarse_defect=-np.asarray(coarse_score,dtype=np.float64)
            baseline=np.argsort(coarse_defect,kind="stable")
            local=np.full((len(COORDS),len(pids)),np.nan,dtype=np.float64)
            if reaction_available[qg]:
                cand=np.flatnonzero(protein_available)
                for c,coord in enumerate(COORDS):
                    sec=reaction_to_protein_section_from_support_distances(
                        np.square(lr[qg,rs_ref]),
                        np.square(coord_dist[coord][np.ix_(cand,es_ref)]),
                        support_pairs,
                        query_index=qg,
                        candidate_indices=cand,
                    )
                    local[c,cand]=sec.defect
                avail=np.repeat(protein_available[None,:],len(COORDS),axis=0)
                res=consensus_stratified_resolution(
                    coarse_defect,local,avail
                )
                refined=res.display_order()
                applied=bool(res.refined_coarse_levels)
            else:
                refined=baseline.copy();applied=False
            positives=set(g.Entry.astype(str))
            for method,order in (("coarse",baseline),("consensus",refined)):
                rows.append({
                    "split_id":sid,"direction":"reaction_to_enzyme",
                    "query_id":rid,"method":method,
                    **metrics_from_order(order,pids,positives),
                    "local_refinement_applied":applied,
                })

        for pid,g in test.groupby("Entry",sort=True):
            qg=pi[pid]
            coarse_score=score_e2r(
                qg,train,Dr2,Dp2,rmap,pmap,ri,pi,len(rids)
            )
            coarse_defect=-np.asarray(coarse_score,dtype=np.float64)
            baseline=np.argsort(coarse_defect,kind="stable")
            if protein_available[qg]:
                cand=np.flatnonzero(reaction_available)
                local=np.full((len(COORDS),len(rids)),np.nan,dtype=np.float64)
                for c,coord in enumerate(COORDS):
                    sec=protein_to_reaction_section_from_support_distances(
                        np.square(coord_dist[coord][qg,es_ref]),
                        np.square(lr[np.ix_(cand,rs_ref)]),
                        support_pairs,
                        query_index=qg,
                        candidate_indices=cand,
                    )
                    local[c,cand]=sec.defect
                avail=np.repeat(reaction_available[None,:],len(COORDS),axis=0)
                res=consensus_stratified_resolution(
                    coarse_defect,local,avail
                )
                refined=res.display_order()
                applied=bool(res.refined_coarse_levels)
            else:
                refined=baseline.copy();applied=False
            positives=set(g.rhea_id.astype(str))
            for method,order in (("coarse",baseline),("consensus",refined)):
                rows.append({
                    "split_id":sid,"direction":"enzyme_to_reaction",
                    "query_id":pid,"method":method,
                    **metrics_from_order(order,rids,positives),
                    "local_refinement_applied":applied,
                })

        split_rows.append({
            "split_id":sid,
            "train_pairs":int(len(train)),
            "local_train_pairs":int(len(local_train)),
            "test_pairs":int(len(test)),
            "protein_overlap":len(set(train.Entry)&set(test.Entry)),
            "reaction_overlap":len(set(train.rhea_id)&set(test.rhea_id)),
            "pocket_reference_count":int(len(lpref)),
            "reaction_local_reference_count":int(len(lrref)),
        })
        print(sid,"done",flush=True)

    q=pd.DataFrame(rows);sp=pd.DataFrame(split_rows)
    metrics=[]
    for (method,direction),g in q.groupby(["method","direction"]):
        metrics.append({
            "method":method,"direction":direction,"n_queries":int(len(g)),
            "mrr":float(g.reciprocal_rank.mean()),
            "median_rank":float(g.best_positive_rank.median()),
            "local_refinement_applied_fraction":float(g.local_refinement_applied.mean()),
        })
    paired=paired_summary(q)
    OUT.mkdir(parents=True,exist_ok=True)
    q.to_csv(OUT/"query_metrics.csv",index=False)
    sp.to_csv(OUT/"split_summary.csv",index=False)
    pd.DataFrame(metrics).to_csv(OUT/"metrics.csv",index=False)
    summary={
        "version":"fibre-consensus-stratified-strict-inductive-v1",
        "protocol":"rebuild coarse, reaction-center and each pocket coordinate atlas without held-out factor entities; attach held-out entities only from query-to-reference observations; refine coarse numerical levels by Pareto consensus",
        "local_coordinates":list(COORDS),
        "missing_policy":"all three pocket coordinates required for compared candidates/query; otherwise coarse relation remains unresolved",
        "motif_role":"mechanistic third stratum; not part of pocket consensus ordering",
        "qualification_rule_selected_on":"construction chosen after internal development analysis; this strict audit is the promotion gate",
        "all_train_test_entity_overlaps_zero":bool(
            ((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()
        ),
        "paired":paired,
        "metrics":metrics,
        "elapsed_seconds":time.time()-t0,
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+chr(10))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
