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
from projects.active.fibre.geometry.stratified import consensus_stratified_resolution

ROOT=Path(__file__).resolve().parents[4]
LOCAL=ROOT/"data/terpene_catalytic_consensus_geometry_v1"
OUT=ROOT/"results/fibre_consensus_stratified_dev_v1"
BOOTSTRAP_REPLICATES=20_000
BOOTSTRAP_SEED=20260919
COORDS=("pocket_local_esmc","pocket_3di","pocket_ot")


def metrics_from_order(order: np.ndarray, ids: list[str], positives: set[str]) -> dict[str,float]:
    ordered=[ids[int(i)] for i in order]
    ranks=[i+1 for i,x in enumerate(ordered) if x in positives]
    if not ranks:
        return {"best_positive_rank":float("inf"),"reciprocal_rank":0.0}
    best=min(ranks)
    return {"best_positive_rank":float(best),"reciprocal_rank":1.0/float(best)}


def paired_summary(q: pd.DataFrame) -> dict:
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    out={}
    for direction,g in q.groupby("direction"):
        a=g[g.method.eq("coarse")][
            ["split_id","query_id","reciprocal_rank","best_positive_rank"]
        ].rename(columns={"reciprocal_rank":"rr_base","best_positive_rank":"rank_base"})
        b=g[g.method.eq("consensus")][
            ["split_id","query_id","reciprocal_rank","best_positive_rank"]
        ].rename(columns={"reciprocal_rank":"rr_refined","best_positive_rank":"rank_refined"})
        z=a.merge(b,on=["split_id","query_id"],validate="one_to_one")
        delta=(z.rr_refined-z.rr_base).to_numpy(float)
        samples=np.empty(BOOTSTRAP_REPLICATES,dtype=np.float64)
        n=len(delta)
        for i in range(BOOTSTRAP_REPLICATES):
            samples[i]=float(np.mean(delta[rng.integers(0,n,size=n)]))
        out[direction]={
            "queries":int(n),
            "mean_rr_delta":float(np.mean(delta)),
            "median_rank_delta_base_minus_refined":float(
                np.median(z.rank_base-z.rank_refined)
            ),
            "rr_improve_tie_worse":[
                int(np.sum(delta>0)),int(np.sum(delta==0)),int(np.sum(delta<0))
            ],
            "bootstrap_95ci":[
                float(np.quantile(samples,.025)),float(np.quantile(samples,.975))
            ],
            "bootstrap_p_delta_gt_0":float(np.mean(samples>0)),
        }
    return out


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
    Dr2=Dr*Dr;De2=De*De

    local_rids=pd.read_csv(LOCAL/"reaction_ids.csv",dtype=str).reaction_id.astype(str).tolist()
    if local_rids!=rids:
        raise RuntimeError("local reaction geometry misaligned")
    global_rows=np.load(LOCAL/"protein_global_rows.npy").astype(int)
    local_pids=pd.read_csv(LOCAL/"protein_ids.csv",dtype=str).protein_id.astype(str).tolist()
    if [pids[int(i)] for i in global_rows]!=local_pids:
        raise RuntimeError("local protein mapping mismatch")
    local_pos={int(g):i for i,g in enumerate(global_rows)}
    common=np.zeros(len(pids),dtype=bool)
    common[global_rows]=True

    DrL,_=intrinsic_geodesic(load_npz(LOCAL/"reaction_local_affinity.npz"))
    DrL2=DrL*DrL
    protein_dist={}
    for coord in COORDS:
        d,_=intrinsic_geodesic(load_npz(LOCAL/f"protein_{coord}_affinity.npz"))
        protein_dist[coord]=d*d

    rows=[];splits=[]
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

            splits.append({
                "split_id":sid,"train_pairs":int(len(train)),
                "local_train_pairs":int(len(local_train)),
                "test_pairs":int(len(test)),
                "protein_overlap":len(set(train.Entry)&set(test.Entry)),
                "reaction_overlap":len(set(train.rhea_id)&set(test.rhea_id)),
            })

            for rid,g in test.groupby("rhea_id",sort=True):
                r=ri[rid]
                coarse_defect=coarse.defect[r]
                baseline=np.argsort(coarse_defect,kind="stable")
                local=np.full((len(COORDS),len(pids)),np.nan,dtype=np.float64)
                for c,coord in enumerate(COORDS):
                    local[c,global_rows]=local_states[coord].defect[r]
                available=np.repeat(common[None,:],len(COORDS),axis=0)
                resolution=consensus_stratified_resolution(
                    coarse_defect,local,available
                )
                refined=resolution.display_order()
                positives=set(g.Entry.astype(str))
                for method,order in (("coarse",baseline),("consensus",refined)):
                    rows.append({
                        "split_id":sid,"direction":"reaction_to_enzyme",
                        "query_id":rid,"method":method,
                        **metrics_from_order(order,pids,positives),
                        "local_refinement_applied":bool(
                            resolution.refined_coarse_levels
                        ),
                        "refined_coarse_level_count":int(
                            len(resolution.refined_coarse_levels)
                        ),
                    })

            for pid,g in test.groupby("Entry",sort=True):
                e=pi[pid]
                coarse_defect=coarse.defect[:,e]
                baseline=np.argsort(coarse_defect,kind="stable")
                if common[e]:
                    le=local_pos[e]
                    local=np.vstack([
                        local_states[coord].defect[:,le]
                        for coord in COORDS
                    ])
                    available=np.ones_like(local,dtype=bool)
                    resolution=consensus_stratified_resolution(
                        coarse_defect,local,available
                    )
                    refined=resolution.display_order()
                    applied=bool(resolution.refined_coarse_levels)
                    nref=int(len(resolution.refined_coarse_levels))
                else:
                    refined=baseline.copy();applied=False;nref=0
                positives=set(g.rhea_id.astype(str))
                for method,order in (("coarse",baseline),("consensus",refined)):
                    rows.append({
                        "split_id":sid,"direction":"enzyme_to_reaction",
                        "query_id":pid,"method":method,
                        **metrics_from_order(order,rids,positives),
                        "local_refinement_applied":applied,
                        "refined_coarse_level_count":nref,
                    })

    q=pd.DataFrame(rows);sp=pd.DataFrame(splits)
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
        "version":"fibre-consensus-stratified-dev-v1",
        "partition":"development_only",
        "coarse_operator":"zero-temperature FIBRE correspondence defect",
        "local_operator":"same defect independently on reaction-center x each pocket coordinate",
        "local_coordinates":list(COORDS),
        "refinement":"Pareto fronts inside coarse numerical levels; candidate dominates only if no worse in all three pocket coordinates and strictly better in at least one",
        "missing_policy":"all three pocket coordinates required for the whole compared coarse level; otherwise unresolved",
        "motif_role":"finer mechanistic stratum, not part of order-bearing pocket consensus",
        "labels_used_to_build_local_geometry":False,
        "all_train_test_entity_overlaps_zero":bool(
            ((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()
        ),
        "paired":paired,
        "metrics":metrics,
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(pd.DataFrame(metrics).to_string(index=False))
    print(json.dumps({k:v for k,v in summary.items() if k!="metrics"},indent=2))


if __name__=="__main__":
    main()
