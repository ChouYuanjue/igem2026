from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from projects.active.fibre.evaluation.correspondence import (
    CACHE, PG, RG, BUD, bools, intrinsic_geodesic, pair_array,
)
from projects.active.fibre.geometry.correspondence import correspondence_state
from projects.active.fibre.geometry.levelset import stable_level_ids
from projects.active.fibre.geometry.stratified import stratified_resolution
from projects.active.fibre.runtime.base_model import rank_metrics

ROOT=Path(__file__).resolve().parents[4]
LOCAL=ROOT/"data/terpene_catalytic_local_geometry_v1"
QUALIFIED_OUT=ROOT/"results/fibre_stratified_correspondence_structure_qualified_dev_v1"
UNQUALIFIED_OUT=ROOT/"results/fibre_stratified_correspondence_unqualified_dev_v1"
BOOTSTRAP_REPLICATES=20_000
BOOTSTRAP_SEED=20260919


def deterministic_order_from_defect(defect: np.ndarray) -> np.ndarray:
    return np.argsort(np.asarray(defect,dtype=np.float64),kind="stable")


def stratified_order(
    coarse_defect: np.ndarray,
    local_defect_by_global_index: np.ndarray,
    local_available: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Thin evaluator wrapper around the canonical StratifiedResolution."""
    resolution=stratified_resolution(
        coarse_defect,local_defect_by_global_index,local_available
    )
    levels=resolution.coarse_level
    refined=set(resolution.locally_refined_coarse_levels)
    return resolution.display_order(),{
        "coarse_numerical_tolerance":float(resolution.coarse_tolerance),
        "coarse_levels":int(levels.max()+1),
        "coarse_nontrivial_levels":int(
            sum(np.sum(levels==k)>1 for k in range(int(levels.max())+1))
        ),
        "locally_refined_levels":int(len(refined)),
        "locally_refined_candidates":int(
            sum(np.sum(levels==k) for k in refined)
        ),
    }


def metrics_from_order(order: np.ndarray, ids: list[str], positives: set[str]) -> dict[str,float]:
    ordered=[ids[int(i)] for i in order]
    positive_ranks=[i+1 for i,x in enumerate(ordered) if x in positives]
    if not positive_ranks:
        return {"best_positive_rank":float("inf"),"reciprocal_rank":0.0}
    best=min(positive_ranks)
    return {"best_positive_rank":float(best),"reciprocal_rank":1.0/float(best)}


def aggregate(q: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (method,direction),g in q.groupby(["method","direction"]):
        rows.append({
            "method":method,
            "direction":direction,
            "n_queries":int(len(g)),
            "mrr":float(g.reciprocal_rank.mean()),
            "median_rank":float(g.best_positive_rank.median()),
            "local_refinement_applied_fraction":float(g.local_refinement_applied.mean()),
            "top_level_refined_fraction":float(g.top_level_refined.mean()),
        })
    return pd.DataFrame(rows)


def paired_summary(q: pd.DataFrame) -> dict:
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    out={}
    for direction,g in q.groupby("direction"):
        base=g[g.method.eq("coarse")][
            ["split_id","query_id","reciprocal_rank","best_positive_rank"]
        ].rename(columns={"reciprocal_rank":"rr_base","best_positive_rank":"rank_base"})
        ref=g[g.method.eq("stratified")][
            ["split_id","query_id","reciprocal_rank","best_positive_rank"]
        ].rename(columns={"reciprocal_rank":"rr_refined","best_positive_rank":"rank_refined"})
        z=base.merge(ref,on=["split_id","query_id"],validate="one_to_one")
        delta=(z.rr_refined-z.rr_base).to_numpy(float)
        samples=np.empty(BOOTSTRAP_REPLICATES,dtype=np.float64)
        n=len(delta)
        for i in range(BOOTSTRAP_REPLICATES):
            samples[i]=float(np.mean(delta[rng.integers(0,n,size=n)]))
        out[direction]={
            "queries":int(n),
            "mean_rr_delta":float(np.mean(delta)),
            "median_rank_delta_base_minus_refined":float(np.median(z.rank_base-z.rank_refined)),
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
    ap=argparse.ArgumentParser()
    ap.add_argument("--qualification",choices=("pocket3di","all_local"),default="pocket3di")
    ap.add_argument("--local-geometry",type=Path,default=LOCAL)
    ap.add_argument("--output",type=Path,default=None)
    args=ap.parse_args()
    local_dir=Path(args.local_geometry)
    out_dir=(Path(args.output) if args.output is not None else (QUALIFIED_OUT if args.qualification=="pocket3di" else UNQUALIFIED_OUT))
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

    local_rids=pd.read_csv(local_dir/"reaction_ids.csv",dtype=str).reaction_id.astype(str).tolist()
    if local_rids != rids:
        raise RuntimeError("local reaction geometry is not aligned to canonical reactions")
    local_protein_rows=np.load(local_dir/"protein_global_rows.npy").astype(int)
    local_pids=pd.read_csv(local_dir/"protein_ids.csv",dtype=str).protein_id.astype(str).tolist()
    if [pids[int(i)] for i in local_protein_rows] != local_pids:
        raise RuntimeError("local protein row mapping mismatch")
    local_pos={int(g):i for i,g in enumerate(local_protein_rows)}
    local_available=np.zeros(len(pids),dtype=bool)
    local_available[local_protein_rows]=True
    pocket3di_observed=np.load(PG/"pocket_3di_relational_available.npy").astype(bool)
    order_bearing_local=(
        local_available & pocket3di_observed
        if args.qualification=="pocket3di"
        else local_available.copy()
    )

    Dr_local,_=intrinsic_geodesic(load_npz(local_dir/"reaction_local_affinity.npz"))
    De_local,_=intrinsic_geodesic(load_npz(local_dir/"protein_local_affinity.npz"))
    Dr_local2=Dr_local*Dr_local;De_local2=De_local*De_local

    rows=[];split_rows=[]
    for pf in range(5):
        for rf in range(5):
            if not (pf==4 or rf==4):
                continue
            split_id=f"p{pf}_r{rf}"
            train=pairs[
                pairs.protein_fold.ne(pf)&pairs.reaction_fold.ne(rf)
            ].drop_duplicates(["rhea_id","Entry"])
            test=pairs[
                pairs.protein_fold.eq(pf)
                & pairs.reaction_fold.eq(rf)
                & ~pairs.protein_seen
                & ~pairs.reaction_seen
            ].drop_duplicates(["rhea_id","Entry"])
            coarse_pairs=pair_array(train,ri,pi)
            coarse=correspondence_state(Dr2,De2,coarse_pairs)

            train_global_rows=train.Entry.map(pi).to_numpy(dtype=int)
            local_train=train[order_bearing_local[train_global_rows]].copy()
            local_pairs=np.asarray([
                (ri[x.rhea_id],local_pos[pi[x.Entry]])
                for x in local_train.itertuples(index=False)
            ],dtype=int)
            local=correspondence_state(Dr_local2,De_local2,local_pairs)

            split_rows.append({
                "split_id":split_id,
                "train_pairs":int(len(train)),
                "local_train_pairs":int(len(local_train)),
                "test_pairs":int(len(test)),
                "protein_overlap":len(set(train.Entry)&set(test.Entry)),
                "reaction_overlap":len(set(train.rhea_id)&set(test.rhea_id)),
            })

            for rid,g in test.groupby("rhea_id",sort=True):
                r=ri[rid]
                coarse_defect=coarse.defect[r]
                baseline=deterministic_order_from_defect(coarse_defect)
                local_global=np.full(len(pids),np.nan,dtype=np.float64)
                local_global[local_protein_rows]=local.defect[r]
                refined,diag=stratified_order(
                    coarse_defect,local_global,order_bearing_local
                )
                positives=set(g.Entry.astype(str))
                for method,order in (("coarse",baseline),("stratified",refined)):
                    m=metrics_from_order(order,pids,positives)
                    rows.append({
                        "split_id":split_id,"direction":"reaction_to_enzyme",
                        "query_id":rid,"method":method,**m,
                        "local_refinement_applied":bool(diag["locally_refined_levels"]>0),
                        "top_level_refined":bool(
                            diag["locally_refined_levels"]>0
                            and np.all(order_bearing_local[np.flatnonzero(
                                stable_level_ids(coarse_defect)[0]==0
                            )])
                            and np.sum(stable_level_ids(coarse_defect)[0]==0)>1
                        ),
                        **diag,
                    })

            for pid,g in test.groupby("Entry",sort=True):
                e=pi[pid]
                coarse_defect=coarse.defect[:,e]
                baseline=deterministic_order_from_defect(coarse_defect)
                if order_bearing_local[e]:
                    local_global=local.defect[:,local_pos[e]]
                    reaction_local_available=np.ones(len(rids),dtype=bool)
                    refined,diag=stratified_order(
                        coarse_defect,local_global,reaction_local_available
                    )
                else:
                    refined=baseline.copy()
                    levels,tol=stable_level_ids(coarse_defect)
                    diag={
                        "coarse_numerical_tolerance":float(tol),
                        "coarse_levels":int(levels.max()+1),
                        "coarse_nontrivial_levels":int(sum(np.sum(levels==k)>1 for k in range(int(levels.max())+1))),
                        "locally_refined_levels":0,
                        "locally_refined_candidates":0,
                    }
                positives=set(g.rhea_id.astype(str))
                for method,order in (("coarse",baseline),("stratified",refined)):
                    m=metrics_from_order(order,rids,positives)
                    rows.append({
                        "split_id":split_id,"direction":"enzyme_to_reaction",
                        "query_id":pid,"method":method,**m,
                        "local_refinement_applied":bool(diag["locally_refined_levels"]>0),
                        "top_level_refined":bool(
                            order_bearing_local[e]
                            and diag["locally_refined_levels"]>0
                            and np.sum(stable_level_ids(coarse_defect)[0]==0)>1
                        ),
                        **diag,
                    })

    q=pd.DataFrame(rows);sp=pd.DataFrame(split_rows)
    metrics=aggregate(q);paired=paired_summary(q)
    out_dir.mkdir(parents=True,exist_ok=True)
    q.to_csv(out_dir/"query_metrics.csv",index=False)
    sp.to_csv(out_dir/"split_summary.csv",index=False)
    metrics.to_csv(out_dir/"metrics.csv",index=False)
    summary={
        "version":"fibre-stratified-correspondence-dev-v1",
        "partition":"development_only",
        "method":"same zero-temperature correspondence defect at coarse and catalytic-local resolutions",
        "coarse_factor":"canonical global reaction x multiresolution protein geometry",
        "local_factor_manifest":json.loads((local_dir/"manifest.json").read_text()),
        "qualification_mode":args.qualification,
        "comparison_rule":(
            "cross-coarse-level order immutable; catalytic-local defect becomes order-bearing only when the protein-side pocket-3Di chart is observed for the query or every candidate in the coarse level; motif/sequence-only local coordinates remain unresolved rather than becoming penalties"
            if args.qualification=="pocket3di" else
            "diagnostic unqualified variant: any observed catalytic-local coordinate may become order-bearing inside a coarse numerical level"
        ),
        "labels_used_to_build_local_geometry":False,
        "qualification_rule_selected_on":"internal development analysis only; no external-retention labels",
        "all_train_test_entity_overlaps_zero":bool(
            ((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()
        ),
        "paired":paired,
        "metrics":metrics.to_dict("records"),
    }
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(metrics.to_string(index=False))
    print(json.dumps({k:v for k,v in summary.items() if k!="metrics"},indent=2))


if __name__=="__main__":
    main()
