from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from projects.active.fibre.evaluation.correspondence import (
    CACHE,
    PG,
    RG,
    BUD,
    intrinsic_geodesic,
    pair_array,
)
from projects.active.fibre.geometry.correspondence import correspondence_state
from projects.active.fibre.runtime.base_model import rank_metrics

ROOT=Path(__file__).resolve().parents[4]
REFINED=ROOT/"data/terpene_reaction_tangent_information_geometry_v1"
OUT=ROOT/"results/fibre_reaction_tangent_information_dev_v1"
BOOTSTRAP_REPLICATES=20_000
BOOTSTRAP_SEED=20260917


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def bools(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"1","true","yes"})


def evaluate_field(
    field: np.ndarray,
    test: pd.DataFrame,
    rids: list[str],
    pids: list[str],
    ri: dict[str,int],
    pi: dict[str,int],
    split_id: str,
    method: str,
) -> list[dict]:
    rows=[]
    for rid,g in test.groupby("rhea_id",sort=True):
        rows.append({
            "method":method,"split_id":split_id,
            "direction":"reaction_to_enzyme","query_id":rid,
            **rank_metrics(field[ri[rid]],pids,set(g.Entry.astype(str)),set(),BUD),
        })
    for pid,g in test.groupby("Entry",sort=True):
        rows.append({
            "method":method,"split_id":split_id,
            "direction":"enzyme_to_reaction","query_id":pid,
            **rank_metrics(field[:,pi[pid]],rids,set(g.rhea_id.astype(str)),set(),BUD),
        })
    return rows


def aggregate(q: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (method,direction),g in q.groupby(["method","direction"]):
        row={
            "method":method,"direction":direction,"n_queries":len(g),
            "mrr":float(g.reciprocal_rank.mean()),
            "median_rank":float(g.best_positive_rank.median()),
        }
        for k in BUD:
            row[f"hit{k}"]=float(g[f"hit_at_{k}"].mean())
            row[f"recall{k}"]=float(g[f"positive_recall_at_{k}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_summary(q: pd.DataFrame) -> dict:
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    out={}
    for direction,g in q.groupby("direction"):
        base=g[g.method.eq("base_global")][
            ["split_id","query_id","reciprocal_rank","best_positive_rank"]
        ].rename(columns={
            "reciprocal_rank":"rr_base","best_positive_rank":"rank_base"
        })
        ref=g[g.method.eq("tangent_information")][
            ["split_id","query_id","reciprocal_rank","best_positive_rank"]
        ].rename(columns={
            "reciprocal_rank":"rr_refined","best_positive_rank":"rank_refined"
        })
        z=base.merge(ref,on=["split_id","query_id"],validate="one_to_one")
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

    Dr_base,base_scale=intrinsic_geodesic(load_npz(RG/"partial_pullback_affinity.npz"))
    Dr_ref,ref_scale=intrinsic_geodesic(load_npz(REFINED/"partial_pullback_affinity.npz"))
    De,protein_scale=intrinsic_geodesic(load_npz(PG/"partial_pullback_affinity.npz"))
    Dr2={"base_global":Dr_base*Dr_base,"tangent_information":Dr_ref*Dr_ref}
    De2=De*De

    rows=[];splits=[]
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
            splits.append({
                "split_id":split_id,"train_pairs":len(train),"test_pairs":len(test),
                "protein_overlap":len(set(train.Entry)&set(test.Entry)),
                "reaction_overlap":len(set(train.rhea_id)&set(test.rhea_id)),
            })
            pp=pair_array(train,ri,pi)
            for method,distance in Dr2.items():
                field=correspondence_state(distance,De2,pp).field
                rows.extend(
                    evaluate_field(field,test,rids,pids,ri,pi,split_id,method)
                )

    q=pd.DataFrame(rows);sp=pd.DataFrame(splits);metrics=aggregate(q)
    paired=paired_summary(q)
    OUT.mkdir(parents=True,exist_ok=True)
    q.to_csv(OUT/"query_metrics.csv",index=False)
    sp.to_csv(OUT/"split_summary.csv",index=False)
    metrics.to_csv(OUT/"metrics.csv",index=False)
    summary={
        "version":"fibre-reaction-tangent-information-dev-v1",
        "partition":"development_only",
        "operator":"canonical zero-temperature FIBRE correspondence defect",
        "comparison":"same protein geometry and positive correspondence; only reaction factor changes",
        "base_reaction_geometry":"global DRFP + reactant + product partial pullback",
        "refined_reaction_geometry":"exact base topology; center W2/token tangent energy weighted by intrinsic entropy-deficit reliability",
        "parameter_selection":"none; no labels, thresholds, learned view weights, or external-retention metrics",
        "characteristic_lengths":{
            "reaction_base":base_scale,
            "reaction_refined":ref_scale,
            "protein":protein_scale,
        },
        "all_train_test_entity_overlaps_zero":bool(
            ((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()
        ),
        "paired":paired,
        "metrics":metrics.to_dict("records"),
        "input_sha256":{
            "split":sha256(CACHE/"marts_pair_folds.csv"),
            "base_reaction_manifest":sha256(RG/"manifest.json"),
            "refined_reaction_manifest":sha256(REFINED/"manifest.json"),
            "protein_manifest":sha256(PG/"manifest.json"),
        },
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(metrics.to_string(index=False))
    print(json.dumps({k:v for k,v in summary.items() if k!="metrics"},indent=2))


if __name__=="__main__":
    main()
