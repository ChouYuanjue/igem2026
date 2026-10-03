from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from projects.active.fibre.evaluation.correspondence import CACHE, PG, RG, intrinsic_geodesic
from scipy.sparse import load_npz

ROOT=Path(__file__).resolve().parents[4]
PROM=ROOT/"results/fibre_promiscuity_holdout_v1/pair_metrics.csv"
OUT=ROOT/"results/fibre_local_fiber_geometry_v1"


def _finite_spearman(x: np.ndarray,y: np.ndarray) -> dict:
    mask=np.isfinite(x)&np.isfinite(y)
    if int(mask.sum())<3:
        return {"n":int(mask.sum()),"rho":None,"pvalue":None}
    result=spearmanr(x[mask],y[mask])
    return {
        "n":int(mask.sum()),
        "rho":float(result.statistic),
        "pvalue":float(result.pvalue),
    }


def main() -> None:
    proteins=pd.read_csv(CACHE/"protein_entities.csv",dtype=str).fillna("")
    reactions=pd.read_csv(CACHE/"reaction_entities.csv",dtype=str).fillna("")
    pairs=pd.read_csv(CACHE/"marts_pair_folds.csv",dtype=str).fillna("").drop_duplicates(["rhea_id","Entry"])
    pids=proteins.protein_id.astype(str).tolist()
    rids=reactions.reaction_id.astype(str).tolist()
    pi={x:i for i,x in enumerate(pids)}
    ri={x:i for i,x in enumerate(rids)}
    Dr,_=intrinsic_geodesic(load_npz(RG/"partial_pullback_affinity.npz"))
    De,_=intrinsic_geodesic(load_npz(PG/"partial_pullback_affinity.npz"))
    Dr2=np.square(Dr,dtype=np.float64)
    De2=np.square(De,dtype=np.float64)

    omega=np.asarray(
        [(ri[str(r.rhea_id)],pi[str(r.Entry)]) for r in pairs.itertuples(index=False)],
        dtype=np.int64,
    )
    by_enzyme={
        pid:sorted(set(g.rhea_id.astype(str)))
        for pid,g in pairs.groupby("Entry",sort=True)
    }
    frame=pd.read_csv(PROM,dtype={"enzyme_id":str,"heldout_reaction_id":str})
    nearest_same=[]
    joint_cover=[]
    marginal_r=[]
    marginal_e=[]
    for row in frame.itertuples(index=False):
        e=pi[str(row.enzyme_id)]
        r=ri[str(row.heldout_reaction_id)]
        remaining=[
            ri[x] for x in by_enzyme[str(row.enzyme_id)]
            if x!=str(row.heldout_reaction_id)
        ]
        nearest_same.append(
            float(np.min(Dr[r,remaining])) if remaining else np.nan
        )
        keep=~((omega[:,0]==r)&(omega[:,1]==e))
        support=omega[keep]
        costs=Dr2[r,support[:,0]]+De2[e,support[:,1]]
        joint_cover.append(float(np.sqrt(np.min(costs))))
        marginal_r.append(float(np.sqrt(np.min(Dr2[r,np.unique(support[:,0])]))))
        marginal_e.append(float(np.sqrt(np.min(De2[e,np.unique(support[:,1])]))))

    frame["nearest_remaining_same_enzyme_reaction_distance"]=nearest_same
    frame["leave_one_out_joint_relation_distance"]=joint_cover
    frame["leave_one_out_reaction_marginal_distance"]=marginal_r
    frame["leave_one_out_enzyme_marginal_distance"]=marginal_e
    frame["warm_rank_improvement"]=frame.cold_expected_rank-frame.warm_expected_rank

    OUT.mkdir(parents=True,exist_ok=True)
    frame.to_csv(OUT/"pair_geometry.csv",index=False)

    # Equal-count quartiles are descriptive only; they are not thresholds used by FIBRE.
    frame["same_enzyme_distance_quartile"]=pd.qcut(
        frame.nearest_remaining_same_enzyme_reaction_distance,
        q=4,labels=False,duplicates="drop",
    )
    bins=[]
    for q,g in frame.groupby("same_enzyme_distance_quartile",sort=True):
        bins.append({
            "quartile":int(q)+1,
            "n":int(len(g)),
            "distance_min":float(g.nearest_remaining_same_enzyme_reaction_distance.min()),
            "distance_median":float(g.nearest_remaining_same_enzyme_reaction_distance.median()),
            "distance_max":float(g.nearest_remaining_same_enzyme_reaction_distance.max()),
            "warm_expected_rank_median":float(g.warm_expected_rank.median()),
            "warm_expected_rr_mean":float(g.warm_expected_rr.mean()),
            "warm_better_fraction":float((g.warm_expected_rank<g.cold_expected_rank).mean()),
        })
    pd.DataFrame(bins).to_csv(OUT/"same_enzyme_distance_quartiles.csv",index=False)

    joint=np.asarray(frame.leave_one_out_joint_relation_distance,dtype=float)
    same=np.asarray(frame.nearest_remaining_same_enzyme_reaction_distance,dtype=float)
    warmrank=np.asarray(frame.warm_expected_rank,dtype=float)
    improvement=np.asarray(frame.warm_rank_improvement,dtype=float)
    summary={
        "schema":"fibre-local-fiber-geometry-audit-v1",
        "role":"assumption diagnostic; not a ranking calibration surface",
        "heldout_verified_positives":int(len(frame)),
        "leave_one_out_relation_coverage":{
            "median_product_distance":float(np.median(joint)),
            "p90_product_distance":float(np.quantile(joint,0.9)),
            "max_product_distance":float(np.max(joint)),
            "interpretation":(
                "empirical positive-only leave-one-out sampling radius on observed MARTS "
                "pairs; not a bound on the unobserved true catalytic relation"
            ),
        },
        "same_enzyme_locality":{
            "median_nearest_other_activity_reaction_distance":float(np.median(same)),
            "p90_nearest_other_activity_reaction_distance":float(np.quantile(same,0.9)),
            "distance_vs_warm_expected_rank":_finite_spearman(same,warmrank),
            "distance_vs_rank_improvement":_finite_spearman(same,improvement),
        },
        "joint_relation_distance_vs_warm_expected_rank":_finite_spearman(joint,warmrank),
        "quartiles":bins,
        "policy":(
            "tests whether positive-only local continuation behavior is associated with "
            "intrinsic locality. It does not estimate the unknown Hausdorff-Lipschitz "
            "constant L and does not turn distance quartiles into deployment thresholds."
        ),
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
