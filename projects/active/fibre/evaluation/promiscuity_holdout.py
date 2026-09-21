from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from projects.active.fibre.evaluation.correspondence import (
    CACHE, PG, RG, intrinsic_geodesic, pair_array,
)
from projects.active.fibre.geometry.correspondence import protein_to_reaction_section
from projects.active.fibre.geometry.levelset import tie_aware_best_positive_rank

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/"results/fibre_promiscuity_holdout_v1"
BUDGETS=(10,20,50)


def _rank(defect: np.ndarray,target_local_index: int):
    return tie_aware_best_positive_rank(
        np.asarray(defect,dtype=np.float64),
        np.asarray([int(target_local_index)],dtype=np.int64),
    )


def _aggregate(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {}
    out={
        "heldout_positive_pairs":int(len(rows)),
        "multi_activity_enzymes":int(rows.enzyme_id.nunique()),
        "median_remaining_known_activities":float(rows.remaining_known_activities.median()),
        "cold_expected_rank_median":float(rows.cold_expected_rank.median()),
        "warm_novel_expected_rank_median":float(rows.warm_expected_rank.median()),
        "cold_expected_reciprocal_rank_mean":float(rows.cold_expected_rr.mean()),
        "warm_novel_expected_reciprocal_rank_mean":float(rows.warm_expected_rr.mean()),
        "warm_better_fraction":float((rows.warm_expected_rank < rows.cold_expected_rank).mean()),
        "warm_equal_fraction":float(np.isclose(rows.warm_expected_rank,rows.cold_expected_rank).mean()),
        "warm_worse_fraction":float((rows.warm_expected_rank > rows.cold_expected_rank).mean()),
    }
    for k in BUDGETS:
        out[f"cold_hit_at_{k}"]=float((rows.cold_optimistic_rank<=k).mean())
        out[f"warm_novel_hit_at_{k}"]=float((rows.warm_optimistic_rank<=k).mean())
    return out


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
    all_pairs=pair_array(pairs,ri,pi)

    multi={
        pid:sorted(set(g.rhea_id.astype(str)))
        for pid,g in pairs.groupby("Entry",sort=True)
        if g.rhea_id.nunique()>=2
    }
    records=[]
    for number,(pid,positive_rids) in enumerate(sorted(multi.items()),start=1):
        e=pi[pid]
        own_pair_set={(ri[r],e) for r in positive_rids}

        # Cold reference: remove every known activity of this enzyme.  This asks
        # how the molecular atlas behaves before any enzyme-specific activity
        # precedent is admitted.
        cold_pairs=np.asarray([
            row for row in all_pairs if tuple(map(int,row)) not in own_pair_set
        ],dtype=np.int64)
        cold=protein_to_reaction_section(Dr2,De2,cold_pairs,e)

        for target_rid in positive_rids:
            target=ri[target_rid]
            heldout=(target,e)

            # Warm leave-one-positive-out reference: every other verified
            # activity of this enzyme remains in Omega; only the target edge is
            # hidden. This is a positive-only promiscuity discovery audit.
            warm_pairs=np.asarray([
                row for row in all_pairs if tuple(map(int,row))!=heldout
            ],dtype=np.int64)
            remaining={ri[r] for r in positive_rids if r!=target_rid}
            candidates=np.asarray([
                i for i in range(len(rids)) if i not in remaining
            ],dtype=np.int64)
            warm=protein_to_reaction_section(
                Dr2,De2,warm_pairs,e,candidate_reaction_indices=candidates
            )
            target_local=int(np.flatnonzero(candidates==target)[0])
            warm_rank=_rank(warm.defect,target_local)
            cold_rank=_rank(cold.defect,target)
            records.append({
                "enzyme_id":pid,
                "heldout_reaction_id":target_rid,
                "known_activity_count":len(positive_rids),
                "remaining_known_activities":len(positive_rids)-1,
                "cold_optimistic_rank":cold_rank.optimistic_rank,
                "cold_expected_rank":cold_rank.expected_rank,
                "cold_pessimistic_rank":cold_rank.pessimistic_rank,
                "cold_expected_rr":cold_rank.expected_reciprocal_rank,
                "warm_optimistic_rank":warm_rank.optimistic_rank,
                "warm_expected_rank":warm_rank.expected_rank,
                "warm_pessimistic_rank":warm_rank.pessimistic_rank,
                "warm_expected_rr":warm_rank.expected_reciprocal_rank,
                "expected_rank_change_warm_minus_cold":(
                    warm_rank.expected_rank-cold_rank.expected_rank
                ),
                "warm_target_defect":float(warm.defect[target_local]),
                "cold_target_defect":float(cold.defect[target]),
                "evaluation_semantics":(
                    "heldout pair is a verified positive; other positive activities "
                    "of the same enzyme are retained only in the warm condition; "
                    "unobserved enzyme-reaction pairs are never labeled negative"
                ),
            })
        if number%50==0:
            print("processed enzymes",number,"records",len(records),flush=True)

    frame=pd.DataFrame(records)
    OUT.mkdir(parents=True,exist_ok=True)
    frame.to_csv(OUT/"pair_metrics.csv",index=False)

    by_activity=[]
    for n,g in frame.groupby("known_activity_count",sort=True):
        row={"known_activity_count":int(n),**_aggregate(g)}
        by_activity.append(row)
    pd.DataFrame(by_activity).to_csv(OUT/"by_activity_count.csv",index=False)

    summary={
        "schema":"fibre-promiscuity-holdout-v1",
        "role":"biological stress test; not a benchmark-selection surface",
        "positive_only":True,
        "negative_pairs_assumed":False,
        "protocol":(
            "for each enzyme with >=2 verified positive reactions, hold out one "
            "positive edge; compare a cold field with all query-enzyme activities "
            "removed against a warm field retaining the enzyme's other verified "
            "activities; remaining known activities are removed from the returned "
            "candidate list so the warm rank measures novel positive recovery"
        ),
        "metrics":_aggregate(frame),
        "interpretation":(
            "measures whether FIBRE's exact positive-set update can exploit observed "
            "enzyme promiscuity without manufacturing negatives"
        ),
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
