from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from projects.active.fibre.evaluation.correspondence import (
    CACHE,PG,RG,BUD,bools,intrinsic_geodesic,pair_array,
)
from projects.active.fibre.geometry.correspondence import (
    reaction_to_protein_section,
    protein_to_reaction_section,
)
from projects.active.fibre.geometry.levelset import (
    section_applicability,
    tie_aware_best_positive_rank,
)
from projects.active.fibre.runtime.base_model import rank_metrics

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/"results/fibre_levelset_uncertainty_dev_v1"


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
            pp=pair_array(train,ri,pi)
            splits.append({
                "split_id":split_id,
                "protein_overlap":len(set(train.Entry)&set(test.Entry)),
                "reaction_overlap":len(set(train.rhea_id)&set(test.rhea_id)),
            })
            for rid,g in test.groupby("rhea_id",sort=True):
                r=ri[rid]
                sec=reaction_to_protein_section(Dr2,De2,pp,r)
                pos=np.asarray([pi[x] for x in g.Entry.astype(str)],dtype=int)
                tie=tie_aware_best_positive_rank(sec.defect,pos)
                app=section_applicability(sec)
                conventional=rank_metrics(
                    sec.field,pids,set(g.Entry.astype(str)),set(),BUD
                )
                rows.append({
                    "split_id":split_id,"direction":"reaction_to_enzyme",
                    "query_id":rid,
                    "conventional_best_positive_rank":conventional["best_positive_rank"],
                    "conventional_reciprocal_rank":conventional["reciprocal_rank"],
                    "tie_optimistic_rank":tie.optimistic_rank,
                    "tie_expected_rank":tie.expected_rank,
                    "tie_pessimistic_rank":tie.pessimistic_rank,
                    "tie_expected_reciprocal_rank":tie.expected_reciprocal_rank,
                    "positive_count_in_best_level":tie.positive_count_in_tie,
                    "best_positive_level_size":tie.tie_size,
                    "query_support_distance":app.query_support_distance,
                    "best_level_size":app.best_level_size,
                    "best_level_fraction":app.best_level_fraction,
                    "next_level_gap":app.next_level_gap,
                    "best_level_candidate_support_distance_min":
                        app.best_level_candidate_support_distance_min,
                    "best_level_candidate_support_distance_median":
                        app.best_level_candidate_support_distance_median,
                    "numerical_tolerance":app.numerical_tolerance,
                })
            for pid,g in test.groupby("Entry",sort=True):
                e=pi[pid]
                sec=protein_to_reaction_section(Dr2,De2,pp,e)
                pos=np.asarray([ri[x] for x in g.rhea_id.astype(str)],dtype=int)
                tie=tie_aware_best_positive_rank(sec.defect,pos)
                app=section_applicability(sec)
                conventional=rank_metrics(
                    sec.field,rids,set(g.rhea_id.astype(str)),set(),BUD
                )
                rows.append({
                    "split_id":split_id,"direction":"enzyme_to_reaction",
                    "query_id":pid,
                    "conventional_best_positive_rank":conventional["best_positive_rank"],
                    "conventional_reciprocal_rank":conventional["reciprocal_rank"],
                    "tie_optimistic_rank":tie.optimistic_rank,
                    "tie_expected_rank":tie.expected_rank,
                    "tie_pessimistic_rank":tie.pessimistic_rank,
                    "tie_expected_reciprocal_rank":tie.expected_reciprocal_rank,
                    "positive_count_in_best_level":tie.positive_count_in_tie,
                    "best_positive_level_size":tie.tie_size,
                    "query_support_distance":app.query_support_distance,
                    "best_level_size":app.best_level_size,
                    "best_level_fraction":app.best_level_fraction,
                    "next_level_gap":app.next_level_gap,
                    "best_level_candidate_support_distance_min":
                        app.best_level_candidate_support_distance_min,
                    "best_level_candidate_support_distance_median":
                        app.best_level_candidate_support_distance_median,
                    "numerical_tolerance":app.numerical_tolerance,
                })

    q=pd.DataFrame(rows);sp=pd.DataFrame(splits)
    OUT.mkdir(parents=True,exist_ok=True)
    q.to_csv(OUT/"query_metrics.csv",index=False)
    summary={
        "version":"fibre-levelset-uncertainty-dev-v1",
        "partition":"development_only",
        "operator":"canonical zero-temperature FIBRE correspondence defect",
        "level_definition":"64 float64 machine eps at whole-section scale; no candidate-ID scientific ordering",
        "tie_model":"neutral uniform ordering within the best positive numerical level set",
        "applicability":"threshold-free intrinsic support distance, best-level ambiguity, and next distinct defect gap",
        "all_train_test_entity_overlaps_zero":bool(
            ((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()
        ),
        "directions":{},
    }
    for direction,g in q.groupby("direction"):
        width=g.tie_pessimistic_rank-g.tie_optimistic_rank
        summary["directions"][direction]={
            "queries":int(len(g)),
            "conventional_mrr":float(g.conventional_reciprocal_rank.mean()),
            "tie_expected_mrr":float(g.tie_expected_reciprocal_rank.mean()),
            "median_expected_rank":float(g.tie_expected_rank.median()),
            "nontrivial_rank_interval_fraction":float(np.mean(width>0)),
            "median_rank_interval_width":float(np.median(width)),
            "p90_rank_interval_width":float(np.quantile(width,.9)),
            "median_best_level_size":float(g.best_level_size.median()),
            "p90_best_level_size":float(g.best_level_size.quantile(.9)),
            "median_query_support_distance":float(g.query_support_distance.median()),
            "spearman_support_distance_vs_conventional_rr":float(
                g.query_support_distance.corr(
                    g.conventional_reciprocal_rank,method="spearman"
                )
            ),
            "spearman_best_level_size_vs_conventional_rr":float(
                g.best_level_size.corr(
                    g.conventional_reciprocal_rank,method="spearman"
                )
            ),
        }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
