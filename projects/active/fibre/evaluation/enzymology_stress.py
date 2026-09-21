from __future__ import annotations

from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[4]
PROMISCUITY=ROOT/"results/fibre_promiscuity_holdout_v1/pair_metrics.csv"
CATALYTIC=ROOT/"results/fibre_catalytic_state_v1/pair_states.jsonl"
ASSAY=ROOT/"results/fibre_assay_context_v1/assay_observations.jsonl"
OUT=ROOT/"results/fibre_enzymology_stress_v1"


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metrics(g: pd.DataFrame) -> dict:
    if g.empty:
        return {"n":0}
    return {
        "n":int(len(g)),
        "cold_expected_rank_median":float(g.cold_expected_rank.median()),
        "warm_expected_rank_median":float(g.warm_expected_rank.median()),
        "cold_expected_rr_mean":float(g.cold_expected_rr.mean()),
        "warm_expected_rr_mean":float(g.warm_expected_rr.mean()),
        "warm_better_fraction":float((g.warm_expected_rank<g.cold_expected_rank).mean()),
        "warm_equal_fraction":float(np.isclose(g.warm_expected_rank,g.cold_expected_rank).mean()),
        "warm_worse_fraction":float((g.warm_expected_rank>g.cold_expected_rank).mean()),
    }


def main() -> None:
    p=pd.read_csv(PROMISCUITY,dtype={"enzyme_id":str,"heldout_reaction_id":str})
    states=_jsonl(CATALYTIC)
    state={(str(x["enzyme_id"]),str(x["reaction_id"])):x for x in states}
    by_enzyme=defaultdict(list)
    for (enzyme,reaction),row in state.items():
        by_enzyme[enzyme].append((reaction,row))

    overlap=[]; source_class=[]
    for row in p.itertuples(index=False):
        target=state.get((str(row.enzyme_id),str(row.heldout_reaction_id)),{})
        target_types=set(target.get("mechanism_reaction_types") or [])
        remaining_types=set()
        for rid,s in by_enzyme.get(str(row.enzyme_id),[]):
            if rid!=str(row.heldout_reaction_id):
                remaining_types.update(s.get("mechanism_reaction_types") or [])
        if target_types and remaining_types:
            j=len(target_types&remaining_types)/len(target_types|remaining_types)
            overlap.append(float(j))
        else:
            overlap.append(np.nan)
        evidence=set(target.get("mechanism_source_evidence") or [])
        if "experiment" in evidence:
            source_class.append("experiment_present")
        elif evidence:
            source_class.append("nonexperimental_or_unspecified")
        else:
            source_class.append("mechanism_unresolved")
    p["mechanism_type_jaccard_to_remaining_activities"]=overlap
    p["target_mechanism_evidence_class"]=source_class

    def overlap_class(x):
        if not np.isfinite(x): return "unresolved"
        if x==0: return "disjoint_observed_step_types"
        if x>=0.75: return "high_overlap"
        return "partial_overlap"
    p["mechanism_overlap_class"]=[overlap_class(float(x)) for x in p.mechanism_type_jaccard_to_remaining_activities]
    OUT.mkdir(parents=True,exist_ok=True)
    p.to_csv(OUT/"promiscuity_mechanism_metrics.csv",index=False)

    mech_rows=[]
    for key,g in p.groupby("mechanism_overlap_class",sort=True):
        mech_rows.append({"mechanism_overlap_class":key,**_metrics(g)})
    pd.DataFrame(mech_rows).to_csv(OUT/"promiscuity_by_mechanism_overlap.csv",index=False)

    evidence_rows=[]
    for key,g in p.groupby("target_mechanism_evidence_class",sort=True):
        evidence_rows.append({"target_mechanism_evidence_class":key,**_metrics(g)})
    pd.DataFrame(evidence_rows).to_csv(OUT/"promiscuity_by_mechanism_evidence.csv",index=False)

    assays=_jsonl(ASSAY)
    assay_by_pair=defaultdict(list)
    explicit_negative=0
    for row in assays:
        assay_by_pair[(str(row["enzyme_id"]),str(row["reaction_id"]))].append(row)
        if str(row.get("outcome") or "") in {"inactive","below_detection","no_conversion"}:
            explicit_negative+=1
    repeated_context_pairs=sum(len(rows)>=2 for rows in assay_by_pair.values())

    high=p[p.mechanism_overlap_class.eq("high_overlap")]
    disjoint=p[p.mechanism_overlap_class.eq("disjoint_observed_step_types")]
    summary={
        "schema":"fibre-enzymology-stress-v1",
        "promiscuity":{
            "all":_metrics(p),
            "mechanism_overlap_coverage_fraction":float(
                np.isfinite(p.mechanism_type_jaccard_to_remaining_activities).mean()
            ),
            "high_mechanism_overlap":_metrics(high),
            "disjoint_observed_step_types":_metrics(disjoint),
            "mechanism_consistent_rescue_count":int(
                (
                    (p.mechanism_overlap_class=="high_overlap")
                    & (p.warm_expected_rank<p.cold_expected_rank)
                ).sum()
            ),
            "interpretation":(
                "mechanism annotations are diagnostic only; FIBRE ranking did not use "
                "these mechanism labels in this audit"
            ),
        },
        "context_shift":{
            "materialized_pair_assay_observations":len(assays),
            "pairs_with_two_or_more_materialized_context_observations":int(repeated_context_pairs),
            "evaluable":bool(repeated_context_pairs>0),
            "status":(
                "evaluable" if repeated_context_pairs>0
                else "blocked_by_pair_specific_assay_coverage"
            ),
        },
        "explicit_negative":{
            "materialized_pair_specific_negative_assays":int(explicit_negative),
            "functional_cliff_ground_truth_available":bool(explicit_negative>0),
            "status":(
                "evaluable" if explicit_negative>0
                else "blocked_without_manufacturing_unknown_pairs_as_negatives"
            ),
        },
        "cofactor_holdout":{
            "status":"not_promoted",
            "reason":(
                "protein-side UniProt cofactor coverage is broad, but a comparably "
                "scoped reaction/assay-side cofactor requirement is not available for "
                "enough canonical pairs; protein annotation alone cannot define a "
                "pair-specific cofactor compatibility ground truth"
            ),
        },
        "policy":(
            "run only stress tests supported by positive or explicitly scoped assay "
            "evidence; unavailable negatives/context shifts are reported as coverage "
            "limits rather than synthesized from missing database edges"
        ),
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
