from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CANONICAL = (
    ROOT / "results/terpene_wetlab_plate_balanced/canonical_balanced_assay_manifest.csv"
)
DEFAULT_RESCUE = (
    ROOT / "results/terpene_wetlab_plate_balanced/uniprot_balanced_assay_manifest.csv"
)
DEFAULT_PLATE_BALANCE = ROOT / "results/terpene_wetlab_plate_balanced/summary.json"
DEFAULT_OUTPUT = ROOT / "results/terpene_wetlab_randomized_layout"


def stable_jitter(seed: int, campaign: str, reaction_id: str, candidate_id: str, slot: str) -> float:
    value = f"{seed}|{campaign}|{reaction_id}|{candidate_id}|{slot}"
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)


def well_parts(well: str) -> tuple[str, int]:
    return str(well)[0], int(str(well)[1:])


def relative_slot(well: str, candidate_columns: list[int]) -> str:
    row, column = well_parts(well)
    local_column = candidate_columns.index(column) + 1
    return f"{row}C{local_column}"


def normalized_entropy(counts: pd.Series, total_slots: int) -> float:
    values = counts[counts > 0].astype(float)
    if values.empty or total_slots <= 1:
        return 1.0
    probabilities = values / values.sum()
    return float(-(probabilities * np.log2(probabilities)).sum() / math.log2(total_slots))


def role_slot_audit(frame: pd.DataFrame, candidate_role: str, role_column: str, label: str) -> pd.DataFrame:
    working = frame[frame["assay_role"].eq(candidate_role)].copy()
    working["row_position"] = working["well"].str[0]
    rows=[]
    for role,group in working.groupby(role_column,sort=True):
        counts=group["relative_candidate_slot"].value_counts()
        rows.append({
            "campaign_scope":label,
            "candidate_role":candidate_role,
            "selection_role":role,
            "n_candidates":len(group),
            "available_slots":working["relative_candidate_slot"].nunique(),
            "occupied_slots":counts.size,
            "maximum_slot_count":int(counts.max()),
            "minimum_slot_count_including_zero":int(counts.reindex(sorted(working["relative_candidate_slot"].unique()),fill_value=0).min()),
            "slot_count_range":int(counts.max()-counts.reindex(sorted(working["relative_candidate_slot"].unique()),fill_value=0).min()),
            "maximum_slot_share":float(counts.max()/len(group)),
            "normalized_slot_entropy":normalized_entropy(counts,working["relative_candidate_slot"].nunique()),
            "occupied_rows":group["row_position"].nunique(),
        })
    return pd.DataFrame(rows)


def randomize_campaign(
    frame: pd.DataFrame,
    campaign_scope: str,
    candidate_role: str,
    role_column: str,
    seed: int,
) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    result=frame.copy().fillna("")
    result["original_well"]=result["well"].astype(str)
    result["randomization_seed"]=seed
    result["randomization_method"]="sequential_hungarian_role_slot_balance_v1"
    result["relative_candidate_slot"]=""
    candidate_mask=result["assay_role"].eq(candidate_role)
    slot_counts: dict[str,dict[str,int]]=defaultdict(lambda:defaultdict(int))
    row_counts: dict[str,dict[str,int]]=defaultdict(lambda:defaultdict(int))
    assignments=[]
    group_columns=["plate_id","reaction_id"]
    groups=[]
    for keys,group in result[candidate_mask].groupby(group_columns,sort=False):
        order=pd.to_numeric(group["reaction_order"],errors="coerce").min()
        groups.append((float(order) if pd.notna(order) else 1e9,str(keys[0]),str(keys[1]),group.index.tolist()))
    groups.sort()
    for _,plate_id,reaction_id,indices in groups:
        candidates=result.loc[indices].copy()
        wells=sorted(candidates["well"].astype(str),key=lambda value:(well_parts(value)[1],well_parts(value)[0]))
        columns=sorted({well_parts(value)[1] for value in wells})
        slots={well:relative_slot(well,columns) for well in wells}
        candidates=candidates.sort_values([role_column,"candidate_id","original_well"]).reset_index()
        cost=np.zeros((len(candidates),len(wells)),dtype=float)
        for i,row in candidates.iterrows():
            role=str(row[role_column])
            for j,well in enumerate(wells):
                slot=slots[well]; row_position=slot[0]
                cost[i,j]=(
                    1000.0*slot_counts[role][slot]
                    +100.0*row_counts[role][row_position]
                    +10.0*sum(slot_counts[other][slot] for other in slot_counts if other==role)
                    +stable_jitter(seed,campaign_scope,reaction_id,str(row["candidate_id"]),slot)
                )
        candidate_indices,slot_indices=linear_sum_assignment(cost)
        for i,j in zip(candidate_indices,slot_indices):
            source_index=int(candidates.iloc[int(i)]["index"])
            well=wells[int(j)]; slot=slots[well]; role=str(candidates.iloc[int(i)][role_column])
            result.at[source_index,"well"]=well
            result.at[source_index,"relative_candidate_slot"]=slot
            slot_counts[role][slot]+=1
            row_counts[role][slot[0]]+=1
            assignments.append({
                "campaign_scope":campaign_scope,"plate_id":plate_id,"reaction_id":reaction_id,
                "candidate_id":str(candidates.iloc[int(i)]["candidate_id"]),"selection_role":role,
                "original_well":str(candidates.iloc[int(i)]["original_well"]),"randomized_well":well,
                "relative_candidate_slot":slot,
            })
    # Controls and blanks keep their original well. Derive their relative block slot for audit only.
    for (plate_id,reaction_id),group in result.groupby(group_columns,sort=False):
        candidate_wells=result.loc[group.index.intersection(result[candidate_mask].index),"well"].astype(str).tolist()
        candidate_columns=sorted({well_parts(value)[1] for value in candidate_wells})
        for index in group.index:
            if result.at[index,"relative_candidate_slot"]:
                continue
            well=str(result.at[index,"well"])
            column=well_parts(well)[1]
            if column in candidate_columns:
                result.at[index,"relative_candidate_slot"]=relative_slot(well,candidate_columns)
            else:
                result.at[index,"relative_candidate_slot"]=f"{well[0]}Ccontrol"
    duplicate=result.duplicated(["plate_id","well"])
    if duplicate.any():
        raise ValueError(f"Randomization created duplicate wells: {result.loc[duplicate,['plate_id','well']].to_dict('records')[:20]}")
    non_candidates=~candidate_mask
    if not result.loc[non_candidates,"well"].equals(result.loc[non_candidates,"original_well"]):
        raise AssertionError("Control or blank wells moved during candidate randomization")
    before=frame.copy().fillna("")
    before["relative_candidate_slot"]=""
    for (plate_id,reaction_id),group in before.groupby(group_columns,sort=False):
        candidate_wells=group.loc[group["assay_role"].eq(candidate_role),"well"].astype(str).tolist()
        columns=sorted({well_parts(value)[1] for value in candidate_wells})
        for index in group.index:
            if before.at[index,"assay_role"]==candidate_role:
                before.at[index,"relative_candidate_slot"]=relative_slot(str(before.at[index,"well"]),columns)
    before_audit=role_slot_audit(before,candidate_role,role_column,campaign_scope).assign(layout="before")
    after_audit=role_slot_audit(result,candidate_role,role_column,campaign_scope).assign(layout="after")
    return result,pd.DataFrame(assignments),pd.concat([before_audit,after_audit],ignore_index=True)


def main() -> None:
    parser=argparse.ArgumentParser(description="Balance candidate-selection roles across fixed reaction-block wells while leaving controls unchanged.")
    parser.add_argument("--canonical-manifest",type=Path,default=DEFAULT_CANONICAL)
    parser.add_argument("--rescue-manifest",type=Path,default=DEFAULT_RESCUE)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--plate-balance-summary",type=Path,default=DEFAULT_PLATE_BALANCE)
    parser.add_argument("--seed",type=int,default=20260723)
    args=parser.parse_args()
    output_dir=args.output_dir.resolve(); output_dir.mkdir(parents=True,exist_ok=True)
    canonical=pd.read_csv(args.canonical_manifest,dtype=str).fillna("")
    rescue=pd.read_csv(args.rescue_manifest,dtype=str).fillna("")
    randomized_c,assign_c,audit_c=randomize_campaign(canonical,"canonical_discovery","discovery_candidate","panel_role",args.seed)
    randomized_r,assign_r,audit_r=randomize_campaign(rescue,"uniprot_rescue","uniprot_rescue_candidate","rescue_role",args.seed)
    randomized_c.to_csv(output_dir/"canonical_randomized_assay_manifest.csv",index=False)
    randomized_r.to_csv(output_dir/"uniprot_randomized_assay_manifest.csv",index=False)
    assignments=pd.concat([assign_c,assign_r],ignore_index=True)
    assignments.to_csv(output_dir/"candidate_well_assignments.csv",index=False)
    audit=pd.concat([audit_c,audit_r],ignore_index=True)
    audit.to_csv(output_dir/"role_slot_balance_audit.csv",index=False)
    for frame in [randomized_c,randomized_r]:
        for plate_id,group in frame.groupby("plate_id",sort=True):
            group.sort_values("well",key=lambda values:values.map(lambda value:(well_parts(value)[1],well_parts(value)[0]))).to_csv(output_dir/f"{plate_id}_randomized_layout.csv",index=False)
    before=audit[audit.layout.eq('before')]; after=audit[audit.layout.eq('after')]
    merged=before.merge(after,on=["campaign_scope","candidate_role","selection_role"],suffixes=("_before","_after"))
    plate_balance = (
        json.loads(args.plate_balance_summary.read_text(encoding="utf-8"))
        if args.plate_balance_summary.exists()
        else {}
    )
    summary={
        "seed":args.seed,
        "reaction_plate_balance":plate_balance,
        "reaction_plate_layout_input":"capacity_constrained_milp_v1",
        "method":"sequential_hungarian_role_slot_balance_v1",
        "control_and_blank_wells_moved":0,
        "candidate_assignments":len(assignments),
        "canonical_candidates":len(assign_c),
        "uniprot_rescue_candidates":len(assign_r),
        "mean_normalized_entropy_before":float(merged.normalized_slot_entropy_before.mean()),
        "mean_normalized_entropy_after":float(merged.normalized_slot_entropy_after.mean()),
        "maximum_slot_share_before":float(merged.maximum_slot_share_before.max()),
        "maximum_slot_share_after":float(merged.maximum_slot_share_after.max()),
        "maximum_role_slot_count_range_before":int(merged.slot_count_range_before.max()),
        "maximum_role_slot_count_range_after":int(merged.slot_count_range_after.max()),
        "all_candidate_ids_preserved":bool(set(canonical.loc[canonical.assay_role.eq('discovery_candidate'),'candidate_id'])==set(randomized_c.loc[randomized_c.assay_role.eq('discovery_candidate'),'candidate_id']) and set(rescue.loc[rescue.assay_role.eq('uniprot_rescue_candidate'),'candidate_id'])==set(randomized_r.loc[randomized_r.assay_role.eq('uniprot_rescue_candidate'),'candidate_id'])),
        "outputs":{
            "canonical_manifest":str(output_dir/"canonical_randomized_assay_manifest.csv"),
            "uniprot_manifest":str(output_dir/"uniprot_randomized_assay_manifest.csv"),
            "assignments":str(output_dir/"candidate_well_assignments.csv"),
            "audit":str(output_dir/"role_slot_balance_audit.csv"),
        },
    }
    (output_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2)); print(audit.to_string(index=False))


if __name__=="__main__":
    main()
