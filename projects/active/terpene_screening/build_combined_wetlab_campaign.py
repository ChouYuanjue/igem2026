from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CANONICAL = (
    ROOT
    / "results/terpene_wetlab_randomized_layout/canonical_randomized_assay_manifest.csv"
)
DEFAULT_RESCUE = (
    ROOT
    / "results/terpene_wetlab_randomized_layout/uniprot_randomized_assay_manifest.csv"
)
DEFAULT_CANONICAL_RESULTS = (
    ROOT
    / "results/terpene_wetlab_randomized_layout/canonical_randomized_assay_results_template.csv"
)
DEFAULT_RESCUE_RESULTS = (
    ROOT
    / "results/terpene_wetlab_randomized_layout/uniprot_randomized_assay_results_template.csv"
)
DEFAULT_RANDOMIZATION_SUMMARY = (
    ROOT / "results/terpene_wetlab_randomized_layout/summary.json"
)
DEFAULT_OUTPUT = ROOT / "results/terpene_combined_wetlab_campaign"
PROTEIN_ROLES = {
    "discovery_candidate",
    "uniprot_rescue_candidate",
    "positive_control_primary",
    "positive_control_replicate",
}


def clean_sequence(value: object) -> str:
    return "".join(str(value).upper().split()).rstrip("*")


def stable_construct_id(sequence: str) -> str:
    return f"TPSMASTER_{hashlib.sha1(sequence.encode('utf-8')).hexdigest()[:12]}"


def normalize_manifest(frame: pd.DataFrame, campaign_scope: str, plate_offset: int) -> pd.DataFrame:
    result = frame.copy().fillna("")
    result.insert(0, "campaign_scope", campaign_scope)
    result.insert(1, "feedback_scope", campaign_scope)
    result.insert(2, "source_plate_id", result["plate_id"].astype(str))
    plate_order = {
        plate_id: plate_offset + index
        for index, plate_id in enumerate(sorted(result["plate_id"].astype(str).unique()), start=1)
    }
    result.insert(3, "master_plate_order", result["plate_id"].map(plate_order).astype(int))
    for column in [
        "panel_role",
        "rescue_role",
        "original_rank",
        "expanded_rank",
        "evidence_quality_tier",
        "domain_family",
        "candidate_source",
        "enzyme_name",
        "species",
        "sequence_length",
        "sequence_construct_id",
    ]:
        if column not in result:
            result[column] = ""
    result["sequence"] = result["sequence"].map(clean_sequence)
    result["sequence_length"] = result["sequence"].str.len().where(
        result["sequence"].ne(""), result["sequence_length"]
    )
    return result


def build_constructs(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    proteins = manifest[manifest["assay_role"].isin(PROTEIN_ROLES)].copy()
    if proteins["sequence"].eq("").any():
        missing = proteins.loc[proteins["sequence"].eq(""), ["plate_id", "well", "candidate_id"]]
        raise ValueError(f"Protein assay wells have missing sequences: {missing.head(20).to_dict('records')}")
    id_sequence_counts = proteins.groupby("candidate_id")["sequence"].nunique()
    conflicting = id_sequence_counts[id_sequence_counts > 1]
    if len(conflicting):
        raise ValueError(f"Candidate IDs map to multiple sequences: {conflicting.to_dict()}")
    rows=[]
    sequence_to_construct={}
    for sequence,group in proteins.groupby("sequence",sort=True):
        construct_id=stable_construct_id(sequence)
        sequence_to_construct[sequence]=construct_id
        scopes=sorted(set(group["campaign_scope"].astype(str)))
        roles=sorted(set(group["assay_role"].astype(str)))
        candidate_ids=sorted(set(group["candidate_id"].astype(str))-{""})
        reaction_ids=sorted(set(group["reaction_id"].astype(str))-{""})
        if set(scopes)=={"canonical_discovery","uniprot_rescue"}:
            scope_class="shared_between_campaigns"
        elif scopes==["canonical_discovery"]:
            scope_class="canonical_only"
        elif scopes==["uniprot_rescue"]:
            scope_class="uniprot_rescue_only"
        else:
            scope_class="other"
        discovery_roles={"discovery_candidate","uniprot_rescue_candidate"}
        usage_class=(
            "positive_control_only"
            if not (set(roles)&discovery_roles)
            else "discovery_and_control"
            if any(role.startswith("positive_control") for role in roles)
            else "discovery_candidate"
        )
        length=len(sequence)
        rows.append({
            "master_construct_id":construct_id,
            "candidate_id_aliases":";".join(candidate_ids),
            "n_candidate_ids":len(candidate_ids),
            "campaign_scopes":";".join(scopes),
            "campaign_scope_class":scope_class,
            "assay_roles":";".join(roles),
            "construct_usage_class":usage_class,
            "reaction_ids":";".join(reaction_ids),
            "reaction_count":len(reaction_ids),
            "assay_well_count":len(group),
            "plate_count":group["plate_id"].nunique(),
            "sequence_length":length,
            "coding_nucleotides_without_stop":3*length,
            "procurement_length_tier":(
                "standard_le_500aa" if length<=500 else "long_501_750aa" if length<=750 else "very_long_751_1000aa"
            ),
            "sequence":sequence,
        })
    constructs=pd.DataFrame(rows).sort_values(["procurement_length_tier","sequence_length","master_construct_id"]).reset_index(drop=True)
    return constructs,sequence_to_construct


def write_fasta(frame: pd.DataFrame,path: Path) -> None:
    with path.open("w",encoding="utf-8") as handle:
        for row in frame.itertuples(index=False):
            handle.write(
                f">{row.master_construct_id}|aliases={row.candidate_id_aliases}|scopes={row.campaign_scopes}|length={row.sequence_length}\n{row.sequence}\n"
            )


def main() -> None:
    parser=argparse.ArgumentParser(description="Build a sequence-deduplicated six-plate TPS wet-lab master campaign.")
    parser.add_argument("--canonical-manifest",type=Path,default=DEFAULT_CANONICAL)
    parser.add_argument("--rescue-manifest",type=Path,default=DEFAULT_RESCUE)
    parser.add_argument("--canonical-result-template",type=Path,default=DEFAULT_CANONICAL_RESULTS)
    parser.add_argument("--rescue-result-template",type=Path,default=DEFAULT_RESCUE_RESULTS)
    parser.add_argument("--randomization-summary",type=Path,default=DEFAULT_RANDOMIZATION_SUMMARY)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    output_dir=args.output_dir.resolve(); output_dir.mkdir(parents=True,exist_ok=True)
    canonical=normalize_manifest(pd.read_csv(args.canonical_manifest,dtype=str),"canonical_discovery",0)
    rescue=normalize_manifest(pd.read_csv(args.rescue_manifest,dtype=str),"uniprot_rescue",canonical["plate_id"].nunique())
    manifest=pd.concat([canonical,rescue],ignore_index=True,sort=False)
    duplicates=manifest.duplicated(["plate_id","well"])
    if duplicates.any():
        raise ValueError(f"Duplicate plate wells across campaigns: {manifest.loc[duplicates,['plate_id','well']].to_dict('records')[:20]}")
    constructs,sequence_to_construct=build_constructs(manifest)
    manifest["master_construct_id"]=manifest["sequence"].map(sequence_to_construct).fillna("")
    protein_wells=manifest["assay_role"].isin(PROTEIN_ROLES)
    if manifest.loc[protein_wells,"master_construct_id"].eq("").any():
        raise ValueError("Some protein wells lack a master construct ID")
    ordered_columns=[
        "campaign_scope","feedback_scope","master_plate_order","plate_id","well","reaction_order","reaction_id",
        "terpene_type","tps_class","substrate_name","product_name","assay_role","candidate_id","master_construct_id",
        "candidate_source","panel_role","rescue_role","original_rank","expanded_rank","evidence_quality_tier","domain_family",
        "sequence_length","sequence","source_plate_id","sequence_construct_id","enzyme_name","species",
    ]
    for column in ordered_columns:
        if column not in manifest: manifest[column]=""
    manifest=manifest[ordered_columns].sort_values(["master_plate_order","plate_id","well"]).reset_index(drop=True)
    manifest.to_csv(output_dir/"master_assay_manifest.csv",index=False)
    constructs.to_csv(output_dir/"master_sequence_constructs.csv",index=False)
    write_fasta(constructs,output_dir/"master_sequence_constructs.fasta")
    plate_summary=(manifest.groupby(["master_plate_order","campaign_scope","plate_id"])
        .agg(n_wells=("well","size"),n_reactions=("reaction_id","nunique"),protein_wells=("master_construct_id",lambda values:int(pd.Series(values).ne("").sum())),unique_constructs=("master_construct_id",lambda values:pd.Series(values)[pd.Series(values).ne("")].nunique()),discovery_wells=("assay_role",lambda values:int(pd.Series(values).isin(["discovery_candidate","uniprot_rescue_candidate"]).sum())),positive_control_wells=("assay_role",lambda values:int(pd.Series(values).astype(str).str.startswith("positive_control").sum())),negative_control_wells=("assay_role",lambda values:int(pd.Series(values).eq("empty_vector_negative").sum())),process_blank_wells=("assay_role",lambda values:int(pd.Series(values).eq("substrate_process_blank").sum())))
        .reset_index().sort_values("master_plate_order"))
    plate_summary.to_csv(output_dir/"master_plate_summary.csv",index=False)
    overlap=(constructs.groupby("campaign_scope_class").agg(unique_constructs=("master_construct_id","size"),total_amino_acids=("sequence_length","sum"),total_coding_nt=("coding_nucleotides_without_stop","sum")).reset_index())
    overlap.to_csv(output_dir/"campaign_sequence_overlap.csv",index=False)
    procurement=(constructs.groupby(["procurement_length_tier","construct_usage_class"])
        .agg(unique_constructs=("master_construct_id","size"),total_amino_acids=("sequence_length","sum"),total_coding_nt=("coding_nucleotides_without_stop","sum"),median_length=("sequence_length","median"),max_length=("sequence_length","max"))
        .reset_index())
    procurement.to_csv(output_dir/"procurement_summary.csv",index=False)
    # Keep feedback batches independent even though procurement is combined.
    feedback_scopes=pd.DataFrame([
        {"feedback_scope":"canonical_discovery","manifest":str(args.canonical_manifest.resolve()),"result_template":str(args.canonical_result_template.resolve()),"analysis_note":"Analyze separately from UniProt rescue to preserve batch-specific QC."},
        {"feedback_scope":"uniprot_rescue","manifest":str(args.rescue_manifest.resolve()),"result_template":str(args.rescue_result_template.resolve()),"analysis_note":"Analyze separately from canonical discovery to preserve batch-specific QC."},
    ])
    feedback_scopes.to_csv(output_dir/"feedback_scopes.csv",index=False)
    randomization = (
        json.loads(args.randomization_summary.read_text(encoding="utf-8"))
        if args.randomization_summary.exists()
        else {}
    )
    summary={
        "n_plates":int(manifest["plate_id"].nunique()),
        "n_wells":len(manifest),
        "campaign_wells":manifest["campaign_scope"].value_counts().to_dict(),
        "n_reactions":int(manifest["reaction_id"].nunique()),
        "reaction_campaign_overlap":int(manifest.groupby("reaction_id")["campaign_scope"].nunique().gt(1).sum()),
        "protein_assay_wells":int(protein_wells.sum()),
        "candidate_id_constructs":int(manifest.loc[protein_wells,"candidate_id"].nunique()),
        "sequence_deduplicated_constructs":len(constructs),
        "constructs_shared_between_campaigns":int(constructs["campaign_scope_class"].eq("shared_between_campaigns").sum()),
        "canonical_only_constructs":int(constructs["campaign_scope_class"].eq("canonical_only").sum()),
        "uniprot_rescue_only_constructs":int(constructs["campaign_scope_class"].eq("uniprot_rescue_only").sum()),
        "total_amino_acids":int(constructs["sequence_length"].sum()),
        "total_coding_nucleotides_without_stop":int(constructs["coding_nucleotides_without_stop"].sum()),
        "feedback_batches_must_remain_separate":True,
        "reaction_plate_layout":randomization.get("reaction_plate_layout_input","unknown"),
        "reaction_plate_balance":randomization.get("reaction_plate_balance",{}),
        "candidate_position_layout":"deterministic_role_balanced_randomization",
        "randomization_seed":int(randomization.get("seed",20260723)),
        "control_and_blank_positions_preserved":True,
        "codon_optimization_performed":False,
        "outputs":{
            "master_manifest":str(output_dir/"master_assay_manifest.csv"),
            "master_constructs":str(output_dir/"master_sequence_constructs.csv"),
            "master_fasta":str(output_dir/"master_sequence_constructs.fasta"),
            "plate_summary":str(output_dir/"master_plate_summary.csv"),
            "procurement_summary":str(output_dir/"procurement_summary.csv"),
            "feedback_scopes":str(output_dir/"feedback_scopes.csv"),
        },
    }
    (output_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2)); print(plate_summary.to_string(index=False)); print(overlap.to_string(index=False)); print(procurement.to_string(index=False))


if __name__=="__main__":
    main()
