from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "results/enzymecage_cleanroom_rdkitplus_v1"
DEFAULT_OUTPUT = ROOT / "results/cleanroom_internal_full_candidate_benchmarks_v1"


def audit(train: pd.DataFrame, dev: pd.DataFrame) -> dict[str, object]:
    train_pairs=set(map(tuple,train[["protein_id","reaction_id"]].itertuples(index=False,name=None)))
    dev_pairs=set(map(tuple,dev[["protein_id","reaction_id"]].itertuples(index=False,name=None)))
    train_p=set(train.protein_id); dev_p=set(dev.protein_id)
    train_r=set(train.reaction_id); dev_r=set(dev.reaction_id)
    return {
        "train_pairs":len(train_pairs),"test_pairs":len(dev_pairs),
        "train_proteins":len(train_p),"test_proteins":len(dev_p),
        "train_reactions":len(train_r),"test_reactions":len(dev_r),
        "exact_train_test_pair_overlap":len(train_pairs & dev_pairs),
        "test_protein_seen_fraction":len(train_p & dev_p)/max(1,len(dev_p)),
        "test_reaction_seen_fraction":len(train_r & dev_r)/max(1,len(dev_r)),
    }


def main() -> None:
    parser=argparse.ArgumentParser(description="Materialize existing clean2023 internal double-cold folds as full-candidate benchmark cells without reading any outer labels.")
    parser.add_argument("--source-root",type=Path,default=DEFAULT_SOURCE)
    parser.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--folds",default="0,1,2")
    args=parser.parse_args()
    source=args.source_root.resolve(); output=args.output_root.resolve(); output.mkdir(parents=True,exist_ok=True)
    cells=[]
    for fold in [int(x) for x in args.folds.split(',') if x.strip()]:
        src=source/f"fold{fold}"
        train=pd.read_csv(src/"training_pairs.csv",dtype=str).fillna("")[["protein_id","reaction_id"]].drop_duplicates()
        dev=pd.read_csv(src/"dev_pairs.csv",dtype=str).fillna("")[["protein_id","reaction_id"]].drop_duplicates()
        measured=audit(train,dev)
        violations=[]
        if measured["exact_train_test_pair_overlap"]: violations.append("exact_pair_overlap")
        if measured["test_protein_seen_fraction"]: violations.append("protein_overlap")
        if measured["test_reaction_seen_fraction"]: violations.append("reaction_overlap")
        name=f"clean2023_internal_double_cold_fold{fold}"
        dst=output/name; dst.mkdir(parents=True,exist_ok=True)
        train.to_csv(dst/"train_pairs.csv",index=False); dev.to_csv(dst/"test_pairs.csv",index=False)
        manifest={
            "name":name,
            "source_protocol":"clean2023 strict protein+reaction entity-disjoint hash fold used before any broad outer reveal",
            "claim_tier":"internal_development_only",
            "outer_benchmark_labels_used":False,
            "dev_fold":fold,"folds":5,
            "audit":measured,
            "valid":not violations,"violations":violations,
            "source_model_dir":str(src),
            "purpose":"full 185,918-protein / 11,081-reaction internal diagnostic for shortlist coverage and representation development; not an external benchmark",
        }
        (dst/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        cells.append(manifest)
        print(json.dumps(manifest),flush=True)
    (output/"summary.json").write_text(json.dumps({"protocol":"internal_full_candidate_cells_v1","outer_labels_used":False,"cells":cells},indent=2),encoding="utf-8")

if __name__=="__main__": main()
