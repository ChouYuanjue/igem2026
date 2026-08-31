from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FULL = ROOT / "data/catalyst_candidate_universes/general_merged/proteins"
DEFAULT_EXISTING = ROOT / "data/external/enzgfm_current/clean2023_650m_mean"
DEFAULT_SCOPE = ROOT / "data/external/enzgfm_current/general_merged_missing_scope_v1.csv"


def sha256(path: Path) -> str:
    d=hashlib.sha256(); d.update(path.read_bytes()); return d.hexdigest()


def id_column(frame: pd.DataFrame) -> str:
    cols=[c for c in ["Entry","protein_id"] if c in frame]
    if len(cols)!=1: raise ValueError(f"expected one protein ID column, got {cols}")
    return cols[0]


def main() -> None:
    p=argparse.ArgumentParser(description="Freeze the label-free EnzGFM completion scope as full registered proteins minus the already embedded clean2023 library.")
    p.add_argument("--full-dir",type=Path,default=DEFAULT_FULL)
    p.add_argument("--existing-dir",type=Path,default=DEFAULT_EXISTING)
    p.add_argument("--output-csv",type=Path,default=DEFAULT_SCOPE)
    args=p.parse_args()
    full_path=args.full_dir.resolve()/"entries.csv"; existing_path=args.existing_dir.resolve()/"entries.csv"
    full=pd.read_csv(full_path,dtype=str).fillna(""); existing=pd.read_csv(existing_path,dtype=str).fillna("")
    fc=id_column(full); ec=id_column(existing)
    full_ids=full[fc].astype(str).tolist(); existing_ids=set(existing[ec].astype(str))
    if len(full_ids)!=len(set(full_ids)): raise ValueError("full registry contains duplicate protein IDs")
    extra=existing_ids-set(full_ids)
    if extra: raise ValueError(f"existing EnzGFM has {len(extra)} IDs outside full registry")
    missing=[value for value in full_ids if value not in existing_ids]
    out=args.output_csv.resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"protein_id":missing}).to_csv(out,index=False)
    manifest={
        "version":"enzgfm-general-merged-missing-scope-v1",
        "outer_labels_used":False,
        "definition":"all proteins in the registered full candidate library minus proteins already present in clean2023 EnzGFM; independent of every benchmark/test split",
        "full_entries":str(full_path),"full_entries_sha256":sha256(full_path),
        "existing_entries":str(existing_path),"existing_entries_sha256":sha256(existing_path),
        "full_protein_count":len(full_ids),"existing_protein_count":len(existing_ids),"missing_protein_count":len(missing),
        "output_csv":str(out),"output_csv_sha256":sha256(out),
    }
    (out.with_suffix('.manifest.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
