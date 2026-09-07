from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE = ROOT / "data/catalyst_candidate_universes/general_merged/proteins"
DEFAULT_AUX = ROOT / "data/external/enzgfm_current/clean2023_650m_mean"
DEFAULT_OUTPUT = ROOT / "data/external/enzgfm_current/clean2023_esmc_enzgfm_equalblock"


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def load_library(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    entries = pd.read_csv(path / "entries.csv", dtype=str).fillna("")
    if "row" not in entries:
        raise ValueError(f"missing row column: {path}")
    id_cols = [column for column in ["Entry", "protein_id"] if column in entries]
    if len(id_cols) != 1:
        raise ValueError(f"expected exactly one protein ID column under {path}; got {id_cols}")
    entries["row"] = pd.to_numeric(entries["row"]).astype(int)
    entries = entries.sort_values("row").reset_index(drop=True)
    matrix = np.load(path / "embeddings.npy", mmap_mode="r")
    if len(entries) != len(matrix):
        raise ValueError(f"entries/matrix length mismatch: {path}")
    entries = entries.rename(columns={id_cols[0]: "Entry"})
    if entries["Entry"].duplicated().any():
        raise ValueError(f"duplicate protein IDs: {path}")
    return entries, matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate two protein feature libraries after independent row normalization so each block has equal norm.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--aux-dir", type=Path, default=DEFAULT_AUX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    base_dir=args.base_dir.resolve(); aux_dir=args.aux_dir.resolve(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    aux_manifest = aux_dir / "manifest.json"
    if not aux_manifest.is_file():
        raise RuntimeError(f"auxiliary feature manifest missing; extraction is not finalized: {aux_manifest}")
    completed_path = aux_dir / "completed.npy"
    if completed_path.is_file():
        completed = np.load(completed_path, mmap_mode="r")
        if not bool(np.asarray(completed).all()):
            raise RuntimeError(f"auxiliary feature extraction incomplete: {int(np.asarray(completed).sum())}/{len(completed)}")
    base_entries,base=load_library(base_dir); aux_entries,aux=load_library(aux_dir)
    base_row=dict(zip(base_entries.Entry.astype(str),base_entries.row.astype(int)))
    ids=aux_entries.Entry.astype(str).tolist()
    missing=[value for value in ids if value not in base_row]
    if missing:
        raise ValueError(f"base library misses {len(missing)} auxiliary IDs; examples={missing[:10]}")
    rows=np.asarray([base_row[value] for value in ids],dtype=np.int64)
    output=np.lib.format.open_memmap(out/"embeddings.npy",mode="w+",dtype=np.float32,shape=(len(ids),base.shape[1]+aux.shape[1]))
    for start in range(0,len(ids),args.chunk_size):
        stop=min(start+args.chunk_size,len(ids))
        output[start:stop,:base.shape[1]]=normalize_rows(np.asarray(base[rows[start:stop]],dtype=np.float32))
        output[start:stop,base.shape[1]:]=normalize_rows(np.asarray(aux[start:stop],dtype=np.float32))
    output.flush()
    pd.DataFrame({"row":np.arange(len(ids),dtype=np.int64),"Entry":ids}).to_csv(out/"entries.csv",index=False)
    manifest={
        "version":"equal-block-normalized-protein-concat-v1",
        "base_dir":str(base_dir),
        "aux_dir":str(aux_dir),
        "protein_count":len(ids),
        "base_dimension":int(base.shape[1]),
        "aux_dimension":int(aux.shape[1]),
        "feature_dimension":int(base.shape[1]+aux.shape[1]),
        "combination":"independent row L2 normalization per block, concatenate; downstream loader may normalize the combined row again",
        "id_order":"auxiliary library entries order",
    }
    manifest["aux_manifest"]=json.loads(aux_manifest.read_text(encoding="utf-8"))
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__": main()
