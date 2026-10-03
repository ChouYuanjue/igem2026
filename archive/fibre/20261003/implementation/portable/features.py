from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from drfp import DrfpEncoder

from .contracts import load_and_validate
from .encoders import canonical_or_raw_reaction, encode_esmc_mean


def _write_feature_csv(ids: list[str], id_column: str, matrix: np.ndarray, output: Path) -> None:
    x=np.asarray(matrix,dtype=np.float32)
    frame=pd.DataFrame(x,columns=[f"f{i}" for i in range(x.shape[1])])
    frame.insert(0,id_column,ids)
    output.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output,index=False)


def build_protein_features(
    proteins_path: str | Path,
    output_path: str | Path,
    *,
    model_name: str="esmc_600m",
    device: str="cpu",
    cache_dir: str | Path | None=None,
) -> dict:
    proteins=pd.read_csv(proteins_path,dtype=str).fillna("")
    if not {"protein_id","sequence"}.issubset(proteins.columns):
        raise ValueError("proteins table requires protein_id and sequence")
    if proteins.protein_id.duplicated().any():
        raise ValueError("protein_id must be unique")
    cache=Path(cache_dir) if cache_dir is not None else None
    matrix,audits=encode_esmc_mean(
        proteins.sequence.astype(str).tolist(),
        device=device,model_name=model_name,cache_dir=cache,
    )
    _write_feature_csv(
        proteins.protein_id.astype(str).tolist(),
        "protein_id",matrix,Path(output_path),
    )
    return {
        "encoder":"ESM-C mean embedding",
        "model":model_name,
        "device":device,
        "rows":len(proteins),
        "dimension":int(matrix.shape[1]),
        "audit_status_counts":pd.Series([x["status"] for x in audits]).value_counts().to_dict(),
    }


def build_reaction_features(
    reactions_path: str | Path,
    output_path: str | Path,
) -> dict:
    reactions=pd.read_csv(reactions_path,dtype=str).fillna("")
    if not {"reaction_id","reaction_smiles"}.issubset(reactions.columns):
        raise ValueError("reactions table requires reaction_id and reaction_smiles")
    if reactions.reaction_id.duplicated().any():
        raise ValueError("reaction_id must be unique")
    rows=[]
    for raw in reactions.reaction_smiles.astype(str):
        canonical=canonical_or_raw_reaction(raw)
        if ">>" not in canonical:
            raise ValueError(f"reaction lacks directed arrow after canonicalization: {raw}")
        rows.append(np.asarray(DrfpEncoder.encode([canonical])[0],dtype=np.float32))
    matrix=np.stack(rows)
    norm=np.linalg.norm(matrix,axis=1,keepdims=True)
    if np.any(norm[:,0]<=1e-12):
        raise ValueError("DRFP returned a zero reaction feature vector")
    matrix=(matrix/norm).astype(np.float32)
    _write_feature_csv(
        reactions.reaction_id.astype(str).tolist(),
        "reaction_id",matrix,Path(output_path),
    )
    return {
        "encoder":"DRFP",
        "rows":len(reactions),
        "dimension":int(matrix.shape[1]),
        "canonicalization":"same canonical_or_raw_reaction path as FIBRE production runtime",
    }


def main() -> None:
    ap=argparse.ArgumentParser(description="Compute generic global FIBRE features.")
    sub=ap.add_subparsers(dest="kind",required=True)
    p=sub.add_parser("proteins")
    p.add_argument("--proteins",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model",default="esmc_600m")
    p.add_argument("--device",default="cpu")
    p.add_argument("--cache-dir",type=Path)
    r=sub.add_parser("reactions")
    r.add_argument("--reactions",type=Path,required=True)
    r.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    info=(
        build_protein_features(
            a.proteins,a.output,model_name=a.model,device=a.device,cache_dir=a.cache_dir
        )
        if a.kind=="proteins"
        else build_reaction_features(a.reactions,a.output)
    )
    print(info)


if __name__=="__main__":
    main()
