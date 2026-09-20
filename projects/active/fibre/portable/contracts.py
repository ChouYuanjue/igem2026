from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


AA_RE=re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$")


@dataclass(frozen=True)
class DatasetTables:
    proteins: pd.DataFrame
    reactions: pd.DataFrame
    pairs: pd.DataFrame


def _read(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path),dtype=str).fillna("")


def _require(frame: pd.DataFrame, names: set[str], label: str) -> None:
    missing=sorted(names-set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def validate_tables(
    proteins: pd.DataFrame,
    reactions: pd.DataFrame,
    pairs: pd.DataFrame,
) -> DatasetTables:
    p=proteins.fillna("").copy()
    r=reactions.fillna("").copy()
    o=pairs.fillna("").copy()
    _require(p,{"protein_id","sequence"},"proteins")
    _require(r,{"reaction_id","reaction_smiles"},"reactions")
    _require(o,{"protein_id","reaction_id"},"pairs")

    p["protein_id"]=p.protein_id.astype(str).str.strip()
    p["sequence"]=p.sequence.astype(str).str.upper().str.replace(r"\s+","",regex=True).str.rstrip("*")
    r["reaction_id"]=r.reaction_id.astype(str).str.strip()
    r["reaction_smiles"]=r.reaction_smiles.astype(str).str.strip()
    o["protein_id"]=o.protein_id.astype(str).str.strip()
    o["reaction_id"]=o.reaction_id.astype(str).str.strip()

    if (p.protein_id=="").any() or p.protein_id.duplicated().any():
        raise ValueError("protein_id must be non-empty and unique")
    if (r.reaction_id=="").any() or r.reaction_id.duplicated().any():
        raise ValueError("reaction_id must be non-empty and unique")
    bad=p[~p.sequence.map(lambda x: bool(x) and bool(AA_RE.fullmatch(x)))]
    if len(bad):
        raise ValueError(f"invalid protein sequences: {bad.protein_id.tolist()[:10]}")
    if (r.reaction_smiles=="").any() or ~r.reaction_smiles.str.contains(">>",regex=False).all():
        raise ValueError("reaction_smiles must be non-empty directed reaction SMILES containing >>")
    unknown_p=sorted(set(o.protein_id)-set(p.protein_id))
    unknown_r=sorted(set(o.reaction_id)-set(r.reaction_id))
    if unknown_p or unknown_r:
        raise ValueError(
            f"pairs reference unknown entities: proteins={unknown_p[:10]} reactions={unknown_r[:10]}"
        )
    if len(o)==0:
        raise ValueError("at least one accepted positive pair is required")
    o=o.drop_duplicates(["protein_id","reaction_id"],keep="first").reset_index(drop=True)
    return DatasetTables(p.reset_index(drop=True),r.reset_index(drop=True),o)


def load_and_validate(
    proteins: str | Path,
    reactions: str | Path,
    pairs: str | Path,
) -> DatasetTables:
    return validate_tables(_read(proteins),_read(reactions),_read(pairs))


def load_feature_csv(
    path: str | Path,
    *,
    id_column: str,
    expected_ids: list[str],
) -> np.ndarray:
    frame=_read(path)
    _require(frame,{id_column},"feature table")
    if frame[id_column].duplicated().any():
        raise ValueError(f"{id_column} duplicated in feature table")
    frame=frame.set_index(id_column)
    missing=[x for x in expected_ids if x not in frame.index]
    extra=sorted(set(frame.index)-set(expected_ids))
    if missing or extra:
        raise ValueError(
            f"feature ids mismatch: missing={missing[:10]} extra={extra[:10]}"
        )
    ordered=frame.loc[expected_ids]
    if ordered.shape[1]==0:
        raise ValueError("feature table has no numeric feature columns")
    try:
        values=ordered.astype(float).to_numpy(dtype=np.float32)
    except ValueError as exc:
        raise ValueError("all feature columns must be numeric") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("feature matrix contains non-finite values")
    norm=np.linalg.norm(values,axis=1,keepdims=True)
    if np.any(norm[:,0]<=1e-12):
        raise ValueError("feature matrix contains zero vectors")
    return (values/norm).astype(np.float32)


def load_partial_feature_csv(
    path: str | Path,
    *,
    id_column: str,
    expected_ids: list[str],
) -> tuple[np.ndarray,np.ndarray]:
    """Load one optional geometric coordinate with explicit missingness.

    The table may contain any non-empty subset of the reference IDs. Missing
    rows mean "coordinate not observed", never a zero vector or negative
    biological observation. Extra/duplicate IDs are rejected.
    """
    frame=_read(path)
    _require(frame,{id_column},"partial feature table")
    if frame[id_column].duplicated().any():
        raise ValueError(f"{id_column} duplicated in partial feature table")
    extra=sorted(set(frame[id_column])-set(expected_ids))
    if extra:
        raise ValueError(f"partial feature table contains unknown ids: {extra[:10]}")
    feature_columns=[x for x in frame.columns if x!=id_column]
    if not feature_columns:
        raise ValueError("partial feature table has no numeric feature columns")
    if len(frame)==0:
        raise ValueError("partial feature table must contain at least one observed row")
    try:
        observed=frame[feature_columns].astype(float).to_numpy(dtype=np.float32)
    except ValueError as exc:
        raise ValueError("all partial feature columns must be numeric") from exc
    if not np.all(np.isfinite(observed)):
        raise ValueError("partial feature matrix contains non-finite values")
    norm=np.linalg.norm(observed,axis=1,keepdims=True)
    if np.any(norm[:,0]<=1e-12):
        raise ValueError("partial feature matrix contains zero vectors")
    observed=(observed/norm).astype(np.float32)
    position={x:i for i,x in enumerate(expected_ids)}
    out=np.zeros((len(expected_ids),observed.shape[1]),dtype=np.float32)
    available=np.zeros(len(expected_ids),dtype=bool)
    for row,identifier in enumerate(frame[id_column].astype(str)):
        idx=position[identifier]
        out[idx]=observed[row]
        available[idx]=True
    return out,available
