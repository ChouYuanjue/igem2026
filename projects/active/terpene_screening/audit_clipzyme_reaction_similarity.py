from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from drfp import DrfpEncoder
from rdkit import Chem
from scipy import sparse

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.audit_clipzyme_outer_overlap import (
    DEFAULT_BENCH,
    DEFAULT_CACHE,
    DEFAULT_CELLS,
    DEFAULT_REACTIONS,
    assign_rule_splits,
    reaction_keys,
    reaction_keys_from_smiles,
)

DEFAULT_OUTPUT = ROOT / "results/clipzyme_reaction_similarity_audit_v1"


def train_reactions(samples: list[dict]) -> list[str]:
    split = assign_rule_splits(samples)
    values: set[str] = set()
    for sample in samples:
        if split[str(sample["rule_id"])] != "train":
            continue
        oriented, _ = reaction_keys(sample.get("reactants", []), sample.get("products", []))
        if oriented is None:
            continue
        values.add(oriented)
        left, right = oriented.split(">>", 1)
        values.add(f"{right}>>{left}")
    return sorted(values)


def stereo_insensitive_reaction(reaction: str) -> str | None:
    if ">>" not in reaction:
        return None
    left, right = reaction.split(">>", 1)

    def clean_side(side: str) -> str | None:
        values=[]
        for part in [value for value in side.split(".") if value]:
            mol=Chem.MolFromSmiles(part)
            if mol is None:
                return None
            Chem.RemoveStereochemistry(mol)
            values.append(Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False))
        return ".".join(sorted(values))

    lhs=clean_side(left); rhs=clean_side(right)
    if lhs is None or rhs is None:
        return None
    return f"{lhs}>>{rhs}"


def encode_binary(reactions: list[str]) -> tuple[sparse.csr_matrix, np.ndarray]:
    if not reactions:
        return sparse.csr_matrix((0, 2048), dtype=np.uint8), np.zeros(0,dtype=bool)
    cleaned=[stereo_insensitive_reaction(value) for value in reactions]
    valid=np.asarray([value is not None for value in cleaned],dtype=bool)
    encoded=np.zeros((len(reactions),2048),dtype=np.uint8)
    if valid.any():
        values=[str(value) for value in cleaned if value is not None]
        dense=np.asarray(DrfpEncoder.encode(values,n_folded_length=2048),dtype=np.uint8)
        encoded[valid]=(dense>0).astype(np.uint8,copy=False)
    return sparse.csr_matrix(encoded),valid


def nearest_tanimoto(
    queries: list[str], train: sparse.csr_matrix, train_nnz: np.ndarray, batch_size: int
) -> np.ndarray:
    output = np.zeros(len(queries), dtype=np.float32)
    for start in range(0, len(queries), batch_size):
        local = queries[start : start + batch_size]
        q, valid = encode_binary(local)
        q_nnz = np.asarray(q.sum(axis=1)).ravel().astype(np.float32)
        intersection = (q @ train.T).toarray().astype(np.float32, copy=False)
        union = q_nnz[:, None] + train_nnz[None, :] - intersection
        sim = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        local_max = sim.max(axis=1) if sim.shape[1] else np.zeros(len(local),dtype=np.float32)
        local_max[~valid] = np.nan
        output[start : start + len(local)] = local_max
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Nearest CLIPZyme-train reaction similarity for frozen RHEA outer cells.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--reactions", type=Path, default=DEFAULT_REACTIONS)
    parser.add_argument("--cell", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    samples = pickle.load(args.cache.open("rb"))
    train_strings = train_reactions(samples)
    print(f"encoding {len(train_strings)} oriented+reversed CLIPZyme train reactions", flush=True)
    train_fp,train_valid = encode_binary(train_strings)
    if not train_valid.all():
        train_fp=train_fp[train_valid]
        train_strings=[value for value,valid in zip(train_strings,train_valid) if valid]
    train_nnz = np.asarray(train_fp.sum(axis=1)).ravel().astype(np.float32)
    reactions = pd.read_csv(args.reactions, dtype=str).fillna("")
    reaction_lookup = dict(zip(reactions.reaction_id.astype(str), reactions.reaction_smiles.astype(str)))
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    summaries=[]
    for cell in args.cell or DEFAULT_CELLS:
        test_path = args.benchmark_root / cell / "test_pairs.csv"
        if not test_path.is_file():
            continue
        ids = sorted(pd.read_csv(test_path, dtype=str).fillna("").reaction_id.unique())
        query_strings=[]; valid_ids=[]; failures=[]
        for rid in ids:
            oriented,_=reaction_keys_from_smiles(reaction_lookup.get(rid,""))
            if oriented is None:
                failures.append(rid); continue
            query_strings.append(oriented); valid_ids.append(rid)
        similarities=nearest_tanimoto(query_strings,train_fp,train_nnz,args.batch_size)
        detail=pd.DataFrame({"reaction_id":valid_ids,"max_clipzyme_train_drfp_tanimoto":similarities})
        detail.to_csv(output/f"{cell}_reaction_similarity.csv",index=False)
        summary={
            "cell":cell,
            "unique_reactions":len(ids),
            "parseable_reactions":len(valid_ids),
            "parse_failures":len(failures),
            "drfp_valid_reactions":int(np.isfinite(similarities).sum()),
            "mean_max_similarity":float(np.nanmean(similarities)) if np.isfinite(similarities).any() else None,
            "median_max_similarity":float(np.nanmedian(similarities)) if np.isfinite(similarities).any() else None,
            "p90_max_similarity":float(np.nanquantile(similarities,0.9)) if np.isfinite(similarities).any() else None,
            "ge_0p9":int(np.nansum(similarities>=0.9)),
            "ge_0p7":int(np.nansum(similarities>=0.7)),
            "ge_0p5":int(np.nansum(similarities>=0.5)),
            "lt_0p3":int(np.nansum(similarities<0.3)),
            "fraction_ge_0p9":float(np.nanmean(similarities>=0.9)) if np.isfinite(similarities).any() else None,
            "fraction_ge_0p7":float(np.nanmean(similarities>=0.7)) if np.isfinite(similarities).any() else None,
            "fraction_lt_0p3":float(np.nanmean(similarities<0.3)) if np.isfinite(similarities).any() else None,
        }
        summaries.append(summary); print(json.dumps(summary),flush=True)
    pd.DataFrame(summaries).to_csv(output/"summary.csv",index=False)
    manifest={
        "protocol":"nearest binary DRFP Tanimoto to CLIPZyme clip_egnn train split",
        "train_reaction_count_oriented_plus_reversed":len(train_strings),
        "fingerprint":"DRFP n_folded_length=2048, binary after removing stereochemistry for conservative near-duplicate detection",
        "reverse_reactions_included":True,
        "split_type":"rule_id",
        "split_probs":[0.8,0.1,0.1],
        "split_seed":0,
        "cells":summaries,
    }
    (output/"summary.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

if __name__=="__main__": main()
