from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from drfp import DrfpEncoder
from scipy import sparse

from projects.active.terpene_screening.audit_clipzyme_outer_overlap import (
    DEFAULT_BENCH,
    DEFAULT_CACHE,
    DEFAULT_CELLS,
    DEFAULT_REACTIONS,
    assign_rule_splits,
    reaction_keys,
    reaction_keys_from_smiles,
)

ROOT = Path(__file__).resolve().parents[3]
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


def encode_binary(reactions: list[str]) -> sparse.csr_matrix:
    if not reactions:
        return sparse.csr_matrix((0, 2048), dtype=np.uint8)
    dense = np.asarray(
        DrfpEncoder.encode(reactions, n_folded_length=2048), dtype=np.uint8
    )
    dense = (dense > 0).astype(np.uint8, copy=False)
    return sparse.csr_matrix(dense)


def nearest_tanimoto(
    queries: list[str], train: sparse.csr_matrix, train_nnz: np.ndarray, batch_size: int
) -> np.ndarray:
    output = np.zeros(len(queries), dtype=np.float32)
    for start in range(0, len(queries), batch_size):
        local = queries[start : start + batch_size]
        q = encode_binary(local)
        q_nnz = np.asarray(q.sum(axis=1)).ravel().astype(np.float32)
        intersection = (q @ train.T).toarray().astype(np.float32, copy=False)
        union = q_nnz[:, None] + train_nnz[None, :] - intersection
        sim = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        output[start : start + len(local)] = sim.max(axis=1) if sim.shape[1] else 0.0
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
    train_fp = encode_binary(train_strings)
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
            "mean_max_similarity":float(np.mean(similarities)) if len(similarities) else None,
            "median_max_similarity":float(np.median(similarities)) if len(similarities) else None,
            "p90_max_similarity":float(np.quantile(similarities,0.9)) if len(similarities) else None,
            "ge_0p9":int(np.sum(similarities>=0.9)),
            "ge_0p7":int(np.sum(similarities>=0.7)),
            "ge_0p5":int(np.sum(similarities>=0.5)),
            "lt_0p3":int(np.sum(similarities<0.3)),
            "fraction_ge_0p9":float(np.mean(similarities>=0.9)) if len(similarities) else None,
            "fraction_ge_0p7":float(np.mean(similarities>=0.7)) if len(similarities) else None,
            "fraction_lt_0p3":float(np.mean(similarities<0.3)) if len(similarities) else None,
        }
        summaries.append(summary); print(json.dumps(summary),flush=True)
    pd.DataFrame(summaries).to_csv(output/"summary.csv",index=False)
    manifest={
        "protocol":"nearest binary DRFP Tanimoto to CLIPZyme clip_egnn train split",
        "train_reaction_count_oriented_plus_reversed":len(train_strings),
        "fingerprint":"DRFP n_folded_length=2048, binary",
        "reverse_reactions_included":True,
        "split_type":"rule_id",
        "split_probs":[0.8,0.1,0.1],
        "split_seed":0,
        "cells":summaries,
    }
    (output/"summary.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

if __name__=="__main__": main()
