from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = ROOT / "external_models/clipzyme_audit/clipzyme_data/cached_enzymemap.p"
DEFAULT_SEQS = ROOT / "external_models/clipzyme_audit/clipzyme_data/uniprot2sequence.p"
DEFAULT_BENCH = ROOT / "results/broad_rhea_fair_benchmarks_v1"
DEFAULT_REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
DEFAULT_PROTEINS = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
DEFAULT_OUTPUT = ROOT / "results/clipzyme_outer_overlap_audit_v1"
DEFAULT_CELLS = [
    "broad_reaction_hash_cold_protein_seen",
    "reactzyme_reaction_projected_double_cold",
    "temporal_post2020_double_cold",
    "reactzyme_enzyme_projected_protein_cold",
    "temporal_post2020_protein_cold",
]


def _canonical_molecule(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonical_side(values: object) -> str | None:
    if isinstance(values, str):
        parts = [part for part in values.split(".") if part]
    elif isinstance(values, (list, tuple)):
        parts = [str(part) for part in values if str(part)]
    else:
        return None
    canonical = [_canonical_molecule(part) for part in parts]
    if any(value is None for value in canonical):
        return None
    return ".".join(sorted(str(value) for value in canonical))


def reaction_keys(reactants: object, products: object) -> tuple[str | None, str | None]:
    left = canonical_side(reactants); right = canonical_side(products)
    if left is None or right is None:
        return None, None
    oriented = f"{left}>>{right}"
    reversed_key = f"{right}>>{left}"
    undirected = min(oriented, reversed_key)
    return oriented, undirected


def reaction_keys_from_smiles(smiles: str) -> tuple[str | None, str | None]:
    if ">>" not in str(smiles):
        return None, None
    left, right = str(smiles).split(">>", 1)
    return reaction_keys(left, right)


def assign_rule_splits(samples: list[dict], split_probs=(0.8, 0.1, 0.1), seed=0) -> dict[str, str]:
    rules = [str(sample["rule_id"]) for sample in samples]
    counts = Counter(rules)
    unique = sorted(counts)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    cumulative = np.cumsum([counts[value] for value in unique])
    cutoffs = [np.searchsorted(cumulative, np.round(q, 3) * cumulative[-1], side="right") for q in np.cumsum(split_probs)]
    cutoffs[-1] = len(unique)
    cuts = np.concatenate([[0], cutoffs])
    mapping: dict[str, str] = {}
    for index, name in enumerate(["train", "dev", "test"]):
        for value in unique[cuts[index] : cuts[index + 1]]:
            mapping[value] = name
    return mapping


def clipzyme_train_sets(samples: list[dict]) -> dict[str, object]:
    split = assign_rule_splits(samples)
    train = [sample for sample in samples if split[str(sample["rule_id"])] == "train"]
    protein_ids: set[str] = set()
    oriented: set[str] = set(); undirected: set[str] = set(); pairs: set[tuple[str, str]] = set(); pairs_undir: set[tuple[str, str]] = set()
    parse_fail = 0
    for sample in train:
        protein = str(sample.get("uniprot_id") or sample.get("protein_id") or "")
        key, key_u = reaction_keys(sample.get("reactants", []), sample.get("products", []))
        if protein:
            protein_ids.add(protein)
        if key is None or key_u is None:
            parse_fail += 1; continue
        oriented.add(key); undirected.add(key_u)
        if protein:
            pairs.add((protein, key)); pairs_undir.add((protein, key_u))
    return {
        "train_samples": train,
        "protein_ids": protein_ids,
        "reaction_oriented": oriented,
        "reaction_undirected": undirected,
        "pairs_oriented": pairs,
        "pairs_undirected": pairs_undir,
        "parse_fail": parse_fail,
        "rule_split": split,
    }


def normalize_sequence(value: str) -> str:
    return "".join(str(value).split()).upper()


def audit_cell(
    *,
    cell: str,
    benchmark_root: Path,
    reaction_lookup: dict[str, str],
    protein_sequence_lookup: dict[str, str],
    train_sets: dict[str, object],
    clip_sequences: dict[str, str] | None,
) -> tuple[dict[str, object], pd.DataFrame]:
    pairs = pd.read_csv(benchmark_root / cell / "test_pairs.csv", dtype=str).fillna("")[["protein_id", "reaction_id"]].drop_duplicates()
    rows=[]
    clip_sequence_set: set[str] = set()
    if clip_sequences is not None:
        train_proteins = train_sets["protein_ids"]
        clip_sequence_set = {
            normalize_sequence(seq) for pid, seq in clip_sequences.items()
            if str(pid) in train_proteins and normalize_sequence(seq)
        }
    reaction_cache: dict[str, tuple[str | None, str | None]] = {}
    for reaction_id in sorted(pairs.reaction_id.unique()):
        reaction_cache[reaction_id] = reaction_keys_from_smiles(reaction_lookup.get(reaction_id, ""))
    for protein_id, reaction_id in pairs.itertuples(index=False):
        key,key_u=reaction_cache[reaction_id]
        sequence=normalize_sequence(protein_sequence_lookup.get(protein_id,""))
        rows.append({
            "protein_id":protein_id,
            "reaction_id":reaction_id,
            "reaction_parseable":key is not None,
            "clipzyme_train_protein_id_seen":protein_id in train_sets["protein_ids"],
            "clipzyme_train_exact_sequence_seen":bool(sequence and sequence in clip_sequence_set),
            "clipzyme_train_reaction_oriented_seen":bool(key and key in train_sets["reaction_oriented"]),
            "clipzyme_train_reaction_undirected_seen":bool(key_u and key_u in train_sets["reaction_undirected"]),
            "clipzyme_train_pair_oriented_seen":bool(key and (protein_id,key) in train_sets["pairs_oriented"]),
            "clipzyme_train_pair_undirected_seen":bool(key_u and (protein_id,key_u) in train_sets["pairs_undirected"]),
        })
    detail=pd.DataFrame(rows)
    unique_reactions=detail.drop_duplicates("reaction_id")
    unique_proteins=detail.drop_duplicates("protein_id")
    summary={
        "cell":cell,
        "test_pair_rows":int(len(detail)),
        "test_unique_proteins":int(detail.protein_id.nunique()),
        "test_unique_reactions":int(detail.reaction_id.nunique()),
        "reaction_parseable_unique":int(unique_reactions.reaction_parseable.sum()),
        "protein_id_seen_unique":int(unique_proteins.clipzyme_train_protein_id_seen.sum()),
        "protein_id_seen_fraction_unique":float(unique_proteins.clipzyme_train_protein_id_seen.mean()),
        "exact_sequence_seen_unique":int(unique_proteins.clipzyme_train_exact_sequence_seen.sum()),
        "exact_sequence_seen_fraction_unique":float(unique_proteins.clipzyme_train_exact_sequence_seen.mean()),
        "reaction_oriented_seen_unique":int(unique_reactions.clipzyme_train_reaction_oriented_seen.sum()),
        "reaction_oriented_seen_fraction_unique":float(unique_reactions.clipzyme_train_reaction_oriented_seen.mean()),
        "reaction_undirected_seen_unique":int(unique_reactions.clipzyme_train_reaction_undirected_seen.sum()),
        "reaction_undirected_seen_fraction_unique":float(unique_reactions.clipzyme_train_reaction_undirected_seen.mean()),
        "pair_oriented_seen_rows":int(detail.clipzyme_train_pair_oriented_seen.sum()),
        "pair_oriented_seen_fraction_rows":float(detail.clipzyme_train_pair_oriented_seen.mean()),
        "pair_undirected_seen_rows":int(detail.clipzyme_train_pair_undirected_seen.sum()),
        "pair_undirected_seen_fraction_rows":float(detail.clipzyme_train_pair_undirected_seen.mean()),
    }
    return summary,detail


def main() -> None:
    parser=argparse.ArgumentParser(description="Audit CLIPZyme's actual train split against frozen RHEA outer cells without using model scores.")
    parser.add_argument("--cache",type=Path,default=DEFAULT_CACHE)
    parser.add_argument("--clip-sequences",type=Path,default=DEFAULT_SEQS)
    parser.add_argument("--benchmark-root",type=Path,default=DEFAULT_BENCH)
    parser.add_argument("--reactions",type=Path,default=DEFAULT_REACTIONS)
    parser.add_argument("--protein-sequences",type=Path,default=DEFAULT_PROTEINS)
    parser.add_argument("--cell",action="append",default=[])
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    cache=args.cache.resolve(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    samples=pickle.load(cache.open("rb"))
    if not isinstance(samples,list) or not samples:
        raise ValueError("CLIPZyme cache must contain a nonempty sample list")
    train_sets=clipzyme_train_sets(samples)
    clip_sequences=None
    if args.clip_sequences.is_file():
        clip_sequences=pickle.load(args.clip_sequences.open("rb"))
    reactions=pd.read_csv(args.reactions,dtype=str).fillna("")
    reaction_lookup=dict(zip(reactions.reaction_id.astype(str),reactions.reaction_smiles.astype(str)))
    proteins=pd.read_csv(args.protein_sequences,sep="\t",dtype=str).fillna("")
    protein_lookup=dict(zip(proteins.protein_id.astype(str),proteins.sequence.astype(str)))
    cells=args.cell or DEFAULT_CELLS
    summaries=[]
    for cell in cells:
        if not (args.benchmark_root/cell/"test_pairs.csv").is_file():
            continue
        summary,detail=audit_cell(cell=cell,benchmark_root=args.benchmark_root,reaction_lookup=reaction_lookup,protein_sequence_lookup=protein_lookup,train_sets=train_sets,clip_sequences=clip_sequences)
        summaries.append(summary); detail.to_csv(out/f"{cell}_detail.csv",index=False)
    frame=pd.DataFrame(summaries); frame.to_csv(out/"summary.csv",index=False)
    split_counts=Counter(train_sets["rule_split"].values())
    payload={
        "protocol":"CLIPZyme clip_egnn public config train-only overlap audit",
        "checkpoint_training_config":"configs/train/clip_egnn.json",
        "processed_cache":str(cache),
        "split_type":"rule_id",
        "split_probs":[0.8,0.1,0.1],
        "split_seed":0,
        "processed_samples":len(samples),
        "train_samples":len(train_sets["train_samples"]),
        "rule_split_counts":dict(split_counts),
        "train_unique_protein_ids":len(train_sets["protein_ids"]),
        "train_unique_reaction_oriented":len(train_sets["reaction_oriented"]),
        "train_reaction_parse_fail":train_sets["parse_fail"],
        "reaction_match":"componentwise RDKit canonical SMILES after removing atom maps; reports both orientation-sensitive and directionless matches",
        "sequence_match":"exact normalized amino-acid sequence among train-split UniProt IDs when uniprot2sequence.p is available",
        "cells":summaries,
    }
    (out/"summary.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(frame.to_string(index=False)); print(json.dumps({k:v for k,v in payload.items() if k!='cells'},indent=2))

if __name__=="__main__": main()
