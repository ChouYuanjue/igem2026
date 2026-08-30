from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.fair_benchmark import sha256_file  # noqa: E402
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    load_feature_schema,
    load_registered_reaction_feature_library,
)

DEFAULT_BENCHMARK_ROOT = ROOT / "results/broad_rhea_fair_benchmarks_v1"
DEFAULT_UNIVERSE = ROOT / "data/catalyst_candidate_universes/general_merged"
DEFAULT_MODEL = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"
DEFAULT_MMSEQS = ROOT / "data/assets/mmseqs2/mmseqs/bin/mmseqs"
DEFAULT_OUTPUT = ROOT / "results/broad_rhea_difficulty_slices_v1"


def protein_identity_bucket(value: float) -> str:
    if not np.isfinite(value):
        return "no_hit"
    if value < 0.20:
        return "lt20"
    if value < 0.40:
        return "20_40"
    if value < 0.60:
        return "40_60"
    if value < 0.80:
        return "60_80"
    return "ge80"


def reaction_similarity_bucket(value: float) -> str:
    if not np.isfinite(value):
        return "no_hit"
    if value < 0.30:
        return "lt0p3"
    if value < 0.50:
        return "0p3_0p5"
    if value < 0.70:
        return "0p5_0p7"
    if value < 0.90:
        return "0p7_0p9"
    return "ge0p9"


def degree_bucket(value: int) -> str:
    if value <= 0:
        return "degree0"
    if value == 1:
        return "degree1"
    if value <= 5:
        return "degree2_5"
    if value <= 20:
        return "degree6_20"
    return "degree21plus"


def _write_fasta(path: Path, identifiers: list[str], sequence_map: dict[str, str]) -> None:
    missing = [value for value in identifiers if not sequence_map.get(value)]
    if missing:
        raise ValueError(f"Missing sequences for {len(missing)} proteins; examples={missing[:5]}")
    with path.open("w", encoding="utf-8") as handle:
        for value in identifiers:
            handle.write(f">{value}\n{sequence_map[value]}\n")


def nearest_train_protein_identity(
    test_ids: list[str],
    train_ids: list[str],
    *,
    sequence_path: Path,
    mmseqs: Path,
    work_dir: Path,
    threads: int,
) -> pd.DataFrame:
    seq = pd.read_csv(sequence_path, sep="\t", dtype=str).fillna("")
    sequence_map = dict(zip(seq["protein_id"].astype(str), seq["sequence"].astype(str)))
    qfa, tfa, out = work_dir / "test.fa", work_dir / "train.fa", work_dir / "hits.tsv"
    _write_fasta(qfa, test_ids, sequence_map)
    _write_fasta(tfa, train_ids, sequence_map)
    tmp = work_dir / "mmseqs_tmp"
    subprocess.run(
        [
            str(mmseqs), "easy-search", str(qfa), str(tfa), str(out), str(tmp),
            "--format-output", "query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits",
            "--max-seqs", "1", "--threads", str(threads), "-s", "7.5",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    columns = ["protein_id", "nearest_train_protein_id", "mmseqs_fident", "alnlen", "qlen", "tlen", "qcov", "tcov", "evalue", "bits"]
    hits = pd.read_csv(out, sep="\t", names=columns, dtype={"protein_id": str, "nearest_train_protein_id": str}) if out.stat().st_size else pd.DataFrame(columns=columns)
    numeric = ["mmseqs_fident", "alnlen", "qlen", "tlen", "qcov", "tcov", "evalue", "bits"]
    for column in numeric:
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
    base = pd.DataFrame({"protein_id": test_ids})
    merged = base.merge(hits, on="protein_id", how="left")
    merged["protein_identity_bucket"] = merged["mmseqs_fident"].map(protein_identity_bucket)
    return merged


def nearest_train_reaction_similarity(
    test_ids: list[str],
    train_ids: list[str],
    *,
    features: np.ndarray,
    reaction_ids: list[str],
    drfp_dim: int,
) -> pd.DataFrame:
    index = {value: i for i, value in enumerate(reaction_ids)}
    missing = (set(test_ids) | set(train_ids)) - set(index)
    if missing:
        raise ValueError(f"Reaction feature universe missing {len(missing)} IDs; examples={sorted(missing)[:5]}")
    train_rows = np.asarray([index[x] for x in train_ids], dtype=np.int64)
    test_rows = np.asarray([index[x] for x in test_ids], dtype=np.int64)
    train = sparse.csr_matrix((features[train_rows, :drfp_dim] > 0).astype(np.float32))
    test = sparse.csr_matrix((features[test_rows, :drfp_dim] > 0).astype(np.float32))
    train_counts = np.asarray(train.sum(axis=1)).reshape(-1).astype(np.float32)
    test_counts = np.asarray(test.sum(axis=1)).reshape(-1).astype(np.float32)
    records: list[dict[str, object]] = []
    train_ids_np = np.asarray(train_ids, dtype=object)
    for row_i, reaction_id in enumerate(test_ids):
        intersections = (test[row_i] @ train.T).tocoo()
        if intersections.nnz == 0:
            records.append({"reaction_id": reaction_id, "nearest_train_reaction_id": "", "max_train_drfp_tanimoto": 0.0})
            continue
        cols = intersections.col.astype(np.int64)
        inter = intersections.data.astype(np.float32)
        union = test_counts[row_i] + train_counts[cols] - inter
        sims = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        best = float(np.max(sims))
        tied = cols[np.isclose(sims, best)]
        best_col = int(tied[np.argmin(train_ids_np[tied].astype(str))]) if len(tied) > 1 else int(tied[0])
        records.append({"reaction_id": reaction_id, "nearest_train_reaction_id": train_ids[best_col], "max_train_drfp_tanimoto": best})
    frame = pd.DataFrame(records)
    frame["reaction_similarity_bucket"] = frame["max_train_drfp_tanimoto"].map(reaction_similarity_bucket)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train-distance and frequency difficulty slices for one leakage-controlled Rhea benchmark cell.")
    parser.add_argument("--cell", required=True)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--universe-dir", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mmseqs", type=Path, default=DEFAULT_MMSEQS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("threads must be positive")

    cell_dir = args.benchmark_root.resolve() / args.cell
    train_path, test_path = cell_dir / "train_pairs.csv", cell_dir / "test_pairs.csv"
    train = pd.read_csv(train_path, dtype=str).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    test = pd.read_csv(test_path, dtype=str).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    train_proteins, test_proteins = sorted(train.protein_id.unique()), sorted(test.protein_id.unique())
    train_reactions, test_reactions = sorted(train.reaction_id.unique()), sorted(test.reaction_id.unique())

    universe = args.universe_dir.resolve()
    schema = load_feature_schema(args.model_dir.resolve())
    reaction_features, reaction_ids = load_registered_reaction_feature_library(
        universe / "reaction_features" / "drfp_categorical_v1", schema
    )
    drfp_dim = int(schema.get("drfp_dimension", 2048))
    out = args.output_root.resolve() / args.cell
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"difficulty_{args.cell}_", dir=str(out)) as temp:
        protein_slice = nearest_train_protein_identity(
            test_proteins, train_proteins,
            sequence_path=universe / "protein_sequences.tsv",
            mmseqs=args.mmseqs.resolve(), work_dir=Path(temp), threads=args.threads,
        )
    reaction_slice = nearest_train_reaction_similarity(
        test_reactions, train_reactions, features=reaction_features,
        reaction_ids=reaction_ids, drfp_dim=drfp_dim,
    )
    protein_degree = train.groupby("protein_id").reaction_id.nunique().to_dict()
    reaction_degree = train.groupby("reaction_id").protein_id.nunique().to_dict()
    protein_slice["train_degree"] = protein_slice.protein_id.map(protein_degree).fillna(0).astype(int)
    protein_slice["train_degree_bucket"] = protein_slice.train_degree.map(degree_bucket)
    reaction_slice["train_degree"] = reaction_slice.reaction_id.map(reaction_degree).fillna(0).astype(int)
    reaction_slice["train_degree_bucket"] = reaction_slice.train_degree.map(degree_bucket)
    protein_slice.to_csv(out / "protein_slices.csv", index=False)
    reaction_slice.to_csv(out / "reaction_slices.csv", index=False)
    pair_slice = test.merge(protein_slice, on="protein_id", how="left").merge(reaction_slice, on="reaction_id", how="left", suffixes=("_protein", "_reaction"))
    pair_slice.to_csv(out / "pair_slices.csv", index=False)
    summary = {
        "cell": args.cell,
        "train_pairs_sha256": sha256_file(train_path),
        "test_pairs_sha256": sha256_file(test_path),
        "n_train_pairs": int(len(train)), "n_test_pairs": int(len(test)),
        "n_test_proteins": len(test_proteins), "n_test_reactions": len(test_reactions),
        "protein_distance": "top-bit-score MMseqs2 hit into train proteins; report local alignment fident plus query/target coverage",
        "reaction_distance": "maximum Tanimoto over the same binary DRFP block used by the active reaction feature schema",
        "protein_identity_bucket_counts": protein_slice.protein_identity_bucket.value_counts(dropna=False).to_dict(),
        "reaction_similarity_bucket_counts": reaction_slice.reaction_similarity_bucket.value_counts(dropna=False).to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
