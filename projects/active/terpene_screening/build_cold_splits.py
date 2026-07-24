from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_PROTEIN_CLUSTERS = ROOT / "data/terpene_sequence_clusters/clusters_id50.csv"
DEFAULT_OUTPUT = ROOT / "data/terpene_cold_splits"
DEFAULT_FOLDS = 5
DEFAULT_REACTION_SIMILARITY = 0.50

from projects.active.terpene_screening.gate_matrix import (  # noqa: E402
    canonical_or_raw_reaction,
    largest_organic_component,
    mol_fp,
    precursor_class_from_reaction,
    product_skeleton_class,
    split_reaction_smiles,
)


def product_fingerprint(reaction_smiles: str):
    _, products = split_reaction_smiles(canonical_or_raw_reaction(reaction_smiles))
    product = largest_organic_component(products)
    return mol_fp(product), product


def butina_clusters(ids: list[str], fingerprints: list[object], similarity_threshold: float) -> dict[str, str]:
    valid_indices = [index for index, fp in enumerate(fingerprints) if fp is not None]
    assignments: dict[str, str] = {}
    if valid_indices:
        distances: list[float] = []
        for offset, index in enumerate(valid_indices):
            if offset == 0:
                continue
            similarities = DataStructs.BulkTanimotoSimilarity(
                fingerprints[index],
                [fingerprints[other] for other in valid_indices[:offset]],
            )
            distances.extend(1.0 - value for value in similarities)
        clusters = Butina.ClusterData(
            distances,
            len(valid_indices),
            1.0 - similarity_threshold,
            isDistData=True,
            reordering=True,
        )
        for cluster_index, members in enumerate(clusters):
            member_ids = sorted(ids[valid_indices[member]] for member in members)
            cluster_id = member_ids[0] if member_ids else f"cluster_{cluster_index:04d}"
            for member in members:
                assignments[ids[valid_indices[member]]] = cluster_id
    for index, reaction_id in enumerate(ids):
        if reaction_id not in assignments:
            assignments[reaction_id] = reaction_id
    return assignments


def assign_groups_to_folds(group_weights: dict[str, int], n_folds: int) -> dict[str, int]:
    fold_weights = [0] * n_folds
    assignments: dict[str, int] = {}
    for group, weight in sorted(group_weights.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(n_folds), key=lambda index: (fold_weights[index], index))
        assignments[group] = fold
        fold_weights[fold] += int(weight)
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cold-start splits for terpene synthase retrieval.")
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--protein-clusters", type=Path, default=DEFAULT_PROTEIN_CLUSTERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--reaction-similarity", type=float, default=DEFAULT_REACTION_SIMILARITY)
    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError("At least two folds are required.")
    if not 0 < args.reaction_similarity <= 1:
        raise ValueError("reaction-similarity must be in (0, 1].")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    positives = pd.read_csv(args.positives, sep="\t", dtype=str).fillna("")
    positives = positives[["rhea_id", "Entry", "smiles_seq"]].drop_duplicates(["rhea_id", "Entry"])
    protein_clusters = pd.read_csv(args.protein_clusters, dtype=str)
    protein_cluster_map = dict(zip(protein_clusters["entry"].astype(str), protein_clusters["cluster_id"].astype(str)))
    positives["protein_cluster"] = positives["Entry"].map(protein_cluster_map)
    missing_protein_cluster = positives["protein_cluster"].isna()
    positives.loc[missing_protein_cluster, "protein_cluster"] = positives.loc[missing_protein_cluster, "Entry"]

    reaction_rows: list[dict[str, object]] = []
    for reaction_id, group in positives.groupby("rhea_id", sort=True):
        reaction_smiles = canonical_or_raw_reaction(group["smiles_seq"].iloc[0])
        fingerprint, product_smiles = product_fingerprint(reaction_smiles)
        reaction_rows.append(
            {
                "reaction_id": reaction_id,
                "reaction_smiles": reaction_smiles,
                "product_smiles": product_smiles,
                "precursor_class": precursor_class_from_reaction(reaction_smiles),
                "product_skeleton_class": product_skeleton_class(reaction_smiles),
                "fingerprint": fingerprint,
                "n_positive_pairs": int(len(group)),
                "n_positive_enzymes": int(group["Entry"].nunique()),
            }
        )
    reactions = pd.DataFrame(reaction_rows)

    reaction_cluster_map: dict[str, str] = {}
    for precursor_class, group in reactions.groupby("precursor_class", sort=True):
        ids = group["reaction_id"].astype(str).tolist()
        fps = group["fingerprint"].tolist()
        local = butina_clusters(ids, fps, args.reaction_similarity)
        for reaction_id, cluster_id in local.items():
            reaction_cluster_map[reaction_id] = f"{precursor_class}::{cluster_id}"
    reactions["reaction_cluster"] = reactions["reaction_id"].map(reaction_cluster_map)

    positive_protein_weights = positives.groupby("protein_cluster").size().astype(int).to_dict()
    protein_weights = {
        cluster_id: max(1, int(positive_protein_weights.get(cluster_id, 0)))
        for cluster_id in sorted(protein_clusters["cluster_id"].astype(str).unique())
    }
    reaction_cluster_weights = (
        positives.assign(reaction_cluster=positives["rhea_id"].map(reaction_cluster_map))
        .groupby("reaction_cluster")
        .size()
        .astype(int)
        .to_dict()
    )
    protein_fold = assign_groups_to_folds(protein_weights, args.folds)
    reaction_fold = assign_groups_to_folds(reaction_cluster_weights, args.folds)

    proteins = protein_clusters.copy()
    proteins["fold"] = proteins["cluster_id"].map(protein_fold).astype(int)
    proteins.to_csv(output_dir / "protein_cluster_folds.csv", index=False)

    reactions["fold"] = reactions["reaction_cluster"].map(reaction_fold).astype(int)
    reactions = reactions.drop(columns=["fingerprint"])
    reactions.to_csv(output_dir / "reaction_cluster_folds.csv", index=False)

    pairs = positives.copy()
    pairs["reaction_cluster"] = pairs["rhea_id"].map(reaction_cluster_map)
    pairs["protein_fold"] = pairs["protein_cluster"].map(protein_fold).astype(int)
    pairs["reaction_fold"] = pairs["reaction_cluster"].map(reaction_fold).astype(int)
    pairs["same_fold_double_cold"] = pairs["protein_fold"] == pairs["reaction_fold"]
    pairs.to_csv(output_dir / "positive_pair_fold_assignments.csv", index=False)

    fold_rows: list[dict[str, object]] = []
    for fold in range(args.folds):
        protein_test = pairs["protein_fold"] == fold
        reaction_test = pairs["reaction_fold"] == fold
        double_test = protein_test & reaction_test
        double_train = (~protein_test) & (~reaction_test)
        fold_rows.append(
            {
                "fold": fold,
                "protein_cold_test_pairs": int(protein_test.sum()),
                "protein_cold_test_enzymes": int(pairs.loc[protein_test, "Entry"].nunique()),
                "reaction_cold_test_pairs": int(reaction_test.sum()),
                "reaction_cold_test_reactions": int(pairs.loc[reaction_test, "rhea_id"].nunique()),
                "double_cold_test_pairs": int(double_test.sum()),
                "double_cold_test_reactions": int(pairs.loc[double_test, "rhea_id"].nunique()),
                "double_cold_test_enzymes": int(pairs.loc[double_test, "Entry"].nunique()),
                "double_cold_train_pairs": int(double_train.sum()),
            }
        )
    fold_summary = pd.DataFrame(fold_rows)
    fold_summary.to_csv(output_dir / "fold_summary.csv", index=False)

    protein_sizes = proteins.groupby("cluster_id")["entry"].size()
    reaction_sizes = reactions.groupby("reaction_cluster")["reaction_id"].size()
    summary = {
        "positives": str(args.positives.resolve()),
        "protein_clusters": str(args.protein_clusters.resolve()),
        "n_folds": args.folds,
        "reaction_similarity_threshold": args.reaction_similarity,
        "n_positive_pairs": int(len(pairs)),
        "n_protein_clusters": int(protein_sizes.size),
        "largest_protein_cluster": int(protein_sizes.max()),
        "n_reaction_clusters": int(reaction_sizes.size),
        "largest_reaction_cluster": int(reaction_sizes.max()),
        "n_unparseable_reactions": int((reactions["product_smiles"] == "").sum()),
        "double_cold_pairs_total": int(pairs["same_fold_double_cold"].sum()),
        "outputs": {
            "protein_cluster_folds": str(output_dir / "protein_cluster_folds.csv"),
            "reaction_cluster_folds": str(output_dir / "reaction_cluster_folds.csv"),
            "positive_pair_fold_assignments": str(output_dir / "positive_pair_fold_assignments.csv"),
            "fold_summary": str(output_dir / "fold_summary.csv"),
        },
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(fold_summary.to_string(index=False))


if __name__ == "__main__":
    main()
