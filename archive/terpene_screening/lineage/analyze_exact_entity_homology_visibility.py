from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = ROOT / "results/terpene_exact_entity_protocols"
DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_PROTEIN_CLUSTERS = ROOT / "data/terpene_sequence_clusters/clusters_id50.csv"
DEFAULT_REACTION_CLUSTERS = ROOT / "data/terpene_cold_splits/reaction_cluster_folds.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_protocol_reassessment/exact_entity_visibility_matrix.csv"
DEFAULT_BUDGETS = (1, 3, 5, 10, 20)


def aggregate(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["protocol", "direction", "same_cluster_evidence"]
    for key, group in frame.groupby(keys, sort=True):
        protocol, direction, visibility = key
        row: dict[str, object] = {
            "protocol": str(protocol),
            "direction": str(direction),
            "same_cluster_evidence": str(visibility),
            "n_query_cells": int(len(group)),
            "n_unique_queries": int(group["query_id"].nunique()),
            "mean_visible_neighbors": float(group["visible_neighbor_count"].mean()),
            "mean_reciprocal_rank": float(group["reciprocal_rank"].mean()),
            "median_best_positive_rank": float(group["best_positive_rank"].median()),
        }
        for budget in budgets:
            row[f"hit_probability_at_{budget}"] = float(group[f"hit_at_{budget}"].mean())
            row[f"expected_hits_at_{budget}"] = float(group[f"hits_at_{budget}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--protein-clusters", type=Path, default=DEFAULT_PROTEIN_CLUSTERS)
    parser.add_argument("--reaction-clusters", type=Path, default=DEFAULT_REACTION_CLUSTERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    query = pd.read_csv(args.results_dir / "query_metrics.csv")
    pairs = pd.read_csv(args.positives, sep="\t", dtype=str).fillna("")
    pairs = pairs[["Entry", "rhea_id"]].drop_duplicates()
    protein_folds = pd.read_csv(args.results_dir / "exact_protein_folds.csv", dtype={"Entry": str})
    reaction_folds = pd.read_csv(args.results_dir / "exact_reaction_folds.csv", dtype={"rhea_id": str})
    protein_fold_map = dict(zip(protein_folds["Entry"].astype(str), protein_folds["exact_fold"].astype(int)))
    reaction_fold_map = dict(zip(reaction_folds["rhea_id"].astype(str), reaction_folds["exact_fold"].astype(int)))

    protein_cluster_frame = pd.read_csv(args.protein_clusters, dtype=str).fillna("")
    protein_cluster = dict(zip(protein_cluster_frame["entry"].astype(str), protein_cluster_frame["cluster_id"].astype(str)))
    reaction_cluster_frame = pd.read_csv(args.reaction_clusters, dtype=str).fillna("")
    reaction_cluster = dict(
        zip(reaction_cluster_frame["reaction_id"].astype(str), reaction_cluster_frame["reaction_cluster"].astype(str))
    )
    pairs["protein_cluster"] = pairs["Entry"].map(lambda value: protein_cluster.get(str(value), str(value)))
    pairs["reaction_cluster"] = pairs["rhea_id"].map(lambda value: reaction_cluster.get(str(value), str(value)))

    visibility_records: list[dict[str, object]] = []
    for protocol in ["protein_exact", "reaction_exact"]:
        for fold in range(5):
            if protocol == "protein_exact":
                test_entities = {entity for entity, assigned in protein_fold_map.items() if assigned == fold}
                train_pairs = pairs[~pairs["Entry"].isin(test_entities)].copy()
                test_pairs = pairs[pairs["Entry"].isin(test_entities)].copy()
                train_cluster_counts = train_pairs.groupby("protein_cluster")["Entry"].nunique().to_dict()
                entity_visible = {
                    entry: int(train_cluster_counts.get(protein_cluster.get(entry, entry), 0))
                    for entry in test_entities
                }
                for entry, group in test_pairs.groupby("Entry", sort=True):
                    count = int(entity_visible[str(entry)])
                    visibility_records.append(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            "direction": "enzyme_to_reaction",
                            "query_id": str(entry),
                            "visible_neighbor_count": count,
                            "same_cluster_evidence": "visible" if count > 0 else "not_visible",
                        }
                    )
                for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
                    counts = [int(entity_visible[str(entry)]) for entry in group["Entry"].astype(str)]
                    total = int(sum(counts))
                    visibility_records.append(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            "direction": "reaction_to_enzyme",
                            "query_id": str(reaction_id),
                            "visible_neighbor_count": total,
                            "same_cluster_evidence": "visible" if any(value > 0 for value in counts) else "not_visible",
                        }
                    )
            else:
                test_entities = {entity for entity, assigned in reaction_fold_map.items() if assigned == fold}
                train_pairs = pairs[~pairs["rhea_id"].isin(test_entities)].copy()
                test_pairs = pairs[pairs["rhea_id"].isin(test_entities)].copy()
                train_cluster_counts = train_pairs.groupby("reaction_cluster")["rhea_id"].nunique().to_dict()
                entity_visible = {
                    reaction_id: int(train_cluster_counts.get(reaction_cluster.get(reaction_id, reaction_id), 0))
                    for reaction_id in test_entities
                }
                for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
                    count = int(entity_visible[str(reaction_id)])
                    visibility_records.append(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            "direction": "reaction_to_enzyme",
                            "query_id": str(reaction_id),
                            "visible_neighbor_count": count,
                            "same_cluster_evidence": "visible" if count > 0 else "not_visible",
                        }
                    )
                for entry, group in test_pairs.groupby("Entry", sort=True):
                    counts = [int(entity_visible[str(reaction_id)]) for reaction_id in group["rhea_id"].astype(str)]
                    total = int(sum(counts))
                    visibility_records.append(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            "direction": "enzyme_to_reaction",
                            "query_id": str(entry),
                            "visible_neighbor_count": total,
                            "same_cluster_evidence": "visible" if any(value > 0 for value in counts) else "not_visible",
                        }
                    )

    visibility = pd.DataFrame(visibility_records)
    if visibility.duplicated(["protocol", "fold", "direction", "query_id"]).any():
        raise AssertionError("Duplicate visibility records")
    merged = query.merge(
        visibility,
        on=["protocol", "fold", "direction", "query_id"],
        how="left",
        validate="one_to_one",
    )
    if merged["same_cluster_evidence"].isna().any():
        raise AssertionError("Visibility annotation did not cover every exact-entity query cell")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output.with_name("exact_entity_query_visibility.csv"), index=False)
    matrix = aggregate(merged, DEFAULT_BUDGETS)
    matrix.to_csv(output, index=False)
    summary = {
        "status": "complete",
        "definition": (
            "visible means at least one training entity from the same 50% protein cluster "
            "or reaction chemical cluster remained after exact-entity holdout"
        ),
        "matrix": str(output),
        "query_rows": str(output.with_name("exact_entity_query_visibility.csv")),
        "n_rows": int(len(merged)),
    }
    output.with_name("exact_entity_visibility_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(matrix.to_string(index=False))


if __name__ == "__main__":
    main()
