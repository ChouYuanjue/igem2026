from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_dual_tower_protocol_comparison import (  # noqa: E402
    masked_rank_metrics,
)
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    ModelConfig,
    build_reaction_features,
    load_protein_features,
    train_model,
)

DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_EMBEDDINGS = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_REACTION_FOLDS = (
    ROOT / "projects/active/terpene_screening/comparison_assets/legacy_exact_reaction_folds.csv"
)
DEFAULT_PROTEIN_CLUSTERS = ROOT / "data/terpene_sequence_clusters/clusters_id50.csv"
DEFAULT_REACTION_CLUSTERS = ROOT / "data/terpene_cold_splits/reaction_cluster_folds.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_exact_entity_protocols"
DEFAULT_BUDGETS = (1, 3, 5, 10, 20)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("Expected at least one integer")
    return result


def stable_tie(seed: int, entity_id: str) -> str:
    return hashlib.sha1(f"{seed}|{entity_id}".encode("utf-8")).hexdigest()


def build_balanced_exact_folds(
    pairs: pd.DataFrame,
    entity_col: str,
    seed: int,
    n_folds: int = 5,
) -> pd.DataFrame:
    degrees = pairs.groupby(entity_col).size().rename("n_pairs").reset_index()
    degrees["tie"] = degrees[entity_col].astype(str).map(lambda value: stable_tie(seed, value))
    degrees = degrees.sort_values(["n_pairs", "tie", entity_col], ascending=[False, True, True])
    fold_load = [0] * n_folds
    fold_entities = [0] * n_folds
    assignments: list[dict[str, object]] = []
    for row in degrees.itertuples(index=False):
        fold = min(range(n_folds), key=lambda value: (fold_load[value], fold_entities[value], value))
        entity_id = str(getattr(row, entity_col))
        n_pairs = int(row.n_pairs)
        assignments.append({entity_col: entity_id, "exact_fold": fold, "n_pairs": n_pairs})
        fold_load[fold] += n_pairs
        fold_entities[fold] += 1
    result = pd.DataFrame(assignments).sort_values(["exact_fold", entity_col]).reset_index(drop=True)
    if result[entity_col].duplicated().any():
        raise AssertionError(f"Duplicate exact-fold entity assignments for {entity_col}")
    return result


def aggregate(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (protocol, direction), group in frame.groupby(["protocol", "direction"], sort=True):
        row: dict[str, object] = {
            "protocol": str(protocol),
            "direction": str(direction),
            "n_query_cells": int(len(group)),
            "n_unique_queries": int(group["query_id"].nunique()),
            "mean_reciprocal_rank": float(group["reciprocal_rank"].mean()),
            "median_best_positive_rank": float(group["best_positive_rank"].median()),
            "mean_test_positives": float(group["n_positives"].mean()),
            "mean_training_associations_masked": float(group["n_masked_known_positives"].mean()),
        }
        for budget in budgets:
            row[f"hit_probability_at_{budget}"] = float(group[f"hit_at_{budget}"].mean())
            row[f"expected_hits_at_{budget}"] = float(group[f"hits_at_{budget}"].mean())
            row[f"precision_at_{budget}"] = float(group[f"precision_at_{budget}"].mean())
            row[f"positive_recall_at_{budget}"] = float(group[f"positive_recall_at_{budget}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exact-protein and exact-reaction holdout protocols while allowing "
            "homologous protein clusters and similar reaction clusters to remain in training."
        )
    )
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--reaction-folds", type=Path, default=DEFAULT_REACTION_FOLDS)
    parser.add_argument("--protein-clusters", type=Path, default=DEFAULT_PROTEIN_CLUSTERS)
    parser.add_argument("--reaction-clusters", type=Path, default=DEFAULT_REACTION_CLUSTERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocols", default="protein_exact,reaction_exact")
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    protocols = tuple(part.strip() for part in args.protocols.split(",") if part.strip())
    unknown = set(protocols) - {"protein_exact", "reaction_exact"}
    if unknown:
        raise ValueError(f"Unknown protocols: {sorted(unknown)}")
    budgets = parse_int_tuple(args.budgets)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    protein_matrix, protein_ids = load_protein_features(args.embedding_dir.resolve())
    positives = pd.read_csv(args.positives, sep="\t", dtype=str).fillna("")
    positives = positives[["Entry", "rhea_id", "smiles_seq"]].drop_duplicates(["Entry", "rhea_id"])
    reaction_matrix, reaction_ids, _reaction_table, feature_schema = build_reaction_features(
        positives, "multiview"
    )
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    pairs = positives[
        positives["Entry"].isin(protein_to_row) & positives["rhea_id"].isin(reaction_to_row)
    ][["Entry", "rhea_id"]].drop_duplicates().copy()

    protein_cluster_frame = pd.read_csv(args.protein_clusters, dtype=str).fillna("")
    protein_group_map = dict(
        zip(protein_cluster_frame["entry"].astype(str), protein_cluster_frame["cluster_id"].astype(str))
    )
    protein_group_map = {value: protein_group_map.get(value, value) for value in protein_ids}
    reaction_cluster_frame = pd.read_csv(args.reaction_clusters, dtype=str).fillna("")
    reaction_group_map = dict(
        zip(
            reaction_cluster_frame["reaction_id"].astype(str),
            reaction_cluster_frame["reaction_cluster"].astype(str),
        )
    )
    reaction_group_map = {value: reaction_group_map.get(value, value) for value in reaction_ids}

    reaction_folds = pd.read_csv(args.reaction_folds, dtype=str).fillna("")
    reaction_folds["exact_fold"] = pd.to_numeric(reaction_folds["legacy_exact_fold"]).astype(int)
    reaction_folds = reaction_folds[["reaction_id", "exact_fold"]].rename(
        columns={"reaction_id": "rhea_id"}
    )
    missing_reactions = sorted(set(reaction_ids) - set(reaction_folds["rhea_id"]))
    if missing_reactions:
        raise ValueError(f"Exact reaction folds miss reactions: {missing_reactions[:10]}")

    protein_folds = build_balanced_exact_folds(pairs, "Entry", args.seed)
    protein_folds.to_csv(output_dir / "exact_protein_folds.csv", index=False)
    reaction_folds.sort_values(["exact_fold", "rhea_id"]).to_csv(
        output_dir / "exact_reaction_folds.csv", index=False
    )

    folds_by_protocol = {
        "protein_exact": dict(zip(protein_folds["Entry"], protein_folds["exact_fold"])),
        "reaction_exact": dict(zip(reaction_folds["rhea_id"], reaction_folds["exact_fold"])),
    }
    all_positive_by_reaction = {
        reaction_id: set(group["Entry"].astype(str))
        for reaction_id, group in pairs.groupby("rhea_id", sort=True)
    }
    all_positive_by_protein = {
        entry: set(group["rhea_id"].astype(str))
        for entry, group in pairs.groupby("Entry", sort=True)
    }

    protein_tensor = torch.as_tensor(protein_matrix, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_matrix, dtype=torch.float32, device=device)
    config = ModelConfig(
        protein_input_dim=int(protein_matrix.shape[1]),
        reaction_input_dim=int(reaction_matrix.shape[1]),
        hidden_dim=512,
        embedding_dim=256,
        dropout=0.1,
    )

    query_records: list[dict[str, object]] = []
    training_records: list[dict[str, object]] = []
    for protocol in protocols:
        fold_map = folds_by_protocol[protocol]
        entity_col = "Entry" if protocol == "protein_exact" else "rhea_id"
        offset = 0 if protocol == "protein_exact" else 100
        for fold in range(5):
            test_entities = {entity for entity, assigned in fold_map.items() if int(assigned) == fold}
            train_pairs = pairs[~pairs[entity_col].isin(test_entities)].copy()
            test_pairs = pairs[pairs[entity_col].isin(test_entities)].copy()
            if train_pairs.empty or test_pairs.empty:
                raise ValueError(f"Empty train/test split for {protocol} fold {fold}")
            model, history = train_model(
                protein_tensor,
                reaction_tensor,
                train_pairs,
                protein_to_row,
                reaction_to_row,
                config,
                args.epochs,
                args.learning_rate,
                args.weight_decay,
                args.temperature,
                args.seed + offset + fold,
                device,
                protein_group_map=protein_group_map,
                reaction_group_map=reaction_group_map,
                exclude_same_group_negatives=True,
                reaction_loss_weight=0.5,
                loss_mode="bidirectional_infonce",
                model_selection="min_loss",
            )
            model.eval()
            with torch.no_grad():
                protein_embeddings = model.encode_proteins(protein_tensor).cpu().numpy()
                reaction_embeddings = model.encode_reactions(reaction_tensor).cpu().numpy()
            training_records.append(
                {
                    "protocol": protocol,
                    "fold": fold,
                    "n_train_pairs": int(len(train_pairs)),
                    "n_test_pairs": int(len(test_pairs)),
                    "seed": int(args.seed + offset + fold),
                    "epochs": int(args.epochs),
                    "final_loss": float(history[-1]["loss"]),
                    "best_loss": float(min(item["loss"] for item in history)),
                }
            )

            for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
                test_positives = set(group["Entry"].astype(str))
                known_train = all_positive_by_reaction.get(str(reaction_id), set()) - test_positives
                scores = reaction_embeddings[reaction_to_row[str(reaction_id)]] @ protein_embeddings.T
                metrics = masked_rank_metrics(
                    scores, protein_ids, test_positives, known_train, budgets
                )
                query_records.append(
                    {
                        "protocol": protocol,
                        "fold": fold,
                        "direction": "reaction_to_enzyme",
                        "query_id": str(reaction_id),
                        **metrics,
                    }
                )

            for entry, group in test_pairs.groupby("Entry", sort=True):
                test_positives = set(group["rhea_id"].astype(str))
                known_train = all_positive_by_protein.get(str(entry), set()) - test_positives
                scores = protein_embeddings[protein_to_row[str(entry)]] @ reaction_embeddings.T
                metrics = masked_rank_metrics(
                    scores, reaction_ids, test_positives, known_train, budgets
                )
                query_records.append(
                    {
                        "protocol": protocol,
                        "fold": fold,
                        "direction": "enzyme_to_reaction",
                        "query_id": str(entry),
                        **metrics,
                    }
                )

    query_frame = pd.DataFrame(query_records)
    metrics = aggregate(query_frame, budgets)
    query_frame.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(training_records).to_csv(output_dir / "training_summary.csv", index=False)
    summary = {
        "status": "complete",
        "protocol_definition": {
            "protein_exact": (
                "Exact test proteins and all their associations are absent from training; "
                "other proteins from the same 50% identity cluster may remain."
            ),
            "reaction_exact": (
                "Exact test reactions and all their associations are absent from training; "
                "other reactions from the same chemical cluster may remain."
            ),
        },
        "model": "current-only multiview dual tower",
        "n_proteins": int(len(protein_ids)),
        "n_reactions": int(len(reaction_ids)),
        "n_pairs": int(len(pairs)),
        "epochs": int(args.epochs),
        "base_seed": int(args.seed),
        "feature_schema": feature_schema,
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "training_summary": str(output_dir / "training_summary.csv"),
            "exact_protein_folds": str(output_dir / "exact_protein_folds.csv"),
            "exact_reaction_folds": str(output_dir / "exact_reaction_folds.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
