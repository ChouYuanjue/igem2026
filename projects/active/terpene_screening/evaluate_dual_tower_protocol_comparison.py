from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    ModelConfig,
    build_reaction_features,
    load_protein_features,
    train_model,
)

DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_EMBEDDINGS = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_STRICT_SPLITS = ROOT / "data/terpene_cold_splits/positive_pair_fold_assignments.csv"
DEFAULT_EXACT_FOLDS = (
    ROOT
    / "projects/active/terpene_screening/comparison_assets/legacy_exact_reaction_folds.csv"
)
DEFAULT_PROTEIN_CLUSTERS = ROOT / "data/terpene_sequence_clusters/clusters_id50.csv"
DEFAULT_REACTION_CLUSTERS = ROOT / "data/terpene_cold_splits/reaction_cluster_folds.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_old_new_comparison/new_dual_tower_protocols"
DEFAULT_SEEDS = (20260723, 20260724, 20260725)
DEFAULT_BUDGETS = (3, 5, 10, 20)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("Expected at least one integer")
    return result


def masked_rank_metrics(
    scores: np.ndarray,
    candidate_ids: list[str],
    positives: set[str],
    masked: set[str],
    budgets: tuple[int, ...],
) -> dict[str, float | int]:
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    id_to_row = {value: index for index, value in enumerate(candidate_ids)}
    for value in masked:
        row = id_to_row.get(value)
        if row is not None:
            adjusted[row] = -np.inf
    order = [
        int(index)
        for index in np.lexsort((np.asarray(candidate_ids), -adjusted))
        if np.isfinite(adjusted[index])
    ]
    ranked_ids = [candidate_ids[index] for index in order]
    positive_ranks = [index + 1 for index, value in enumerate(ranked_ids) if value in positives]
    best_rank = min(positive_ranks) if positive_ranks else np.nan
    result: dict[str, float | int] = {
        "n_positives": int(len(positives)),
        "n_masked_known_positives": int(len(masked)),
        "best_positive_rank": float(best_rank),
        "reciprocal_rank": 0.0 if not positive_ranks else float(1.0 / best_rank),
    }
    for budget in budgets:
        selected = set(ranked_ids[:budget])
        hits = len(selected & positives)
        result[f"hit_at_{budget}"] = int(hits > 0)
        result[f"hits_at_{budget}"] = int(hits)
        result[f"precision_at_{budget}"] = float(hits / budget)
        result[f"positive_recall_at_{budget}"] = float(hits / len(positives)) if positives else 0.0
    return result


def aggregate(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for protocol, group in frame.groupby("protocol", sort=True):
        row: dict[str, object] = {
            "protocol": protocol,
            "n_query_cells": int(len(group)),
            "n_unique_reactions": int(group["reaction_id"].nunique()),
            "mean_reciprocal_rank": float(group["reciprocal_rank"].mean()),
            "median_best_positive_rank": float(group["best_positive_rank"].median()),
            "mean_masked_known_positives": float(group["n_masked_known_positives"].mean()),
        }
        for budget in budgets:
            row[f"hit_probability_at_{budget}"] = float(group[f"hit_at_{budget}"].mean())
            row[f"expected_hits_at_{budget}"] = float(group[f"hits_at_{budget}"].mean())
            row[f"precision_at_{budget}"] = float(group[f"precision_at_{budget}"].mean())
            row[f"positive_recall_at_{budget}"] = float(group[f"positive_recall_at_{budget}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def train_ensemble(
    *,
    protein_tensor: torch.Tensor,
    reaction_tensor: torch.Tensor,
    train_pairs: pd.DataFrame,
    protein_to_row: dict[str, int],
    reaction_to_row: dict[str, int],
    protein_groups: dict[str, str],
    reaction_groups: dict[str, str],
    config: ModelConfig,
    seeds: tuple[int, ...],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    reaction_loss_weight: float,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, object]]]:
    protein_embeddings: list[np.ndarray] = []
    reaction_embeddings: list[np.ndarray] = []
    histories: list[dict[str, object]] = []
    for seed in seeds:
        model, history = train_model(
            protein_tensor,
            reaction_tensor,
            train_pairs,
            protein_to_row,
            reaction_to_row,
            config,
            epochs,
            learning_rate,
            weight_decay,
            temperature,
            seed,
            device,
            protein_group_map=protein_groups,
            reaction_group_map=reaction_groups,
            exclude_same_group_negatives=True,
            reaction_loss_weight=reaction_loss_weight,
        )
        model.eval()
        with torch.no_grad():
            protein_embeddings.append(model.encode_proteins(protein_tensor).cpu().numpy())
            reaction_embeddings.append(model.encode_reactions(reaction_tensor).cpu().numpy())
        histories.append(
            {
                "seed": seed,
                "epochs": epochs,
                "final_loss": float(history[-1]["loss"]),
                "best_loss": float(min(item["loss"] for item in history)),
            }
        )
    return protein_embeddings, reaction_embeddings, histories


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the new TPS dual tower under legacy exact-reaction and strict 25-cell protocols."
    )
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--strict-splits", type=Path, default=DEFAULT_STRICT_SPLITS)
    parser.add_argument("--exact-folds", type=Path, default=DEFAULT_EXACT_FOLDS)
    parser.add_argument("--protein-clusters", type=Path, default=DEFAULT_PROTEIN_CLUSTERS)
    parser.add_argument("--reaction-clusters", type=Path, default=DEFAULT_REACTION_CLUSTERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocols", default="legacy_exact,double_cold_25cell")
    parser.add_argument("--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--reaction-loss-weight", type=float, default=0.75)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    protocols = tuple(part.strip() for part in args.protocols.split(",") if part.strip())
    unknown = set(protocols) - {"legacy_exact", "double_cold_25cell"}
    if unknown:
        raise ValueError(f"Unknown protocols: {sorted(unknown)}")
    seeds = parse_int_tuple(args.seeds)
    budgets = parse_int_tuple(args.budgets)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protein_matrix, protein_ids = load_protein_features(args.embedding_dir.resolve())
    positives = pd.read_csv(args.positives, sep="\t", dtype=str).fillna("")
    positives = positives[["Entry", "rhea_id", "smiles_seq"]].drop_duplicates(["Entry", "rhea_id"])
    reaction_matrix, reaction_ids, reaction_table, feature_schema = build_reaction_features(
        positives, "drfp_categorical"
    )
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    positives = positives[
        positives["Entry"].isin(protein_to_row) & positives["rhea_id"].isin(reaction_to_row)
    ].copy()

    strict = pd.read_csv(args.strict_splits, dtype=str).fillna("")
    strict["protein_fold"] = pd.to_numeric(strict["protein_fold"]).astype(int)
    strict["reaction_fold"] = pd.to_numeric(strict["reaction_fold"]).astype(int)
    strict = strict[["Entry", "rhea_id", "protein_cluster", "reaction_cluster", "protein_fold", "reaction_fold"]].drop_duplicates(["Entry", "rhea_id"])
    pairs = positives[["Entry", "rhea_id"]].merge(
        strict,
        on=["Entry", "rhea_id"],
        how="left",
        validate="one_to_one",
    )
    if pairs[["protein_fold", "reaction_fold"]].isna().any().any():
        raise ValueError("Strict fold assignments do not cover every current positive pair")
    pairs["protein_fold"] = pairs["protein_fold"].astype(int)
    pairs["reaction_fold"] = pairs["reaction_fold"].astype(int)

    protein_clusters = pd.read_csv(args.protein_clusters, dtype=str).fillna("")
    protein_groups = dict(zip(protein_clusters["entry"].astype(str), protein_clusters["cluster_id"].astype(str)))
    protein_groups = {value: protein_groups.get(value, value) for value in protein_ids}
    reaction_clusters = pd.read_csv(args.reaction_clusters, dtype=str).fillna("")
    reaction_groups = dict(
        zip(reaction_clusters["reaction_id"].astype(str), reaction_clusters["reaction_cluster"].astype(str))
    )
    reaction_groups = {value: reaction_groups.get(value, value) for value in reaction_ids}

    exact = pd.read_csv(args.exact_folds, dtype=str).fillna("")
    exact["legacy_exact_fold"] = pd.to_numeric(exact["legacy_exact_fold"]).astype(int)
    exact_fold_by_reaction = dict(zip(exact["reaction_id"].astype(str), exact["legacy_exact_fold"].astype(int)))
    missing_exact = sorted(set(reaction_ids) - set(exact_fold_by_reaction))
    if missing_exact:
        raise ValueError(f"Legacy exact folds miss reactions: {missing_exact[:10]}")

    protein_tensor = torch.as_tensor(protein_matrix, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_matrix, dtype=torch.float32, device=device)
    config = ModelConfig(
        protein_input_dim=int(protein_matrix.shape[1]),
        reaction_input_dim=int(reaction_matrix.shape[1]),
        hidden_dim=512,
        embedding_dim=256,
        dropout=0.1,
    )
    all_positive_by_reaction = {
        reaction_id: set(group["Entry"].astype(str))
        for reaction_id, group in pairs.groupby("rhea_id", sort=True)
    }
    records: list[dict[str, object]] = []
    training_records: list[dict[str, object]] = []

    if "legacy_exact" in protocols:
        for fold in range(5):
            test_reactions = {value for value, local_fold in exact_fold_by_reaction.items() if local_fold == fold}
            train_pairs = pairs[~pairs["rhea_id"].isin(test_reactions)][["Entry", "rhea_id"]].drop_duplicates()
            protein_sets, reaction_sets, histories = train_ensemble(
                protein_tensor=protein_tensor,
                reaction_tensor=reaction_tensor,
                train_pairs=train_pairs,
                protein_to_row=protein_to_row,
                reaction_to_row=reaction_to_row,
                protein_groups=protein_groups,
                reaction_groups=reaction_groups,
                config=config,
                seeds=seeds,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                temperature=args.temperature,
                reaction_loss_weight=args.reaction_loss_weight,
                device=device,
            )
            for item in histories:
                training_records.append({"protocol": "legacy_exact", "fold": fold, "protein_fold": "", "reaction_fold": fold, "n_train_pairs": len(train_pairs), **item})
            for reaction_id in sorted(test_reactions):
                positives_for_query = all_positive_by_reaction.get(reaction_id, set())
                if not positives_for_query:
                    continue
                reaction_row = reaction_to_row[reaction_id]
                scores = np.mean(
                    [reaction_sets[index][reaction_row] @ protein_sets[index].T for index in range(len(seeds))],
                    axis=0,
                )
                records.append(
                    {
                        "protocol": "legacy_exact",
                        "protein_fold": "",
                        "reaction_fold": fold,
                        "reaction_id": reaction_id,
                        **masked_rank_metrics(scores, protein_ids, positives_for_query, set(), budgets),
                    }
                )

    if "double_cold_25cell" in protocols:
        for protein_fold in range(5):
            for reaction_fold in range(5):
                train_pairs = pairs[
                    (pairs["protein_fold"] != protein_fold)
                    & (pairs["reaction_fold"] != reaction_fold)
                ][["Entry", "rhea_id"]].drop_duplicates()
                test_pairs = pairs[
                    (pairs["protein_fold"] == protein_fold)
                    & (pairs["reaction_fold"] == reaction_fold)
                ].copy()
                if test_pairs.empty:
                    continue
                protein_sets, reaction_sets, histories = train_ensemble(
                    protein_tensor=protein_tensor,
                    reaction_tensor=reaction_tensor,
                    train_pairs=train_pairs,
                    protein_to_row=protein_to_row,
                    reaction_to_row=reaction_to_row,
                    protein_groups=protein_groups,
                    reaction_groups=reaction_groups,
                    config=config,
                    seeds=seeds,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    temperature=args.temperature,
                    reaction_loss_weight=args.reaction_loss_weight,
                    device=device,
                )
                split_id = f"p{protein_fold}_r{reaction_fold}"
                for item in histories:
                    training_records.append({"protocol": "double_cold_25cell", "fold": split_id, "protein_fold": protein_fold, "reaction_fold": reaction_fold, "n_train_pairs": len(train_pairs), **item})
                for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
                    positives_for_query = set(group["Entry"].astype(str))
                    known_other = all_positive_by_reaction.get(reaction_id, set()) - positives_for_query
                    reaction_row = reaction_to_row[reaction_id]
                    scores = np.mean(
                        [reaction_sets[index][reaction_row] @ protein_sets[index].T for index in range(len(seeds))],
                        axis=0,
                    )
                    records.append(
                        {
                            "protocol": "double_cold_25cell",
                            "protein_fold": protein_fold,
                            "reaction_fold": reaction_fold,
                            "reaction_id": reaction_id,
                            **masked_rank_metrics(scores, protein_ids, positives_for_query, known_other, budgets),
                        }
                    )

    query_metrics = pd.DataFrame(records)
    metrics = aggregate(query_metrics, budgets)
    training = pd.DataFrame(training_records)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    training.to_csv(output_dir / "training_summary.csv", index=False)
    summary = {
        "method": "new_dual_tower_controlled_current_only",
        "protein_features": "ESM-C 600M mean, 1152 dimensions",
        "reaction_features": "DRFP plus precursor/product-skeleton categories, 2115 dimensions",
        "model_config": asdict(config),
        "protocols": list(protocols),
        "seeds": list(seeds),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "reaction_loss_weight": args.reaction_loss_weight,
        "pu_group_mask": True,
        "n_current_proteins": len(protein_ids),
        "n_current_reactions": len(reaction_ids),
        "n_positive_pairs": len(pairs),
        "feature_schema": feature_schema,
        "outputs": {
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "metrics": str(output_dir / "metrics.csv"),
            "training_summary": str(output_dir / "training_summary.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
