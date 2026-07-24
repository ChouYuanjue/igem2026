from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    ensemble_similarity,
    load_auxiliary_reaction_library,
    load_feature_schema,
    load_models,
    load_protein_library,
    load_reaction_library,
    models_require_auxiliary_reaction_features,
)
from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics  # noqa: E402

DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_PROTEINS = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_BASE = ROOT / "results/terpene_production_models/drfp_categorical"
DEFAULT_ADAPTED = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"
DEFAULT_OUTPUT = ROOT / "results/terpene_production_retention"
DEFAULT_BUDGETS = (1, 3, 5, 10, 20)


def aggregate_query_metrics(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n_queries": ("query_id", "size"),
        "mean_positive_count": ("n_positives", "mean"),
        "mean_reciprocal_rank": ("reciprocal_rank", "mean"),
        "median_best_positive_rank": ("best_positive_rank", "median"),
    }
    for budget in budgets:
        aggregations[f"hit_probability_at_{budget}"] = (f"hit_at_{budget}", "mean")
        aggregations[f"positive_recall_at_{budget}"] = (f"positive_recall_at_{budget}", "mean")
    return frame.groupby(["model", "direction", "evaluation_level"]).agg(**aggregations).reset_index()


def score_matrix(
    production_dir: Path,
    protein_features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    schema = load_feature_schema(production_dir)
    reaction_features, reaction_ids = load_reaction_library(production_dir, schema)
    models = load_models(production_dir / "models", "production", device)
    auxiliary = (
        load_auxiliary_reaction_library(production_dir, reaction_ids)
        if models_require_auxiliary_reaction_features(models)
        else None
    )
    return (
        ensemble_similarity(
            models,
            protein_features,
            reaction_features,
            device,
            auxiliary,
        ),
        reaction_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate current-database retention before and after MARTS adaptation.")
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--protein-dir", type=Path, default=DEFAULT_PROTEINS)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapted-dir", type=Path, default=DEFAULT_ADAPTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    positives = pd.read_csv(args.positives, sep="\t", dtype=str).fillna("")
    positives = positives[["rhea_id", "Entry"]].drop_duplicates()
    protein_features, protein_ids = load_protein_library(args.protein_dir.resolve())
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    positives = positives[positives["Entry"].isin(protein_to_row)].copy()

    records: list[dict[str, object]] = []
    pair_records: list[dict[str, object]] = []
    matrices: dict[str, np.ndarray] = {}
    reaction_ids_by_model: dict[str, list[str]] = {}
    for model_name, directory in [
        ("base_current_production", args.base_dir.resolve()),
        ("marts_pu_rehearsal_production", args.adapted_dir.resolve()),
    ]:
        matrix, reaction_ids = score_matrix(directory, protein_features, device)
        matrices[model_name] = matrix
        reaction_ids_by_model[model_name] = reaction_ids
        reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
        current_pairs = positives[positives["rhea_id"].isin(reaction_to_row)].copy()
        by_reaction = {
            reaction_id: set(group["Entry"].astype(str))
            for reaction_id, group in current_pairs.groupby("rhea_id", sort=True)
        }
        by_protein = {
            protein_id: set(group["rhea_id"].astype(str))
            for protein_id, group in current_pairs.groupby("Entry", sort=True)
        }

        for reaction_id, positive_ids in by_reaction.items():
            metrics = rank_metrics(
                matrix[reaction_to_row[reaction_id]],
                protein_ids,
                positive_ids,
                set(),
                budgets,
            )
            records.append(
                {
                    "model": model_name,
                    "direction": "reaction_to_enzyme",
                    "evaluation_level": "query_all_known_positives",
                    "query_id": reaction_id,
                    **metrics,
                }
            )
            for target in sorted(positive_ids):
                metrics = rank_metrics(
                    matrix[reaction_to_row[reaction_id]],
                    protein_ids,
                    {target},
                    positive_ids - {target},
                    budgets,
                )
                pair_records.append(
                    {
                        "model": model_name,
                        "direction": "reaction_to_enzyme",
                        "evaluation_level": "pair_leave_other_known_masked",
                        "query_id": f"{reaction_id}::{target}",
                        "reaction_id": reaction_id,
                        "protein_id": target,
                        **metrics,
                    }
                )

        for protein_id, positive_ids in by_protein.items():
            metrics = rank_metrics(
                matrix[:, protein_to_row[protein_id]],
                reaction_ids,
                positive_ids,
                set(),
                budgets,
            )
            records.append(
                {
                    "model": model_name,
                    "direction": "enzyme_to_reaction",
                    "evaluation_level": "query_all_known_positives",
                    "query_id": protein_id,
                    **metrics,
                }
            )
            for target in sorted(positive_ids):
                metrics = rank_metrics(
                    matrix[:, protein_to_row[protein_id]],
                    reaction_ids,
                    {target},
                    positive_ids - {target},
                    budgets,
                )
                pair_records.append(
                    {
                        "model": model_name,
                        "direction": "enzyme_to_reaction",
                        "evaluation_level": "pair_leave_other_known_masked",
                        "query_id": f"{protein_id}::{target}",
                        "reaction_id": target,
                        "protein_id": protein_id,
                        **metrics,
                    }
                )

    query_frame = pd.DataFrame(records)
    pair_frame = pd.DataFrame(pair_records)
    all_frame = pd.concat([query_frame, pair_frame], ignore_index=True, sort=False)
    all_frame.to_csv(output_dir / "metrics_long.csv", index=False)
    query_frame.to_csv(output_dir / "query_metrics.csv", index=False)
    pair_frame.to_csv(output_dir / "pair_metrics.csv", index=False)
    metrics = aggregate_query_metrics(all_frame, budgets)
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    pivot_rows = []
    for (direction, level), group in metrics.groupby(["direction", "evaluation_level"]):
        base = group[group["model"].eq("base_current_production")].iloc[0]
        adapted = group[group["model"].eq("marts_pu_rehearsal_production")].iloc[0]
        row: dict[str, object] = {
            "direction": direction,
            "evaluation_level": level,
            "base_mrr": base["mean_reciprocal_rank"],
            "adapted_mrr": adapted["mean_reciprocal_rank"],
            "delta_mrr": adapted["mean_reciprocal_rank"] - base["mean_reciprocal_rank"],
        }
        for budget in budgets:
            row[f"base_hit_at_{budget}"] = base[f"hit_probability_at_{budget}"]
            row[f"adapted_hit_at_{budget}"] = adapted[f"hit_probability_at_{budget}"]
            row[f"delta_hit_at_{budget}"] = (
                adapted[f"hit_probability_at_{budget}"] - base[f"hit_probability_at_{budget}"]
            )
        pivot_rows.append(row)
    comparison = pd.DataFrame(pivot_rows)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    summary = {
        "n_positive_pairs": len(positives),
        "n_proteins": len(protein_ids),
        "n_reactions": len(reaction_ids_by_model["base_current_production"]),
        "budgets": budgets,
        "note": "This is a retention/memorization sanity check, not an unbiased cold-start estimate.",
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "comparison": str(output_dir / "comparison.csv"),
            "metrics_long": str(output_dir / "metrics_long.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
