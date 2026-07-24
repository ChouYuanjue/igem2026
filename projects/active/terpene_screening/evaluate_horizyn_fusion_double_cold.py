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

from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    ModelConfig,
    TerpeneDualTower,
    rank_metrics,
)
from projects.active.terpene_screening.train_horizyn_reaction_adapter_double_cold import (  # noqa: E402
    AdapterConfig,
    ProteinAdapter,
)

DEFAULT_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_CURRENT = ROOT / "results/terpene_marts_domain_adaptation_cartesian_pu"
DEFAULT_HORIZYN = ROOT / "results/terpene_horizyn_adapter_full"
DEFAULT_OUTPUT = ROOT / "results/terpene_horizyn_fusion_double_cold"
DEFAULT_WEIGHTS = (0.1, 0.25, 0.5, 0.75, 0.9)
DEFAULT_BUDGETS = (3, 10, 20)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    denominator[denominator == 0] = 1
    return matrix / denominator


def tied_rank_percentile(scores: np.ndarray, ids: list[str]) -> np.ndarray:
    order = np.lexsort((np.asarray(ids), -scores))
    sorted_scores = scores[order]
    result = np.empty(len(scores), dtype=np.float32)
    if len(scores) == 1:
        result[0] = 1.0
        return result
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_position = (start + end - 1) / 2
        result[order[start:end]] = 1.0 - average_position / (len(scores) - 1)
        start = end
    return result


def boolean_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"1", "true", "yes"})


def load_current_score_matrix(
    model_dir: Path,
    split_id: str,
    protein_features: torch.Tensor,
    reaction_features: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    paths = sorted(model_dir.glob(f"adapted_{split_id}_model*.pt"))
    if not paths:
        raise FileNotFoundError(f"No current models for {split_id}")
    total = np.zeros(
        (reaction_features.shape[0], protein_features.shape[0]), dtype=np.float32
    )
    for path in paths:
        payload = torch.load(path, map_location=device, weights_only=False)
        model = TerpeneDualTower(ModelConfig(**payload["model_config"])).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        with torch.no_grad():
            proteins = model.encode_proteins(protein_features)
            reactions = model.encode_reactions(reaction_features)
            total += (reactions @ proteins.T).cpu().numpy()
    return total / len(paths)


def load_horizyn_score_matrix(
    model_dir: Path,
    split_id: str,
    protein_features: torch.Tensor,
    reaction_embeddings: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    paths = sorted(model_dir.glob(f"adapter_{split_id}_seed*.pt"))
    if not paths:
        raise FileNotFoundError(f"No Horizyn adapters for {split_id}")
    total = np.zeros(
        (len(reaction_embeddings), protein_features.shape[0]), dtype=np.float32
    )
    for path in paths:
        payload = torch.load(path, map_location=device, weights_only=False)
        adapter = ProteinAdapter(AdapterConfig(**payload["adapter_config"])).to(device)
        adapter.load_state_dict(payload["adapter_state_dict"])
        adapter.eval()
        with torch.no_grad():
            proteins = adapter(protein_features).cpu().numpy()
        total += reaction_embeddings @ proteins.T
    return total / len(paths)


def append_metrics(
    records: list[dict[str, object]],
    split_id: str,
    method: str,
    score_matrix: np.ndarray,
    test_pairs: pd.DataFrame,
    protein_ids: list[str],
    reaction_ids: list[str],
    budgets: tuple[int, ...],
) -> None:
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
        positives = set(group["Entry"].astype(str))
        metrics = rank_metrics(
            score_matrix[reaction_to_row[str(reaction_id)]],
            protein_ids,
            positives,
            set(),
            budgets,
        )
        records.append(
            {
                "split_id": split_id,
                "method": method,
                "direction": "reaction_to_enzyme",
                "query_id": reaction_id,
                **metrics,
            }
        )
    for protein_id, group in test_pairs.groupby("Entry", sort=True):
        positives = set(group["rhea_id"].astype(str))
        metrics = rank_metrics(
            score_matrix[:, protein_to_row[str(protein_id)]],
            reaction_ids,
            positives,
            set(),
            budgets,
        )
        records.append(
            {
                "split_id": split_id,
                "method": method,
                "direction": "enzyme_to_reaction",
                "query_id": protein_id,
                **metrics,
            }
        )


def partition(split_id: str, development_fold: int) -> str:
    protein_fold, reaction_fold = map(
        int, str(split_id).replace("p", "").split("_r")
    )
    return (
        "development_9_cells"
        if development_fold in {protein_fold, reaction_fold}
        else "frozen_16_cells"
    )


def aggregate(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n_query_cells": ("query_id", "size"),
        "n_unique_queries": ("query_id", "nunique"),
        "mean_reciprocal_rank": ("reciprocal_rank", "mean"),
        "median_best_positive_rank": ("best_positive_rank", "median"),
    }
    for budget in budgets:
        aggregations[f"hit_probability_at_{budget}"] = (f"hit_at_{budget}", "mean")
        aggregations[f"positive_recall_at_{budget}"] = (
            f"positive_recall_at_{budget}",
            "mean",
        )
    return (
        frame.groupby(["partition", "direction", "method"])
        .agg(**aggregations)
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse current PU and Horizyn-adapter rankings under strict double-cold splits."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--current-dir", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--horizyn-dir", type=Path, default=DEFAULT_HORIZYN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weights", default=",".join(map(str, DEFAULT_WEIGHTS)))
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--development-fold", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    weights = tuple(float(value) for value in args.weights.split(",") if value)
    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    if any(value < 0 or value > 1 for value in weights):
        raise ValueError("Fusion weights must lie in [0, 1]")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cache_dir = args.cache_dir.resolve()
    proteins = pd.read_csv(cache_dir / "protein_entities.csv", dtype=str).fillna("")
    reactions = pd.read_csv(cache_dir / "reaction_entities.csv", dtype=str).fillna("")
    pairs = pd.read_csv(cache_dir / "marts_pair_folds.csv", dtype=str).fillna("")
    pairs["protein_fold"] = pd.to_numeric(pairs["protein_fold"]).astype(int)
    pairs["reaction_fold"] = pd.to_numeric(pairs["reaction_fold"]).astype(int)
    pairs["protein_seen"] = boolean_series(pairs["protein_seen"])
    pairs["reaction_seen"] = boolean_series(pairs["reaction_seen"])
    protein_ids = proteins["protein_id"].astype(str).tolist()
    reaction_ids = reactions["reaction_id"].astype(str).tolist()
    protein_features_np = normalize_rows(np.load(cache_dir / "protein_features.npy"))
    reaction_features_np = np.load(cache_dir / "reaction_features.npy").astype(np.float32)
    horizyn_reactions = normalize_rows(
        np.load(args.horizyn_dir / "horizyn_reaction_embeddings.npy")
    )
    protein_features = torch.as_tensor(
        protein_features_np, dtype=torch.float32, device=device
    )
    reaction_features = torch.as_tensor(
        reaction_features_np, dtype=torch.float32, device=device
    )

    records: list[dict[str, object]] = []
    for protein_fold in range(5):
        for reaction_fold in range(5):
            split_id = f"p{protein_fold}_r{reaction_fold}"
            test_pairs = pairs[
                (pairs["protein_fold"] == protein_fold)
                & (pairs["reaction_fold"] == reaction_fold)
                & (~pairs["protein_seen"])
                & (~pairs["reaction_seen"])
            ].drop_duplicates(["rhea_id", "Entry"])
            if test_pairs.empty:
                continue
            current = load_current_score_matrix(
                args.current_dir.resolve() / "models",
                split_id,
                protein_features,
                reaction_features,
                device,
            )
            horizyn = load_horizyn_score_matrix(
                args.horizyn_dir.resolve() / "models",
                split_id,
                protein_features,
                horizyn_reactions,
                device,
            )
            append_metrics(
                records,
                split_id,
                "current_direct",
                current,
                test_pairs,
                protein_ids,
                reaction_ids,
                budgets,
            )
            append_metrics(
                records,
                split_id,
                "horizyn_direct",
                horizyn,
                test_pairs,
                protein_ids,
                reaction_ids,
                budgets,
            )
            current_ranks = np.stack(
                [tied_rank_percentile(row, protein_ids) for row in current]
            )
            horizyn_ranks = np.stack(
                [tied_rank_percentile(row, protein_ids) for row in horizyn]
            )
            current_ranks_e2r = np.stack(
                [tied_rank_percentile(current[:, column], reaction_ids) for column in range(current.shape[1])],
                axis=1,
            )
            horizyn_ranks_e2r = np.stack(
                [tied_rank_percentile(horizyn[:, column], reaction_ids) for column in range(horizyn.shape[1])],
                axis=1,
            )
            for weight in weights:
                fused = weight * current_ranks + (1 - weight) * horizyn_ranks
                fused_e2r = (
                    weight * current_ranks_e2r
                    + (1 - weight) * horizyn_ranks_e2r
                )
                method = f"rank_fusion_current_{weight:g}"
                protein_to_row = {
                    value: index for index, value in enumerate(protein_ids)
                }
                reaction_to_row = {
                    value: index for index, value in enumerate(reaction_ids)
                }
                for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
                    metrics = rank_metrics(
                        fused[reaction_to_row[str(reaction_id)]],
                        protein_ids,
                        set(group["Entry"].astype(str)),
                        set(),
                        budgets,
                    )
                    records.append(
                        {
                            "split_id": split_id,
                            "method": method,
                            "direction": "reaction_to_enzyme",
                            "query_id": reaction_id,
                            **metrics,
                        }
                    )
                for protein_id, group in test_pairs.groupby("Entry", sort=True):
                    metrics = rank_metrics(
                        fused_e2r[:, protein_to_row[str(protein_id)]],
                        reaction_ids,
                        set(group["rhea_id"].astype(str)),
                        set(),
                        budgets,
                    )
                    records.append(
                        {
                            "split_id": split_id,
                            "method": method,
                            "direction": "enzyme_to_reaction",
                            "query_id": protein_id,
                            **metrics,
                        }
                    )

    query_metrics = pd.DataFrame(records)
    query_metrics["partition"] = query_metrics["split_id"].map(
        lambda value: partition(value, args.development_fold)
    )
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics = aggregate(query_metrics, budgets)
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    selected_rows: list[dict[str, object]] = []
    for direction in metrics["direction"].unique():
        development = metrics[
            metrics["partition"].eq("development_9_cells")
            & metrics["direction"].eq(direction)
        ]
        frozen = metrics[
            metrics["partition"].eq("frozen_16_cells")
            & metrics["direction"].eq(direction)
        ]
        for budget in budgets:
            chosen = development.sort_values(
                [f"hit_probability_at_{budget}", "mean_reciprocal_rank"],
                ascending=[False, False],
            ).iloc[0]
            frozen_row = frozen[frozen["method"].eq(chosen["method"])].iloc[0]
            selected_rows.append(
                {
                    "direction": direction,
                    "budget": budget,
                    "selected_method": chosen["method"],
                    "development_hit": chosen[f"hit_probability_at_{budget}"],
                    "frozen_hit": frozen_row[f"hit_probability_at_{budget}"],
                    "frozen_mrr": frozen_row["mean_reciprocal_rank"],
                    "frozen_median_best_rank": frozen_row["median_best_positive_rank"],
                }
            )
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(output_dir / "development_selected_frozen_results.csv", index=False)
    summary = {
        "current_dir": str(args.current_dir.resolve()),
        "horizyn_dir": str(args.horizyn_dir.resolve()),
        "development_fold": args.development_fold,
        "weights": weights,
        "budgets": budgets,
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "selected": str(
                output_dir / "development_selected_frozen_results.csv"
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
