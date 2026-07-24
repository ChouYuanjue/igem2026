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

from projects.active.terpene_screening.evaluate_marts_adapted_neighbor_hybrid import (  # noqa: E402
    encode_models,
    enzyme_to_reaction_transfer,
    load_models,
    normalize_rows,
)
from projects.active.terpene_screening.rank_open_world import tied_rank_percentile  # noqa: E402
from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics  # noqa: E402

DEFAULT_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_PRIMARY_MODEL = ROOT / "results/terpene_marts_domain_adaptation_freeze_reaction"
DEFAULT_SECONDARY_MODEL = ROOT / "results/terpene_marts_domain_adaptation_hardneg128_e50"
DEFAULT_OUTPUT = ROOT / "results/terpene_e2r_route_interleaving"
DEFAULT_BUDGETS = (3, 10, 20)


def route_scores(
    protein_id: str,
    direct_matrix: np.ndarray,
    reaction_embedding_sets: list[np.ndarray],
    protein_features: np.ndarray,
    protein_to_row: dict[str, int],
    reaction_to_row: dict[str, int],
    train_by_protein: dict[str, list[str]],
    reaction_ids: list[str],
    neighbor_k: int,
    direct_weight: float,
) -> np.ndarray:
    direct = direct_matrix[:, protein_to_row[protein_id]]
    transfer = enzyme_to_reaction_transfer(
        protein_id,
        train_by_protein,
        protein_features,
        protein_to_row,
        reaction_to_row,
        reaction_embedding_sets,
        neighbor_k,
    )
    if transfer is None:
        return tied_rank_percentile(direct, reaction_ids)
    return (
        direct_weight * tied_rank_percentile(direct, reaction_ids)
        + (1.0 - direct_weight) * tied_rank_percentile(transfer, reaction_ids)
    ).astype(np.float32)


def rank_positions(scores: np.ndarray, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((np.asarray(ids), -scores))
    positions = np.empty(len(order), dtype=np.int64)
    positions[order] = np.arange(1, len(order) + 1)
    return order, positions


def ordered_panel_scores(order: list[int], candidate_count: int) -> np.ndarray:
    seen: set[int] = set()
    unique: list[int] = []
    for value in order:
        index = int(value)
        if index not in seen:
            seen.add(index)
            unique.append(index)
    if len(unique) != candidate_count:
        raise ValueError(
            f"Interleaved order contains {len(unique)} unique candidates; expected {candidate_count}"
        )
    scores = np.empty(candidate_count, dtype=np.float32)
    for position, index in enumerate(unique):
        scores[index] = float(candidate_count - position)
    return scores


def prefix_interleave(
    primary_order: np.ndarray,
    secondary_order: np.ndarray,
    primary_prefix: int,
) -> np.ndarray:
    return ordered_panel_scores(
        [*primary_order[:primary_prefix], *secondary_order, *primary_order[primary_prefix:]],
        len(primary_order),
    )


def alternating_interleave(
    primary_order: np.ndarray,
    secondary_order: np.ndarray,
    start_primary: bool,
) -> np.ndarray:
    result: list[int] = []
    primary_index = 0
    secondary_index = 0
    turn_primary = start_primary
    seen: set[int] = set()
    while len(result) < len(primary_order):
        source = primary_order if turn_primary else secondary_order
        index_pointer = primary_index if turn_primary else secondary_index
        while index_pointer < len(source) and int(source[index_pointer]) in seen:
            index_pointer += 1
        if index_pointer < len(source):
            value = int(source[index_pointer])
            result.append(value)
            seen.add(value)
            index_pointer += 1
        if turn_primary:
            primary_index = index_pointer
        else:
            secondary_index = index_pointer
        turn_primary = not turn_primary
        if primary_index >= len(primary_order) and secondary_index >= len(secondary_order):
            break
    for source in (primary_order, secondary_order):
        for value in source:
            if int(value) not in seen:
                result.append(int(value))
                seen.add(int(value))
    return ordered_panel_scores(result, len(primary_order))


def reciprocal_rank_fusion(
    primary_positions: np.ndarray,
    secondary_positions: np.ndarray,
    primary_weight: float,
    constant: float,
) -> np.ndarray:
    return (
        primary_weight / (constant + primary_positions)
        + (1.0 - primary_weight) / (constant + secondary_positions)
    ).astype(np.float32)


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
    return frame.groupby("method").agg(**aggregations).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interleave complementary E2R routes without retraining their encoders."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--primary-model-dir", type=Path, default=DEFAULT_PRIMARY_MODEL)
    parser.add_argument("--secondary-model-dir", type=Path, default=DEFAULT_SECONDARY_MODEL)
    parser.add_argument("--primary-neighbor-k", type=int, default=5)
    parser.add_argument("--primary-direct-weight", type=float, default=0.5)
    parser.add_argument("--secondary-neighbor-k", type=int, default=3)
    parser.add_argument("--secondary-direct-weight", type=float, default=0.9)
    parser.add_argument("--prefixes", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--rrf-weights", default="0.2,0.35,0.5,0.65,0.8")
    parser.add_argument("--rrf-constants", default="0,10,60")
    parser.add_argument("--budgets", default="3,10,20")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    prefixes = tuple(int(value) for value in args.prefixes.split(",") if value)
    rrf_weights = tuple(float(value) for value in args.rrf_weights.split(",") if value)
    rrf_constants = tuple(float(value) for value in args.rrf_constants.split(",") if value)
    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    if any(value <= 0 for value in prefixes):
        raise ValueError("Prefixes must be positive")
    if any(not 0 <= value <= 1 for value in rrf_weights):
        raise ValueError("RRF weights must be within [0, 1]")
    if any(value < 0 for value in rrf_constants):
        raise ValueError("RRF constants must be non-negative")

    cache = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    protein_table = pd.read_csv(cache / "protein_entities.csv", dtype=str).fillna("")
    reaction_table = pd.read_csv(cache / "reaction_entities.csv", dtype=str).fillna("")
    pairs = pd.read_csv(cache / "marts_pair_folds.csv", dtype=str).fillna("")
    pairs["protein_fold"] = pd.to_numeric(pairs["protein_fold"]).astype(int)
    pairs["reaction_fold"] = pd.to_numeric(pairs["reaction_fold"]).astype(int)
    pairs["protein_seen"] = pairs["protein_seen"].str.lower().eq("true")
    pairs["reaction_seen"] = pairs["reaction_seen"].str.lower().eq("true")
    protein_ids = protein_table["protein_id"].astype(str).tolist()
    reaction_ids = reaction_table["reaction_id"].astype(str).tolist()
    protein_features = normalize_rows(
        np.load(cache / "protein_features.npy").astype(np.float32)
    )
    reaction_features = np.load(cache / "reaction_features.npy").astype(np.float32)
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}

    records: list[dict[str, object]] = []
    top_panels: list[dict[str, object]] = []
    for protein_fold in sorted(pairs["protein_fold"].unique()):
        for reaction_fold in sorted(pairs["reaction_fold"].unique()):
            split_id = f"p{protein_fold}_r{reaction_fold}"
            train_pairs = pairs[
                pairs["protein_fold"].ne(protein_fold)
                & pairs["reaction_fold"].ne(reaction_fold)
            ]
            test_pairs = pairs[
                pairs["protein_fold"].eq(protein_fold)
                & pairs["reaction_fold"].eq(reaction_fold)
                & ~pairs["protein_seen"]
                & ~pairs["reaction_seen"]
            ]
            if test_pairs.empty:
                continue
            primary_models = load_models(args.primary_model_dir.resolve(), split_id, device)
            secondary_models = load_models(args.secondary_model_dir.resolve(), split_id, device)
            _, primary_reactions, primary_direct = encode_models(
                primary_models, protein_features, reaction_features, device
            )
            _, secondary_reactions, secondary_direct = encode_models(
                secondary_models, protein_features, reaction_features, device
            )
            train_by_protein = {
                protein_id: sorted(set(group["rhea_id"].astype(str)))
                for protein_id, group in train_pairs.groupby("Entry")
            }
            for protein_id, group in test_pairs.groupby("Entry", sort=True):
                positives = set(group["rhea_id"].astype(str))
                primary = route_scores(
                    protein_id,
                    primary_direct,
                    primary_reactions,
                    protein_features,
                    protein_to_row,
                    reaction_to_row,
                    train_by_protein,
                    reaction_ids,
                    args.primary_neighbor_k,
                    args.primary_direct_weight,
                )
                secondary = route_scores(
                    protein_id,
                    secondary_direct,
                    secondary_reactions,
                    protein_features,
                    protein_to_row,
                    reaction_to_row,
                    train_by_protein,
                    reaction_ids,
                    args.secondary_neighbor_k,
                    args.secondary_direct_weight,
                )
                primary_order, primary_positions = rank_positions(primary, reaction_ids)
                secondary_order, secondary_positions = rank_positions(secondary, reaction_ids)
                score_map: dict[str, np.ndarray] = {
                    "primary": primary,
                    "secondary": secondary,
                    "max_percentile": np.maximum(primary, secondary),
                    "alternating_primary_first": alternating_interleave(
                        primary_order, secondary_order, True
                    ),
                    "alternating_secondary_first": alternating_interleave(
                        primary_order, secondary_order, False
                    ),
                }
                for prefix in prefixes:
                    score_map[f"primary_prefix_{prefix}"] = prefix_interleave(
                        primary_order, secondary_order, prefix
                    )
                    score_map[f"secondary_prefix_{prefix}"] = prefix_interleave(
                        secondary_order, primary_order, prefix
                    )
                for weight in rrf_weights:
                    for constant in rrf_constants:
                        score_map[f"rrf_primary_{weight:g}_c{constant:g}"] = reciprocal_rank_fusion(
                            primary_positions,
                            secondary_positions,
                            weight,
                            constant,
                        )
                for method, scores in score_map.items():
                    metrics = rank_metrics(scores, reaction_ids, positives, set(), budgets)
                    records.append(
                        {
                            "split_id": split_id,
                            "protein_fold": protein_fold,
                            "reaction_fold": reaction_fold,
                            "method": method,
                            "query_id": protein_id,
                            **metrics,
                        }
                    )
                    order, _ = rank_positions(scores, reaction_ids)
                    for rank, candidate_index in enumerate(order[:20], start=1):
                        top_panels.append(
                            {
                                "split_id": split_id,
                                "method": method,
                                "query_id": protein_id,
                                "rank": rank,
                                "reaction_id": reaction_ids[int(candidate_index)],
                                "is_positive": int(reaction_ids[int(candidate_index)] in positives),
                            }
                        )
            del primary_models, secondary_models
            if device.type == "cuda":
                torch.cuda.empty_cache()

    query_metrics = pd.DataFrame(records)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    pd.DataFrame(top_panels).to_csv(output_dir / "top20_panels.csv", index=False)
    metrics = aggregate(query_metrics, budgets)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    best_rows: list[pd.DataFrame] = []
    for budget in budgets:
        selected = metrics.sort_values(
            [f"hit_probability_at_{budget}", "mean_reciprocal_rank", "method"],
            ascending=[False, False, True],
        ).head(1).copy()
        selected.insert(1, "selection_budget", budget)
        best_rows.append(selected)
    best = pd.concat(best_rows, ignore_index=True)
    best.to_csv(output_dir / "best_methods.csv", index=False)
    summary = {
        "cache_dir": str(cache),
        "primary_model_dir": str(args.primary_model_dir.resolve()),
        "secondary_model_dir": str(args.secondary_model_dir.resolve()),
        "primary_route": {
            "neighbor_k": args.primary_neighbor_k,
            "direct_weight": args.primary_direct_weight,
        },
        "secondary_route": {
            "neighbor_k": args.secondary_neighbor_k,
            "direct_weight": args.secondary_direct_weight,
        },
        "prefixes": prefixes,
        "rrf_weights": rrf_weights,
        "rrf_constants": rrf_constants,
        "budgets": budgets,
        "n_methods": int(query_metrics["method"].nunique()),
        "n_query_method_rows": int(len(query_metrics)),
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "top20_panels": str(output_dir / "top20_panels.csv"),
            "best_methods": str(output_dir / "best_methods.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
