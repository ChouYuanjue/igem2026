from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_model_rank_fusion_double_cold import (  # noqa: E402
    load_score_matrix,
    parse_source,
    rank_percentiles,
    split_partition,
)
from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics  # noqa: E402

DEFAULT_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_OUTPUT = ROOT / "results/terpene_lambdarank_stacking_double_cold"
DEFAULT_BUDGETS = (3, 10, 20)


def row_zscore(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((matrix - mean) / std).astype(np.float32)


def build_feature_tensor(source_scores: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    labels = list(source_scores)
    rank_blocks = [rank_percentiles(source_scores[label]) for label in labels]
    z_blocks = [row_zscore(source_scores[label]) for label in labels]
    rank_stack = np.stack(rank_blocks, axis=-1)
    z_stack = np.stack(z_blocks, axis=-1)
    derived = np.stack(
        [
            rank_stack.mean(axis=-1),
            rank_stack.std(axis=-1),
            rank_stack.min(axis=-1),
            rank_stack.max(axis=-1),
            z_stack.mean(axis=-1),
            z_stack.std(axis=-1),
            z_stack.max(axis=-1),
        ],
        axis=-1,
    )
    names = (
        [f"rank_{label}" for label in labels]
        + [f"zscore_{label}" for label in labels]
        + [
            "rank_mean",
            "rank_std",
            "rank_min",
            "rank_max",
            "zscore_mean",
            "zscore_std",
            "zscore_max",
        ]
    )
    return np.concatenate([rank_stack, z_stack, derived], axis=-1).astype(np.float32), names


def prepare_rank_training_data(
    features: np.ndarray,
    train_pairs: pd.DataFrame,
    query_column: str,
    candidate_column: str,
    query_to_row: dict[str, int],
    candidate_to_row: dict[str, int],
    candidate_groups: dict[str, str],
    hard_negatives: int,
    random_negatives: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_candidates = sorted(set(train_pairs[candidate_column].astype(str)) & set(candidate_to_row))
    train_candidate_rows = np.asarray(
        [candidate_to_row[value] for value in train_candidates], dtype=np.int64
    )
    candidate_group_array = np.asarray(
        [candidate_groups.get(value, "") for value in train_candidates], dtype=object
    )
    feature_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    group_sizes: list[int] = []
    audit_rows: list[dict[str, object]] = []
    for query_id, group in train_pairs.groupby(query_column, sort=True):
        query_id = str(query_id)
        if query_id not in query_to_row:
            continue
        positive_ids = sorted(set(group[candidate_column].astype(str)) & set(candidate_to_row))
        if not positive_ids:
            continue
        positive_rows = np.asarray([candidate_to_row[value] for value in positive_ids], dtype=np.int64)
        positive_groups = {candidate_groups.get(value, "") for value in positive_ids} - {""}
        eligible = np.ones(len(train_candidates), dtype=bool)
        positive_local = np.isin(train_candidate_rows, positive_rows)
        eligible[positive_local] = False
        if positive_groups:
            eligible &= ~np.isin(candidate_group_array, list(positive_groups))
        eligible_indices = np.flatnonzero(eligible)
        if len(eligible_indices) == 0:
            continue
        query_row = query_to_row[query_id]
        hardness = features[query_row, train_candidate_rows, : len(features.shape) * 0 + 1].reshape(-1)
        # The first feature is a source-model rank percentile. Use the mean of all rank
        # features when available; callers order rank features first.
        n_rank_features = (features.shape[-1] - 7) // 2
        hardness = features[query_row, train_candidate_rows, :n_rank_features].mean(axis=1)
        hard_order = eligible_indices[np.argsort(-hardness[eligible_indices], kind="stable")]
        selected_hard = hard_order[:hard_negatives]
        remaining = np.setdiff1d(eligible_indices, selected_hard, assume_unique=False)
        random_count = min(random_negatives, len(remaining))
        selected_random = (
            rng.choice(remaining, size=random_count, replace=False)
            if random_count
            else np.empty(0, dtype=np.int64)
        )
        negative_rows = train_candidate_rows[np.concatenate([selected_hard, selected_random])]
        rows = np.concatenate([positive_rows, negative_rows])
        y = np.concatenate(
            [np.ones(len(positive_rows), dtype=np.float32), np.zeros(len(negative_rows), dtype=np.float32)]
        )
        feature_rows.append(features[query_row, rows])
        labels.append(y)
        group_sizes.append(len(rows))
        audit_rows.append(
            {
                "query_id": query_id,
                "positive_count": len(positive_rows),
                "hard_negative_count": len(selected_hard),
                "random_negative_count": len(selected_random),
                "group_size": len(rows),
            }
        )
    if not feature_rows:
        raise ValueError("No ranking groups could be built")
    return (
        np.concatenate(feature_rows, axis=0),
        np.concatenate(labels, axis=0),
        group_sizes,
        pd.DataFrame(audit_rows),
    )


def train_ranker(
    features: np.ndarray,
    labels: np.ndarray,
    group_sizes: list[int],
    objective: str,
    rounds: int,
    max_depth: int,
    learning_rate: float,
    min_child_weight: float,
    reg_lambda: float,
    nthread: int,
    seed: int,
) -> xgb.Booster:
    matrix = xgb.DMatrix(features, label=labels)
    matrix.set_group(group_sizes)
    params: dict[str, object] = {
        "objective": objective,
        "eval_metric": "ndcg@10",
        "tree_method": "hist",
        "max_depth": max_depth,
        "eta": learning_rate,
        "min_child_weight": min_child_weight,
        "lambda": reg_lambda,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": seed,
        "nthread": nthread,
        "verbosity": 0,
    }
    if objective == "rank:ndcg":
        params.update(
            {
                "lambdarank_pair_method": "topk",
                "lambdarank_num_pair_per_sample": 20,
            }
        )
    return xgb.train(params, matrix, num_boost_round=rounds)


def evaluate_direction(
    records: list[dict[str, object]],
    booster: xgb.Booster,
    features: np.ndarray,
    test_pairs: pd.DataFrame,
    split_id: str,
    direction: str,
    query_column: str,
    candidate_column: str,
    query_to_row: dict[str, int],
    candidate_ids: list[str],
    budgets: tuple[int, ...],
    method: str,
) -> None:
    for query_id, group in test_pairs.groupby(query_column, sort=True):
        query_id = str(query_id)
        if query_id not in query_to_row:
            continue
        positives = set(group[candidate_column].astype(str))
        query_features = features[query_to_row[query_id]]
        scores = booster.predict(xgb.DMatrix(query_features))
        metrics = rank_metrics(scores, candidate_ids, positives, set(), budgets)
        records.append(
            {
                "split_id": split_id,
                "method": method,
                "direction": direction,
                "query_id": query_id,
                **metrics,
            }
        )


def aggregate(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n_queries": ("query_id", "size"),
        "mean_reciprocal_rank": ("reciprocal_rank", "mean"),
        "median_best_positive_rank": ("best_positive_rank", "median"),
    }
    for budget in budgets:
        aggregations[f"hit_probability_at_{budget}"] = (f"hit_at_{budget}", "mean")
        aggregations[f"positive_recall_at_{budget}"] = (
            f"positive_recall_at_{budget}",
            "mean",
        )
    return frame.groupby(["method", "direction"]).agg(**aggregations).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-split LambdaRank stacking of strict TPS source models."
    )
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--partition", choices=["development", "frozen", "all"], default="development")
    parser.add_argument("--development-fold", type=int, default=4)
    parser.add_argument("--objective", choices=["rank:pairwise", "rank:ndcg"], default="rank:pairwise")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-child-weight", type=float, default=5.0)
    parser.add_argument("--reg-lambda", type=float, default=10.0)
    parser.add_argument("--hard-negatives", type=int, default=64)
    parser.add_argument("--random-negatives", type=int, default=32)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sources = dict(parse_source(value) for value in args.source)
    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    method = (
        f"lambdarank_{args.objective.replace(':', '_')}_d{args.max_depth}_r{args.rounds}"
        f"_h{args.hard_negatives}_n{args.random_negatives}"
    )

    protein_matrix = np.load(cache_dir / "protein_features.npy").astype(np.float32)
    reaction_matrix = np.load(cache_dir / "reaction_features.npy").astype(np.float32)
    protein_table = pd.read_csv(cache_dir / "protein_entities.csv", dtype=str).fillna("")
    reaction_table = pd.read_csv(cache_dir / "reaction_entities.csv", dtype=str).fillna("")
    pairs = pd.read_csv(cache_dir / "marts_pair_folds.csv", dtype=str).fillna("")
    pairs["protein_fold"] = pd.to_numeric(pairs["protein_fold"]).astype(int)
    pairs["reaction_fold"] = pd.to_numeric(pairs["reaction_fold"]).astype(int)
    protein_ids = protein_table["protein_id"].astype(str).tolist()
    reaction_ids = reaction_table["reaction_id"].astype(str).tolist()
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    protein_groups = dict(zip(protein_table["protein_id"], protein_table["cluster_id"]))
    reaction_groups = dict(zip(reaction_table["reaction_id"], reaction_table["cluster_id"]))
    device = torch.device(args.device)
    protein_tensor = torch.as_tensor(protein_matrix, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_matrix, dtype=torch.float32, device=device)

    records: list[dict[str, object]] = []
    audits: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    split_rows: list[dict[str, object]] = []
    for protein_fold in range(5):
        for reaction_fold in range(5):
            split_id = f"p{protein_fold}_r{reaction_fold}"
            partition = split_partition(split_id, args.development_fold)
            if args.partition == "development" and partition != "development_9_cells":
                continue
            if args.partition == "frozen" and partition != "frozen_16_cells":
                continue
            train_pairs = pairs[
                pairs["protein_fold"].ne(protein_fold)
                & pairs["reaction_fold"].ne(reaction_fold)
            ].copy()
            test_pairs = pairs[
                pairs["protein_fold"].eq(protein_fold)
                & pairs["reaction_fold"].eq(reaction_fold)
                & pairs["protein_seen"].str.lower().eq("false")
                & pairs["reaction_seen"].str.lower().eq("false")
            ].copy()
            if test_pairs.empty:
                continue
            raw_scores = {
                label: load_score_matrix(
                    result_dir, split_id, protein_tensor, reaction_tensor, device
                )
                for label, result_dir in sources.items()
            }
            r2e_features, feature_names = build_feature_tensor(raw_scores)
            e2r_features, _ = build_feature_tensor(
                {label: matrix.T for label, matrix in raw_scores.items()}
            )
            r2e_x, r2e_y, r2e_groups, r2e_audit = prepare_rank_training_data(
                r2e_features,
                train_pairs,
                "rhea_id",
                "Entry",
                reaction_to_row,
                protein_to_row,
                protein_groups,
                args.hard_negatives,
                args.random_negatives,
                args.seed + protein_fold * 100 + reaction_fold,
            )
            e2r_x, e2r_y, e2r_groups, e2r_audit = prepare_rank_training_data(
                e2r_features,
                train_pairs,
                "Entry",
                "rhea_id",
                protein_to_row,
                reaction_to_row,
                reaction_groups,
                args.hard_negatives,
                args.random_negatives,
                args.seed + 1000 + protein_fold * 100 + reaction_fold,
            )
            r2e_model = train_ranker(
                r2e_x,
                r2e_y,
                r2e_groups,
                args.objective,
                args.rounds,
                args.max_depth,
                args.learning_rate,
                args.min_child_weight,
                args.reg_lambda,
                args.threads,
                args.seed + protein_fold * 10 + reaction_fold,
            )
            e2r_model = train_ranker(
                e2r_x,
                e2r_y,
                e2r_groups,
                args.objective,
                args.rounds,
                args.max_depth,
                args.learning_rate,
                args.min_child_weight,
                args.reg_lambda,
                args.threads,
                args.seed + 1000 + protein_fold * 10 + reaction_fold,
            )
            evaluate_direction(
                records,
                r2e_model,
                r2e_features,
                test_pairs,
                split_id,
                "reaction_to_enzyme",
                "rhea_id",
                "Entry",
                reaction_to_row,
                protein_ids,
                budgets,
                method,
            )
            evaluate_direction(
                records,
                e2r_model,
                e2r_features,
                test_pairs,
                split_id,
                "enzyme_to_reaction",
                "Entry",
                "rhea_id",
                protein_to_row,
                reaction_ids,
                budgets,
                method,
            )
            for direction, booster in [
                ("reaction_to_enzyme", r2e_model),
                ("enzyme_to_reaction", e2r_model),
            ]:
                gain = booster.get_score(importance_type="gain")
                frame = pd.DataFrame(
                    {
                        "feature": feature_names,
                        "gain": [float(gain.get(f"f{index}", 0.0)) for index in range(len(feature_names))],
                    }
                )
                frame.insert(0, "direction", direction)
                frame.insert(0, "split_id", split_id)
                importances.append(frame)
            r2e_audit.insert(0, "direction", "reaction_to_enzyme")
            e2r_audit.insert(0, "direction", "enzyme_to_reaction")
            r2e_audit.insert(0, "split_id", split_id)
            e2r_audit.insert(0, "split_id", split_id)
            audits.extend([r2e_audit, e2r_audit])
            split_rows.append(
                {
                    "split_id": split_id,
                    "partition": partition,
                    "train_pairs": len(train_pairs),
                    "test_pairs": len(test_pairs),
                    "r2e_training_rows": len(r2e_y),
                    "e2r_training_rows": len(e2r_y),
                }
            )

    query_metrics = pd.DataFrame(records)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics = aggregate(query_metrics, budgets)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(audits, ignore_index=True).to_csv(output_dir / "training_group_audit.csv", index=False)
    pd.concat(importances, ignore_index=True).to_csv(output_dir / "feature_importance.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_dir / "split_summary.csv", index=False)
    summary = {
        "method": method,
        "sources": {label: str(path) for label, path in sources.items()},
        "partition": args.partition,
        "development_fold": args.development_fold,
        "objective": args.objective,
        "rounds": args.rounds,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "hard_negatives": args.hard_negatives,
        "random_negatives": args.random_negatives,
        "training_protocol": (
            "Each split ranker is trained only on that split's training pairs and training candidate universe; "
            "held-out labels are never used for fitting or negative sampling."
        ),
        "budgets": budgets,
        "device": str(device),
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "training_group_audit": str(output_dir / "training_group_audit.csv"),
            "feature_importance": str(output_dir / "feature_importance.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
