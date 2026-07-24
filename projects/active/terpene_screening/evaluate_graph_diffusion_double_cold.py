from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics  # noqa: E402

DEFAULT_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_OUTPUT = ROOT / "results/terpene_graph_diffusion_double_cold"
DEFAULT_BUDGETS = (3, 10, 20)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    denominator[denominator == 0] = 1
    return matrix / denominator


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def boolean_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"1", "true", "yes"})


def topk_transition(similarity: np.ndarray, k: int, temperature: float) -> csr_matrix:
    if k <= 0:
        raise ValueError("graph k must be positive")
    if temperature <= 0:
        raise ValueError("graph temperature must be positive")
    n = similarity.shape[0]
    k = min(k, max(1, n - 1))
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row in range(n):
        scores = similarity[row].astype(np.float64, copy=True)
        scores[row] = -np.inf
        indices = np.argpartition(scores, -k)[-k:]
        indices = indices[np.argsort(-scores[indices], kind="stable")]
        finite = np.isfinite(scores[indices])
        indices = indices[finite]
        if not len(indices):
            continue
        local = scores[indices]
        local = np.exp((local - local.max()) / temperature)
        local /= local.sum()
        rows.extend([row] * len(indices))
        columns.extend(indices.tolist())
        values.extend(local.tolist())
    return csr_matrix((values, (rows, columns)), shape=(n, n), dtype=np.float32)


def query_seed_vector(
    query_row: int,
    similarity: np.ndarray,
    allowed_rows: np.ndarray,
    topk: int,
    temperature: float,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("seed temperature must be positive")
    allowed_rows = np.asarray(allowed_rows, dtype=np.int64)
    allowed_rows = allowed_rows[allowed_rows != query_row]
    result = np.zeros(similarity.shape[0], dtype=np.float32)
    if not len(allowed_rows):
        return result
    scores = similarity[query_row, allowed_rows].astype(np.float64)
    k = min(topk, len(allowed_rows))
    selected_local = np.argpartition(scores, -k)[-k:]
    selected_local = selected_local[np.argsort(-scores[selected_local], kind="stable")]
    selected_rows = allowed_rows[selected_local]
    selected_scores = scores[selected_local]
    weights = np.exp((selected_scores - selected_scores.max()) / temperature)
    weights /= weights.sum()
    result[selected_rows] = weights.astype(np.float32)
    return result


def propagate(
    initial: np.ndarray,
    transition: csr_matrix,
    steps: int,
    restart: float,
) -> np.ndarray:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not 0 <= restart <= 1:
        raise ValueError("restart must be within [0, 1]")
    score = initial.astype(np.float32, copy=True)
    for _ in range(steps):
        score = restart * initial + (1 - restart) * np.asarray(transition @ score).reshape(-1)
    return score.astype(np.float32)


def aggregate_metrics(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n_queries": ("query_id", "size"),
        "mean_reciprocal_rank": ("reciprocal_rank", "mean"),
        "median_best_positive_rank": ("best_positive_rank", "median"),
    }
    for budget in budgets:
        aggregations[f"hit_probability_at_{budget}"] = (f"hit_at_{budget}", "mean")
        aggregations[f"positive_recall_at_{budget}"] = (f"positive_recall_at_{budget}", "mean")
    return frame.groupby(["direction", "method"]).agg(**aggregations).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict double-cold two-graph label propagation benchmark.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reaction-seed-k", default="5,10,20")
    parser.add_argument("--protein-seed-k", default="5,10,20")
    parser.add_argument("--protein-graph-k", default="16,32")
    parser.add_argument("--reaction-graph-k", default="16,32")
    parser.add_argument("--steps", default="0,1,2")
    parser.add_argument("--restart", default="0.5")
    parser.add_argument("--seed-temperature", type=float, default=0.08)
    parser.add_argument("--graph-temperature", type=float, default=0.08)
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    args = parser.parse_args()

    reaction_seed_ks = parse_ints(args.reaction_seed_k)
    protein_seed_ks = parse_ints(args.protein_seed_k)
    protein_graph_ks = parse_ints(args.protein_graph_k)
    reaction_graph_ks = parse_ints(args.reaction_graph_k)
    steps_values = parse_ints(args.steps)
    restarts = parse_floats(args.restart)
    budgets = parse_ints(args.budgets)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    protein_features = normalize_rows(np.load(cache_dir / "protein_features.npy"))
    protein_similarity = np.clip(protein_features @ protein_features.T, -1.0, 1.0).astype(np.float32)
    reaction_similarity = np.load(cache_dir / "reaction_zero_shot_similarity.npy").astype(np.float32)
    if reaction_similarity.shape != (len(reaction_ids), len(reaction_ids)):
        raise ValueError("Reaction similarity matrix does not match reaction entity order")

    protein_transitions = {
        k: topk_transition(protein_similarity, k, args.graph_temperature)
        for k in protein_graph_ks
    }
    reaction_transitions = {
        k: topk_transition(reaction_similarity, k, args.graph_temperature)
        for k in reaction_graph_ks
    }

    records: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for protein_fold in range(5):
        for reaction_fold in range(5):
            split_id = f"p{protein_fold}_r{reaction_fold}"
            train = pairs[
                (pairs["protein_fold"] != protein_fold)
                & (pairs["reaction_fold"] != reaction_fold)
            ].drop_duplicates(["rhea_id", "Entry"])
            test = pairs[
                (pairs["protein_fold"] == protein_fold)
                & (pairs["reaction_fold"] == reaction_fold)
                & (~pairs["protein_seen"])
                & (~pairs["reaction_seen"])
            ].drop_duplicates(["rhea_id", "Entry"])
            split_rows.append(
                {
                    "split_id": split_id,
                    "train_pairs": len(train),
                    "test_pairs": len(test),
                    "test_reactions": test["rhea_id"].nunique(),
                    "test_proteins": test["Entry"].nunique(),
                }
            )
            if test.empty:
                continue

            adjacency = np.zeros((len(reaction_ids), len(protein_ids)), dtype=np.float32)
            for row in train.itertuples(index=False):
                reaction_row = reaction_to_row.get(str(row.rhea_id))
                protein_row = protein_to_row.get(str(row.Entry))
                if reaction_row is not None and protein_row is not None:
                    adjacency[reaction_row, protein_row] = 1.0
            reaction_degree = adjacency.sum(axis=1, keepdims=True)
            reaction_degree[reaction_degree == 0] = 1
            reaction_normalized = adjacency / reaction_degree
            protein_degree = adjacency.sum(axis=0, keepdims=True)
            protein_degree[protein_degree == 0] = 1
            protein_normalized = adjacency / protein_degree
            train_reaction_rows = np.flatnonzero(adjacency.sum(axis=1) > 0)
            train_protein_rows = np.flatnonzero(adjacency.sum(axis=0) > 0)

            for reaction_id, group in test.groupby("rhea_id", sort=True):
                query_row = reaction_to_row[str(reaction_id)]
                positives = set(group["Entry"].astype(str))
                for seed_k in reaction_seed_ks:
                    reaction_seed = query_seed_vector(
                        query_row,
                        reaction_similarity,
                        train_reaction_rows,
                        seed_k,
                        args.seed_temperature,
                    )
                    initial_protein = reaction_seed @ reaction_normalized
                    for graph_k, transition in protein_transitions.items():
                        for steps in steps_values:
                            for restart in restarts:
                                score = propagate(initial_protein, transition, steps, restart)
                                method = f"rseed{seed_k}_pgraph{graph_k}_s{steps}_a{restart:g}"
                                metrics = rank_metrics(score, protein_ids, positives, set(), budgets)
                                records.append(
                                    {
                                        "split_id": split_id,
                                        "direction": "reaction_to_enzyme",
                                        "query_id": reaction_id,
                                        "method": method,
                                        **metrics,
                                    }
                                )

            for protein_id, group in test.groupby("Entry", sort=True):
                query_row = protein_to_row[str(protein_id)]
                positives = set(group["rhea_id"].astype(str))
                for seed_k in protein_seed_ks:
                    protein_seed = query_seed_vector(
                        query_row,
                        protein_similarity,
                        train_protein_rows,
                        seed_k,
                        args.seed_temperature,
                    )
                    initial_reaction = protein_seed @ protein_normalized.T
                    for graph_k, transition in reaction_transitions.items():
                        for steps in steps_values:
                            for restart in restarts:
                                score = propagate(initial_reaction, transition, steps, restart)
                                method = f"pseed{seed_k}_rgraph{graph_k}_s{steps}_a{restart:g}"
                                metrics = rank_metrics(score, reaction_ids, positives, set(), budgets)
                                records.append(
                                    {
                                        "split_id": split_id,
                                        "direction": "enzyme_to_reaction",
                                        "query_id": protein_id,
                                        "method": method,
                                        **metrics,
                                    }
                                )

    query_metrics = pd.DataFrame(records)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics = aggregate_metrics(query_metrics, budgets)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    best_rows: list[pd.DataFrame] = []
    for direction, group in metrics.groupby("direction"):
        for budget in budgets:
            ordered = group.sort_values(
                [f"hit_probability_at_{budget}", "mean_reciprocal_rank"],
                ascending=[False, False],
            ).head(1).copy()
            ordered.insert(2, "selection_budget", budget)
            best_rows.append(ordered)
    best = pd.concat(best_rows, ignore_index=True)
    best.to_csv(output_dir / "best_methods.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_dir / "split_summary.csv", index=False)
    summary = {
        "cache_dir": str(cache_dir),
        "n_proteins": len(protein_ids),
        "n_reactions": len(reaction_ids),
        "reaction_seed_k": reaction_seed_ks,
        "protein_seed_k": protein_seed_ks,
        "protein_graph_k": protein_graph_ks,
        "reaction_graph_k": reaction_graph_ks,
        "steps": steps_values,
        "restart": restarts,
        "seed_temperature": args.seed_temperature,
        "graph_temperature": args.graph_temperature,
        "budgets": budgets,
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "best_methods": str(output_dir / "best_methods.csv"),
            "split_summary": str(output_dir / "split_summary.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
