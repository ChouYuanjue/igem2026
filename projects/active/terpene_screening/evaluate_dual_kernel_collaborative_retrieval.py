from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_dual_tower_protocol_comparison import (  # noqa: E402
    DEFAULT_EMBEDDINGS,
    DEFAULT_POSITIVES,
    DEFAULT_STRICT_SPLITS,
    masked_rank_metrics,
)
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    build_reaction_features,
    load_protein_features,
)

DEFAULT_OUTPUT = ROOT / "results/terpene_dual_kernel_collaborative"
DEFAULT_BUDGETS = (3, 5, 10, 20)


@dataclass(frozen=True)
class KernelConfig:
    reaction_k: int
    protein_k: int
    temperature: float
    degree_power: float

    @property
    def name(self) -> str:
        return (
            f"rk{self.reaction_k}_pk{self.protein_k}_"
            f"t{self.temperature:g}_d{self.degree_power:g}"
        )


def parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result or any(item <= 0 for item in result):
        raise ValueError("Expected positive integers")
    return result


def parse_float_tuple(value: str) -> tuple[float, ...]:
    result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("Expected floats")
    return result


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    denominator[denominator == 0] = 1.0
    return matrix / denominator


def topk_affinity(
    similarity: np.ndarray,
    allowed_columns: np.ndarray,
    k: int,
    temperature: float,
) -> csr_matrix:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    allowed_columns = np.asarray(allowed_columns, dtype=np.int64)
    if not len(allowed_columns):
        raise ValueError("No allowed affinity columns")
    k = min(k, len(allowed_columns))
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row in range(similarity.shape[0]):
        local = np.asarray(similarity[row, allowed_columns], dtype=np.float64)
        selected = np.argpartition(local, -k)[-k:]
        selected = selected[np.argsort(-local[selected], kind="stable")]
        selected_columns = allowed_columns[selected]
        selected_scores = local[selected]
        weights = np.exp((selected_scores - selected_scores.max()) / temperature)
        denominator = weights.sum()
        if denominator <= 0 or not np.isfinite(denominator):
            weights = np.ones(len(selected), dtype=np.float64) / len(selected)
        else:
            weights /= denominator
        rows.extend([row] * len(selected_columns))
        columns.extend(selected_columns.tolist())
        values.extend(weights.astype(np.float32).tolist())
    return csr_matrix(
        (values, (rows, columns)),
        shape=similarity.shape,
        dtype=np.float32,
    )


def normalized_adjacency(
    pairs: pd.DataFrame,
    reaction_to_row: dict[str, int],
    protein_to_row: dict[str, int],
    shape: tuple[int, int],
    degree_power: float,
) -> csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    for row in pairs[["rhea_id", "Entry"]].drop_duplicates().itertuples(index=False):
        reaction = reaction_to_row.get(str(row.rhea_id))
        protein = protein_to_row.get(str(row.Entry))
        if reaction is not None and protein is not None:
            rows.append(reaction)
            columns.append(protein)
    data = np.ones(len(rows), dtype=np.float32)
    adjacency = csr_matrix((data, (rows, columns)), shape=shape, dtype=np.float32)
    if degree_power == 0:
        return adjacency
    reaction_degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    protein_degree = np.asarray(adjacency.sum(axis=0)).reshape(-1)
    reaction_scale = np.zeros_like(reaction_degree, dtype=np.float32)
    protein_scale = np.zeros_like(protein_degree, dtype=np.float32)
    reaction_scale[reaction_degree > 0] = reaction_degree[reaction_degree > 0] ** (-degree_power)
    protein_scale[protein_degree > 0] = protein_degree[protein_degree > 0] ** (-degree_power)
    return diags(reaction_scale) @ adjacency @ diags(protein_scale)


def aggregate_metrics(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (partition, config), group in frame.groupby(["partition", "config"], sort=True):
        row: dict[str, object] = {
            "partition": partition,
            "config": config,
            "n_queries": len(group),
            "mrr": group.reciprocal_rank.mean(),
            "median_rank": group.best_positive_rank.median(),
        }
        for budget in budgets:
            row[f"hit{budget}"] = group[f"hit_at_{budget}"].mean()
        row["pareto_score"] = sum(row[f"hit{budget}"] for budget in budgets) + row["mrr"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fold-local dual-kernel collaborative TPS retrieval."
    )
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--strict-splits", type=Path, default=DEFAULT_STRICT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict-partition", choices=["development", "frozen"], default="development")
    parser.add_argument("--reaction-k", default="5,10,20,50")
    parser.add_argument("--protein-k", default="5,10,20,50")
    parser.add_argument("--temperatures", default="0.03,0.08")
    parser.add_argument("--degree-powers", default="0.5,1")
    parser.add_argument("--configs", default="")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--ranking-depth", type=int, default=100)
    parser.add_argument("--max-cells", type=int, default=0)
    args = parser.parse_args()

    budgets = parse_int_tuple(args.budgets)
    if args.ranking_depth < max(budgets):
        raise ValueError("ranking depth must cover all budgets")
    if args.configs:
        configs = []
        for value in args.configs.split(";"):
            reaction_k, protein_k, temperature, degree_power = value.split(":")
            configs.append(
                KernelConfig(
                    int(reaction_k), int(protein_k), float(temperature), float(degree_power)
                )
            )
    else:
        configs = [
            KernelConfig(reaction_k, protein_k, temperature, degree_power)
            for reaction_k in parse_int_tuple(args.reaction_k)
            for protein_k in parse_int_tuple(args.protein_k)
            for temperature in parse_float_tuple(args.temperatures)
            for degree_power in parse_float_tuple(args.degree_powers)
        ]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    positives = pd.read_csv(args.positives, sep="\t", dtype=str).fillna("")
    positives = positives[["Entry", "rhea_id", "smiles_seq"]].drop_duplicates(
        ["Entry", "rhea_id"]
    )
    protein_features, protein_ids = load_protein_features(args.embedding_dir.resolve())
    reaction_features, reaction_ids, _, feature_schema = build_reaction_features(
        positives, "multiview"
    )
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    protein_similarity = np.clip(
        normalize_rows(protein_features) @ normalize_rows(protein_features).T,
        -1.0,
        1.0,
    ).astype(np.float32)
    reaction_similarity = np.clip(
        normalize_rows(reaction_features) @ normalize_rows(reaction_features).T,
        -1.0,
        1.0,
    ).astype(np.float32)

    strict = pd.read_csv(args.strict_splits, dtype=str).fillna("")
    strict[["protein_fold", "reaction_fold"]] = strict[
        ["protein_fold", "reaction_fold"]
    ].astype(int)
    strict = strict[["Entry", "rhea_id", "protein_fold", "reaction_fold"]].drop_duplicates(
        ["Entry", "rhea_id"]
    )
    all_positive_by_reaction = {
        reaction: set(group.Entry.astype(str))
        for reaction, group in strict.groupby("rhea_id", sort=True)
    }
    query_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    completed = 0
    stop = False
    identifiers = np.asarray(protein_ids)
    for protein_fold in range(5):
        for reaction_fold in range(5):
            development = protein_fold == 4 or reaction_fold == 4
            if args.strict_partition == "development" and not development:
                continue
            if args.strict_partition == "frozen" and development:
                continue
            if args.max_cells and completed >= args.max_cells:
                stop = True
                break
            train = strict[
                strict.protein_fold.ne(protein_fold)
                & strict.reaction_fold.ne(reaction_fold)
            ]
            test = strict[
                strict.protein_fold.eq(protein_fold)
                & strict.reaction_fold.eq(reaction_fold)
            ]
            if test.empty:
                completed += 1
                continue
            train_reaction_rows = np.asarray(
                sorted({reaction_to_row[value] for value in train.rhea_id.astype(str)}),
                dtype=np.int64,
            )
            train_protein_rows = np.asarray(
                sorted({protein_to_row[value] for value in train.Entry.astype(str)}),
                dtype=np.int64,
            )
            adjacency_by_degree = {
                degree_power: normalized_adjacency(
                    train,
                    reaction_to_row,
                    protein_to_row,
                    (len(reaction_ids), len(protein_ids)),
                    degree_power,
                )
                for degree_power in sorted({config.degree_power for config in configs})
            }
            reaction_affinity = {
                (config.reaction_k, config.temperature): topk_affinity(
                    reaction_similarity,
                    train_reaction_rows,
                    config.reaction_k,
                    config.temperature,
                )
                for config in configs
            }
            protein_affinity = {
                (config.protein_k, config.temperature): topk_affinity(
                    protein_similarity,
                    train_protein_rows,
                    config.protein_k,
                    config.temperature,
                )
                for config in configs
            }
            test_reactions = sorted(test.rhea_id.astype(str).unique())
            test_reaction_rows = np.asarray(
                [reaction_to_row[value] for value in test_reactions], dtype=np.int64
            )
            for config in configs:
                left = reaction_affinity[(config.reaction_k, config.temperature)][
                    test_reaction_rows
                ]
                right = protein_affinity[(config.protein_k, config.temperature)]
                scores = (left @ adjacency_by_degree[config.degree_power] @ right.T).toarray()
                for local_row, reaction_id in enumerate(test_reactions):
                    positives_for_query = set(
                        test.loc[test.rhea_id.eq(reaction_id), "Entry"].astype(str)
                    )
                    known_other = all_positive_by_reaction.get(reaction_id, set()) - positives_for_query
                    score = scores[local_row]
                    metrics = masked_rank_metrics(
                        score,
                        protein_ids,
                        positives_for_query,
                        known_other,
                        budgets,
                    )
                    query_rows.append(
                        {
                            "partition": args.strict_partition,
                            "protein_fold": protein_fold,
                            "reaction_fold": reaction_fold,
                            "reaction_id": reaction_id,
                            "config": config.name,
                            **metrics,
                        }
                    )
                    adjusted = score.copy()
                    for candidate in known_other:
                        row = protein_to_row.get(candidate)
                        if row is not None:
                            adjusted[row] = -np.inf
                    order = np.lexsort((identifiers, -adjusted))
                    rank = 0
                    for candidate_row in order:
                        if not np.isfinite(adjusted[candidate_row]):
                            continue
                        rank += 1
                        if rank > args.ranking_depth:
                            break
                        ranking_rows.append(
                            {
                                "partition": args.strict_partition,
                                "protein_fold": protein_fold,
                                "reaction_fold": reaction_fold,
                                "reaction_id": reaction_id,
                                "config": config.name,
                                "candidate_id": protein_ids[int(candidate_row)],
                                "rank": rank,
                                "score": float(adjusted[candidate_row]),
                            }
                        )
            completed += 1
        if stop:
            break

    query_frame = pd.DataFrame(query_rows)
    ranking_frame = pd.DataFrame(ranking_rows)
    metrics = aggregate_metrics(query_frame, budgets)
    query_frame.to_csv(output_dir / "query_metrics.csv", index=False)
    ranking_frame.to_csv(output_dir / "rankings.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    summary = {
        "method": "fold_local_dual_kernel_collaborative_retrieval",
        "strict_partition": args.strict_partition,
        "configs": [config.__dict__ for config in configs],
        "completed_cells": completed,
        "protein_count": len(protein_ids),
        "reaction_count": len(reaction_ids),
        "reaction_feature_schema": feature_schema,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics.sort_values(["pareto_score", "mrr"], ascending=False).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
