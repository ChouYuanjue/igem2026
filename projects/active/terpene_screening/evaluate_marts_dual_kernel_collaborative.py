from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_dual_kernel_collaborative_retrieval import (  # noqa: E402
    KernelConfig,
    normalized_adjacency,
    normalize_rows,
    parse_float_tuple,
    parse_int_tuple,
    topk_affinity,
)
from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics  # noqa: E402

DEFAULT_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_OUTPUT = ROOT / "results/terpene_marts_dual_kernel_collaborative"
DEFAULT_BUDGETS = (3, 10, 20)


def boolean_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"1", "true", "yes"})


def aggregate_metrics(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (partition, direction, config), group in frame.groupby(
        ["partition", "direction", "config"], sort=True
    ):
        row: dict[str, object] = {
            "partition": partition,
            "direction": direction,
            "config": config,
            "n_queries": len(group),
            "mrr": group.reciprocal_rank.mean(),
            "median_rank": group.best_positive_rank.median(),
        }
        for budget in budgets:
            row[f"hit{budget}"] = group[f"hit_at_{budget}"].mean()
            row[f"recall{budget}"] = group[f"positive_recall_at_{budget}"].mean()
        row["pareto_score"] = (
            sum(row[f"hit{budget}"] for budget in budgets) + row["mrr"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bidirectional strict MARTS dual-kernel collaborative retrieval."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict-partition", choices=["development", "frozen", "all"], default="development")
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
        configs: list[KernelConfig] = []
        for value in args.configs.split(";"):
            reaction_k, protein_k, temperature, degree_power = value.split(":")
            configs.append(
                KernelConfig(
                    int(reaction_k),
                    int(protein_k),
                    float(temperature),
                    float(degree_power),
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

    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    proteins = pd.read_csv(cache_dir / "protein_entities.csv", dtype=str).fillna("")
    reactions = pd.read_csv(cache_dir / "reaction_entities.csv", dtype=str).fillna("")
    pairs = pd.read_csv(cache_dir / "marts_pair_folds.csv", dtype=str).fillna("")
    pairs[["protein_fold", "reaction_fold"]] = pairs[
        ["protein_fold", "reaction_fold"]
    ].astype(int)
    pairs["protein_seen"] = boolean_series(pairs["protein_seen"])
    pairs["reaction_seen"] = boolean_series(pairs["reaction_seen"])

    protein_ids = proteins.protein_id.astype(str).tolist()
    reaction_ids = reactions.reaction_id.astype(str).tolist()
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    protein_features = normalize_rows(np.load(cache_dir / "protein_features.npy"))
    protein_similarity = np.clip(
        protein_features @ protein_features.T, -1.0, 1.0
    ).astype(np.float32)
    reaction_similarity = np.load(
        cache_dir / "reaction_zero_shot_similarity.npy"
    ).astype(np.float32)
    if reaction_similarity.shape != (len(reaction_ids), len(reaction_ids)):
        raise ValueError("Reaction similarity matrix shape mismatch")

    query_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    completed = 0
    protein_array = np.asarray(protein_ids)
    reaction_array = np.asarray(reaction_ids)
    stop = False
    for protein_fold in range(5):
        for reaction_fold in range(5):
            development = protein_fold == 4 or reaction_fold == 4
            partition = "development" if development else "frozen"
            if args.strict_partition != "all" and args.strict_partition != partition:
                continue
            if args.max_cells and completed >= args.max_cells:
                stop = True
                break
            split_id = f"p{protein_fold}_r{reaction_fold}"
            train = pairs[
                pairs.protein_fold.ne(protein_fold)
                & pairs.reaction_fold.ne(reaction_fold)
            ].drop_duplicates(["rhea_id", "Entry"])
            test = pairs[
                pairs.protein_fold.eq(protein_fold)
                & pairs.reaction_fold.eq(reaction_fold)
                & ~pairs.protein_seen
                & ~pairs.reaction_seen
            ].drop_duplicates(["rhea_id", "Entry"])
            split_rows.append(
                {
                    "split_id": split_id,
                    "partition": partition,
                    "train_pairs": len(train),
                    "test_pairs": len(test),
                    "test_reactions": test.rhea_id.nunique(),
                    "test_proteins": test.Entry.nunique(),
                }
            )
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
            adjacency = {
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
            test_proteins = sorted(test.Entry.astype(str).unique())
            test_reaction_rows = np.asarray(
                [reaction_to_row[value] for value in test_reactions], dtype=np.int64
            )
            test_protein_rows = np.asarray(
                [protein_to_row[value] for value in test_proteins], dtype=np.int64
            )
            for config in configs:
                reaction_left = reaction_affinity[
                    (config.reaction_k, config.temperature)
                ][test_reaction_rows]
                protein_right = protein_affinity[
                    (config.protein_k, config.temperature)
                ]
                reaction_query_scores = (
                    reaction_left @ adjacency[config.degree_power] @ protein_right.T
                ).toarray()

                reaction_left_all = reaction_affinity[
                    (config.reaction_k, config.temperature)
                ]
                protein_right_test = protein_affinity[
                    (config.protein_k, config.temperature)
                ][test_protein_rows]
                protein_query_scores = (
                    reaction_left_all
                    @ adjacency[config.degree_power]
                    @ protein_right_test.T
                ).toarray()

                for local_row, reaction_id in enumerate(test_reactions):
                    positives = set(
                        test.loc[test.rhea_id.eq(reaction_id), "Entry"].astype(str)
                    )
                    score = reaction_query_scores[local_row]
                    metrics = rank_metrics(score, protein_ids, positives, set(), budgets)
                    query_rows.append(
                        {
                            "split_id": split_id,
                            "partition": partition,
                            "direction": "reaction_to_enzyme",
                            "query_id": reaction_id,
                            "config": config.name,
                            **metrics,
                        }
                    )
                    order = np.lexsort((protein_array, -score))[: args.ranking_depth]
                    ranking_rows.extend(
                        {
                            "split_id": split_id,
                            "partition": partition,
                            "direction": "reaction_to_enzyme",
                            "query_id": reaction_id,
                            "config": config.name,
                            "rank": rank,
                            "candidate_id": protein_ids[int(index)],
                            "score": float(score[int(index)]),
                            "is_positive": int(protein_ids[int(index)] in positives),
                        }
                        for rank, index in enumerate(order, start=1)
                    )

                for local_column, protein_id in enumerate(test_proteins):
                    positives = set(
                        test.loc[test.Entry.eq(protein_id), "rhea_id"].astype(str)
                    )
                    score = protein_query_scores[:, local_column]
                    metrics = rank_metrics(score, reaction_ids, positives, set(), budgets)
                    query_rows.append(
                        {
                            "split_id": split_id,
                            "partition": partition,
                            "direction": "enzyme_to_reaction",
                            "query_id": protein_id,
                            "config": config.name,
                            **metrics,
                        }
                    )
                    order = np.lexsort((reaction_array, -score))[: args.ranking_depth]
                    ranking_rows.extend(
                        {
                            "split_id": split_id,
                            "partition": partition,
                            "direction": "enzyme_to_reaction",
                            "query_id": protein_id,
                            "config": config.name,
                            "rank": rank,
                            "candidate_id": reaction_ids[int(index)],
                            "score": float(score[int(index)]),
                            "is_positive": int(reaction_ids[int(index)] in positives),
                        }
                        for rank, index in enumerate(order, start=1)
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
    pd.DataFrame(split_rows).to_csv(output_dir / "split_summary.csv", index=False)
    summary = {
        "method": "bidirectional_fold_local_marts_dual_kernel_collaborative",
        "strict_partition": args.strict_partition,
        "configs": [config.__dict__ for config in configs],
        "completed_cells": completed,
        "protein_count": len(protein_ids),
        "reaction_count": len(reaction_ids),
        "budgets": list(budgets),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        metrics.sort_values(
            ["direction", "pareto_score", "mrr"],
            ascending=[True, False, False],
        ).groupby("direction", as_index=False).head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
