from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_STRICT_SPLITS = ROOT / "data/terpene_cold_splits/positive_pair_fold_assignments.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_pareto_rank_fusion"
DEFAULT_BUDGETS = (3, 5, 10, 20)


@dataclass(frozen=True)
class QueryKey:
    protocol: str
    protein_fold: int
    reaction_fold: int
    reaction_id: str


def parse_source(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise ValueError("Each --source must use LABEL=RESULT_DIR")
    return label.strip(), Path(path.strip()).resolve()


def parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result or any(item <= 0 for item in result):
        raise ValueError("Expected positive comma-separated integers")
    return result


def parse_float_tuple(value: str) -> tuple[float, ...]:
    result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not result or any(item < 0 for item in result):
        raise ValueError("Expected non-negative comma-separated floats")
    return result


def canonical_key(row: object) -> QueryKey:
    protein_fold = getattr(row, "protein_fold")
    reaction_fold = getattr(row, "reaction_fold")
    return QueryKey(
        protocol=str(getattr(row, "protocol")),
        protein_fold=-1 if pd.isna(protein_fold) or str(protein_fold) == "" else int(float(protein_fold)),
        reaction_fold=-1 if pd.isna(reaction_fold) or str(reaction_fold) == "" else int(float(reaction_fold)),
        reaction_id=str(getattr(row, "reaction_id")),
    )


def integer_compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    result: list[tuple[int, ...]] = []
    for head in range(total + 1):
        for tail in integer_compositions(total - head, parts - 1):
            result.append((head, *tail))
    return result


def weight_grid(n_sources: int) -> np.ndarray:
    values: set[tuple[float, ...]] = set()
    for composition in integer_compositions(4, n_sources):
        if sum(composition):
            values.add(tuple(round(value / 4.0, 8) for value in composition))
    for first in range(n_sources):
        for second in range(first + 1, n_sources):
            for step in range(1, 10):
                vector = [0.0] * n_sources
                vector[first] = step / 10.0
                vector[second] = 1.0 - step / 10.0
                values.add(tuple(round(value, 8) for value in vector))
    ordered = sorted(values, key=lambda row: (sum(value > 0 for value in row), row))
    return np.asarray(ordered, dtype=np.float64)


def load_rankings(
    sources: list[tuple[str, Path]],
) -> tuple[list[str], dict[QueryKey, dict[str, dict[str, int]]], int]:
    labels = [label for label, _ in sources]
    rankings: dict[QueryKey, dict[str, dict[str, int]]] = {}
    common_keys: set[QueryKey] | None = None
    depths: list[int] = []
    for label, directory in sources:
        path = directory / "rankings.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"reaction_id": str, "candidate_id": str})
        required = {
            "protocol",
            "protein_fold",
            "reaction_fold",
            "reaction_id",
            "candidate_id",
            "rank",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} misses columns {sorted(missing)}")
        local_keys: set[QueryKey] = set()
        frame["rank"] = pd.to_numeric(frame["rank"]).astype(int)
        depths.append(int(frame["rank"].max()))
        for group_key, group in frame.groupby(
            ["protocol", "protein_fold", "reaction_fold", "reaction_id"],
            dropna=False,
            sort=False,
        ):
            protocol, protein_fold, reaction_fold, reaction_id = group_key
            key = QueryKey(
                protocol=str(protocol),
                protein_fold=-1 if pd.isna(protein_fold) else int(float(protein_fold)),
                reaction_fold=-1 if pd.isna(reaction_fold) else int(float(reaction_fold)),
                reaction_id=str(reaction_id),
            )
            if group["candidate_id"].duplicated().any():
                raise ValueError(f"Duplicate candidates for {label} {key}")
            local_keys.add(key)
            rankings.setdefault(key, {})[label] = dict(
                zip(group["candidate_id"].astype(str), group["rank"].astype(int))
            )
        common_keys = local_keys if common_keys is None else common_keys & local_keys
    if common_keys is None or not common_keys:
        raise ValueError("Sources have no common query keys")
    if len(set(depths)) != 1:
        raise ValueError(f"Sources use different ranking depths: {depths}")
    filtered = {key: rankings[key] for key in sorted(common_keys, key=lambda item: (item.protocol, item.protein_fold, item.reaction_fold, item.reaction_id))}
    for key, source_map in filtered.items():
        if set(source_map) != set(labels):
            raise ValueError(f"Incomplete sources for {key}")
    return labels, filtered, depths[0]


def build_positive_map(
    keys: list[QueryKey],
    positives_path: Path,
    strict_path: Path,
) -> dict[QueryKey, set[str]]:
    positives = pd.read_csv(positives_path, sep="\t", dtype=str).fillna("")
    positives = positives[["Entry", "rhea_id"]].drop_duplicates()
    all_by_reaction = {
        reaction_id: set(group["Entry"].astype(str))
        for reaction_id, group in positives.groupby("rhea_id", sort=False)
    }
    strict = pd.read_csv(strict_path, dtype=str).fillna("")
    strict["protein_fold"] = pd.to_numeric(strict["protein_fold"]).astype(int)
    strict["reaction_fold"] = pd.to_numeric(strict["reaction_fold"]).astype(int)
    strict_map = {
        (int(protein_fold), int(reaction_fold), str(reaction_id)): set(group["Entry"].astype(str))
        for (protein_fold, reaction_fold, reaction_id), group in strict.groupby(
            ["protein_fold", "reaction_fold", "rhea_id"], sort=False
        )
    }
    result: dict[QueryKey, set[str]] = {}
    for key in keys:
        if key.protocol == "legacy_exact":
            result[key] = all_by_reaction.get(key.reaction_id, set())
        elif key.protocol == "double_cold_25cell":
            result[key] = strict_map.get(
                (key.protein_fold, key.reaction_fold, key.reaction_id), set()
            )
        else:
            raise ValueError(f"Unknown protocol {key.protocol}")
        if not result[key]:
            raise ValueError(f"No positives for {key}")
    return result


def precompute_predictions(
    *,
    labels: list[str],
    rankings: dict[QueryKey, dict[str, dict[str, int]]],
    positives: dict[QueryKey, set[str]],
    weights: np.ndarray,
    constants: tuple[float, ...],
    powers: tuple[float, ...],
    budgets: tuple[int, ...],
) -> tuple[list[dict[str, object]], np.ndarray, dict[int, np.ndarray], np.ndarray]:
    keys = list(rankings)
    parameters: list[dict[str, object]] = []
    blocks: list[tuple[float, float, int, int]] = []
    start = 0
    for constant in constants:
        for power in powers:
            for local_index, vector in enumerate(weights):
                parameters.append(
                    {
                        "parameter_index": len(parameters),
                        "constant": constant,
                        "power": power,
                        **{f"weight_{label}": float(vector[index]) for index, label in enumerate(labels)},
                    }
                )
            blocks.append((constant, power, start, start + len(weights)))
            start += len(weights)
    n_parameters = len(parameters)
    hit_arrays = {
        budget: np.zeros((len(keys), n_parameters), dtype=np.uint8) for budget in budgets
    }
    reciprocal_rank = np.zeros((len(keys), n_parameters), dtype=np.float32)
    max_budget = max(budgets)
    for query_index, key in enumerate(keys):
        source_maps = rankings[key]
        candidates = sorted(
            set().union(*(set(source_maps[label]) for label in labels))
        )
        candidate_to_row = {candidate: index for index, candidate in enumerate(candidates)}
        rank_matrix = np.full((len(candidates), len(labels)), np.inf, dtype=np.float64)
        for source_index, label in enumerate(labels):
            for candidate, rank in source_maps[label].items():
                rank_matrix[candidate_to_row[candidate], source_index] = rank
        positive_mask = np.asarray(
            [candidate in positives[key] for candidate in candidates], dtype=bool
        )
        lexicographic_epsilon = 1e-14 * np.arange(len(candidates), 0, -1, dtype=np.float64)
        for constant, power, begin, end in blocks:
            contribution = np.zeros_like(rank_matrix)
            present = np.isfinite(rank_matrix)
            contribution[present] = 1.0 / np.power(constant + rank_matrix[present], power)
            score_matrix = contribution @ weights.T
            score_matrix += lexicographic_epsilon[:, None]
            local_k = min(max_budget, len(candidates))
            selected = np.argpartition(-score_matrix, kth=local_k - 1, axis=0)[:local_k]
            selected_scores = np.take_along_axis(score_matrix, selected, axis=0)
            selected_order = np.argsort(-selected_scores, axis=0, kind="stable")
            selected = np.take_along_axis(selected, selected_order, axis=0)
            selected_positive = positive_mask[selected]
            for budget in budgets:
                hit_arrays[budget][query_index, begin:end] = selected_positive[:budget].any(axis=0)
            if positive_mask.any():
                best_positive_score = score_matrix[positive_mask].max(axis=0)
                valid = best_positive_score > lexicographic_epsilon.min()
                ranks = 1 + (score_matrix > best_positive_score[None, :]).sum(axis=0)
                reciprocal_rank[query_index, begin:end] = np.where(valid, 1.0 / ranks, 0.0)
    return parameters, np.asarray(keys, dtype=object), hit_arrays, reciprocal_rank


def select_parameter(
    candidate_indices: np.ndarray,
    hit_values: np.ndarray,
    reciprocal_rank: np.ndarray,
    parameters: list[dict[str, object]],
) -> int:
    hit_mean = hit_values[candidate_indices].mean(axis=0)
    rr_mean = reciprocal_rank[candidate_indices].mean(axis=0)
    active_sources = np.asarray(
        [
            sum(float(value) > 0 for key, value in row.items() if key.startswith("weight_"))
            for row in parameters
        ],
        dtype=np.int64,
    )
    order = np.lexsort((active_sources, -rr_mean, -hit_mean))
    return int(order[0])


def aggregate_selected(
    *,
    evaluation: str,
    budget: int,
    query_indices: np.ndarray,
    parameter_by_query: np.ndarray,
    hit_values: np.ndarray,
    reciprocal_rank: np.ndarray,
) -> dict[str, object]:
    local_params = parameter_by_query[query_indices]
    local_hits = hit_values[query_indices, local_params]
    local_rr = reciprocal_rank[query_indices, local_params]
    return {
        "evaluation": evaluation,
        "budget": budget,
        "n_queries": int(len(query_indices)),
        "hit_probability": float(local_hits.mean()),
        "mean_reciprocal_rank": float(local_rr.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-controlled objective-specific weighted RRF for TPS retrieval."
    )
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--strict-splits", type=Path, default=DEFAULT_STRICT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--constants", default="0,10,30,60,100")
    parser.add_argument("--powers", default="0.5,1")
    args = parser.parse_args()

    sources = [parse_source(value) for value in args.source]
    labels, rankings, ranking_depth = load_rankings(sources)
    budgets = parse_int_tuple(args.budgets)
    constants = parse_float_tuple(args.constants)
    powers = parse_float_tuple(args.powers)
    weights = weight_grid(len(labels))
    keys = list(rankings)
    positives = build_positive_map(
        keys, args.positives.resolve(), args.strict_splits.resolve()
    )
    parameters, key_array, hit_arrays, reciprocal_rank = precompute_predictions(
        labels=labels,
        rankings=rankings,
        positives=positives,
        weights=weights,
        constants=constants,
        powers=powers,
        budgets=budgets,
    )
    parameter_frame = pd.DataFrame(parameters)
    exact_indices = np.asarray(
        [index for index, key in enumerate(keys) if key.protocol == "legacy_exact"],
        dtype=np.int64,
    )
    strict_indices = np.asarray(
        [index for index, key in enumerate(keys) if key.protocol == "double_cold_25cell"],
        dtype=np.int64,
    )
    development_indices = np.asarray(
        [
            index
            for index in strict_indices
            if keys[index].protein_fold == 4 or keys[index].reaction_fold == 4
        ],
        dtype=np.int64,
    )
    frozen_indices = np.asarray(
        [
            index
            for index in strict_indices
            if keys[index].protein_fold != 4 and keys[index].reaction_fold != 4
        ],
        dtype=np.int64,
    )
    selected_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for budget in budgets:
        exact_parameter_by_query = np.full(len(keys), -1, dtype=np.int64)
        for target_fold in range(5):
            train_indices = np.asarray(
                [
                    index
                    for index in exact_indices
                    if keys[index].reaction_fold != target_fold
                ],
                dtype=np.int64,
            )
            test_indices = np.asarray(
                [
                    index
                    for index in exact_indices
                    if keys[index].reaction_fold == target_fold
                ],
                dtype=np.int64,
            )
            selected = select_parameter(
                train_indices,
                hit_arrays[budget],
                reciprocal_rank,
                parameters,
            )
            exact_parameter_by_query[test_indices] = selected
            selected_rows.append(
                {
                    "selection_protocol": "legacy_exact_nested",
                    "budget": budget,
                    "target_fold": target_fold,
                    "n_selection_queries": len(train_indices),
                    "selection_hit_probability": float(
                        hit_arrays[budget][train_indices, selected].mean()
                    ),
                    "selection_mrr": float(reciprocal_rank[train_indices, selected].mean()),
                    **parameters[selected],
                }
            )
        if (exact_parameter_by_query[exact_indices] < 0).any():
            raise RuntimeError("Nested exact selection did not assign every query")
        metric_rows.append(
            aggregate_selected(
                evaluation="legacy_exact_nested",
                budget=budget,
                query_indices=exact_indices,
                parameter_by_query=exact_parameter_by_query,
                hit_values=hit_arrays[budget],
                reciprocal_rank=reciprocal_rank,
            )
        )
        for index in exact_indices:
            selected = int(exact_parameter_by_query[index])
            key = keys[index]
            query_rows.append(
                {
                    "evaluation": "legacy_exact_nested",
                    "budget": budget,
                    "protocol": key.protocol,
                    "protein_fold": key.protein_fold,
                    "reaction_fold": key.reaction_fold,
                    "reaction_id": key.reaction_id,
                    "parameter_index": selected,
                    "hit": int(hit_arrays[budget][index, selected]),
                    "reciprocal_rank": float(reciprocal_rank[index, selected]),
                }
            )

        strict_selected = select_parameter(
            development_indices,
            hit_arrays[budget],
            reciprocal_rank,
            parameters,
        )
        strict_parameter_by_query = np.full(len(keys), strict_selected, dtype=np.int64)
        selected_rows.append(
            {
                "selection_protocol": "double_cold_development_locked",
                "budget": budget,
                "target_fold": "frozen_16_cells",
                "n_selection_queries": len(development_indices),
                "selection_hit_probability": float(
                    hit_arrays[budget][development_indices, strict_selected].mean()
                ),
                "selection_mrr": float(
                    reciprocal_rank[development_indices, strict_selected].mean()
                ),
                **parameters[strict_selected],
            }
        )
        for evaluation, indices in [
            ("double_cold_development_9_cells", development_indices),
            ("double_cold_frozen_16_cells", frozen_indices),
            ("double_cold_all_25_cells", strict_indices),
        ]:
            metric_rows.append(
                aggregate_selected(
                    evaluation=evaluation,
                    budget=budget,
                    query_indices=indices,
                    parameter_by_query=strict_parameter_by_query,
                    hit_values=hit_arrays[budget],
                    reciprocal_rank=reciprocal_rank,
                )
            )
            for index in indices:
                key = keys[index]
                query_rows.append(
                    {
                        "evaluation": evaluation,
                        "budget": budget,
                        "protocol": key.protocol,
                        "protein_fold": key.protein_fold,
                        "reaction_fold": key.reaction_fold,
                        "reaction_id": key.reaction_id,
                        "parameter_index": strict_selected,
                        "hit": int(hit_arrays[budget][index, strict_selected]),
                        "reciprocal_rank": float(reciprocal_rank[index, strict_selected]),
                    }
                )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_frame = pd.DataFrame(selected_rows)
    query_frame = pd.DataFrame(query_rows)
    metrics = pd.DataFrame(metric_rows)
    parameter_frame.to_csv(output_dir / "parameter_grid.csv", index=False)
    selected_frame.to_csv(output_dir / "selected_parameters.csv", index=False)
    query_frame.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    summary = {
        "method": "objective_specific_weighted_rrf",
        "source_labels": labels,
        "source_directories": {label: str(path) for label, path in sources},
        "ranking_depth": ranking_depth,
        "weight_vectors": len(weights),
        "constants": list(constants),
        "powers": list(powers),
        "parameter_count": len(parameters),
        "selection": {
            "legacy_exact": "nested five-fold: each target fold uses weights selected on the other four OOF folds",
            "double_cold": "weights selected on protein_fold==4 OR reaction_fold==4, locked on remaining 16 cells",
        },
        "outputs": {
            "parameter_grid": str(output_dir / "parameter_grid.csv"),
            "selected_parameters": str(output_dir / "selected_parameters.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "metrics": str(output_dir / "metrics.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(selected_frame.to_string(index=False))


if __name__ == "__main__":
    main()
