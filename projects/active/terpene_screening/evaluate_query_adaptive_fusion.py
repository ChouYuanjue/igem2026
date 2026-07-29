from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.gate_matrix import (  # noqa: E402
    canonical_or_raw_reaction,
    precursor_class_from_reaction,
    product_skeleton_class,
)
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    reaction_multiview_features,
)

DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_STRICT = ROOT / "data/terpene_cold_splits/positive_pair_fold_assignments.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_query_adaptive_fusion"
DEFAULT_BUDGETS = (3, 5, 10, 20)


@dataclass(frozen=True)
class QueryKey:
    protein_fold: int
    reaction_fold: int
    reaction_id: str


def parse_source(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise ValueError("Each --source must use LABEL=RESULT_DIR")
    return label.strip(), Path(path.strip()).resolve()


def parse_float_tuple(value: str) -> tuple[float, ...]:
    result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("Expected comma-separated floats")
    return result


def parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("Expected comma-separated integers")
    return result


def load_sources(
    sources: list[tuple[str, Path]],
) -> tuple[
    list[str],
    pd.DataFrame,
    dict[str, dict[QueryKey, pd.DataFrame]],
]:
    labels = [label for label, _ in sources]
    ranking_maps: dict[str, dict[QueryKey, pd.DataFrame]] = {}
    query_frame: pd.DataFrame | None = None
    key_columns = ["protein_fold", "reaction_fold", "reaction_id"]
    for label, directory in sources:
        rankings = pd.read_csv(
            directory / "rankings.csv",
            dtype={"reaction_id": str, "candidate_id": str},
        )
        rankings = rankings[rankings["protocol"].eq("double_cold_25cell")].copy()
        rankings["protein_fold"] = pd.to_numeric(rankings["protein_fold"]).astype(int)
        rankings["reaction_fold"] = pd.to_numeric(rankings["reaction_fold"]).astype(int)
        rankings["rank"] = pd.to_numeric(rankings["rank"]).astype(int)
        rankings["score"] = pd.to_numeric(rankings["score"]).astype(float)
        local: dict[QueryKey, pd.DataFrame] = {}
        for values, group in rankings.groupby(key_columns, sort=False):
            protein_fold, reaction_fold, reaction_id = values
            key = QueryKey(int(protein_fold), int(reaction_fold), str(reaction_id))
            ordered = group.sort_values(["rank", "candidate_id"], kind="stable")[
                ["candidate_id", "rank", "score"]
            ].reset_index(drop=True)
            if ordered["candidate_id"].duplicated().any():
                raise ValueError(f"Duplicate candidates for {label} {key}")
            local[key] = ordered
        ranking_maps[label] = local

        metrics = pd.read_csv(directory / "query_metrics.csv", dtype={"reaction_id": str})
        metrics = metrics[metrics["protocol"].eq("double_cold_25cell")].copy()
        metrics["protein_fold"] = pd.to_numeric(metrics["protein_fold"]).astype(int)
        metrics["reaction_fold"] = pd.to_numeric(metrics["reaction_fold"]).astype(int)
        keep = key_columns + [column for column in metrics if column.startswith("hit_at_")]
        metrics = metrics[keep].rename(
            columns={column: f"{label}_{column}" for column in keep if column.startswith("hit_at_")}
        )
        query_frame = (
            metrics
            if query_frame is None
            else query_frame.merge(metrics, on=key_columns, validate="one_to_one")
        )
    if query_frame is None:
        raise ValueError("No sources loaded")
    common_keys = set(ranking_maps[labels[0]])
    for label in labels[1:]:
        common_keys &= set(ranking_maps[label])
    frame_keys = {
        QueryKey(int(row.protein_fold), int(row.reaction_fold), str(row.reaction_id))
        for row in query_frame.itertuples(index=False)
    }
    if common_keys != frame_keys:
        raise ValueError(
            f"Ranking/query key mismatch: common={len(common_keys)} metrics={len(frame_keys)}"
        )
    return labels, query_frame, ranking_maps


def build_feature_frame(
    labels: list[str],
    queries: pd.DataFrame,
    ranking_maps: dict[str, dict[QueryKey, pd.DataFrame]],
    positives_path: Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    positives = pd.read_csv(positives_path, sep="\t", dtype=str).fillna("")
    reaction_smiles = positives.groupby("rhea_id")["smiles_seq"].first().to_dict()
    records: list[dict[str, object]] = []
    for row in queries.itertuples(index=False):
        key = QueryKey(int(row.protein_fold), int(row.reaction_fold), str(row.reaction_id))
        record: dict[str, object] = {
            "protein_fold": key.protein_fold,
            "reaction_fold": key.reaction_fold,
            "reaction_id": key.reaction_id,
        }
        reaction = canonical_or_raw_reaction(reaction_smiles.get(key.reaction_id, ""))
        record["precursor_class"] = precursor_class_from_reaction(reaction)
        record["skeleton_class"] = product_skeleton_class(reaction)
        try:
            descriptors = reaction_multiview_features(reaction)[3]
        except Exception:
            descriptors = np.zeros(11, dtype=np.float32)
        for index, value in enumerate(descriptors):
            record[f"chem_{index}"] = float(value)

        candidate_lists: dict[str, list[str]] = {}
        rank_lookup: dict[str, dict[str, int]] = {}
        for label in labels:
            ranking = ranking_maps[label][key]
            scores = ranking["score"].to_numpy(dtype=np.float64)
            candidates = ranking["candidate_id"].astype(str).tolist()
            candidate_lists[label] = candidates
            rank_lookup[label] = dict(zip(candidates, ranking["rank"].astype(int)))
            record[f"{label}_top1_score"] = float(scores[0])
            record[f"{label}_margin_1_2"] = float(scores[0] - scores[1])
            record[f"{label}_margin_1_5"] = float(scores[0] - scores[4])
            record[f"{label}_margin_1_10"] = float(scores[0] - scores[9])
            record[f"{label}_mean_10"] = float(scores[:10].mean())
            record[f"{label}_std_10"] = float(scores[:10].std())
            record[f"{label}_mean_all"] = float(scores.mean())
            record[f"{label}_std_all"] = float(scores.std())
            record[f"{label}_range_all"] = float(scores[0] - scores[-1])
            scale = max(float(scores.std()), 1e-8)
            record[f"{label}_normalized_margin_1_2"] = float((scores[0] - scores[1]) / scale)
            record[f"{label}_normalized_margin_1_10"] = float((scores[0] - scores[9]) / scale)

        for first, second in itertools.combinations(labels, 2):
            for cutoff in (3, 5, 10, 20):
                first_set = set(candidate_lists[first][:cutoff])
                second_set = set(candidate_lists[second][:cutoff])
                record[f"jaccard_{first}_{second}_{cutoff}"] = float(
                    len(first_set & second_set) / len(first_set | second_set)
                )
        record["top1_unique_count"] = len(
            {candidate_lists[label][0] for label in labels}
        )
        for label in labels:
            top_candidate = candidate_lists[label][0]
            reciprocal_ranks = [
                1.0 / rank_lookup[other].get(top_candidate, 101)
                for other in labels
                if other != label
            ]
            record[f"{label}_top1_other_mean_rr"] = float(np.mean(reciprocal_ranks))
            record[f"{label}_top5_consensus"] = int(
                sum(
                    top_candidate in set(candidate_lists[other][:5])
                    for other in labels
                    if other != label
                )
            )
        records.append(record)
    features = pd.DataFrame(records)
    merged = queries.merge(
        features,
        on=["protein_fold", "reaction_fold", "reaction_id"],
        validate="one_to_one",
    )
    categorical = ["precursor_class", "skeleton_class"]
    excluded = {
        "protein_fold",
        "reaction_fold",
        "reaction_id",
        *categorical,
    }
    numeric = [
        column
        for column in features.columns
        if column not in excluded
    ]
    return merged, numeric, categorical


def build_positive_map(strict_path: Path) -> dict[QueryKey, set[str]]:
    strict = pd.read_csv(strict_path, dtype=str).fillna("")
    strict["protein_fold"] = pd.to_numeric(strict["protein_fold"]).astype(int)
    strict["reaction_fold"] = pd.to_numeric(strict["reaction_fold"]).astype(int)
    return {
        QueryKey(int(protein_fold), int(reaction_fold), str(reaction_id)): set(
            group["Entry"].astype(str)
        )
        for (protein_fold, reaction_fold, reaction_id), group in strict.groupby(
            ["protein_fold", "reaction_fold", "rhea_id"], sort=False
        )
    }


def classifier_pipeline(
    numeric: list[str],
    categorical: list[str],
    c_value: float,
    balanced: bool,
    seed: int,
) -> object:
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
        ]
    )
    return make_pipeline(
        preprocessing,
        LogisticRegression(
            C=c_value,
            class_weight="balanced" if balanced else None,
            max_iter=1000,
            random_state=seed,
        ),
    )


def predict_source_reliability(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    labels: list[str],
    budget: int,
    numeric: list[str],
    categorical: list[str],
    c_value: float,
    balanced: bool,
    seed: int,
) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    feature_columns = numeric + categorical
    for source_index, label in enumerate(labels):
        target = train[f"{label}_hit_at_{budget}"].astype(int).to_numpy()
        if len(np.unique(target)) == 1:
            prediction = np.full(len(test), float(target[0]), dtype=np.float64)
        else:
            model = classifier_pipeline(
                numeric, categorical, c_value, balanced, seed + source_index
            )
            model.fit(train[feature_columns], target)
            prediction = model.predict_proba(test[feature_columns])[:, 1]
        probabilities.append(prediction)
    return np.stack(probabilities, axis=1)


def query_contributions(
    key: QueryKey,
    labels: list[str],
    ranking_maps: dict[str, dict[QueryKey, pd.DataFrame]],
    positive_ids: set[str],
    constant: float,
    power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = sorted(
        set().union(
            *(
                set(ranking_maps[label][key]["candidate_id"].astype(str))
                for label in labels
            )
        )
    )
    candidate_to_row = {candidate: index for index, candidate in enumerate(candidates)}
    contribution = np.zeros((len(candidates), len(labels)), dtype=np.float64)
    for source_index, label in enumerate(labels):
        ranking = ranking_maps[label][key]
        for row in ranking.itertuples(index=False):
            contribution[candidate_to_row[str(row.candidate_id)], source_index] = (
                1.0 / (constant + int(row.rank)) ** power
            )
    positive_mask = np.asarray(
        [candidate in positive_ids for candidate in candidates], dtype=bool
    )
    lexicographic_epsilon = 1e-14 * np.arange(
        len(candidates), 0, -1, dtype=np.float64
    )
    return contribution, positive_mask, lexicographic_epsilon


def transform_weights(
    probabilities: np.ndarray,
    mode: str,
    gamma: float,
) -> np.ndarray:
    values = np.maximum(probabilities, 1e-6) ** gamma
    if mode == "select":
        result = np.zeros_like(values)
        result[np.arange(len(values)), np.argmax(values, axis=1)] = 1.0
        return result
    if mode == "top2":
        selected = np.argpartition(-values, kth=1, axis=1)[:, :2]
        result = np.zeros_like(values)
        rows = np.arange(len(values))[:, None]
        result[rows, selected] = values[rows, selected]
        denominator = result.sum(axis=1, keepdims=True)
        return result / np.maximum(denominator, 1e-12)
    if mode != "weighted":
        raise ValueError(f"Unknown fusion mode {mode}")
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


def evaluate_probabilities(
    *,
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    labels: list[str],
    ranking_maps: dict[str, dict[QueryKey, pd.DataFrame]],
    positives: dict[QueryKey, set[str]],
    contribution_cache: dict[
        tuple[QueryKey, float, float], tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    budget: int,
    constant: float,
    power: float,
    gamma: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    weights = transform_weights(probabilities, mode, gamma)
    hits = np.zeros(len(frame), dtype=np.uint8)
    reciprocal_ranks = np.zeros(len(frame), dtype=np.float64)
    for index, row in enumerate(frame.itertuples(index=False)):
        key = QueryKey(int(row.protein_fold), int(row.reaction_fold), str(row.reaction_id))
        cache_key = (key, float(constant), float(power))
        cached = contribution_cache.get(cache_key)
        if cached is None:
            cached = query_contributions(
                key,
                labels,
                ranking_maps,
                positives[key],
                constant,
                power,
            )
            contribution_cache[cache_key] = cached
        contribution, positive_mask, epsilon = cached
        scores = contribution @ weights[index] + epsilon
        if positive_mask.any():
            best_positive_score = float(scores[positive_mask].max())
            best_rank = 1 + int((scores > best_positive_score).sum())
            reciprocal_ranks[index] = 1.0 / best_rank
            hits[index] = int(best_rank <= budget)
    return hits, reciprocal_ranks


def development_cells(frame: pd.DataFrame) -> list[tuple[int, int]]:
    cells = sorted(
        {
            (int(row.protein_fold), int(row.reaction_fold))
            for row in frame.itertuples(index=False)
        }
    )
    expected = {(4, reaction_fold) for reaction_fold in range(5)} | {
        (protein_fold, 4) for protein_fold in range(4)
    }
    if set(cells) != expected:
        raise ValueError(f"Unexpected development cells: {cells}")
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query-adaptive expert reliability gating for strict TPS double-cold retrieval."
    )
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--strict-splits", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--c-values", default="0.01,0.1,1,10")
    parser.add_argument("--constants", default="0,10,30,60")
    parser.add_argument("--powers", default="0.5,1")
    parser.add_argument("--gammas", default="0.5,1,2,4")
    parser.add_argument("--modes", default="weighted,top2,select")
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    sources = [parse_source(value) for value in args.source]
    labels, queries, ranking_maps = load_sources(sources)
    features, numeric, categorical = build_feature_frame(
        labels, queries, ranking_maps, args.positives.resolve()
    )
    positives = build_positive_map(args.strict_splits.resolve())
    development = features[
        features["protein_fold"].eq(4) | features["reaction_fold"].eq(4)
    ].reset_index(drop=True)
    frozen = features[
        features["protein_fold"].ne(4) & features["reaction_fold"].ne(4)
    ].reset_index(drop=True)
    cells = development_cells(development)
    budgets = parse_int_tuple(args.budgets)
    c_values = parse_float_tuple(args.c_values)
    constants = parse_float_tuple(args.constants)
    powers = parse_float_tuple(args.powers)
    gammas = parse_float_tuple(args.gammas)
    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())

    selection_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    contribution_cache: dict[
        tuple[QueryKey, float, float], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for budget in budgets:
        candidates: list[dict[str, object]] = []
        probability_cache: dict[tuple[float, bool], np.ndarray] = {}
        for c_value in c_values:
            for balanced in (False, True):
                oof = np.zeros((len(development), len(labels)), dtype=np.float64)
                for cell_index, (protein_fold, reaction_fold) in enumerate(cells):
                    validation_mask = development["protein_fold"].eq(protein_fold) & development[
                        "reaction_fold"
                    ].eq(reaction_fold)
                    train = development[~validation_mask]
                    validation = development[validation_mask]
                    oof[validation_mask.to_numpy()] = predict_source_reliability(
                        train=train,
                        test=validation,
                        labels=labels,
                        budget=budget,
                        numeric=numeric,
                        categorical=categorical,
                        c_value=c_value,
                        balanced=balanced,
                        seed=args.seed + cell_index * 31 + budget,
                    )
                probability_cache[(c_value, balanced)] = oof
                for constant in constants:
                    for power in powers:
                        for gamma in gammas:
                            for mode in modes:
                                hits, rr = evaluate_probabilities(
                                    frame=development,
                                    probabilities=oof,
                                    labels=labels,
                                    ranking_maps=ranking_maps,
                                    positives=positives,
                                    contribution_cache=contribution_cache,
                                    budget=budget,
                                    constant=constant,
                                    power=power,
                                    gamma=gamma,
                                    mode=mode,
                                )
                                candidates.append(
                                    {
                                        "budget": budget,
                                        "c_value": c_value,
                                        "balanced": balanced,
                                        "constant": constant,
                                        "power": power,
                                        "gamma": gamma,
                                        "mode": mode,
                                        "development_hit": float(hits.mean()),
                                        "development_mrr": float(rr.mean()),
                                    }
                                )
        candidate_frame = pd.DataFrame(candidates).sort_values(
            ["development_hit", "development_mrr", "balanced", "c_value", "constant", "power", "gamma", "mode"],
            ascending=[False, False, True, True, True, True, True, True],
            kind="stable",
        )
        best = candidate_frame.iloc[0]
        selection_rows.extend(candidate_frame.head(50).to_dict("records"))
        final_probabilities = predict_source_reliability(
            train=development,
            test=frozen,
            labels=labels,
            budget=budget,
            numeric=numeric,
            categorical=categorical,
            c_value=float(best.c_value),
            balanced=bool(best.balanced),
            seed=args.seed + 1000 + budget,
        )
        frozen_hits, frozen_rr = evaluate_probabilities(
            frame=frozen,
            probabilities=final_probabilities,
            labels=labels,
            ranking_maps=ranking_maps,
            positives=positives,
            contribution_cache=contribution_cache,
            budget=budget,
            constant=float(best.constant),
            power=float(best.power),
            gamma=float(best.gamma),
            mode=str(best["mode"]),
        )
        metric_rows.append(
            {
                "evaluation": "development_9_cells_oof",
                "budget": budget,
                "n_queries": len(development),
                "hit_probability": float(best.development_hit),
                "mean_reciprocal_rank": float(best.development_mrr),
                "c_value": float(best.c_value),
                "balanced": bool(best.balanced),
                "constant": float(best.constant),
                "power": float(best.power),
                "gamma": float(best.gamma),
                "mode": str(best["mode"]),
            }
        )
        metric_rows.append(
            {
                "evaluation": "frozen_16_cells_locked",
                "budget": budget,
                "n_queries": len(frozen),
                "hit_probability": float(frozen_hits.mean()),
                "mean_reciprocal_rank": float(frozen_rr.mean()),
                "c_value": float(best.c_value),
                "balanced": bool(best.balanced),
                "constant": float(best.constant),
                "power": float(best.power),
                "gamma": float(best.gamma),
                "mode": str(best["mode"]),
            }
        )
        transformed = transform_weights(
            final_probabilities, str(best["mode"]), float(best.gamma)
        )
        for index, row in enumerate(frozen.itertuples(index=False)):
            query_rows.append(
                {
                    "budget": budget,
                    "protein_fold": int(row.protein_fold),
                    "reaction_fold": int(row.reaction_fold),
                    "reaction_id": str(row.reaction_id),
                    "hit": int(frozen_hits[index]),
                    "reciprocal_rank": float(frozen_rr[index]),
                    **{
                        f"probability_{label}": float(final_probabilities[index, source_index])
                        for source_index, label in enumerate(labels)
                    },
                    **{
                        f"weight_{label}": float(transformed[index, source_index])
                        for source_index, label in enumerate(labels)
                    },
                }
            )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    selections = pd.DataFrame(selection_rows)
    query_metrics = pd.DataFrame(query_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    selections.to_csv(output_dir / "development_candidate_configurations.csv", index=False)
    query_metrics.to_csv(output_dir / "frozen_query_metrics.csv", index=False)
    feature_manifest = {
        "numeric_features": numeric,
        "categorical_features": categorical,
    }
    (output_dir / "feature_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "method": "query_adaptive_source_reliability_gating",
        "sources": {label: str(path) for label, path in sources},
        "development_partition": "protein_fold==4 OR reaction_fold==4",
        "development_selection": "leave-one-cell-out OOF across the nine development cells",
        "frozen_partition": "protein_fold in 0..3 AND reaction_fold in 0..3",
        "frozen_policy": "all gate and fusion hyperparameters locked before evaluation",
        "classifier": "per-source logistic probability of Hit@K from deployable query diagnostics",
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "development_candidates": str(
                output_dir / "development_candidate_configurations.csv"
            ),
            "frozen_query_metrics": str(output_dir / "frozen_query_metrics.csv"),
            "feature_manifest": str(output_dir / "feature_manifest.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
