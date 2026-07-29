from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    build_reaction_features,
)

DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_STRICT = ROOT / "data/terpene_cold_splits/positive_pair_fold_assignments.csv"
DEFAULT_PFAM = (
    ROOT / "data/terpene_current_pfam_uniprot_v2/current_pfam_annotations.csv"
)
DEFAULT_RANKINGS = (
    ROOT / "results/terpene_current_me8_fusion_rankings_v1/r2e075/rankings.csv"
)
DEFAULT_OUTPUT = ROOT / "results/terpene_current_pfam_architecture_reranking"
DEFAULT_BUDGETS = (3, 5, 10, 20)


def parse_float_tuple(value: str) -> tuple[float, ...]:
    result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("Expected at least one float")
    return result


def parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("Expected at least one integer")
    return result


def normalize_pfam(value: object) -> str:
    return ";".join(sorted({part for part in str(value).split(";") if part}))


def choose_architecture_vocabulary(
    annotations: pd.DataFrame,
    minimum_count: int,
    maximum_classes: int,
) -> tuple[str, ...]:
    counts = annotations.loc[
        annotations["pfam_combination"].ne(""), "pfam_combination"
    ].value_counts()
    values = counts[counts >= minimum_count].head(maximum_classes).index.astype(str)
    return tuple(values)


def architecture_group(value: str, vocabulary: tuple[str, ...]) -> str:
    if not value:
        return ""
    return value if value in vocabulary else "__OTHER_PFAM__"


def build_reaction_labels(
    train_pairs: pd.DataFrame,
    protein_architecture: dict[str, str],
    reaction_ids: list[str],
    classes: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    class_to_column = {value: index for index, value in enumerate(classes)}
    labels = np.zeros((len(reaction_ids), len(classes)), dtype=np.int8)
    labelled = np.zeros(len(reaction_ids), dtype=bool)
    for reaction, group in train_pairs.groupby("rhea_id", sort=False):
        reaction = str(reaction)
        row = reaction_to_row.get(reaction)
        if row is None:
            continue
        values = {
            protein_architecture.get(protein, "")
            for protein in group.Entry.astype(str)
        } - {""}
        if not values:
            continue
        labelled[row] = True
        for value in values:
            column = class_to_column.get(value)
            if column is not None:
                labels[row, column] = 1
    return labels, labelled


def fit_predictors(
    features: np.ndarray,
    labels: np.ndarray,
    labelled: np.ndarray,
    c_value: float,
    seed: int,
) -> np.ndarray:
    result = np.zeros_like(labels, dtype=np.float32)
    x = features[labelled]
    if len(x) == 0:
        raise ValueError("No fold-local reactions have Pfam labels")
    for column in range(labels.shape[1]):
        y = labels[labelled, column]
        positives = int(y.sum())
        negatives = int(len(y) - positives)
        if positives == 0:
            result[:, column] = 0.0
        elif negatives == 0:
            result[:, column] = 1.0
        else:
            model = LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=3000,
                solver="liblinear",
                random_state=seed + column,
            )
            model.fit(x, y)
            result[:, column] = model.predict_proba(features)[:, 1].astype(np.float32)
    return result


def tied_percentile(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.ones(len(values), dtype=np.float32)
    ranks = rankdata(-values, method="average")
    return (1.0 - (ranks - 1.0) / (len(values) - 1.0)).astype(np.float32)


def reciprocal_rank_and_hits(
    ranked: list[str], positives: set[str], budgets: tuple[int, ...]
) -> dict[str, float | int]:
    positions = [index + 1 for index, value in enumerate(ranked) if value in positives]
    best = min(positions) if positions else np.nan
    row: dict[str, float | int] = {
        "best_positive_rank": float(best),
        "reciprocal_rank": 0.0 if not positions else float(1.0 / best),
    }
    for budget in budgets:
        row[f"hit_at_{budget}"] = int(bool(set(ranked[:budget]) & positives))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fold-local current TPS Pfam architecture reranking over an existing Top-N list."
    )
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--strict-splits", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--pfam-annotations", type=Path, default=DEFAULT_PFAM)
    parser.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--c-values", default="0.03,0.1,0.3,1.0")
    parser.add_argument("--fusion-weights", default="0,0.03,0.05,0.1,0.2,0.3")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--minimum-class-count", type=int, default=8)
    parser.add_argument("--maximum-classes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    c_values = parse_float_tuple(args.c_values)
    fusion_weights = parse_float_tuple(args.fusion_weights)
    budgets = parse_int_tuple(args.budgets)
    if any(weight < 0 or weight > 1 for weight in fusion_weights):
        raise ValueError("Fusion weights must be within [0, 1]")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    positives = pd.read_csv(args.positives, sep="\t", dtype=str).fillna("")
    positives = positives[["Entry", "rhea_id", "smiles_seq"]].drop_duplicates(
        ["Entry", "rhea_id"]
    )
    reaction_features, reaction_ids, _, feature_schema = build_reaction_features(
        positives, "multiview"
    )
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    strict = pd.read_csv(args.strict_splits, dtype=str).fillna("")
    strict[["protein_fold", "reaction_fold"]] = strict[
        ["protein_fold", "reaction_fold"]
    ].astype(int)
    strict = strict[
        ["Entry", "rhea_id", "protein_fold", "reaction_fold"]
    ].drop_duplicates(["Entry", "rhea_id"])

    annotations = pd.read_csv(args.pfam_annotations, dtype=str).fillna("")
    annotations["pfam_combination"] = annotations.pfam_combination.map(normalize_pfam)
    vocabulary = choose_architecture_vocabulary(
        annotations, args.minimum_class_count, args.maximum_classes
    )
    classes = vocabulary + ("__OTHER_PFAM__",)
    annotations["architecture_group"] = annotations.pfam_combination.map(
        lambda value: architecture_group(value, vocabulary)
    )
    protein_architecture = dict(
        zip(annotations.Entry.astype(str), annotations.architecture_group.astype(str))
    )

    rankings = pd.read_csv(args.rankings, dtype=str).fillna("")
    rankings = rankings[rankings.protocol.eq("double_cold_25cell")].copy()
    rankings[["protein_fold", "reaction_fold", "rank"]] = rankings[
        ["protein_fold", "reaction_fold", "rank"]
    ].astype(int)
    rankings["score"] = pd.to_numeric(rankings.score, errors="coerce").fillna(0.0)
    keys = ["protein_fold", "reaction_fold", "reaction_id"]
    base_lists = {
        key: group.sort_values(["rank", "candidate_id"])[
            ["candidate_id", "rank", "score"]
        ].copy()
        for key, group in rankings.groupby(keys, sort=True)
    }
    positives_by_query = {
        (protein_fold, reaction_fold, str(reaction)): set(group.Entry.astype(str))
        for (protein_fold, reaction_fold, reaction), group in strict.rename(
            columns={"rhea_id": "reaction_id"}
        ).groupby(keys, sort=True)
    }
    if set(base_lists) != set(positives_by_query):
        missing = set(positives_by_query) - set(base_lists)
        raise ValueError(f"Ranking query set mismatch; missing={list(sorted(missing))[:5]}")

    parameter_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    fit_audit: list[dict[str, object]] = []
    for protein_fold in range(5):
        for reaction_fold in range(5):
            split_id = f"p{protein_fold}_r{reaction_fold}"
            partition = (
                "development" if protein_fold == 4 or reaction_fold == 4 else "frozen"
            )
            train_pairs = strict[
                strict.protein_fold.ne(protein_fold)
                & strict.reaction_fold.ne(reaction_fold)
            ].copy()
            labels, labelled = build_reaction_labels(
                train_pairs, protein_architecture, reaction_ids, classes
            )
            fit_audit.append(
                {
                    "split_id": split_id,
                    "partition": partition,
                    "train_pairs": len(train_pairs),
                    "labelled_train_reactions": int(labelled.sum()),
                    "architecture_classes": len(classes),
                }
            )
            local_keys = [
                key
                for key in base_lists
                if key[0] == protein_fold and key[1] == reaction_fold
            ]
            for c_value in c_values:
                probabilities = fit_predictors(
                    reaction_features,
                    labels,
                    labelled,
                    c_value,
                    args.seed + protein_fold * 100 + reaction_fold,
                )
                class_to_column = {value: index for index, value in enumerate(classes)}
                for weight in fusion_weights:
                    method = f"pfam_c{c_value:g}_w{weight:g}"
                    for key in local_keys:
                        reaction = key[2]
                        base = base_lists[key]
                        candidates = base.candidate_id.astype(str).tolist()
                        base_percentile = tied_percentile(-base["rank"].to_numpy(float))
                        neutral = float(
                            probabilities[reaction_to_row[reaction]].mean()
                        )
                        architecture_scores = np.asarray(
                            [
                                probabilities[
                                    reaction_to_row[reaction],
                                    class_to_column[protein_architecture[candidate]],
                                ]
                                if protein_architecture.get(candidate, "")
                                in class_to_column
                                else neutral
                                for candidate in candidates
                            ],
                            dtype=np.float32,
                        )
                        architecture_percentile = tied_percentile(architecture_scores)
                        score = (
                            (1.0 - weight) * base_percentile
                            + weight * architecture_percentile
                        )
                        order = np.lexsort((np.asarray(candidates), -score))
                        ranked = [candidates[index] for index in order]
                        query_rows.append(
                            {
                                "split_id": split_id,
                                "partition": partition,
                                "protein_fold": protein_fold,
                                "reaction_fold": reaction_fold,
                                "reaction_id": reaction,
                                "c_value": c_value,
                                "fusion_weight": weight,
                                "method": method,
                                **reciprocal_rank_and_hits(
                                    ranked, positives_by_query[key], budgets
                                ),
                            }
                        )
                    parameter_rows.append(
                        {
                            "split_id": split_id,
                            "partition": partition,
                            "c_value": c_value,
                            "fusion_weight": weight,
                            "method": method,
                        }
                    )

    query_frame = pd.DataFrame(query_rows)
    development = query_frame[query_frame.partition.eq("development")]
    selected_rows: list[dict[str, object]] = []
    frozen_rows: list[dict[str, object]] = []
    for budget in budgets:
        column = f"hit_at_{budget}"
        summary = (
            development.groupby(["c_value", "fusion_weight", "method"], as_index=False)
            .agg(hit_probability=(column, "mean"), mrr=("reciprocal_rank", "mean"))
            .sort_values(
                ["hit_probability", "mrr", "fusion_weight", "c_value"],
                ascending=[False, False, True, True],
            )
        )
        selected = summary.iloc[0]
        selected_rows.append(
            {
                "budget": budget,
                "c_value": float(selected.c_value),
                "fusion_weight": float(selected.fusion_weight),
                "method": str(selected.method),
                "development_hit_probability": float(selected.hit_probability),
                "development_mrr": float(selected.mrr),
            }
        )
        frozen = query_frame[
            query_frame.partition.eq("frozen")
            & query_frame.method.eq(str(selected.method))
        ].copy()
        baseline = query_frame[
            query_frame.partition.eq("frozen")
            & query_frame.c_value.eq(float(selected.c_value))
            & query_frame.fusion_weight.eq(0.0)
        ].copy()
        merge_keys = ["split_id", "reaction_id"]
        paired = frozen[merge_keys + [column]].merge(
            baseline[merge_keys + [column]],
            on=merge_keys,
            suffixes=("_selected", "_baseline"),
            validate="one_to_one",
        )
        difference = paired[f"{column}_selected"] - paired[f"{column}_baseline"]
        frozen_rows.append(
            {
                "budget": budget,
                "selected_method": str(selected.method),
                "n_queries": len(paired),
                "baseline_hit_probability": float(
                    paired[f"{column}_baseline"].mean()
                ),
                "selected_hit_probability": float(
                    paired[f"{column}_selected"].mean()
                ),
                "difference": float(difference.mean()),
                "new_hits": int((difference == 1).sum()),
                "lost_hits": int((difference == -1).sum()),
            }
        )

    selected_frame = pd.DataFrame(selected_rows)
    frozen_frame = pd.DataFrame(frozen_rows)
    query_frame.to_csv(output_dir / "query_metrics.csv", index=False)
    selected_frame.to_csv(output_dir / "selected_parameters.csv", index=False)
    frozen_frame.to_csv(output_dir / "frozen_metrics.csv", index=False)
    pd.DataFrame(fit_audit).to_csv(output_dir / "fit_audit.csv", index=False)
    annotations.to_csv(output_dir / "candidate_architecture_groups.csv", index=False)
    summary = {
        "method": "fold_local_current_pfam_architecture_reranking",
        "base_rankings": str(args.rankings.resolve()),
        "pfam_annotations": str(args.pfam_annotations.resolve()),
        "vocabulary": list(vocabulary),
        "classes": list(classes),
        "minimum_class_count": args.minimum_class_count,
        "maximum_classes": args.maximum_classes,
        "c_values": list(c_values),
        "fusion_weights": list(fusion_weights),
        "selection": "development cells have protein_fold==4 or reaction_fold==4; frozen cells are never used for parameter selection",
        "reaction_feature_schema": feature_schema,
        "outputs": {
            "selected_parameters": str(output_dir / "selected_parameters.csv"),
            "frozen_metrics": str(output_dir / "frozen_metrics.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("SELECTED")
    print(selected_frame.to_string(index=False))
    print("FROZEN")
    print(frozen_frame.to_string(index=False))


if __name__ == "__main__":
    main()
