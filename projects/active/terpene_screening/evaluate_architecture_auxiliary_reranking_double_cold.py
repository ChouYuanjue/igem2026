from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_model_rank_fusion_double_cold import (  # noqa: E402
    load_score_matrix,
    parse_source,
    split_partition,
)
from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics  # noqa: E402

DEFAULT_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_ARCHITECTURES = DEFAULT_CACHE / "protein_architecture_annotations.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_architecture_auxiliary_reranking_double_cold"
DEFAULT_BUDGETS = (3, 10, 20)
ARCHITECTURE_CLASSES = (
    "plant_full",
    "bacterial_classI",
    "osc_full",
    "plant_single",
    "classII_single",
    "classI_hybrid",
)


def architecture_class(value: str) -> str:
    normalized = ";".join(part for part in str(value).split(";") if part)
    mapping = {
        "PF01397;PF03936": "plant_full",
        "PF19086": "bacterial_classI",
        "PF13243;PF13249": "osc_full",
        "PF01397": "plant_single",
        "PF13243": "classII_single",
        "PF01397;PF19086": "classI_hybrid",
    }
    return mapping.get(normalized, "")


def tied_rank_percentiles(matrix: np.ndarray) -> np.ndarray:
    result = np.empty_like(matrix, dtype=np.float32)
    n_candidates = matrix.shape[1]
    if n_candidates == 1:
        result.fill(1.0)
        return result
    for row_index in range(matrix.shape[0]):
        ranks = rankdata(-matrix[row_index], method="average")
        result[row_index] = 1.0 - (ranks - 1.0) / (n_candidates - 1.0)
    return result


def build_reaction_labels(
    pairs: pd.DataFrame,
    protein_to_architecture: dict[str, str],
    reaction_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    labels = np.zeros((len(reaction_ids), len(ARCHITECTURE_CLASSES)), dtype=np.int8)
    labelled = np.zeros(len(reaction_ids), dtype=bool)
    class_to_column = {value: index for index, value in enumerate(ARCHITECTURE_CLASSES)}
    for reaction_id, group in pairs.groupby("rhea_id", sort=False):
        reaction_id = str(reaction_id)
        if reaction_id not in reaction_to_row:
            continue
        classes = {
            protein_to_architecture.get(str(protein_id), "")
            for protein_id in group["Entry"].astype(str)
        } - {""}
        if not classes:
            continue
        row = reaction_to_row[reaction_id]
        labelled[row] = True
        for value in classes:
            labels[row, class_to_column[value]] = 1
    return labels, labelled


def fit_architecture_predictors(
    reaction_features: np.ndarray,
    labels: np.ndarray,
    labelled: np.ndarray,
    c_value: float,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    predictions = np.zeros_like(labels, dtype=np.float32)
    audit: list[dict[str, object]] = []
    x = reaction_features[labelled]
    if len(x) == 0:
        raise ValueError("No training reactions have architecture labels")
    for column, class_name in enumerate(ARCHITECTURE_CLASSES):
        y = labels[labelled, column]
        positives = int(y.sum())
        negatives = int(len(y) - positives)
        if positives == 0:
            probability = np.zeros(len(reaction_features), dtype=np.float32)
            fit_mode = "constant_zero"
        elif negatives == 0:
            probability = np.ones(len(reaction_features), dtype=np.float32)
            fit_mode = "constant_one"
        else:
            model = LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=3000,
                solver="liblinear",
                random_state=seed + column,
            )
            model.fit(x, y)
            probability = model.predict_proba(reaction_features)[:, 1].astype(np.float32)
            fit_mode = "logistic_regression"
        predictions[:, column] = probability
        audit.append(
            {
                "architecture_class": class_name,
                "c_value": c_value,
                "labelled_reactions": int(len(y)),
                "positive_reactions": positives,
                "negative_reactions": negatives,
                "fit_mode": fit_mode,
                "mean_prediction": float(probability.mean()),
            }
        )
    return predictions, audit


def architecture_score_matrices(
    probabilities: np.ndarray,
    protein_architectures: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    class_to_column = {value: index for index, value in enumerate(ARCHITECTURE_CLASSES)}
    neutral_by_reaction = probabilities.mean(axis=1)
    r2e = np.empty((len(probabilities), len(protein_architectures)), dtype=np.float32)
    e2r = np.empty((len(protein_architectures), len(probabilities)), dtype=np.float32)
    for protein_index, architecture in enumerate(protein_architectures):
        column = class_to_column.get(architecture)
        if column is None:
            r2e[:, protein_index] = neutral_by_reaction
            e2r[protein_index, :] = 0.5
        else:
            r2e[:, protein_index] = probabilities[:, column]
            e2r[protein_index, :] = probabilities[:, column]
    return r2e, e2r


def classification_audit(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    labelled: np.ndarray,
    split_id: str,
    c_value: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column, class_name in enumerate(ARCHITECTURE_CLASSES):
        y = true_labels[labelled, column]
        score = probabilities[labelled, column]
        positives = int(y.sum())
        negatives = int(len(y) - positives)
        average_precision = (
            float(average_precision_score(y, score)) if positives and negatives else np.nan
        )
        roc_auc = float(roc_auc_score(y, score)) if positives and negatives else np.nan
        rows.append(
            {
                "split_id": split_id,
                "c_value": c_value,
                "architecture_class": class_name,
                "labelled_test_reactions": int(len(y)),
                "positive_test_reactions": positives,
                "average_precision": average_precision,
                "roc_auc": roc_auc,
            }
        )
    if labelled.any():
        top_class = probabilities[labelled].argmax(axis=1)
        top_hit = true_labels[labelled][np.arange(labelled.sum()), top_class]
        rows.append(
            {
                "split_id": split_id,
                "c_value": c_value,
                "architecture_class": "macro_top1_true_class",
                "labelled_test_reactions": int(labelled.sum()),
                "positive_test_reactions": int(top_hit.sum()),
                "average_precision": float(top_hit.mean()),
                "roc_auc": np.nan,
            }
        )
    return rows


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


def paired_bootstrap(
    frame: pd.DataFrame,
    selected_method: str,
    reference_method: str,
    direction: str,
    budget: int,
    samples: int,
    seed: int,
) -> dict[str, object]:
    keys = ["split_id", "direction", "query_id"]
    column = f"hit_at_{budget}"
    selected = frame[
        frame["method"].eq(selected_method) & frame["direction"].eq(direction)
    ][keys + [column]].rename(columns={column: "selected_hit"})
    reference = frame[
        frame["method"].eq(reference_method) & frame["direction"].eq(direction)
    ][keys + [column]].rename(columns={column: "reference_hit"})
    paired = selected.merge(reference, on=keys, validate="one_to_one")
    if paired.empty:
        raise ValueError("No paired frozen rows for bootstrap comparison")
    paired["difference"] = paired["selected_hit"] - paired["reference_hit"]
    cells = sorted(paired["split_id"].unique())
    by_cell = {
        cell: paired.loc[paired["split_id"].eq(cell), "difference"].to_numpy(float)
        for cell in cells
    }
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled = rng.choice(cells, size=len(cells), replace=True)
        boot[index] = np.concatenate([by_cell[str(cell)] for cell in sampled]).mean()
    return {
        "selected_method": selected_method,
        "reference_method": reference_method,
        "direction": direction,
        "selection_budget": budget,
        "n_paired_query_cells": int(len(paired)),
        "selected_hit_probability": float(paired["selected_hit"].mean()),
        "reference_hit_probability": float(paired["reference_hit"].mean()),
        "absolute_hit_delta": float(paired["difference"].mean()),
        "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
        "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
        "selected_only_hits": int(
            ((paired["selected_hit"] == 1) & (paired["reference_hit"] == 0)).sum()
        ),
        "reference_only_hits": int(
            ((paired["selected_hit"] == 0) & (paired["reference_hit"] == 1)).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Soft Pfam-architecture auxiliary reranking under strict double-cold splits."
    )
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--reference-label", default="base")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--architectures", type=Path, default=DEFAULT_ARCHITECTURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--c-values", default="0.03,0.1,0.3,1.0")
    parser.add_argument("--fusion-weights", default="0,0.05,0.1,0.2,0.3")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--development-fold", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sources = dict(parse_source(value) for value in args.source)
    if args.reference_label not in sources:
        raise ValueError(f"Reference source missing: {args.reference_label}")
    c_values = tuple(float(value) for value in args.c_values.split(",") if value)
    fusion_weights = tuple(float(value) for value in args.fusion_weights.split(",") if value)
    if any(value < 0 or value > 1 for value in fusion_weights):
        raise ValueError("Fusion weights must be within [0, 1]")
    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    protein_features = np.load(cache_dir / "protein_features.npy").astype(np.float32)
    reaction_features = np.load(cache_dir / "reaction_features.npy").astype(np.float32)
    protein_table = pd.read_csv(cache_dir / "protein_entities.csv", dtype=str).fillna("")
    reaction_table = pd.read_csv(cache_dir / "reaction_entities.csv", dtype=str).fillna("")
    pairs = pd.read_csv(cache_dir / "marts_pair_folds.csv", dtype=str).fillna("")
    pairs["protein_fold"] = pd.to_numeric(pairs["protein_fold"]).astype(int)
    pairs["reaction_fold"] = pd.to_numeric(pairs["reaction_fold"]).astype(int)
    architecture_table = pd.read_csv(args.architectures, dtype=str).fillna("")
    architecture_table["architecture_class"] = architecture_table["pfam_combination"].map(
        architecture_class
    )
    protein_ids = protein_table["protein_id"].astype(str).tolist()
    reaction_ids = reaction_table["reaction_id"].astype(str).tolist()
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    protein_to_architecture = dict(
        zip(architecture_table["protein_id"], architecture_table["architecture_class"])
    )
    protein_architectures = [protein_to_architecture.get(value, "") for value in protein_ids]
    protein_tensor = torch.as_tensor(protein_features, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_features, dtype=torch.float32, device=device)

    query_records: list[dict[str, object]] = []
    classifier_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for protein_fold in range(5):
        for reaction_fold in range(5):
            split_id = f"p{protein_fold}_r{reaction_fold}"
            partition = split_partition(split_id, args.development_fold)
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
            train_labels, train_labelled = build_reaction_labels(
                train_pairs, protein_to_architecture, reaction_ids
            )
            test_labels, test_labelled = build_reaction_labels(
                test_pairs, protein_to_architecture, reaction_ids
            )
            source_scores = {
                label: load_score_matrix(
                    result_dir, split_id, protein_tensor, reaction_tensor, device
                )
                for label, result_dir in sources.items()
            }
            source_r2e_ranks = {
                label: tied_rank_percentiles(matrix) for label, matrix in source_scores.items()
            }
            source_e2r_ranks = {
                label: tied_rank_percentiles(matrix.T) for label, matrix in source_scores.items()
            }
            for c_value in c_values:
                probabilities, audit = fit_architecture_predictors(
                    reaction_features,
                    train_labels,
                    train_labelled,
                    c_value,
                    args.seed + protein_fold * 100 + reaction_fold,
                )
                for row in audit:
                    row.update(
                        {
                            "split_id": split_id,
                            "partition": partition,
                            "train_pairs": len(train_pairs),
                            "labelled_train_reactions": int(train_labelled.sum()),
                        }
                    )
                fit_rows.extend(audit)
                classifier_rows.extend(
                    classification_audit(
                        probabilities,
                        test_labels,
                        test_labelled,
                        split_id,
                        c_value,
                    )
                )
                architecture_r2e, architecture_e2r = architecture_score_matrices(
                    probabilities, protein_architectures
                )
                architecture_r2e_rank = tied_rank_percentiles(architecture_r2e)
                architecture_e2r_rank = tied_rank_percentiles(architecture_e2r)
                for source_label in sources:
                    for fusion_weight in fusion_weights:
                        method = (
                            f"{source_label}__arch_c{c_value:g}_w{fusion_weight:g}"
                        )
                        r2e_scores = (
                            (1.0 - fusion_weight) * source_r2e_ranks[source_label]
                            + fusion_weight * architecture_r2e_rank
                        )
                        e2r_scores = (
                            (1.0 - fusion_weight) * source_e2r_ranks[source_label]
                            + fusion_weight * architecture_e2r_rank
                        )
                        for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
                            positives = set(group["Entry"].astype(str))
                            metrics = rank_metrics(
                                r2e_scores[reaction_to_row[str(reaction_id)]],
                                protein_ids,
                                positives,
                                set(),
                                budgets,
                            )
                            query_records.append(
                                {
                                    "split_id": split_id,
                                    "partition": partition,
                                    "method": method,
                                    "source_label": source_label,
                                    "c_value": c_value,
                                    "fusion_weight": fusion_weight,
                                    "direction": "reaction_to_enzyme",
                                    "query_id": str(reaction_id),
                                    **metrics,
                                }
                            )
                        for protein_id, group in test_pairs.groupby("Entry", sort=True):
                            positives = set(group["rhea_id"].astype(str))
                            metrics = rank_metrics(
                                e2r_scores[protein_to_row[str(protein_id)]],
                                reaction_ids,
                                positives,
                                set(),
                                budgets,
                            )
                            query_records.append(
                                {
                                    "split_id": split_id,
                                    "partition": partition,
                                    "method": method,
                                    "source_label": source_label,
                                    "c_value": c_value,
                                    "fusion_weight": fusion_weight,
                                    "direction": "enzyme_to_reaction",
                                    "query_id": str(protein_id),
                                    **metrics,
                                }
                            )
            split_rows.append(
                {
                    "split_id": split_id,
                    "partition": partition,
                    "train_pairs": len(train_pairs),
                    "test_pairs": len(test_pairs),
                    "labelled_train_reactions": int(train_labelled.sum()),
                    "labelled_test_reactions": int(test_labelled.sum()),
                }
            )

    query_metrics = pd.DataFrame(query_records)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics = aggregate(query_metrics, budgets)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(classifier_rows).to_csv(
        output_dir / "architecture_classifier_audit.csv", index=False
    )
    pd.DataFrame(fit_rows).to_csv(output_dir / "architecture_fit_audit.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_dir / "split_summary.csv", index=False)

    development = metrics[metrics["partition"].eq("development_9_cells")]
    frozen = metrics[metrics["partition"].eq("frozen_16_cells")]
    selections: list[pd.DataFrame] = []
    frozen_rows: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, object]] = []
    reference_method = f"{args.reference_label}__arch_c{c_values[0]:g}_w0"
    for direction in sorted(development["direction"].unique()):
        group = development[development["direction"].eq(direction)]
        for budget in budgets:
            selected = group.sort_values(
                [f"hit_probability_at_{budget}", "mean_reciprocal_rank", "method"],
                ascending=[False, False, True],
            ).head(1).copy()
            selected.insert(3, "selection_budget", budget)
            selections.append(selected)
            method = str(selected.iloc[0]["method"])
            frozen_row = frozen[
                frozen["direction"].eq(direction) & frozen["method"].eq(method)
            ].copy()
            frozen_row.insert(3, "selection_budget", budget)
            frozen_rows.append(frozen_row)
            bootstrap_rows.append(
                paired_bootstrap(
                    query_metrics[query_metrics["partition"].eq("frozen_16_cells")],
                    method,
                    reference_method,
                    direction,
                    budget,
                    args.bootstrap_samples,
                    args.seed + budget + (1000 if direction == "enzyme_to_reaction" else 0),
                )
            )
    development_selection = pd.concat(selections, ignore_index=True)
    frozen_evaluation = pd.concat(frozen_rows, ignore_index=True)
    frozen_bootstrap = pd.DataFrame(bootstrap_rows)
    development_selection.to_csv(output_dir / "development_selection.csv", index=False)
    frozen_evaluation.to_csv(output_dir / "frozen_evaluation.csv", index=False)
    frozen_bootstrap.to_csv(output_dir / "frozen_paired_bootstrap.csv", index=False)

    summary = {
        "sources": {label: str(path) for label, path in sources.items()},
        "reference_method": reference_method,
        "architecture_classes": ARCHITECTURE_CLASSES,
        "annotated_proteins": int(sum(bool(value) for value in protein_architectures)),
        "unknown_architecture_proteins": int(
            sum(not bool(value) for value in protein_architectures)
        ),
        "c_values": c_values,
        "fusion_weights": fusion_weights,
        "development_fold": args.development_fold,
        "selection_rule": (
            "Architecture classifier and reranking hyperparameters use training pairs and the nine development cells only; "
            "the selected method is evaluated unchanged on sixteen frozen cells."
        ),
        "unknown_architecture_policy": (
            "Unknown candidate architectures receive the reaction-wise mean architecture probability; "
            "unknown enzyme queries receive a constant score and therefore no architecture reranking."
        ),
        "budgets": budgets,
        "device": str(device),
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "development_selection": str(output_dir / "development_selection.csv"),
            "frozen_evaluation": str(output_dir / "frozen_evaluation.csv"),
            "frozen_paired_bootstrap": str(output_dir / "frozen_paired_bootstrap.csv"),
            "architecture_classifier_audit": str(
                output_dir / "architecture_classifier_audit.csv"
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("\nDEVELOPMENT SELECTION")
    print(development_selection.to_string(index=False))
    print("\nFROZEN EVALUATION")
    print(frozen_evaluation.to_string(index=False))
    print("\nFROZEN PAIRED BOOTSTRAP")
    print(frozen_bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
