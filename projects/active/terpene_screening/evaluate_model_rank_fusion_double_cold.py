from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict
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

DEFAULT_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_OUTPUT = ROOT / "results/terpene_model_rank_fusion_double_cold"
DEFAULT_BUDGETS = (3, 10, 20)


def parse_source(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise ValueError("Each source must use LABEL=RESULT_DIR")
    return label, Path(path).resolve()


def split_partition(split_id: str, development_fold: int) -> str:
    protein_fold, reaction_fold = map(
        int, str(split_id).replace("p", "").split("_r")
    )
    return (
        "development_9_cells"
        if protein_fold == development_fold or reaction_fold == development_fold
        else "frozen_16_cells"
    )


def rank_percentiles(scores: np.ndarray) -> np.ndarray:
    if scores.ndim != 2:
        raise ValueError("scores must be rank-2")
    n_candidates = scores.shape[1]
    if n_candidates == 1:
        return np.ones_like(scores, dtype=np.float32)
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    row_indices = np.arange(scores.shape[0])[:, None]
    ranks[row_indices, order] = np.arange(n_candidates)[None, :]
    return (1.0 - ranks / (n_candidates - 1)).astype(np.float32)


def load_score_matrix(
    result_dir: Path,
    split_id: str,
    protein_tensor: torch.Tensor,
    reaction_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    paths = sorted((result_dir / "models").glob(f"adapted_{split_id}_model*.pt"))
    if not paths:
        raise FileNotFoundError(
            f"No adapted checkpoints for {split_id} under {result_dir / 'models'}"
        )
    total = np.zeros(
        (reaction_tensor.shape[0], protein_tensor.shape[0]), dtype=np.float32
    )
    for path in paths:
        payload = torch.load(path, map_location=device, weights_only=False)
        model = TerpeneDualTower(ModelConfig(**payload["model_config"])).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        with torch.no_grad():
            protein_embeddings = model.encode_proteins(protein_tensor)
            reaction_embeddings = model.encode_reactions(reaction_tensor)
            total += (reaction_embeddings @ protein_embeddings.T).float().cpu().numpy()
        del model, payload, protein_embeddings, reaction_embeddings
    return total / len(paths)


def method_specifications(
    labels: list[str], pair_weights: tuple[float, ...], include_triples: bool
) -> dict[str, dict[str, float]]:
    methods: dict[str, dict[str, float]] = {
        f"single_{label}": {label: 1.0} for label in labels
    }
    for left, right in itertools.combinations(labels, 2):
        for left_weight in pair_weights:
            right_weight = 1.0 - left_weight
            name = f"pair_{left}_{left_weight:g}__{right}_{right_weight:g}"
            methods[name] = {left: left_weight, right: right_weight}
    if include_triples:
        for triple in itertools.combinations(labels, 3):
            name = "triple_equal__" + "__".join(triple)
            methods[name] = {label: 1.0 / 3.0 for label in triple}
    return methods


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


def bootstrap_delta(
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
    paired["difference"] = paired["selected_hit"] - paired["reference_hit"]
    cells = sorted(paired["split_id"].unique())
    values_by_cell = {
        cell: paired.loc[paired["split_id"].eq(cell), "difference"].to_numpy(float)
        for cell in cells
    }
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled = rng.choice(cells, size=len(cells), replace=True)
        boot[index] = np.concatenate([values_by_cell[str(cell)] for cell in sampled]).mean()
    return {
        "selected_method": selected_method,
        "reference_method": reference_method,
        "direction": direction,
        "selection_budget": budget,
        "n_paired_query_cells": len(paired),
        "selected_hit_probability": paired["selected_hit"].mean(),
        "reference_hit_probability": paired["reference_hit"].mean(),
        "absolute_hit_delta": paired["difference"].mean(),
        "bootstrap_ci_low": np.quantile(boot, 0.025),
        "bootstrap_ci_high": np.quantile(boot, 0.975),
        "selected_only_hits": int(
            ((paired["selected_hit"] == 1) & (paired["reference_hit"] == 0)).sum()
        ),
        "reference_only_hits": int(
            ((paired["selected_hit"] == 0) & (paired["reference_hit"] == 1)).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Development-selected rank fusion of heterogeneous TPS dual towers."
    )
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--pair-weights", default="0.25,0.5,0.75")
    parser.add_argument("--include-triples", action="store_true")
    parser.add_argument("--development-fold", type=int, default=4)
    parser.add_argument("--reference-label", default="base")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sources = dict(parse_source(value) for value in args.source)
    if len(sources) < 2:
        raise ValueError("At least two distinct model sources are required")
    if args.reference_label not in sources:
        raise ValueError(f"Reference label is not a source: {args.reference_label}")
    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    pair_weights = tuple(float(value) for value in args.pair_weights.split(",") if value)
    if any(not 0 < value < 1 for value in pair_weights):
        raise ValueError("Pair weights must be strictly within (0, 1)")
    methods = method_specifications(list(sources), pair_weights, args.include_triples)

    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
    device = torch.device(args.device)
    protein_tensor = torch.as_tensor(protein_matrix, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_matrix, dtype=torch.float32, device=device)

    records: list[dict[str, object]] = []
    split_summary: list[dict[str, object]] = []
    for protein_fold in range(5):
        for reaction_fold in range(5):
            split_id = f"p{protein_fold}_r{reaction_fold}"
            test_pairs = pairs[
                pairs["protein_fold"].eq(protein_fold)
                & pairs["reaction_fold"].eq(reaction_fold)
                & pairs["protein_seen"].str.lower().eq("false")
                & pairs["reaction_seen"].str.lower().eq("false")
            ].copy()
            if test_pairs.empty:
                continue
            source_r2e: dict[str, np.ndarray] = {}
            source_e2r: dict[str, np.ndarray] = {}
            for label, result_dir in sources.items():
                raw = load_score_matrix(
                    result_dir, split_id, protein_tensor, reaction_tensor, device
                )
                source_r2e[label] = rank_percentiles(raw)
                source_e2r[label] = rank_percentiles(raw.T)
            fused_r2e = {
                name: sum(weight * source_r2e[label] for label, weight in weights.items())
                for name, weights in methods.items()
            }
            fused_e2r = {
                name: sum(weight * source_e2r[label] for label, weight in weights.items())
                for name, weights in methods.items()
            }
            partition = split_partition(split_id, args.development_fold)
            for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
                positives = set(group["Entry"].astype(str))
                row = reaction_to_row[reaction_id]
                for method, matrix in fused_r2e.items():
                    metrics = rank_metrics(
                        matrix[row], protein_ids, positives, set(), budgets
                    )
                    records.append(
                        {
                            "split_id": split_id,
                            "partition": partition,
                            "direction": "reaction_to_enzyme",
                            "method": method,
                            "query_id": reaction_id,
                            **metrics,
                        }
                    )
            for protein_id, group in test_pairs.groupby("Entry", sort=True):
                positives = set(group["rhea_id"].astype(str))
                row = protein_to_row[protein_id]
                for method, matrix in fused_e2r.items():
                    metrics = rank_metrics(
                        matrix[row], reaction_ids, positives, set(), budgets
                    )
                    records.append(
                        {
                            "split_id": split_id,
                            "partition": partition,
                            "direction": "enzyme_to_reaction",
                            "method": method,
                            "query_id": protein_id,
                            **metrics,
                        }
                    )
            split_summary.append(
                {
                    "split_id": split_id,
                    "partition": partition,
                    "test_pairs": len(test_pairs),
                    "test_reactions": test_pairs["rhea_id"].nunique(),
                    "test_proteins": test_pairs["Entry"].nunique(),
                }
            )

    query_metrics = pd.DataFrame(records)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics = aggregate(query_metrics, budgets)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(split_summary).to_csv(output_dir / "split_summary.csv", index=False)

    development = metrics[metrics["partition"].eq("development_9_cells")]
    frozen = metrics[metrics["partition"].eq("frozen_16_cells")]
    selections: list[pd.DataFrame] = []
    frozen_rows: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, object]] = []
    reference_method = f"single_{args.reference_label}"
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
                bootstrap_delta(
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
    paired_bootstrap = pd.DataFrame(bootstrap_rows)
    development_selection.to_csv(output_dir / "development_selection.csv", index=False)
    frozen_evaluation.to_csv(output_dir / "frozen_evaluation.csv", index=False)
    paired_bootstrap.to_csv(output_dir / "frozen_paired_bootstrap.csv", index=False)

    summary = {
        "sources": {label: str(path) for label, path in sources.items()},
        "method_count": len(methods),
        "methods": methods,
        "development_fold": args.development_fold,
        "selection_rule": "All fusion methods are selected only on the nine development cells.",
        "reference_method": reference_method,
        "budgets": budgets,
        "pair_weights": pair_weights,
        "include_triples": args.include_triples,
        "device": str(device),
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "development_selection": str(output_dir / "development_selection.csv"),
            "frozen_evaluation": str(output_dir / "frozen_evaluation.csv"),
            "frozen_paired_bootstrap": str(output_dir / "frozen_paired_bootstrap.csv"),
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
    print("\nPAIRED BOOTSTRAP")
    print(paired_bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
