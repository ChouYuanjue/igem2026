from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.train_dual_tower_cold import (
    ModelConfig,
    TerpeneDualTower,
    rank_metrics,
)
from projects.active.terpene_screening.train_horizyn_reaction_adapter_double_cold import (
    AdapterConfig,
    ProteinAdapter,
)

DEFAULT_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_OUTPUT = ROOT / "results/terpene_cross_model_rank_fusion"
DEFAULT_BUDGETS = (3, 10, 20)
_SPLIT_RE = re.compile(r"p(\d+)_r(\d+)")


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    denominator[denominator == 0] = 1.0
    return matrix / denominator


def rank_percentile(scores: np.ndarray, ids: list[str]) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.asarray(ids, dtype=object), -scores))
    result = np.empty(len(scores), dtype=np.float32)
    if len(scores) <= 1:
        result.fill(1.0)
        return result
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_position = (start + end - 1) / 2.0
        result[order[start:end]] = 1.0 - average_position / (len(scores) - 1)
        start = end
    return result


def parse_source(value: str) -> tuple[str, Path, str]:
    if "=" not in value:
        raise ValueError("Source must be LABEL=PATH[:standard|horizyn]")
    label, payload = value.split("=", 1)
    source_type = "standard"
    path_text = payload
    if payload.endswith(":horizyn"):
        path_text = payload[: -len(":horizyn")]
        source_type = "horizyn"
    elif payload.endswith(":standard"):
        path_text = payload[: -len(":standard")]
    path = Path(path_text).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return label, path, source_type


def split_ids(frame: pd.DataFrame) -> list[str]:
    return sorted(
        {
            f"p{int(p)}_r{int(r)}"
            for p, r in zip(frame["protein_fold"], frame["reaction_fold"])
        }
    )


def partition(split_id: str, development_fold: int) -> str:
    match = _SPLIT_RE.fullmatch(split_id)
    if match is None:
        raise ValueError(f"Invalid split id: {split_id}")
    p_fold, r_fold = map(int, match.groups())
    return "development_9_cells" if p_fold == development_fold or r_fold == development_fold else "frozen_16_cells"


def load_standard_scores(
    source_dir: Path,
    split_id: str,
    protein_tensor: torch.Tensor,
    reaction_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    paths = sorted((source_dir / "models").glob(f"adapted_{split_id}_model*.pt"))
    if not paths:
        raise FileNotFoundError(f"No standard checkpoints for {split_id} under {source_dir}")
    total: np.ndarray | None = None
    for path in paths:
        payload = torch.load(path, map_location=device, weights_only=False)
        model = TerpeneDualTower(ModelConfig(**payload["model_config"])).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        with torch.no_grad():
            proteins = model.encode_proteins(protein_tensor).cpu().numpy()
            reactions = model.encode_reactions(reaction_tensor).cpu().numpy()
        scores = reactions @ proteins.T
        total = scores if total is None else total + scores
    assert total is not None
    return total / len(paths)


def load_horizyn_scores(
    source_dir: Path,
    split_id: str,
    protein_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    paths = sorted((source_dir / "models").glob(f"adapter_{split_id}_seed*.pt"))
    if not paths:
        raise FileNotFoundError(f"No Horizyn adapter checkpoints for {split_id} under {source_dir}")
    reaction_embeddings = normalize_rows(np.load(source_dir / "horizyn_reaction_embeddings.npy"))
    total: np.ndarray | None = None
    for path in paths:
        payload = torch.load(path, map_location=device, weights_only=False)
        adapter = ProteinAdapter(AdapterConfig(**payload["adapter_config"])).to(device)
        adapter.load_state_dict(payload["adapter_state_dict"])
        adapter.eval()
        with torch.no_grad():
            proteins = adapter(protein_tensor).cpu().numpy()
        scores = reaction_embeddings @ proteins.T
        total = scores if total is None else total + scores
    assert total is not None
    return total / len(paths)


def method_weights(labels: list[str]) -> dict[str, dict[str, float]]:
    methods: dict[str, dict[str, float]] = {label: {label: 1.0} for label in labels}
    for a, b in itertools.combinations(labels, 2):
        for a_weight in (0.25, 0.5, 0.75):
            name = f"{a}:{a_weight:g}+{b}:{1-a_weight:g}"
            methods[name] = {a: a_weight, b: 1.0 - a_weight}
    for combo in itertools.combinations(labels, 3):
        name = "+".join(combo) + ":equal3"
        methods[name] = {label: 1.0 / 3.0 for label in combo}
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
        aggregations[f"positive_recall_at_{budget}"] = (f"positive_recall_at_{budget}", "mean")
    return frame.groupby(["partition", "direction", "method"]).agg(**aggregations).reset_index()


def select_on_development(metrics: pd.DataFrame, budgets: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[dict[str, object]] = []
    frozen_rows: list[pd.Series] = []
    dev = metrics[metrics["partition"].eq("development_9_cells")]
    frozen = metrics[metrics["partition"].eq("frozen_16_cells")]
    for direction in sorted(dev["direction"].unique()):
        direction_dev = dev[dev["direction"].eq(direction)]
        for budget in budgets:
            hit_col = f"hit_probability_at_{budget}"
            best = direction_dev.sort_values(
                [hit_col, "mean_reciprocal_rank", "median_best_positive_rank", "method"],
                ascending=[False, False, True, True],
            ).iloc[0]
            selected_rows.append(
                {
                    "direction": direction,
                    "selection_budget": budget,
                    "selected_method": best["method"],
                    "development_hit": best[hit_col],
                    "development_mrr": best["mean_reciprocal_rank"],
                }
            )
            frozen_match = frozen[
                frozen["direction"].eq(direction) & frozen["method"].eq(best["method"])
            ]
            if len(frozen_match) != 1:
                raise ValueError(f"Frozen row missing for {direction} {best['method']}")
            row = frozen_match.iloc[0].copy()
            row["selection_budget"] = budget
            row["selected_on"] = "development_9_cells"
            frozen_rows.append(row)
    return pd.DataFrame(selected_rows), pd.DataFrame(frozen_rows)


def leave_one_cell_out_stability(
    query_metrics: pd.DataFrame,
    budgets: tuple[int, ...],
    development_fold: int,
) -> pd.DataFrame:
    development = query_metrics[query_metrics["partition"].eq("development_9_cells")]
    cells = sorted(development["split_id"].unique())
    rows: list[dict[str, object]] = []
    for held_cell in cells:
        train = development[~development["split_id"].eq(held_cell)]
        metrics = aggregate(train, budgets)
        for direction in sorted(train["direction"].unique()):
            group = metrics[metrics["direction"].eq(direction)]
            for budget in budgets:
                hit_col = f"hit_probability_at_{budget}"
                best = group.sort_values(
                    [hit_col, "mean_reciprocal_rank", "median_best_positive_rank", "method"],
                    ascending=[False, False, True, True],
                ).iloc[0]
                rows.append(
                    {
                        "held_development_cell": held_cell,
                        "direction": direction,
                        "selection_budget": budget,
                        "selected_method": best["method"],
                    }
                )
    result = pd.DataFrame(rows)
    counts = (
        result.groupby(["direction", "selection_budget", "selected_method"])
        .size()
        .rename("selection_count")
        .reset_index()
    )
    counts["n_development_cells"] = len(cells)
    counts["selection_fraction"] = counts["selection_count"] / len(cells)
    return counts.sort_values(
        ["direction", "selection_budget", "selection_count", "selected_method"],
        ascending=[True, True, False, True],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate-level rank fusion selected on 9 development cells and evaluated on 16 frozen cells.")
    parser.add_argument("--source", action="append", required=True, help="LABEL=PATH[:standard|horizyn]")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--development-fold", type=int, default=4)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    sources = [parse_source(value) for value in args.source]
    labels = [label for label, _, _ in sources]
    if len(set(labels)) != len(labels):
        raise ValueError("Source labels must be unique")
    methods = method_weights(labels)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve()
    device = torch.device(args.device)

    protein_features = normalize_rows(np.load(cache_dir / "protein_features.npy"))
    reaction_features = np.load(cache_dir / "reaction_features.npy").astype(np.float32)
    protein_table = pd.read_csv(cache_dir / "protein_entities.csv", dtype=str).fillna("")
    reaction_table = pd.read_csv(cache_dir / "reaction_entities.csv", dtype=str).fillna("")
    pairs = pd.read_csv(cache_dir / "marts_pair_folds.csv", dtype=str).fillna("")
    pairs["protein_fold"] = pd.to_numeric(pairs["protein_fold"]).astype(int)
    pairs["reaction_fold"] = pd.to_numeric(pairs["reaction_fold"]).astype(int)
    protein_ids = protein_table["protein_id"].astype(str).tolist()
    reaction_ids = reaction_table["reaction_id"].astype(str).tolist()
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    protein_tensor = torch.as_tensor(protein_features, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_features, dtype=torch.float32, device=device)

    records: list[dict[str, object]] = []
    for split_id in split_ids(pairs):
        match = _SPLIT_RE.fullmatch(split_id)
        assert match is not None
        p_fold, r_fold = map(int, match.groups())
        test_pairs = pairs[
            pairs["protein_fold"].eq(p_fold)
            & pairs["reaction_fold"].eq(r_fold)
            & pairs["protein_seen"].astype(str).str.lower().isin({"false", "0"})
            & pairs["reaction_seen"].astype(str).str.lower().isin({"false", "0"})
        ]
        if test_pairs.empty:
            continue
        source_scores: dict[str, np.ndarray] = {}
        for label, path, source_type in sources:
            if source_type == "horizyn":
                source_scores[label] = load_horizyn_scores(path, split_id, protein_tensor, device)
            else:
                source_scores[label] = load_standard_scores(path, split_id, protein_tensor, reaction_tensor, device)

        for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
            positives = set(group["Entry"].astype(str))
            reaction_row = reaction_to_row[reaction_id]
            source_ranks = {
                label: rank_percentile(scores[reaction_row], protein_ids)
                for label, scores in source_scores.items()
            }
            for method, weights in methods.items():
                fused = sum(weights[label] * source_ranks[label] for label in weights)
                metrics = rank_metrics(fused, protein_ids, positives, set(), budgets)
                records.append(
                    {
                        "split_id": split_id,
                        "partition": partition(split_id, args.development_fold),
                        "direction": "reaction_to_enzyme",
                        "query_id": reaction_id,
                        "method": method,
                        **metrics,
                    }
                )

        for protein_id, group in test_pairs.groupby("Entry", sort=True):
            positives = set(group["rhea_id"].astype(str))
            protein_row = protein_to_row[protein_id]
            source_ranks = {
                label: rank_percentile(scores[:, protein_row], reaction_ids)
                for label, scores in source_scores.items()
            }
            for method, weights in methods.items():
                fused = sum(weights[label] * source_ranks[label] for label in weights)
                metrics = rank_metrics(fused, reaction_ids, positives, set(), budgets)
                records.append(
                    {
                        "split_id": split_id,
                        "partition": partition(split_id, args.development_fold),
                        "direction": "enzyme_to_reaction",
                        "query_id": protein_id,
                        "method": method,
                        **metrics,
                    }
                )

    query_metrics = pd.DataFrame(records)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics = aggregate(query_metrics, budgets)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    selected, frozen_selected = select_on_development(metrics, budgets)
    selected.to_csv(output_dir / "development_selected_methods.csv", index=False)
    frozen_selected.to_csv(output_dir / "frozen_performance_of_development_selection.csv", index=False)
    stability = leave_one_cell_out_stability(query_metrics, budgets, args.development_fold)
    stability.to_csv(output_dir / "development_leave_one_cell_out_stability.csv", index=False)
    method_table = pd.DataFrame(
        [
            {"method": method, "weights_json": json.dumps(weights, sort_keys=True)}
            for method, weights in methods.items()
        ]
    )
    method_table.to_csv(output_dir / "method_weights.csv", index=False)
    summary = {
        "sources": [
            {"label": label, "path": str(path), "type": source_type}
            for label, path, source_type in sources
        ],
        "n_methods": len(methods),
        "development_fold": args.development_fold,
        "development_cells": 9,
        "frozen_cells": 16,
        "budgets": budgets,
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "development_selected_methods": str(output_dir / "development_selected_methods.csv"),
            "frozen_performance": str(output_dir / "frozen_performance_of_development_selection.csv"),
            "stability": str(output_dir / "development_leave_one_cell_out_stability.csv"),
            "method_weights": str(output_dir / "method_weights.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nDEVELOPMENT SELECTION")
    print(selected.to_string(index=False))
    print("\nFROZEN PERFORMANCE")
    cols = [
        "direction", "selection_budget", "method", "n_query_cells", "mean_reciprocal_rank",
        "median_best_positive_rank", "hit_probability_at_3", "hit_probability_at_10", "hit_probability_at_20",
    ]
    print(frozen_selected[cols].to_string(index=False))
    print("\nSTABILITY TOP")
    print(stability.groupby(["direction", "selection_budget"], as_index=False).head(3).to_string(index=False))


if __name__ == "__main__":
    main()
