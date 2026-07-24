from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_uniprot_expanded_double_cold import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_LONG_MODELS,
    DEFAULT_SHORT_MODELS,
    ensemble_scores,
    load_fold_models,
    normalize_rows,
)
from projects.active.terpene_screening.evaluate_uniprot_tiered_double_cold import (  # noqa: E402
    DEFAULT_METADATA,
    DEFAULT_UNIPROT,
    TIER_GROUPS,
    load_uniprot_metadata,
)
from projects.active.terpene_screening.rank_open_world import load_protein_library  # noqa: E402
from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics  # noqa: E402

DEFAULT_OUTPUT = ROOT / "results/terpene_candidate_hub_normalization_double_cold"
DEFAULT_LOCAL_K = 20


def normalization_parameters(
    score_matrix: np.ndarray,
    train_rows: np.ndarray,
    local_k: int,
) -> dict[str, np.ndarray]:
    train = score_matrix[train_rows]
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-6] = 1.0
    k = min(int(local_k), len(train_rows))
    if k <= 0:
        raise ValueError("No training reactions available for hub normalization")
    local = np.partition(train, len(train) - k, axis=0)[-k:].mean(axis=0)
    return {"mean": mean, "std": std, "local_topk_mean": local}


def normalized_scores(
    raw: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    centered = raw - parameters["mean"]
    return {
        "raw": raw,
        "candidate_mean_centered": centered,
        "candidate_zscore": centered / parameters["std"],
        "candidate_local_density": raw - parameters["local_topk_mean"],
    }


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["candidate_universe", "score_normalization", "budget"])
        .agg(
            n_queries=("query_id", "size"),
            hit_probability=("hit", "mean"),
            positive_recall=("positive_recall", "mean"),
            mean_reciprocal_rank=("reciprocal_rank", "mean"),
            median_best_positive_rank=("best_positive_rank", "median"),
            mean_added_above_best_positive=("added_above_best_positive", "mean"),
            median_added_above_best_positive=("added_above_best_positive", "median"),
            added_top1_fraction=("top1_is_added", "mean"),
        )
        .reset_index()
    )


def compare_to_canonical_raw(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for budget, budget_frame in frame.groupby("budget", sort=True):
        baseline = budget_frame[
            budget_frame["candidate_universe"].eq("canonical")
            & budget_frame["score_normalization"].eq("raw")
        ].set_index(["split_id", "query_id"])
        for (universe, normalization), group in budget_frame.groupby(
            ["candidate_universe", "score_normalization"], sort=True
        ):
            candidate = group.set_index(["split_id", "query_id"])
            aligned = baseline[["hit", "best_positive_rank", "reciprocal_rank"]].join(
                candidate[["hit", "best_positive_rank", "reciprocal_rank"]],
                lsuffix="_baseline",
                rsuffix="_candidate",
                how="inner",
            )
            base_hit = aligned["hit_baseline"].astype(int)
            cand_hit = aligned["hit_candidate"].astype(int)
            retained = int(((base_hit == 1) & (cand_hit == 1)).sum())
            rows.append(
                {
                    "budget": int(budget),
                    "candidate_universe": universe,
                    "score_normalization": normalization,
                    "n_queries": len(aligned),
                    "baseline_hits": int(base_hit.sum()),
                    "candidate_hits": int(cand_hit.sum()),
                    "retained_baseline_hits": retained,
                    "baseline_hits_lost": int(((base_hit == 1) & (cand_hit == 0)).sum()),
                    "hit_retention_fraction": (
                        retained / base_hit.sum() if base_hit.sum() else np.nan
                    ),
                    "mrr_ratio_to_baseline": float(
                        aligned["reciprocal_rank_candidate"].mean()
                        / aligned["reciprocal_rank_baseline"].mean()
                    ),
                    "median_rank_change": float(
                        (
                            aligned["best_positive_rank_candidate"]
                            - aligned["best_positive_rank_baseline"]
                        ).median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict double-cold candidate-hub normalization benchmark."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--uniprot-protein-dir", type=Path, default=DEFAULT_UNIPROT)
    parser.add_argument("--uniprot-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--short-model-dir", type=Path, default=DEFAULT_SHORT_MODELS)
    parser.add_argument("--long-model-dir", type=Path, default=DEFAULT_LONG_MODELS)
    parser.add_argument("--local-k", type=int, default=DEFAULT_LOCAL_K)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    canonical_features = normalize_rows(np.load(cache_dir / "protein_features.npy").astype(np.float32))
    reaction_features = np.load(cache_dir / "reaction_features.npy").astype(np.float32)
    protein_table = pd.read_csv(cache_dir / "protein_entities.csv", dtype=str).fillna("")
    reaction_table = pd.read_csv(cache_dir / "reaction_entities.csv", dtype=str).fillna("")
    pairs = pd.read_csv(cache_dir / "marts_pair_folds.csv", dtype=str).fillna("")
    pairs["protein_fold"] = pd.to_numeric(pairs["protein_fold"]).astype(int)
    pairs["reaction_fold"] = pd.to_numeric(pairs["reaction_fold"]).astype(int)
    pairs["protein_seen"] = pairs["protein_seen"].astype(str).str.lower().eq("true")
    pairs["reaction_seen"] = pairs["reaction_seen"].astype(str).str.lower().eq("true")
    canonical_ids = protein_table["protein_id"].astype(str).tolist()
    reaction_ids = reaction_table["reaction_id"].astype(str).tolist()
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}

    uniprot_features, uniprot_ids = load_protein_library(args.uniprot_protein_dir.resolve())
    metadata = load_uniprot_metadata(args.uniprot_metadata.resolve(), uniprot_ids)
    all_features = np.concatenate([canonical_features, uniprot_features], axis=0)
    all_ids = canonical_ids + uniprot_ids
    canonical_count = len(canonical_ids)
    ab_local = np.flatnonzero(
        metadata["evidence_quality_tier"].isin(TIER_GROUPS["ab"]).to_numpy()
    )
    universe_indices = {
        "canonical": np.arange(canonical_count, dtype=np.int64),
        "expanded_ab": np.concatenate(
            [np.arange(canonical_count, dtype=np.int64), canonical_count + ab_local]
        ),
        "expanded_abcd": np.arange(len(all_ids), dtype=np.int64),
    }
    records: list[dict[str, object]] = []

    for protein_fold in range(5):
        for reaction_fold in range(5):
            split_id = f"p{protein_fold}_r{reaction_fold}"
            train_pairs = pairs[
                (pairs["protein_fold"] != protein_fold)
                & (pairs["reaction_fold"] != reaction_fold)
            ]
            test = pairs[
                (pairs["protein_fold"] == protein_fold)
                & (pairs["reaction_fold"] == reaction_fold)
                & (~pairs["protein_seen"])
                & (~pairs["reaction_seen"])
            ]
            if test.empty:
                continue
            train_reaction_rows = np.asarray(
                sorted(
                    {
                        reaction_to_row[value]
                        for value in train_pairs["rhea_id"].astype(str)
                        if value in reaction_to_row
                    }
                ),
                dtype=np.int64,
            )
            score_cache: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
            for budget, model_dir in [
                (3, args.short_model_dir.resolve()),
                (10, args.short_model_dir.resolve()),
                (20, args.long_model_dir.resolve()),
            ]:
                key = str(model_dir)
                if key not in score_cache:
                    models = load_fold_models(model_dir, split_id, device)
                    score_matrix = ensemble_scores(
                        models, all_features, reaction_features, device
                    )
                    parameters = normalization_parameters(
                        score_matrix, train_reaction_rows, args.local_k
                    )
                    score_cache[key] = (score_matrix, parameters)
                score_matrix, parameters = score_cache[key]
                for reaction_id, group in test.groupby("rhea_id", sort=True):
                    positives = set(group["Entry"].astype(str))
                    reaction_row = reaction_to_row[reaction_id]
                    method_scores = normalized_scores(score_matrix[reaction_row], parameters)
                    for normalization, full_scores in method_scores.items():
                        for universe, indices in universe_indices.items():
                            candidate_ids = [all_ids[int(index)] for index in indices]
                            scores = full_scores[indices]
                            metrics = rank_metrics(
                                scores,
                                candidate_ids,
                                positives,
                                set(),
                                (budget,),
                            )
                            order = np.lexsort((np.asarray(candidate_ids), -scores))
                            is_added = indices >= canonical_count
                            best_rank = int(metrics["best_positive_rank"])
                            records.append(
                                {
                                    "split_id": split_id,
                                    "query_id": reaction_id,
                                    "budget": budget,
                                    "candidate_universe": universe,
                                    "score_normalization": normalization,
                                    "candidate_count": len(indices),
                                    "n_positives": len(positives),
                                    "hit": int(metrics[f"hit_at_{budget}"]),
                                    "positive_recall": float(
                                        metrics[f"positive_recall_at_{budget}"]
                                    ),
                                    "best_positive_rank": best_rank,
                                    "reciprocal_rank": float(metrics["reciprocal_rank"]),
                                    "top1_is_added": bool(is_added[order[0]]),
                                    "added_above_best_positive": int(
                                        np.sum(is_added[order[: max(best_rank - 1, 0)]])
                                    ),
                                    "train_reactions_for_normalization": len(train_reaction_rows),
                                }
                            )

    query_metrics = pd.DataFrame(records)
    metrics = aggregate(query_metrics)
    comparison = compare_to_canonical_raw(query_metrics)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    comparison.to_csv(output_dir / "comparison_to_canonical_raw.csv", index=False)
    best = (
        comparison.sort_values(
            ["budget", "hit_retention_fraction", "mrr_ratio_to_baseline"],
            ascending=[True, False, False],
        )
        .groupby(["budget", "candidate_universe"], as_index=False)
        .head(1)
    )
    best.to_csv(output_dir / "best_normalization_by_universe.csv", index=False)
    summary = {
        "strict_external_double_cold": True,
        "candidate_universes": {
            "canonical": canonical_count,
            "expanded_ab": len(universe_indices["expanded_ab"]),
            "expanded_abcd": len(universe_indices["expanded_abcd"]),
        },
        "normalizations": [
            "raw",
            "candidate_mean_centered",
            "candidate_zscore",
            "candidate_local_density",
        ],
        "local_k": args.local_k,
        "best_by_universe": best.to_dict("records"),
        "outputs": {
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "metrics": str(output_dir / "metrics.csv"),
            "comparison": str(output_dir / "comparison_to_canonical_raw.csv"),
            "best": str(output_dir / "best_normalization_by_universe.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
