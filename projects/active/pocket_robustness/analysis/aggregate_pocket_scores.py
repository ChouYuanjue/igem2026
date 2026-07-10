from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ENZYME_ID_COLUMNS = ["enzyme_id", "UniprotID", "uniprot_id", "protein_id"]
SCORE_COLUMNS = ["cage_score", "score", "pred", "prediction", "catalytic_score", "y_pred"]
SUPPORTED_METHODS = [
    "max",
    "mean",
    "rank_weighted",
    "softmax_pool",
    "source_weighted",
    "source_balanced_mean",
    "source_balanced_rank_weighted",
    "source_balanced_softmax_pool",
    "catalytic_residue_weighted",
    "residue_prior_weighted",
    "none_or_max",
]


def _normalize_column(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {_normalize_column(column): column for column in columns}
    for candidate in candidates:
        key = _normalize_column(candidate)
        if key in normalized:
            return normalized[key]
    return None


def _standardize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    enzyme_col = _find_column(list(result.columns), ENZYME_ID_COLUMNS)
    if enzyme_col is None:
        raise ValueError(f"Could not find enzyme id column. Accepted: {ENZYME_ID_COLUMNS}")
    if enzyme_col != "enzyme_id":
        result = result.rename(columns={enzyme_col: "enzyme_id"})

    score_col = _find_column(list(result.columns), SCORE_COLUMNS)
    if score_col is None:
        raise ValueError(f"Could not find score column. Accepted: {SCORE_COLUMNS}")
    if score_col != "cage_score":
        result = result.rename(columns={score_col: "cage_score"})

    result["cage_score"] = pd.to_numeric(result["cage_score"], errors="coerce")
    result = result.dropna(subset=["enzyme_id", "cage_score"]).copy()
    result["enzyme_id"] = result["enzyme_id"].astype(str)
    return result


def _attach_manifest_metadata(predictions: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    if manifest.empty:
        return result

    manifest = manifest.copy()
    enzyme_col = _find_column(list(manifest.columns), ENZYME_ID_COLUMNS)
    if enzyme_col and enzyme_col != "enzyme_id":
        manifest = manifest.rename(columns={enzyme_col: "enzyme_id"})

    merge_cols = []
    if "pocket_global_id" in result.columns and "pocket_global_id" in manifest.columns:
        merge_cols = ["pocket_global_id"]
    elif "pocket_pdb_path" in result.columns and "pocket_pdb_path" in manifest.columns:
        merge_cols = ["enzyme_id", "pocket_pdb_path"]

    if merge_cols:
        metadata_cols = [
            column
            for column in [
                "enzyme_id",
                "pocket_global_id",
                "pocket_source",
                "pocket_rank",
                "pocket_score",
                "pocket_pdb_path",
            ]
            if column in manifest.columns
        ]
        metadata = manifest[metadata_cols].drop_duplicates()
        result = result.merge(metadata, on=merge_cols, how="left", suffixes=("", "_manifest"))
        for column in ["pocket_source", "pocket_rank", "pocket_score", "pocket_global_id"]:
            manifest_col = f"{column}_manifest"
            if manifest_col in result.columns:
                if column not in result.columns:
                    result[column] = result[manifest_col]
                else:
                    result[column] = result[column].fillna(result[manifest_col])
                result = result.drop(columns=[manifest_col])
        return result

    if len(result) == len(manifest):
        print("[warning] Attaching manifest metadata by row order.")
        for column in ["pocket_global_id", "pocket_source", "pocket_rank", "pocket_score"]:
            if column in manifest.columns and column not in result.columns:
                result[column] = manifest[column].to_numpy()

    return result


def _ensure_pocket_metadata(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "pocket_rank" in result.columns:
        result["pocket_rank"] = pd.to_numeric(result["pocket_rank"], errors="coerce")
    else:
        result["pocket_rank"] = np.nan
    if "pocket_source" not in result.columns:
        result["pocket_source"] = "unknown"
    result["pocket_source"] = result["pocket_source"].fillna("unknown").astype(str)

    group_cols = ["enzyme_id"]
    if "reaction_id" in result.columns:
        group_cols = ["reaction_id", "enzyme_id"]

    missing_rank = result["pocket_rank"].isna()
    if missing_rank.any():
        fallback_rank = result.groupby(group_cols).cumcount() + 1
        result.loc[missing_rank, "pocket_rank"] = fallback_rank.loc[missing_rank]
    result["pocket_rank"] = result["pocket_rank"].astype(int)

    if "pocket_global_id" not in result.columns:
        result["pocket_global_id"] = [
            f"{enzyme_id}__{source}__rank{rank}"
            for enzyme_id, source, rank in zip(
                result["enzyme_id"],
                result["pocket_source"],
                result["pocket_rank"],
            )
        ]
    result["pocket_global_id"] = result["pocket_global_id"].fillna("").astype(str)
    empty_global_id = result["pocket_global_id"].str.len() == 0
    if empty_global_id.any():
        result.loc[empty_global_id, "pocket_global_id"] = [
            f"{enzyme_id}__{source}__rank{rank}"
            for enzyme_id, source, rank in zip(
                result.loc[empty_global_id, "enzyme_id"],
                result.loc[empty_global_id, "pocket_source"],
                result.loc[empty_global_id, "pocket_rank"],
            )
        ]
    return result


def _softmax_pool(scores: np.ndarray, temperature: float) -> float:
    if temperature <= 0:
        raise ValueError("temperature must be positive for softmax_pool")
    scaled = scores / temperature
    scaled = scaled - np.max(scaled)
    weights = np.exp(scaled)
    weights = weights / weights.sum()
    return float(np.sum(weights * scores))


def _weighted_average(scores: np.ndarray, weights: np.ndarray) -> float:
    if np.all(weights == 0):
        return float(np.mean(scores))
    return float(np.average(scores, weights=weights))


def _source_weighted(group: pd.DataFrame, source_weights: dict[str, float] | None) -> tuple[float, bool, str]:
    if not source_weights:
        return _rank_weighted(group), True, "source_weights missing; fallback to rank_weighted"
    weights = group["pocket_source"].map(source_weights).fillna(0.0).to_numpy(dtype=float)
    if np.all(weights == 0):
        return _rank_weighted(group), True, "no matching source weights; fallback to rank_weighted"
    return _weighted_average(group["cage_score"].to_numpy(dtype=float), weights), False, ""


def _source_level_aggregate(group: pd.DataFrame, inner_method: str, temperature: float) -> tuple[float, bool, str]:
    source_scores: list[float] = []
    for _, source_group in group.groupby("pocket_source", dropna=False):
        scores = source_group["cage_score"].to_numpy(dtype=float)
        if inner_method == "mean":
            source_score = float(np.mean(scores))
        elif inner_method == "rank_weighted":
            source_score = _rank_weighted(source_group)
        elif inner_method == "softmax_pool":
            source_score = _softmax_pool(scores, temperature)
        else:
            raise ValueError(f"Unsupported source-level inner method: {inner_method}")
        source_scores.append(source_score)

    if not source_scores:
        return _rank_weighted(group), True, "no pocket_source groups for source-balanced aggregation"
    return float(np.mean(source_scores)), False, ""


def _rank_weighted(group: pd.DataFrame) -> float:
    scores = group["cage_score"].to_numpy(dtype=float)
    ranks = group["pocket_rank"].to_numpy(dtype=float)
    return _weighted_average(scores, 1.0 / ranks)


def _prior_weighted(group: pd.DataFrame, prior_col: str, method_name: str) -> tuple[float, bool, str]:
    if prior_col not in group.columns:
        return _rank_weighted(group), True, f"{prior_col} missing for {method_name}; fallback to rank_weighted"
    priors = pd.to_numeric(group[prior_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if np.all(priors == 0):
        return _rank_weighted(group), True, f"{prior_col} empty for {method_name}; fallback to rank_weighted"
    ranks = group["pocket_rank"].to_numpy(dtype=float)
    weights = priors / ranks
    return _weighted_average(group["cage_score"].to_numpy(dtype=float), weights), False, ""


def aggregate_scores(
    df: pd.DataFrame,
    method: str,
    temperature: float = 0.2,
    source_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    if method == "none_or_max":
        method = "max"
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported aggregation method: {method}")

    standardized = _standardize_predictions(df)
    standardized = _ensure_pocket_metadata(standardized)
    group_cols = ["enzyme_id"]
    if "reaction_id" in standardized.columns:
        group_cols = ["reaction_id", "enzyme_id"]

    rows = []
    for group_key, group in standardized.groupby(group_cols, dropna=False):
        scores = group["cage_score"].to_numpy(dtype=float)
        used_fallback = False
        fallback_reason = ""

        if method == "max":
            aggregated_score = float(np.max(scores))
        elif method == "mean":
            aggregated_score = float(np.mean(scores))
        elif method == "rank_weighted":
            aggregated_score = _rank_weighted(group)
        elif method == "softmax_pool":
            aggregated_score = _softmax_pool(scores, temperature)
        elif method == "source_weighted":
            aggregated_score, used_fallback, fallback_reason = _source_weighted(group, source_weights)
        elif method == "source_balanced_mean":
            aggregated_score, used_fallback, fallback_reason = _source_level_aggregate(
                group,
                "mean",
                temperature,
            )
        elif method == "source_balanced_rank_weighted":
            aggregated_score, used_fallback, fallback_reason = _source_level_aggregate(
                group,
                "rank_weighted",
                temperature,
            )
        elif method == "source_balanced_softmax_pool":
            aggregated_score, used_fallback, fallback_reason = _source_level_aggregate(
                group,
                "softmax_pool",
                temperature,
            )
        elif method == "catalytic_residue_weighted":
            aggregated_score, used_fallback, fallback_reason = _prior_weighted(
                group,
                "catalytic_prior_score",
                method,
            )
        elif method == "residue_prior_weighted":
            aggregated_score, used_fallback, fallback_reason = _prior_weighted(
                group,
                "residue_prior_score",
                method,
            )
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

        best_idx = group["cage_score"].idxmax()
        best_row = group.loc[best_idx]

        if len(group_cols) == 1:
            enzyme_id = group_key[0] if isinstance(group_key, tuple) else group_key
            row: dict[str, Any] = {"enzyme_id": enzyme_id}
        else:
            reaction_id, enzyme_id = group_key
            row = {"reaction_id": reaction_id, "enzyme_id": enzyme_id}

        row.update(
            {
                "aggregated_score": aggregated_score,
                "aggregation_method": method,
                "n_pockets": int(len(group)),
                "best_pocket_global_id": str(best_row["pocket_global_id"]),
                "best_pocket_source": str(best_row["pocket_source"]),
                "best_pocket_rank": int(best_row["pocket_rank"]),
                "best_pocket_cage_score": float(best_row["cage_score"]),
                "used_fallback": bool(used_fallback),
                "fallback_reason": fallback_reason,
            }
        )
        rows.append(row)

    output_cols = group_cols + [
        "aggregated_score",
        "aggregation_method",
        "n_pockets",
        "best_pocket_global_id",
        "best_pocket_source",
        "best_pocket_rank",
        "best_pocket_cage_score",
        "used_fallback",
        "fallback_reason",
    ]
    return pd.DataFrame(rows, columns=output_cols)


def _parse_source_weights(raw: str | None) -> dict[str, float] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return {str(key): float(value) for key, value in parsed.items()}
    except json.JSONDecodeError:
        weights = {}
        for token in raw.split(","):
            if not token.strip():
                continue
            key, value = token.split("=", 1)
            weights[key.strip()] = float(value)
        return weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate pocket-level prediction scores into enzyme-level scores.")
    parser.add_argument("--prediction_csv", required=True)
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--source_weights")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prediction_csv = Path(args.prediction_csv)
    manifest_csv = Path(args.manifest_csv)
    output_csv = Path(args.output_csv)

    predictions = pd.read_csv(prediction_csv)
    manifest = pd.read_csv(manifest_csv) if manifest_csv.exists() else pd.DataFrame()
    if not manifest_csv.exists():
        print(f"[warning] Manifest CSV does not exist: {manifest_csv}")

    predictions = _attach_manifest_metadata(predictions, manifest)
    aggregated = aggregate_scores(
        predictions,
        method=args.method,
        temperature=args.temperature,
        source_weights=_parse_source_weights(args.source_weights),
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output_csv, index=False)
    print(f"[done] Wrote aggregated scores to {output_csv}")


if __name__ == "__main__":
    main()
