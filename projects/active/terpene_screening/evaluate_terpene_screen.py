from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projects.active.terpene_screening.common import (
    TERPENE_DATA_DIR,
    TERPENE_RESULTS_DIR,
    coerce_text,
    identify_terpene_columns,
    read_table,
    safe_json_dump,
    write_table,
)


PREDICTIONS_CSV = TERPENE_RESULTS_DIR / "predictions" / "all_pair_scores.csv"
PAIRS_CSV = TERPENE_DATA_DIR / "terpene_candidate_pairs.csv"
REACTION_MANIFEST_CSV = TERPENE_RESULTS_DIR / "all_rhea_gate_reaction_manifest.csv"
METRICS_JSON = TERPENE_RESULTS_DIR / "metrics" / "topk_metrics.json"
REACTION_RESULTS_CSV = TERPENE_RESULTS_DIR / "reaction_level_results.csv"


def _load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "cage_score" not in df.columns and "pred" in df.columns:
        df = df.rename(columns={"pred": "cage_score"})
    if "reaction_id" not in df.columns:
        raise ValueError(f"Predictions file missing `reaction_id`: {path}")
    return df


def _load_pairs(path: Path) -> pd.DataFrame:
    df = read_table(path)
    cols = identify_terpene_columns(df)
    if "reaction_id" not in df.columns:
        raise ValueError(f"Pair file missing `reaction_id`: {path}")
    if "label" not in df.columns and "Label" in df.columns:
        df["label"] = df["Label"]
    if "enzyme_id" not in df.columns and cols["enzyme_id"]["column"] in df.columns:
        df["enzyme_id"] = df[cols["enzyme_id"]["column"]]
    if "uniprot_id" not in df.columns and cols["uniprot_id"]["column"] in df.columns:
        df["uniprot_id"] = df[cols["uniprot_id"]["column"]]
    return df


def _load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "reaction_id" not in df.columns or "status" not in df.columns:
        raise ValueError(f"Reaction manifest missing required columns: {path}")
    if "n_true_enzymes" not in df.columns and "n_positive_enzymes" in df.columns:
        df["n_true_enzymes"] = df["n_positive_enzymes"]
    if "n_gate_candidates" not in df.columns:
        df["n_gate_candidates"] = 0
    if "n_gate_positive_candidates" not in df.columns:
        df["n_gate_positive_candidates"] = 0
    if "gate_hit" not in df.columns:
        df["gate_hit"] = False
    return df


def _compute_reaction_rows(pairs_df: pd.DataFrame, preds_df: pd.DataFrame) -> list[dict[str, Any]]:
    reaction_rows: list[dict[str, Any]] = []
    pred_grouped = {reaction_id: group.copy() for reaction_id, group in preds_df.groupby("reaction_id")}

    for reaction_id, group in pairs_df.groupby("reaction_id", sort=False):
        group = group.copy()
        scored_group = pred_grouped.get(reaction_id, pd.DataFrame()).copy()
        if not scored_group.empty:
            scored_group = scored_group.sort_values(
                by=["cage_score", "uniprot_id", "enzyme_id"],
                ascending=[False, True, True],
                kind="mergesort",
            ).reset_index(drop=True)
            scored_group["rank_within_reaction"] = range(1, len(scored_group) + 1)

        positives = group[group["label"].astype(int) == 1].copy()
        n_candidates = int(len(group))
        n_scored_candidates = int(len(scored_group))
        n_positive_enzymes = int(len(positives))
        n_scored_positive_enzymes = int(
            len(scored_group[scored_group["label"].astype(int) == 1]) if not scored_group.empty else 0
        )
        status = "ok" if n_positive_enzymes > 0 else "no_positive_label"

        best_positive_row: dict[str, Any] | None = None
        if n_scored_positive_enzymes > 0:
            positive_scored = scored_group[scored_group["label"].astype(int) == 1].sort_values(
                by=["cage_score", "uniprot_id", "enzyme_id"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            best_positive_row = positive_scored.iloc[0].to_dict()

        if best_positive_row is not None:
            best_positive_rank = int(best_positive_row["rank_within_reaction"])
            best_positive_enzyme_id = coerce_text(best_positive_row.get("enzyme_id"))
            best_positive_score = float(best_positive_row.get("cage_score"))
            reciprocal_rank = 1.0 / best_positive_rank
        else:
            best_positive_rank = None
            best_positive_enzyme_id = ""
            best_positive_score = None
            reciprocal_rank = 0.0 if status == "ok" else None

        top10 = scored_group.head(10)
        top10_enzyme_ids = top10["enzyme_id"].astype(str).tolist() if not top10.empty else []
        top10_scores = [float(value) for value in top10["cage_score"].tolist()] if not top10.empty else []

        top1_hit = bool(best_positive_rank is not None and best_positive_rank <= 1)
        top5_hit = bool(best_positive_rank is not None and best_positive_rank <= 5)
        top10_hit = bool(best_positive_rank is not None and best_positive_rank <= 10)

        reaction_rows.append(
            {
                "reaction_id": reaction_id,
                "rhea_id": coerce_text(group["rhea_id"].iloc[0]) if "rhea_id" in group.columns else "",
                "status": status,
                "n_candidates": n_candidates,
                "n_scored_candidates": n_scored_candidates,
                "n_positive_enzymes": n_positive_enzymes,
                "n_scored_positive_enzymes": n_scored_positive_enzymes,
                "best_positive_rank": best_positive_rank,
                "best_positive_enzyme_id": best_positive_enzyme_id,
                "best_positive_score": best_positive_score,
                "top1_hit": top1_hit,
                "top5_hit": top5_hit,
                "top10_hit": top10_hit,
                "top10_enzyme_ids": json.dumps(top10_enzyme_ids, ensure_ascii=False),
                "top10_scores": json.dumps(top10_scores, ensure_ascii=False),
                "reciprocal_rank": reciprocal_rank,
            }
        )
    return reaction_rows


def _compute_reaction_rows_from_manifest(
    manifest_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    preds_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    reaction_rows: list[dict[str, Any]] = []
    pair_grouped = {reaction_id: group.copy() for reaction_id, group in pairs_df.groupby("reaction_id")}
    pred_grouped = {reaction_id: group.copy() for reaction_id, group in preds_df.groupby("reaction_id")}

    for _, manifest_row in manifest_df.iterrows():
        reaction_id = coerce_text(manifest_row.get("reaction_id"))
        status = coerce_text(manifest_row.get("status")) or "ok"
        rhea_id = coerce_text(manifest_row.get("rhea_id"))
        n_positive_enzymes = int(manifest_row.get("n_true_enzymes") or 0)
        n_candidates = int(manifest_row.get("n_gate_candidates") or 0)
        gate_hit = bool(manifest_row.get("gate_hit")) if status == "ok" else False

        group = pair_grouped.get(reaction_id, pd.DataFrame()).copy()
        if not group.empty:
            n_candidates = int(len(group))

        scored_group = pred_grouped.get(reaction_id, pd.DataFrame()).copy()
        if not scored_group.empty:
            scored_group = scored_group.sort_values(
                by=["cage_score", "uniprot_id", "enzyme_id"],
                ascending=[False, True, True],
                kind="mergesort",
            ).reset_index(drop=True)
            scored_group["rank_within_reaction"] = range(1, len(scored_group) + 1)

        n_scored_candidates = int(len(scored_group))
        n_scored_positive_enzymes = int(
            len(scored_group[scored_group["label"].astype(int) == 1]) if not scored_group.empty else 0
        )

        best_positive_row: dict[str, Any] | None = None
        if n_scored_positive_enzymes > 0:
            positive_scored = scored_group[scored_group["label"].astype(int) == 1].sort_values(
                by=["cage_score", "uniprot_id", "enzyme_id"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            best_positive_row = positive_scored.iloc[0].to_dict()

        if best_positive_row is not None:
            best_positive_rank = int(best_positive_row["rank_within_reaction"])
            best_positive_enzyme_id = coerce_text(best_positive_row.get("enzyme_id"))
            best_positive_score = float(best_positive_row.get("cage_score"))
            reciprocal_rank = 1.0 / best_positive_rank
        else:
            best_positive_rank = None
            best_positive_enzyme_id = ""
            best_positive_score = None
            reciprocal_rank = 0.0 if status == "ok" else None

        top10 = scored_group.head(10)
        top10_enzyme_ids = top10["enzyme_id"].astype(str).tolist() if not top10.empty else []
        top10_scores = [float(value) for value in top10["cage_score"].tolist()] if not top10.empty else []

        reaction_rows.append(
            {
                "reaction_id": reaction_id,
                "rhea_id": rhea_id,
                "status": status,
                "gate_hit": gate_hit,
                "n_candidates": n_candidates,
                "n_scored_candidates": n_scored_candidates,
                "n_positive_enzymes": n_positive_enzymes,
                "n_scored_positive_enzymes": n_scored_positive_enzymes,
                "best_positive_rank": best_positive_rank,
                "best_positive_enzyme_id": best_positive_enzyme_id,
                "best_positive_score": best_positive_score,
                "top1_hit": bool(best_positive_rank is not None and best_positive_rank <= 1),
                "top3_hit": bool(best_positive_rank is not None and best_positive_rank <= 3),
                "top5_hit": bool(best_positive_rank is not None and best_positive_rank <= 5),
                "top10_hit": bool(best_positive_rank is not None and best_positive_rank <= 10),
                "top10_enzyme_ids": json.dumps(top10_enzyme_ids, ensure_ascii=False),
                "top10_scores": json.dumps(top10_scores, ensure_ascii=False),
                "reciprocal_rank": reciprocal_rank,
            }
        )

    return reaction_rows


def evaluate(
    pairs_csv: Path = PAIRS_CSV,
    predictions_csv: Path = PREDICTIONS_CSV,
    reaction_manifest_csv: Path | None = None,
    metrics_json: Path = METRICS_JSON,
    reaction_results_csv: Path = REACTION_RESULTS_CSV,
) -> dict[str, Any]:
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")
    if not pairs_csv.exists():
        raise FileNotFoundError(f"Candidate pairs CSV not found: {pairs_csv}")

    preds_df = _load_predictions(predictions_csv)
    pairs_df = _load_pairs(pairs_csv)
    manifest_df = _load_manifest(reaction_manifest_csv) if reaction_manifest_csv and reaction_manifest_csv.exists() else None

    if manifest_df is not None:
        reaction_rows = _compute_reaction_rows_from_manifest(manifest_df, pairs_df, preds_df)
    else:
        reaction_rows = _compute_reaction_rows(pairs_df, preds_df)
    reaction_results_df = pd.DataFrame(reaction_rows)
    reaction_results_df = reaction_results_df.sort_values("reaction_id", kind="mergesort").reset_index(drop=True)
    write_table(reaction_results_df, reaction_results_csv, sep=",")

    if "status" in reaction_results_df.columns:
        eligible = reaction_results_df[reaction_results_df["status"] == "ok"].copy()
    else:
        eligible = reaction_results_df.copy()
    if eligible.empty:
        eligible = reaction_results_df.copy()
    positives = eligible[eligible["n_positive_enzymes"] > 0].copy()
    if positives.empty:
        positives = eligible.copy()

    if "gate_hit" in reaction_results_df.columns:
        gate_hit_series = reaction_results_df["gate_hit"].astype(bool)
    else:
        gate_hit_series = reaction_results_df["n_scored_positive_enzymes"] > 0
    if "status" in reaction_results_df.columns:
        ok_series = reaction_results_df["status"] == "ok"
        no_smiles_series = reaction_results_df["status"] == "no_smiles"
    else:
        ok_series = pd.Series([True] * len(reaction_results_df), index=reaction_results_df.index)
        no_smiles_series = pd.Series([False] * len(reaction_results_df), index=reaction_results_df.index)

    metrics = {
        "n_query_reactions": int(len(reaction_results_df)),
        "n_reactions_with_positive_label": int((reaction_results_df["n_positive_enzymes"] > 0).sum()),
        "n_reactions_with_smiles": int(ok_series.sum()),
        "n_reactions_without_smiles": int(no_smiles_series.sum()),
        "n_reactions_with_gate_hit": int((ok_series & gate_hit_series).sum()),
        "n_reactions_without_gate_hit": int((ok_series & ~gate_hit_series).sum()),
        "n_reactions_with_scored_positive_label": int((reaction_results_df["n_scored_positive_enzymes"] > 0).sum()),
        "n_reactions_missing_scored_positive_label": int(
            ((reaction_results_df["n_positive_enzymes"] > 0) & (reaction_results_df["n_scored_positive_enzymes"] == 0)).sum()
        ),
        "n_candidate_enzymes": int(pairs_df["uniprot_id"].astype(str).nunique()) if "uniprot_id" in pairs_df.columns else 0,
        "n_pairs_scored": int(len(preds_df)),
        "top3_recall": float((positives["top3_hit"].astype(bool)).mean()) if len(positives) else 0.0,
        "top1_recall": float((positives["top1_hit"].astype(bool)).mean()) if len(positives) else 0.0,
        "top5_recall": float((positives["top5_hit"].astype(bool)).mean()) if len(positives) else 0.0,
        "top10_recall": float((positives["top10_hit"].astype(bool)).mean()) if len(positives) else 0.0,
        "mean_reciprocal_rank": float(positives["reciprocal_rank"].astype(float).mean()) if len(positives) else 0.0,
        "median_best_positive_rank": float(positives["best_positive_rank"].dropna().astype(float).median())
        if positives["best_positive_rank"].notna().any()
        else None,
    }
    safe_json_dump(metrics, metrics_json)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate terpene screening ranking metrics.")
    parser.add_argument("--pairs_csv", type=str, default=str(PAIRS_CSV))
    parser.add_argument("--predictions_csv", type=str, default=str(PREDICTIONS_CSV))
    parser.add_argument("--reaction_manifest_csv", type=str, default=str(REACTION_MANIFEST_CSV))
    parser.add_argument("--metrics_json", type=str, default=str(METRICS_JSON))
    parser.add_argument("--reaction_results_csv", type=str, default=str(REACTION_RESULTS_CSV))
    args = parser.parse_args()
    reaction_manifest_csv = Path(args.reaction_manifest_csv) if args.reaction_manifest_csv else None
    metrics = evaluate(
        pairs_csv=Path(args.pairs_csv),
        predictions_csv=Path(args.predictions_csv),
        reaction_manifest_csv=reaction_manifest_csv,
        metrics_json=Path(args.metrics_json),
        reaction_results_csv=Path(args.reaction_results_csv),
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
