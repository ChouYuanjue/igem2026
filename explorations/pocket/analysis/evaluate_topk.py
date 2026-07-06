from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _label_path_is_missing(label_csv: str) -> bool:
    return label_csv.strip().lower() in {"", "null", "none"}


def evaluate_topk(
    predictions: pd.DataFrame,
    topk_values: list[int],
    group_col: str,
    score_col: str,
    label_col: str,
) -> dict[str, object]:
    if group_col not in predictions.columns:
        raise ValueError(f"Missing group column: {group_col}")
    if score_col not in predictions.columns:
        raise ValueError(f"Missing score column: {score_col}")
    if label_col not in predictions.columns:
        raise ValueError(f"Missing label column: {label_col}")

    predictions = predictions.copy()
    predictions[score_col] = pd.to_numeric(predictions[score_col], errors="coerce")
    predictions[label_col] = pd.to_numeric(predictions[label_col], errors="coerce").fillna(0)
    predictions = predictions.dropna(subset=[group_col, score_col])
    predictions["has_positive"] = predictions[label_col] == 1

    metrics: dict[str, object] = {
        "n_groups": int(predictions[group_col].nunique()),
        "n_pairs": int(len(predictions)),
        "n_positive_pairs": int((predictions[label_col] == 1).sum()),
    }

    for k in topk_values:
        successes = []
        valid_groups = 0
        groups_without_positive = 0
        for _, group in predictions.groupby(group_col):
            if not (group[label_col] == 1).any():
                groups_without_positive += 1
                continue
            valid_groups += 1
            ranked = group.sort_values(score_col, ascending=False).head(k)
            successes.append(int((ranked[label_col] == 1).any()))
        metric_name = f"top{k}_success_rate"
        metrics[metric_name] = float(sum(successes) / len(successes)) if successes else 0.0
        metrics["n_valid_reactions"] = int(valid_groups)
        metrics["reactions_without_positive"] = int(groups_without_positive)

    return metrics


def _merge_labels(
    prediction_df: pd.DataFrame,
    label_df: pd.DataFrame,
    group_col: str,
    enzyme_col: str,
    label_col: str,
) -> pd.DataFrame:
    if label_col in prediction_df.columns:
        return prediction_df

    merge_cols = [group_col, enzyme_col]
    missing_prediction = [column for column in merge_cols if column not in prediction_df.columns]
    missing_label = [column for column in merge_cols + [label_col] if column not in label_df.columns]
    if missing_prediction:
        raise ValueError(f"Prediction CSV missing columns for label merge: {missing_prediction}")
    if missing_label:
        raise ValueError(f"Label CSV missing columns for label merge: {missing_label}")

    return prediction_df.merge(
        label_df[merge_cols + [label_col]],
        on=merge_cols,
        how="left",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate top-k retrieval success.")
    parser.add_argument("--prediction_csv", required=True)
    parser.add_argument("--label_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--group_col", default="reaction_id")
    parser.add_argument("--enzyme_col", default="enzyme_id")
    parser.add_argument("--score_col", default="aggregated_score")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--topk", nargs="+", type=int, default=[5, 10])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prediction_csv = Path(args.prediction_csv)
    label_csv = Path(args.label_csv) if not _label_path_is_missing(args.label_csv) else None
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if label_csv is None or not label_csv.exists():
        reason = "label_csv is null or does not exist"
        print(f"[warning] Skipping top-k evaluation: {reason}")
        result = {
            "status": "skipped",
            "reason": reason,
            "prediction_csv": str(prediction_csv),
            "label_csv": args.label_csv,
        }
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return

    predictions = pd.read_csv(prediction_csv)
    labels = pd.read_csv(label_csv)
    merged = _merge_labels(
        prediction_df=predictions,
        label_df=labels,
        group_col=args.group_col,
        enzyme_col=args.enzyme_col,
        label_col=args.label_col,
    )
    result = evaluate_topk(
        predictions=merged,
        topk_values=args.topk,
        group_col=args.group_col,
        score_col=args.score_col,
        label_col=args.label_col,
    )
    result["status"] = "ok"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[done] Wrote top-k metrics to {output_json}")


if __name__ == "__main__":
    main()
