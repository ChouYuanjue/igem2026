from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _with_rank(
    df: pd.DataFrame,
    group_col: str,
    enzyme_col: str,
    score_col: str,
    score_output_col: str,
    rank_output_col: str,
) -> pd.DataFrame:
    missing = [column for column in [enzyme_col, score_col] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = df.copy()
    result[score_col] = pd.to_numeric(result[score_col], errors="coerce")
    result = result.dropna(subset=[enzyme_col, score_col])

    if group_col not in result.columns:
        print(f"[warning] Missing {group_col}; using global ranking.")
        result[group_col] = "global"

    result[rank_output_col] = (
        result.groupby(group_col)[score_col]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return result[[group_col, enzyme_col, score_col, rank_output_col]].rename(
        columns={score_col: score_output_col}
    )


def compare_rank_shift(
    baseline: pd.DataFrame,
    new: pd.DataFrame,
    group_col: str = "reaction_id",
    enzyme_col: str = "enzyme_id",
    score_col: str = "aggregated_score",
) -> pd.DataFrame:
    baseline_ranked = _with_rank(
        baseline,
        group_col=group_col,
        enzyme_col=enzyme_col,
        score_col=score_col,
        score_output_col="baseline_score",
        rank_output_col="baseline_rank",
    )
    new_ranked = _with_rank(
        new,
        group_col=group_col,
        enzyme_col=enzyme_col,
        score_col=score_col,
        score_output_col="new_score",
        rank_output_col="new_rank",
    )

    merged = baseline_ranked.merge(
        new_ranked,
        on=[group_col, enzyme_col],
        how="inner",
    )
    merged["rank_shift"] = merged["baseline_rank"] - merged["new_rank"]
    return merged[
        [
            group_col,
            enzyme_col,
            "baseline_score",
            "new_score",
            "baseline_rank",
            "new_rank",
            "rank_shift",
        ]
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare rank shifts between two baselines.")
    parser.add_argument("--baseline_csv", required=True)
    parser.add_argument("--new_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--group_col", default="reaction_id")
    parser.add_argument("--enzyme_col", default="enzyme_id")
    parser.add_argument("--score_col", default="aggregated_score")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    baseline = pd.read_csv(Path(args.baseline_csv))
    new = pd.read_csv(Path(args.new_csv))
    output = compare_rank_shift(
        baseline=baseline,
        new=new,
        group_col=args.group_col,
        enzyme_col=args.enzyme_col,
        score_col=args.score_col,
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    print(f"[done] Wrote rank shift CSV to {output_csv}")


if __name__ == "__main__":
    main()
