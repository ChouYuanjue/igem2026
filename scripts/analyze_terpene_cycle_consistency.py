from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.core.evidence import cycle_consistency_score
from projects.active.terpene_screening.rank_open_world import build_parser, execute_ranking


def _execute(argv: list[str]) -> pd.DataFrame:
    args = build_parser().parse_args(argv)
    return execute_ranking(args)


def _candidate_rank(frame: pd.DataFrame, candidate_id: str) -> int | None:
    matched = frame[frame["candidate_id"].astype(str).eq(str(candidate_id))]
    if matched.empty:
        return None
    return int(matched.iloc[0]["rank"])


def _forward_arguments(args: argparse.Namespace) -> tuple[list[str], str, str]:
    common = ["--top-k", str(args.top_k), "--device", args.device]
    if args.ranking_objective != "auto":
        common.extend(["--ranking-objective", args.ranking_objective])
    if args.direction == "reaction_to_enzyme":
        if bool(args.reaction_id) == bool(args.reaction_smiles):
            raise ValueError("Provide exactly one of --reaction-id or --reaction-smiles")
        query_id = args.reaction_id or args.query_id or "cycle_external_reaction"
        query_kind = "current" if args.reaction_id else "external"
        query = ["--reaction-id", args.reaction_id] if args.reaction_id else [
            "--query-id", query_id,
            "--reaction-smiles", args.reaction_smiles,
        ]
        return ["rank-enzymes", *query, *common], str(query_id), query_kind
    if bool(args.enzyme_id) == bool(args.enzyme_sequence):
        raise ValueError("Provide exactly one of --enzyme-id or --enzyme-sequence")
    query_id = args.enzyme_id or args.query_id or "cycle_external_enzyme"
    query_kind = "current" if args.enzyme_id else "external"
    query = ["--enzyme-id", args.enzyme_id] if args.enzyme_id else [
        "--query-id", query_id,
        "--enzyme-sequence", args.enzyme_sequence,
    ]
    return ["rank-reactions", *query, *common], str(query_id), query_kind


def _reverse_arguments(
    args: argparse.Namespace,
    candidate_id: str,
    target_query_id: str,
    query_kind: str,
    temporary_dir: Path,
) -> list[str]:
    common = ["--top-k", str(args.reverse_top_k), "--device", args.device]
    if args.direction == "reaction_to_enzyme":
        argv = ["rank-reactions", "--enzyme-id", candidate_id, *common]
        if query_kind == "external":
            extension = temporary_dir / "cycle_external_reaction.csv"
            pd.DataFrame(
                [{"reaction_id": target_query_id, "reaction_smiles": args.reaction_smiles}]
            ).to_csv(extension, index=False)
            argv.extend(["--external-reactions-csv", str(extension)])
        return argv
    argv = ["rank-enzymes", "--reaction-id", candidate_id, *common]
    if query_kind == "external":
        extension = temporary_dir / "cycle_external_enzyme.csv"
        pd.DataFrame(
            [{"enzyme_id": target_query_id, "sequence": args.enzyme_sequence}]
        ).to_csv(extension, index=False)
        argv.extend(["--external-enzymes-csv", str(extension)])
    return argv


def _cycle_rerank(frame: pd.DataFrame, weight: float, constant: float) -> pd.DataFrame:
    if not 0.0 <= weight <= 0.25:
        raise ValueError("--cycle-rerank-weight must be within [0, 0.25]")
    result = frame.copy()
    reverse_rank = result["cycle_reverse_rank"].fillna(result["cycle_reverse_search_k"] + 1).astype(float)
    result["cycle_rerank_score"] = (
        (1.0 - weight) / (constant + result["rank"].astype(float))
        + weight / (constant + reverse_rank)
    )
    order = np.lexsort(
        (
            result["candidate_id"].astype(str).to_numpy(),
            -result["cycle_rerank_score"].to_numpy(),
        )
    )
    reranked = np.empty(len(result), dtype=np.int64)
    reranked[order] = np.arange(1, len(result) + 1, dtype=np.int64)
    result["cycle_reranked_rank"] = reranked
    result["cycle_rerank_weight"] = weight
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run optional bidirectional semantic-closure checks for a terpene ranking."
    )
    parser.add_argument(
        "--direction",
        choices=["reaction_to_enzyme", "enzyme_to_reaction"],
        required=True,
    )
    parser.add_argument("--reaction-id")
    parser.add_argument("--reaction-smiles")
    parser.add_argument("--enzyme-id")
    parser.add_argument("--enzyme-sequence")
    parser.add_argument("--query-id")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cycle-top-n", type=int, default=5)
    parser.add_argument("--reverse-top-k", type=int, default=50)
    parser.add_argument("--ranking-objective", choices=["auto", "top3", "top10", "top20"], default="auto")
    parser.add_argument("--cycle-rerank-weight", type=float, default=0.15)
    parser.add_argument("--rrf-constant", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/terpene_cycle_consistency/cycle_consistency.csv",
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()
    if args.top_k <= 0 or args.cycle_top_n <= 0 or args.reverse_top_k <= 0:
        raise ValueError("top-k values must be positive")

    forward_argv, target_query_id, query_kind = _forward_arguments(args)
    forward = _execute(forward_argv).copy()
    evaluated = min(args.cycle_top_n, len(forward))
    reverse_ranks: list[float] = [float("nan")] * len(forward)
    reverse_routes: list[str] = [""] * len(forward)
    reverse_sources: list[str] = [""] * len(forward)
    cycle_scores: list[float] = [float("nan")] * len(forward)

    with tempfile.TemporaryDirectory(prefix="terpene_cycle_") as temp:
        temp_path = Path(temp)
        for local_index in range(evaluated):
            row = forward.iloc[local_index]
            candidate_id = str(row["candidate_id"])
            reverse = _execute(
                _reverse_arguments(args, candidate_id, target_query_id, query_kind, temp_path)
            )
            reverse_rank = _candidate_rank(reverse, target_query_id)
            reverse_ranks[local_index] = float(reverse_rank) if reverse_rank is not None else float("nan")
            reverse_routes[local_index] = str(reverse.iloc[0].get("route_id", ""))
            reverse_sources[local_index] = str(reverse.iloc[0].get("score_source", ""))
            cycle_scores[local_index] = cycle_consistency_score(
                int(row["rank"]), reverse_rank, reciprocal_rank_constant=args.rrf_constant
            )

    forward["cycle_evaluated"] = [index < evaluated for index in range(len(forward))]
    forward["cycle_target_query_id"] = target_query_id
    forward["cycle_reverse_rank"] = reverse_ranks
    forward["cycle_recovered"] = [
        bool(index < evaluated and np.isfinite(reverse_ranks[index])) for index in range(len(forward))
    ]
    forward["cycle_consistency_score"] = cycle_scores
    forward["cycle_reverse_route_id"] = reverse_routes
    forward["cycle_reverse_score_source"] = reverse_sources
    forward["cycle_reverse_search_k"] = args.reverse_top_k
    forward["cycle_consistency_interpretation"] = "bidirectional_rank_closure_not_activity_probability"
    forward = _cycle_rerank(forward, args.cycle_rerank_weight, args.rrf_constant)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    forward.to_csv(args.output, index=False)
    evaluated_frame = forward[forward["cycle_evaluated"]]
    recovered = int(evaluated_frame["cycle_recovered"].sum())
    summary = {
        "status": "completed",
        "direction": args.direction,
        "query_id": target_query_id,
        "query_kind": query_kind,
        "forward_route_id": str(forward.iloc[0].get("route_id", "")),
        "forward_score_source": str(forward.iloc[0].get("score_source", "")),
        "query_applicability_score": float(forward.iloc[0]["query_applicability_score"]),
        "query_applicability_tier": str(forward.iloc[0]["query_applicability_tier"]),
        "evaluated_candidates": evaluated,
        "recovered_candidates": recovered,
        "recovery_fraction": recovered / evaluated if evaluated else 0.0,
        "mean_cycle_consistency": float(
            evaluated_frame["cycle_consistency_score"].dropna().mean()
        ) if evaluated_frame["cycle_consistency_score"].notna().any() else None,
        "cycle_rerank_weight": args.cycle_rerank_weight,
        "production_ranking_modified": False,
        "output": str(args.output.resolve()),
    }
    summary_output = args.summary_output or args.output.with_suffix(".summary.json")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
