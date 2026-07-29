from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRODUCTION = ROOT / "results/terpene_neighbor_route_rankings_v1/freeze_k5/rankings.csv"
DEFAULT_KERNEL_DEVELOPMENT = ROOT / "results/terpene_marts_dual_kernel_development_v1/rankings.csv"
DEFAULT_KERNEL_FROZEN = ROOT / "results/terpene_marts_dual_kernel_frozen_v1/rankings.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_marts_dual_kernel_rescue_route"
DIRECTION = "enzyme_to_reaction"
PRODUCTION_METHOD = "rank_hybrid_direct_0.75"
KERNEL_CONFIG = "rk50_pk5_t0.03_d1"
BUDGET = 20


def parse_partition(split_id: str) -> str:
    protein_fold, reaction_fold = split_id.removeprefix("p").split("_r")
    return "development" if int(protein_fold) == 4 or int(reaction_fold) == 4 else "frozen"


def ordered_lists(frame: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    return {
        key: group.sort_values(["rank", "candidate_id"]).candidate_id.astype(str).tolist()
        for key, group in frame.groupby(["split_id", "query_id"], sort=True)
    }


def positive_sets(frame: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    return {
        key: set(group.loc[group.is_positive.astype(int).eq(1), "candidate_id"].astype(str))
        for key, group in frame.groupby(["split_id", "query_id"], sort=True)
    }


def rescue(base: list[str], auxiliary: list[str], slots: int) -> list[str]:
    result = list(base[: BUDGET - slots])
    for candidate in auxiliary:
        if candidate not in result:
            result.append(candidate)
        if len(result) >= BUDGET:
            break
    return result[:BUDGET]


def rrf(base: list[str], auxiliary: list[str], weight: float, constant: float = 60.0) -> list[str]:
    universe = list(dict.fromkeys(base + auxiliary))
    base_rank = {value: index + 1 for index, value in enumerate(base)}
    auxiliary_rank = {value: index + 1 for index, value in enumerate(auxiliary)}
    return sorted(
        universe,
        key=lambda value: (
            -(
                (1.0 - weight) / (constant + base_rank.get(value, 10001))
                + weight / (constant + auxiliary_rank.get(value, 10001))
            ),
            value,
        ),
    )[:BUDGET]


def ranking_metrics(ranking: list[str], positives: set[str]) -> tuple[int, float, float]:
    position = next((index + 1 for index, value in enumerate(ranking) if value in positives), None)
    return (
        int(position is not None and position <= BUDGET),
        0.0 if position is None else 1.0 / position,
        np.nan if position is None else float(position),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked E2R Top-20 dual-kernel rescue route.")
    parser.add_argument("--production-rankings", type=Path, default=DEFAULT_PRODUCTION)
    parser.add_argument("--kernel-development-rankings", type=Path, default=DEFAULT_KERNEL_DEVELOPMENT)
    parser.add_argument("--kernel-frozen-rankings", type=Path, default=DEFAULT_KERNEL_FROZEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    production = pd.read_csv(args.production_rankings, dtype=str).fillna("")
    production = production[
        production.direction.eq(DIRECTION) & production.method.eq(PRODUCTION_METHOD)
    ].copy()
    production["rank"] = production["rank"].astype(int)
    production["is_positive"] = production["is_positive"].astype(int)
    kernel_development = pd.read_csv(args.kernel_development_rankings, dtype=str).fillna("")
    kernel_frozen = pd.read_csv(args.kernel_frozen_rankings, dtype=str).fillna("")
    kernel = pd.concat([kernel_development, kernel_frozen], ignore_index=True)
    kernel = kernel[
        kernel.direction.eq(DIRECTION) & kernel.config.eq(KERNEL_CONFIG)
    ].copy()
    kernel["rank"] = kernel["rank"].astype(int)
    kernel["is_positive"] = kernel["is_positive"].astype(int)

    production_lists = ordered_lists(production)
    kernel_lists = ordered_lists(kernel)
    positives = positive_sets(production)
    if set(production_lists) != set(kernel_lists):
        raise ValueError("Production and kernel query sets differ")

    configs: list[tuple[str, str, float]] = [("production", "production", 0.0), ("kernel", "kernel", 0.0)]
    configs.extend((f"rescue_{slots}", "rescue", float(slots)) for slots in range(1, 6))
    configs.extend((f"rrf_{weight:g}", "rrf", weight) for weight in (0.05, 0.1, 0.2, 0.3))
    query_rows: list[dict[str, object]] = []
    for key in sorted(production_lists):
        partition = parse_partition(key[0])
        for name, method, parameter in configs:
            if method == "production":
                ranking = production_lists[key][:BUDGET]
            elif method == "kernel":
                ranking = kernel_lists[key][:BUDGET]
            elif method == "rescue":
                ranking = rescue(production_lists[key], kernel_lists[key], int(parameter))
            else:
                ranking = rrf(production_lists[key], kernel_lists[key], parameter)
            hit, reciprocal_rank, best_rank = ranking_metrics(ranking, positives[key])
            query_rows.append(
                {
                    "split_id": key[0],
                    "query_id": key[1],
                    "partition": partition,
                    "config": name,
                    "method": method,
                    "parameter": parameter,
                    "hit_at_20": hit,
                    "reciprocal_rank": reciprocal_rank,
                    "best_positive_rank": best_rank,
                }
            )
    query_frame = pd.DataFrame(query_rows)
    development = (
        query_frame[query_frame.partition.eq("development")]
        .groupby(["config", "method", "parameter"], as_index=False)
        .agg(
            hit_at_20=("hit_at_20", "mean"),
            mrr=("reciprocal_rank", "mean"),
            median_rank=("best_positive_rank", "median"),
        )
    )
    complexity = {
        "production": 0,
        "kernel": 1,
        "rescue": 2,
        "rrf": 3,
    }
    development["complexity"] = development.method.map(complexity)
    selected = development.sort_values(
        ["hit_at_20", "mrr", "complexity", "parameter", "config"],
        ascending=[False, False, True, True, True],
    ).iloc[0]
    selected_name = str(selected.config)

    selected_frozen = query_frame[
        query_frame.partition.eq("frozen") & query_frame.config.eq(selected_name)
    ]
    baseline_frozen = query_frame[
        query_frame.partition.eq("frozen") & query_frame.config.eq("production")
    ]
    keys = ["split_id", "query_id"]
    paired = selected_frozen[keys + ["hit_at_20", "reciprocal_rank"]].merge(
        baseline_frozen[keys + ["hit_at_20", "reciprocal_rank"]],
        on=keys,
        suffixes=("_selected", "_production"),
        validate="one_to_one",
    )
    difference = paired.hit_at_20_selected.to_numpy() - paired.hit_at_20_production.to_numpy()
    rng = np.random.default_rng(args.seed)
    bootstrap = np.asarray(
        [
            rng.choice(difference, size=len(difference), replace=True).mean()
            for _ in range(args.bootstrap_samples)
        ]
    )
    frozen_summary = {
        "selected_config": selected_name,
        "n": len(paired),
        "production_hit": float(paired.hit_at_20_production.mean()),
        "selected_hit": float(paired.hit_at_20_selected.mean()),
        "difference": float(difference.mean()),
        "new_hits": int((difference == 1).sum()),
        "lost_hits": int((difference == -1).sum()),
        "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
        "production_mrr": float(paired.reciprocal_rank_production.mean()),
        "selected_mrr": float(paired.reciprocal_rank_selected.mean()),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    query_frame.to_csv(output_dir / "query_metrics.csv", index=False)
    development.sort_values(["hit_at_20", "mrr"], ascending=False).to_csv(
        output_dir / "development_grid.csv", index=False
    )
    paired.to_csv(output_dir / "frozen_paired.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "direction": DIRECTION,
                "budget": BUDGET,
                "production_method": PRODUCTION_METHOD,
                "kernel_config": KERNEL_CONFIG,
                "selected_development": selected.to_dict(),
                "frozen": frozen_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("DEVELOPMENT")
    print(development.sort_values(["hit_at_20", "mrr"], ascending=False).to_string(index=False))
    print("FROZEN")
    print(json.dumps(frozen_summary, indent=2))


if __name__ == "__main__":
    main()
