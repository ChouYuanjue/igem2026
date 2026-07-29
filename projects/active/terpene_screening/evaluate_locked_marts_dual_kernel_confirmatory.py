from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRODUCTION = ROOT / "results/terpene_marts_freeze_reaction_neighbor_confirmatory20260726/rankings.csv"
DEFAULT_KERNEL = ROOT / "results/terpene_marts_dual_kernel_confirmatory20260726/rankings.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_marts_dual_kernel_confirmatory20260726"
DIRECTION = "enzyme_to_reaction"
PRODUCTION_METHOD = "rank_hybrid_direct_0.75"
KERNEL_CONFIG = "rk50_pk5_t0.03_d1"
KERNEL_WEIGHT = 0.30
RRF_CONSTANT = 60.0
BUDGET = 20


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


def rrf(base: list[str], auxiliary: list[str]) -> list[str]:
    universe = list(dict.fromkeys(base + auxiliary))
    base_rank = {value: index + 1 for index, value in enumerate(base)}
    auxiliary_rank = {value: index + 1 for index, value in enumerate(auxiliary)}
    return sorted(
        universe,
        key=lambda value: (
            -(
                (1.0 - KERNEL_WEIGHT) / (RRF_CONSTANT + base_rank.get(value, 10001))
                + KERNEL_WEIGHT / (RRF_CONSTANT + auxiliary_rank.get(value, 10001))
            ),
            value,
        ),
    )


def rank_metrics(ranking: list[str], positives: set[str]) -> tuple[int, float, float]:
    position = next((index + 1 for index, value in enumerate(ranking) if value in positives), None)
    return (
        int(position is not None and position <= BUDGET),
        0.0 if position is None else 1.0 / position,
        np.nan if position is None else float(position),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked confirmatory E2R dual-kernel RRF.")
    parser.add_argument("--production-rankings", type=Path, default=DEFAULT_PRODUCTION)
    parser.add_argument("--kernel-rankings", type=Path, default=DEFAULT_KERNEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    production = pd.read_csv(args.production_rankings, dtype=str).fillna("")
    production = production[
        production.direction.eq(DIRECTION) & production.method.eq(PRODUCTION_METHOD)
    ].copy()
    production["rank"] = production["rank"].astype(int)
    production["is_positive"] = production["is_positive"].astype(int)
    kernel = pd.read_csv(args.kernel_rankings, dtype=str).fillna("")
    kernel = kernel[
        kernel.direction.eq(DIRECTION) & kernel.config.eq(KERNEL_CONFIG)
    ].copy()
    kernel["rank"] = kernel["rank"].astype(int)
    kernel["is_positive"] = kernel["is_positive"].astype(int)

    production_lists = ordered_lists(production)
    kernel_lists = ordered_lists(kernel)
    positives = positive_sets(production)
    if set(production_lists) != set(kernel_lists):
        missing_production = sorted(set(kernel_lists) - set(production_lists))
        missing_kernel = sorted(set(production_lists) - set(kernel_lists))
        raise ValueError(
            f"Query mismatch: missing production={missing_production[:5]}, "
            f"missing kernel={missing_kernel[:5]}"
        )

    rows: list[dict[str, object]] = []
    for key in sorted(production_lists):
        production_ranking = production_lists[key]
        fused_ranking = rrf(production_ranking, kernel_lists[key])
        production_hit, production_rr, production_rank = rank_metrics(
            production_ranking, positives[key]
        )
        fused_hit, fused_rr, fused_rank = rank_metrics(fused_ranking, positives[key])
        rows.append(
            {
                "split_id": key[0],
                "query_id": key[1],
                "production_hit": production_hit,
                "fused_hit": fused_hit,
                "difference": fused_hit - production_hit,
                "production_rr": production_rr,
                "fused_rr": fused_rr,
                "production_rank": production_rank,
                "fused_rank": fused_rank,
            }
        )
    paired = pd.DataFrame(rows)
    difference = paired.difference.to_numpy()
    rng = np.random.default_rng(args.seed)
    bootstrap = np.asarray(
        [
            rng.choice(difference, size=len(difference), replace=True).mean()
            for _ in range(args.bootstrap_samples)
        ]
    )
    summary = {
        "status": "locked_confirmatory",
        "direction": DIRECTION,
        "budget": BUDGET,
        "production_method": PRODUCTION_METHOD,
        "kernel_config": KERNEL_CONFIG,
        "kernel_weight": KERNEL_WEIGHT,
        "rrf_constant": RRF_CONSTANT,
        "n_query_cells": len(paired),
        "n_unique_queries": paired.query_id.nunique(),
        "production_hit": float(paired.production_hit.mean()),
        "fused_hit": float(paired.fused_hit.mean()),
        "difference": float(difference.mean()),
        "new_hits": int((difference == 1).sum()),
        "lost_hits": int((difference == -1).sum()),
        "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
        "production_mrr": float(paired.production_rr.mean()),
        "fused_mrr": float(paired.fused_rr.mean()),
        "confirmed_nonnegative": bool(np.quantile(bootstrap, 0.025) >= 0),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_dir / "locked_confirmatory_paired.csv", index=False)
    (output_dir / "locked_confirmatory_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame([summary]).to_csv(
        output_dir / "locked_confirmatory_summary.csv", index=False
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
