from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE = ROOT / "results/terpene_current_me8_top10_locked_base_v1/rankings.csv"
DEFAULT_PFAM = ROOT / "results/terpene_current_pfam_fixed_rankings_v1/rankings.csv"
DEFAULT_KERNEL_DEVELOPMENT = ROOT / "results/terpene_dual_kernel_development_v1/rankings.csv"
DEFAULT_KERNEL_FROZEN = ROOT / "results/terpene_dual_kernel_frozen_v1/rankings.csv"
DEFAULT_STRICT = ROOT / "data/terpene_cold_splits/positive_pair_fold_assignments.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_locked_tps_multisource_route"
KERNEL_CONFIG = "rk50_pk10_t0.08_d1"
BUDGET = 10

ROUTES: dict[str, tuple[str, tuple[tuple[str, int], ...]]] = {
    "base": ("base", ()),
    "pfam": ("pfam", ()),
    "kernel": ("kernel", ()),
    "base_k1": ("base", (("kernel", 1),)),
    "base_k2": ("base", (("kernel", 2),)),
    "base_p1": ("base", (("pfam", 1),)),
    "base_p2": ("base", (("pfam", 2),)),
    "pfam_k1": ("pfam", (("kernel", 1),)),
    "pfam_k2": ("pfam", (("kernel", 2),)),
    "kernel_p1": ("kernel", (("pfam", 1),)),
    "tri_base8_p1_k1": ("base", (("pfam", 1), ("kernel", 1))),
    "tri_base7_p1_k2": ("base", (("pfam", 1), ("kernel", 2))),
    "tri_base7_p2_k1": ("base", (("pfam", 2), ("kernel", 1))),
}


def load_rankings(path: Path, source: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if "protocol" in frame.columns:
        frame = frame[frame.protocol.eq("double_cold_25cell")].copy()
    frame[["protein_fold", "reaction_fold", "rank"]] = frame[
        ["protein_fold", "reaction_fold", "rank"]
    ].astype(int)
    frame["source"] = source
    return frame


def ordered_lists(frame: pd.DataFrame) -> dict[tuple[int, int, str], list[str]]:
    keys = ["protein_fold", "reaction_fold", "reaction_id"]
    return {
        key: group.sort_values(["rank", "candidate_id"]).candidate_id.astype(str).tolist()
        for key, group in frame.groupby(keys, sort=True)
    }


def quota_route(
    sources: dict[str, list[str]],
    primary: str,
    quotas: tuple[tuple[str, int], ...],
    budget: int,
) -> list[str]:
    total_auxiliary = sum(count for _, count in quotas)
    result = list(sources[primary][: max(0, budget - total_auxiliary)])
    for source, count in quotas:
        added = 0
        for candidate in sources[source]:
            if candidate in result:
                continue
            result.append(candidate)
            added += 1
            if added >= count:
                break
    for source in (primary, "base", "pfam", "kernel"):
        for candidate in sources[source]:
            if candidate not in result:
                result.append(candidate)
    return result


def metrics_for_ranking(ranking: list[str], positives: set[str]) -> tuple[int, float, float]:
    position = next((index + 1 for index, candidate in enumerate(ranking) if candidate in positives), None)
    hit = int(position is not None and position <= BUDGET)
    reciprocal_rank = 0.0 if position is None else 1.0 / position
    return hit, reciprocal_rank, float(position) if position is not None else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked TPS Top-10 multi-source quota route.")
    parser.add_argument("--base-rankings", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--pfam-rankings", type=Path, default=DEFAULT_PFAM)
    parser.add_argument("--kernel-development-rankings", type=Path, default=DEFAULT_KERNEL_DEVELOPMENT)
    parser.add_argument("--kernel-frozen-rankings", type=Path, default=DEFAULT_KERNEL_FROZEN)
    parser.add_argument("--strict-splits", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    base = load_rankings(args.base_rankings, "base")
    pfam = load_rankings(args.pfam_rankings, "pfam")
    kernel_development = load_rankings(args.kernel_development_rankings, "kernel")
    kernel_frozen = load_rankings(args.kernel_frozen_rankings, "kernel")
    kernel = pd.concat(
        [
            kernel_development[kernel_development.config.eq(KERNEL_CONFIG)],
            kernel_frozen[kernel_frozen.config.eq(KERNEL_CONFIG)],
        ],
        ignore_index=True,
    )
    base_lists = ordered_lists(base)
    pfam_lists = ordered_lists(pfam)
    kernel_lists = ordered_lists(kernel)
    if not (set(base_lists) == set(pfam_lists) == set(kernel_lists)):
        raise ValueError("Source query sets differ")

    strict = pd.read_csv(args.strict_splits, dtype=str).fillna("").rename(
        columns={"rhea_id": "reaction_id"}
    )
    strict[["protein_fold", "reaction_fold"]] = strict[
        ["protein_fold", "reaction_fold"]
    ].astype(int)
    keys = ["protein_fold", "reaction_fold", "reaction_id"]
    positives = {
        key: set(group.Entry.astype(str))
        for key, group in strict.groupby(keys, sort=True)
    }

    query_rows: list[dict[str, object]] = []
    for key in sorted(base_lists):
        partition = "development" if key[0] == 4 or key[1] == 4 else "frozen"
        sources = {
            "base": base_lists[key],
            "pfam": pfam_lists[key],
            "kernel": kernel_lists[key],
        }
        for route, (primary, quotas) in ROUTES.items():
            ranking = quota_route(sources, primary, quotas, BUDGET)
            hit, reciprocal_rank, best_rank = metrics_for_ranking(ranking, positives[key])
            query_rows.append(
                {
                    **dict(zip(keys, key)),
                    "partition": partition,
                    "route": route,
                    "hit_at_10": hit,
                    "reciprocal_rank": reciprocal_rank,
                    "best_positive_rank": best_rank,
                }
            )
    query_frame = pd.DataFrame(query_rows)
    development = (
        query_frame[query_frame.partition.eq("development")]
        .groupby("route", as_index=False)
        .agg(
            hit_at_10=("hit_at_10", "mean"),
            mrr=("reciprocal_rank", "mean"),
            median_rank=("best_positive_rank", "median"),
        )
    )
    complexity = {
        "base": 0,
        "pfam": 1,
        "kernel": 1,
        "base_k1": 2,
        "base_p1": 2,
        "pfam_k1": 2,
        "kernel_p1": 2,
        "base_k2": 3,
        "base_p2": 3,
        "pfam_k2": 3,
        "tri_base8_p1_k1": 4,
        "tri_base7_p1_k2": 5,
        "tri_base7_p2_k1": 5,
    }
    development["complexity"] = development.route.map(complexity)
    selected = development.sort_values(
        ["hit_at_10", "mrr", "complexity", "route"],
        ascending=[False, False, True, True],
    ).iloc[0]
    selected_route = str(selected.route)

    frozen_selected = query_frame[
        query_frame.partition.eq("frozen") & query_frame.route.eq(selected_route)
    ]
    frozen_baseline = query_frame[
        query_frame.partition.eq("frozen") & query_frame.route.eq("base")
    ]
    merge_keys = ["protein_fold", "reaction_fold", "reaction_id"]
    paired = frozen_selected[merge_keys + ["hit_at_10"]].merge(
        frozen_baseline[merge_keys + ["hit_at_10"]],
        on=merge_keys,
        suffixes=("_selected", "_baseline"),
        validate="one_to_one",
    )
    difference = paired.hit_at_10_selected.to_numpy() - paired.hit_at_10_baseline.to_numpy()
    rng = np.random.default_rng(args.seed)
    bootstrap = np.asarray(
        [
            rng.choice(difference, size=len(difference), replace=True).mean()
            for _ in range(args.bootstrap_samples)
        ]
    )
    frozen_summary = {
        "selected_route": selected_route,
        "n": len(paired),
        "baseline_hit": float(paired.hit_at_10_baseline.mean()),
        "selected_hit": float(paired.hit_at_10_selected.mean()),
        "difference": float(difference.mean()),
        "new_hits": int((difference == 1).sum()),
        "lost_hits": int((difference == -1).sum()),
        "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    query_frame.to_csv(output_dir / "query_metrics.csv", index=False)
    development.sort_values(
        ["hit_at_10", "mrr"], ascending=False
    ).to_csv(output_dir / "development_routes.csv", index=False)
    paired.to_csv(output_dir / "frozen_paired.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "budget": BUDGET,
                "kernel_config": KERNEL_CONFIG,
                "selected_development": selected.to_dict(),
                "frozen": frozen_summary,
                "routes": {
                    name: {"primary": primary, "quotas": quotas}
                    for name, (primary, quotas) in ROUTES.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("DEVELOPMENT")
    print(development.sort_values(["hit_at_10", "mrr"], ascending=False).to_string(index=False))
    print("FROZEN")
    print(json.dumps(frozen_summary, indent=2))


if __name__ == "__main__":
    main()
