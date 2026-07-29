from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KERNEL = ROOT / "results/terpene_marts_dual_kernel_frozen_v1/query_metrics.csv"
DEFAULT_FREEZE = ROOT / "results/terpene_marts_freeze_reaction_neighbor_hybrid/query_metrics.csv"
DEFAULT_R2E075 = ROOT / "results/terpene_marts_r2e075_neighbor_hybrid/query_metrics.csv"
DEFAULT_EXACT = ROOT / "results/terpene_horizyn_residual_canonical_exact/query_metrics.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_marts_dual_kernel_frozen_v1"

LOCKED = {
    ("enzyme_to_reaction", 20): (
        "rk50_pk5_t0.03_d1",
        "freeze",
        "rank_hybrid_direct_0.75",
    ),
    ("reaction_to_enzyme", 3): (
        "rk50_pk10_t0.08_d1",
        "r2e075",
        "adapted_direct",
    ),
    ("reaction_to_enzyme", 10): (
        "rk50_pk10_t0.08_d1",
        "exact",
        "horizyn_reaction_residual",
    ),
    ("reaction_to_enzyme", 20): (
        "rk50_pk10_t0.08_d0.5",
        "exact",
        "horizyn_reaction_residual",
    ),
}


def normalize_split(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "split_id" not in frame.columns:
        if "fold" not in frame.columns:
            raise ValueError(f"No split identifier in {frame.columns.tolist()}")
        frame["split_id"] = frame["fold"].astype(str)
    folds = frame["split_id"].str.extract(
        r"p(?P<protein_fold>\d+)_r(?P<reaction_fold>\d+)"
    )
    if folds.isna().any().any():
        raise ValueError("Unparseable split identifier")
    frame["protein_fold"] = folds.protein_fold.astype(int)
    frame["reaction_fold"] = folds.reaction_fold.astype(int)
    frame["partition"] = np.where(
        frame.protein_fold.eq(4) | frame.reaction_fold.eq(4),
        "development",
        "frozen",
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired frozen MARTS dual-kernel comparison.")
    parser.add_argument("--kernel-query-metrics", type=Path, default=DEFAULT_KERNEL)
    parser.add_argument("--freeze-query-metrics", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--r2e075-query-metrics", type=Path, default=DEFAULT_R2E075)
    parser.add_argument("--exact-query-metrics", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    kernel = normalize_split(pd.read_csv(args.kernel_query_metrics))
    sources = {
        "freeze": normalize_split(pd.read_csv(args.freeze_query_metrics)),
        "r2e075": normalize_split(pd.read_csv(args.r2e075_query_metrics)),
        "exact": normalize_split(pd.read_csv(args.exact_query_metrics)),
    }
    keys = ["split_id", "query_id"]
    rows: list[dict[str, object]] = []
    paired_rows: list[pd.DataFrame] = []
    for (direction, budget), (config, source_name, method) in LOCKED.items():
        hit_column = f"hit_at_{budget}"
        candidate = kernel[
            kernel.direction.eq(direction) & kernel.config.eq(config)
        ][keys + [hit_column, "reciprocal_rank"]].rename(
            columns={hit_column: "kernel_hit", "reciprocal_rank": "kernel_rr"}
        )
        source = sources[source_name]
        source = source[source.partition.eq("frozen")]
        if "direction" in source.columns:
            source = source[source.direction.eq(direction)]
        source = source[source.method.eq(method)][
            keys + [hit_column, "reciprocal_rank"]
        ].rename(
            columns={hit_column: "production_hit", "reciprocal_rank": "production_rr"}
        )
        paired = candidate.merge(source, on=keys, validate="one_to_one")
        if len(paired) != len(candidate) or len(paired) != len(source):
            raise ValueError(
                f"Query mismatch for {direction} Top-{budget}: "
                f"kernel={len(candidate)}, production={len(source)}, paired={len(paired)}"
            )
        difference = paired.kernel_hit.to_numpy() - paired.production_hit.to_numpy()
        rng = np.random.default_rng(args.seed + budget + (0 if direction == "enzyme_to_reaction" else 100))
        bootstrap = np.asarray(
            [
                rng.choice(difference, size=len(difference), replace=True).mean()
                for _ in range(args.bootstrap_samples)
            ]
        )
        paired["direction"] = direction
        paired["budget"] = budget
        paired["kernel_config"] = config
        paired["production_source"] = source_name
        paired["production_method"] = method
        paired["difference"] = difference
        paired_rows.append(paired)
        rows.append(
            {
                "direction": direction,
                "budget": budget,
                "kernel_config": config,
                "production_source": source_name,
                "production_method": method,
                "n": len(paired),
                "production_hit": float(paired.production_hit.mean()),
                "kernel_hit": float(paired.kernel_hit.mean()),
                "difference": float(difference.mean()),
                "new_hits": int((difference == 1).sum()),
                "lost_hits": int((difference == -1).sum()),
                "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
                "production_mrr": float(paired.production_rr.mean()),
                "kernel_mrr": float(paired.kernel_rr.mean()),
            }
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    paired = pd.concat(paired_rows, ignore_index=True)
    metrics.to_csv(output_dir / "locked_production_comparison.csv", index=False)
    paired.to_csv(output_dir / "locked_production_paired.csv", index=False)
    (output_dir / "locked_production_comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
