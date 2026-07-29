from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE = ROOT / "results/terpene_current_me8_fusion_rankings_v1/r2e075/rankings.csv"
DEFAULT_KERNEL = ROOT / "results/terpene_dual_kernel_frozen_v1/rankings.csv"
DEFAULT_STRICT = ROOT / "data/terpene_cold_splits/positive_pair_fold_assignments.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_dual_kernel_frozen_v1"
ROUTES = {
    3: ("rk10_pk10_t0.08_d1", "kernel", 0),
    5: ("rk10_pk10_t0.08_d1", "kernel", 0),
    10: ("rk50_pk10_t0.08_d1", "rescue", 1),
    20: ("rk20_pk50_t0.08_d1", "rescue", 2),
}


def rescue(base: list[str], auxiliary: list[str], budget: int, slots: int) -> list[str]:
    result = list(base[: max(0, budget - slots)])
    for candidate in auxiliary:
        if candidate not in result:
            result.append(candidate)
        if len(result) >= budget:
            break
    return result[:budget]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen locked dual-kernel routes.")
    parser.add_argument("--base-rankings", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--kernel-rankings", type=Path, default=DEFAULT_KERNEL)
    parser.add_argument("--strict-splits", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    base = pd.read_csv(args.base_rankings, dtype=str).fillna("")
    kernel = pd.read_csv(args.kernel_rankings, dtype=str).fillna("")
    strict = pd.read_csv(args.strict_splits, dtype=str).fillna("").rename(
        columns={"rhea_id": "reaction_id"}
    )
    strict[["protein_fold", "reaction_fold"]] = strict[
        ["protein_fold", "reaction_fold"]
    ].astype(int)
    keys = ["protein_fold", "reaction_fold", "reaction_id"]
    base = base[base.protocol.eq("double_cold_25cell")].copy()
    base[["protein_fold", "reaction_fold", "rank"]] = base[
        ["protein_fold", "reaction_fold", "rank"]
    ].astype(int)
    base = base[
        base.protein_fold.ne(4) & base.reaction_fold.ne(4)
    ].copy()
    kernel[["protein_fold", "reaction_fold", "rank"]] = kernel[
        ["protein_fold", "reaction_fold", "rank"]
    ].astype(int)
    base_lists = {
        key: group.sort_values(["rank", "candidate_id"]).candidate_id.astype(str).tolist()
        for key, group in base.groupby(keys, sort=True)
    }
    positives = {
        key: set(group.Entry.astype(str))
        for key, group in strict.groupby(keys, sort=True)
    }

    rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for budget, (config, method, slots) in ROUTES.items():
        local = kernel[kernel.config.eq(config)]
        kernel_lists = {
            key: group.sort_values(["rank", "candidate_id"]).candidate_id.astype(str).tolist()
            for key, group in local.groupby(keys, sort=True)
        }
        if set(base_lists) != set(kernel_lists):
            raise ValueError(f"Query set mismatch for {config}")
        baseline_values: list[int] = []
        selected_values: list[int] = []
        for key in sorted(base_lists):
            baseline_ranking = base_lists[key][:budget]
            selected_ranking = (
                kernel_lists[key][:budget]
                if method == "kernel"
                else rescue(base_lists[key], kernel_lists[key], budget, slots)
            )
            baseline_hit = int(bool(set(baseline_ranking) & positives[key]))
            selected_hit = int(bool(set(selected_ranking) & positives[key]))
            baseline_values.append(baseline_hit)
            selected_values.append(selected_hit)
            paired_rows.append(
                {
                    **dict(zip(keys, key)),
                    "budget": budget,
                    "kernel_config": config,
                    "method": method,
                    "slots": slots,
                    "baseline_hit": baseline_hit,
                    "selected_hit": selected_hit,
                    "difference": selected_hit - baseline_hit,
                }
            )
        baseline_array = np.asarray(baseline_values)
        selected_array = np.asarray(selected_values)
        difference = selected_array - baseline_array
        rng = np.random.default_rng(args.seed + budget)
        bootstrap = np.asarray(
            [
                rng.choice(difference, size=len(difference), replace=True).mean()
                for _ in range(args.bootstrap_samples)
            ]
        )
        rows.append(
            {
                "budget": budget,
                "kernel_config": config,
                "method": method,
                "slots": slots,
                "n": len(difference),
                "baseline_hit": float(baseline_array.mean()),
                "selected_hit": float(selected_array.mean()),
                "difference": float(difference.mean()),
                "new_hits": int(((selected_array == 1) & (baseline_array == 0)).sum()),
                "lost_hits": int(((selected_array == 0) & (baseline_array == 1)).sum()),
                "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
            }
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    paired = pd.DataFrame(paired_rows)
    metrics.to_csv(output_dir / "locked_route_metrics.csv", index=False)
    paired.to_csv(output_dir / "locked_route_paired.csv", index=False)
    (output_dir / "locked_route_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
