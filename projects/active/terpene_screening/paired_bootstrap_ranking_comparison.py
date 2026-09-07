from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_METRICS = {
    "hit_at_1": "hit_at_1",
    "hit_at_10": "hit_at_10",
    "hit_at_50": "hit_at_50",
    "mrr": "reciprocal_rank",
    "map": "average_precision",
    "macro_roc_auc": "roc_auc",
    "ndcg_at_10": "ndcg_at_10",
}


def paired_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    direction: str,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    left = baseline.loc[baseline["direction"].eq(direction)].set_index("query_id").sort_index()
    right = candidate.loc[candidate["direction"].eq(direction)].set_index("query_id").sort_index()
    if not left.index.is_unique or not right.index.is_unique:
        raise ValueError("query_id must be unique inside one direction")
    if set(left.index) != set(right.index):
        raise ValueError("paired comparison requires identical query IDs")
    common = left.index
    n = len(common)
    if n == 0:
        raise ValueError("no paired queries")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n, size=(bootstrap_samples, n), endpoint=False)
    rows: list[dict[str, object]] = []
    for metric_name, column in DEFAULT_METRICS.items():
        if column not in left or column not in right:
            continue
        delta = right.loc[common, column].to_numpy(float) - left.loc[common, column].to_numpy(float)
        observed = float(np.nanmean(delta))
        boot = np.nanmean(delta[sampled], axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        p_two_sided = float(min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))))
        rows.append(
            {
                "metric": metric_name,
                "n_queries": n,
                "baseline_mean": float(np.nanmean(left.loc[common, column].to_numpy(float))),
                "candidate_mean": float(np.nanmean(right.loc[common, column].to_numpy(float))),
                "delta": observed,
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "bootstrap_p_two_sided": p_two_sided,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired query-level bootstrap comparison for frozen ranking outputs.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--direction", choices=["reaction_to_enzyme", "enzyme_to_reaction"], required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    baseline = pd.read_csv(args.baseline, dtype={"query_id": str})
    candidate = pd.read_csv(args.candidate, dtype={"query_id": str})
    result = paired_bootstrap(
        baseline,
        candidate,
        direction=args.direction,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "paired_bootstrap.csv", index=False)
    payload = {
        "baseline": args.baseline_name,
        "candidate": args.candidate_name,
        "baseline_path": str(args.baseline.resolve()),
        "candidate_path": str(args.candidate.resolve()),
        "direction": args.direction,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "inference_unit": "query; paired resampling preserves per-query dependence across models",
        "metrics": result.to_dict("records"),
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
