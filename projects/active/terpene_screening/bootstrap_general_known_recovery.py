from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.terpene_screening.compare_general_known_recovery import (
    KEY_COLUMNS,
    _validate_unique,
    evaluation_model_signature,
)

DEFAULT_METRICS = ("reciprocal_rank", "hit_at_10", "hit_at_20")


def paired_bootstrap_delta(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    samples: int,
    seed: int,
    chunk_size: int = 1000,
) -> dict[str, float | int]:
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if baseline.shape != candidate.shape or baseline.ndim != 1 or baseline.size == 0:
        raise ValueError("baseline and candidate must be non-empty paired 1D arrays")
    if samples <= 0 or chunk_size <= 0:
        raise ValueError("samples and chunk_size must be positive")
    delta = candidate - baseline
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float64)
    written = 0
    while written < samples:
        n = min(chunk_size, samples - written)
        indices = rng.integers(0, delta.size, size=(n, delta.size))
        boot[written : written + n] = delta[indices].mean(axis=1)
        written += n
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "n_queries": int(delta.size),
        "baseline_mean": float(baseline.mean()),
        "candidate_mean": float(candidate.mean()),
        "delta": float(delta.mean()),
        "ratio": float(candidate.mean() / baseline.mean()) if baseline.mean() != 0 else float("nan"),
        "bootstrap_ci_low": float(low),
        "bootstrap_ci_high": float(high),
        "bootstrap_probability_nonpositive": float(np.mean(boot <= 0.0)),
        "bootstrap_samples": int(samples),
    }


def compare_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    stratum: str,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    samples: int = 30000,
    seed: int = 20260723,
) -> pd.DataFrame:
    _validate_unique(baseline, "baseline")
    _validate_unique(candidate, "candidate")
    joined = baseline.merge(
        candidate,
        on=list(KEY_COLUMNS),
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
        how="inner",
    )
    if len(joined) != len(baseline) or len(joined) != len(candidate):
        raise ValueError("baseline and candidate query populations do not match exactly")
    rows: list[dict[str, object]] = []
    for direction_index, direction in enumerate(sorted(joined["direction"].unique())):
        group = joined[(joined["direction"] == direction) & (joined["stratum"] == stratum)]
        if group.empty:
            continue
        for metric_index, metric in enumerate(metrics):
            if f"{metric}_baseline" not in group or f"{metric}_candidate" not in group:
                raise ValueError(f"missing metric: {metric}")
            result = paired_bootstrap_delta(
                group[f"{metric}_baseline"].to_numpy(),
                group[f"{metric}_candidate"].to_numpy(),
                samples=samples,
                seed=seed + 100 * direction_index + metric_index,
            )
            rows.append({"direction": direction, "stratum": stratum, "metric": metric, **result})
    if not rows:
        raise ValueError(f"no rows found for stratum {stratum}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired query-bootstrap comparison for broad known-recovery models.")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stratum", default="unseen_to_historical_training")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    baseline_signature = evaluation_model_signature(args.baseline_dir)
    candidate_signature = evaluation_model_signature(args.candidate_dir)
    if baseline_signature != candidate_signature:
        raise ValueError(
            "model ensemble signatures do not match; "
            f"baseline={baseline_signature}, candidate={candidate_signature}"
        )
    baseline = pd.read_csv(args.baseline_dir / "query_metrics.csv")
    candidate = pd.read_csv(args.candidate_dir / "query_metrics.csv")
    metrics = tuple(value.strip() for value in args.metrics.split(",") if value.strip())
    frame = compare_bootstrap(
        baseline,
        candidate,
        stratum=args.stratum,
        metrics=metrics,
        samples=args.samples,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    summary = {
        "baseline_dir": str(args.baseline_dir.resolve()),
        "candidate_dir": str(args.candidate_dir.resolve()),
        "model_signature": list(baseline_signature),
        "stratum": args.stratum,
        "metrics": list(metrics),
        "samples": args.samples,
        "seed": args.seed,
        "output": str((args.output_dir / "paired_bootstrap.csv").resolve()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
