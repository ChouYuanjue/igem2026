from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "results/enzymecage_family_response_v1"
FAMILIES = ("p450", "phosphatase", "terpene")
FEATURE_SUFFIXES = (
    "response_top10pct_mean",
    "response_rms",
    "fused_shift_mean",
    "hidden2_shift_mean",
    "attention_entropy_delta_mean",
    "attention_peak_delta_mean",
)


def robust_stats(x: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(x, errors="coerce").dropna().to_numpy(float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return {
        "median": median,
        "mad": mad,
        "iqr": float(q75 - q25),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "n": int(len(values)),
    }


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(reference.astype(float))
    return np.searchsorted(ref, values.astype(float), side="right") / max(len(ref), 1)


def calibrate_frame(frame: pd.DataFrame, calibration: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = frame.copy()
    family_primary = {}
    for fam in FAMILIES:
        for suffix in FEATURE_SUFFIXES:
            col = f"{fam}_{suffix}"
            if col not in calibration.columns or col not in out.columns:
                continue
            st = stats["features"][col]
            values = pd.to_numeric(out[col], errors="coerce").to_numpy(float)
            ref = pd.to_numeric(calibration[col], errors="coerce").dropna().to_numpy(float)
            scale = max(1.4826 * st["mad"], st["std"], 1e-8)
            out[f"{col}_robust_z"] = (values - st["median"]) / scale
            out[f"{col}_percentile"] = empirical_percentile(ref, values)
        family_primary[fam] = out[f"{fam}_response_top10pct_mean_percentile"].to_numpy(float)

    matrix = np.column_stack([family_primary[f] for f in FAMILIES])
    safe = np.clip(matrix, 1e-6, 1 - 1e-6)
    logits = np.log(safe / (1 - safe))
    prob = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    ordered = np.sort(logits, axis=1)
    out["family_calibrated_winner"] = [FAMILIES[i] for i in logits.argmax(axis=1)]
    out["family_calibrated_margin"] = ordered[:, -1] - ordered[:, -2]
    out["family_calibrated_entropy"] = -(prob * np.log(np.clip(prob, 1e-12, None))).sum(axis=1)
    out["family_calibrated_max_percentile"] = matrix.max(axis=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-tag", default="calibration_author_valid")
    ap.add_argument("--apply-tag", action="append", default=[])
    args = ap.parse_args()

    calibration_path = BASE / args.calibration_tag / "query_features.csv"
    calibration = pd.read_csv(calibration_path)
    stats = {
        "schema": "enzymecage-family-response-calibration-stats-v1",
        "calibration_tag": args.calibration_tag,
        "queries": int(len(calibration)),
        "features": {},
        "family_feature_quality": {},
    }
    for fam in FAMILIES:
        quality = {}
        for suffix in FEATURE_SUFFIXES:
            col = f"{fam}_{suffix}"
            st = robust_stats(calibration[col])
            stats["features"][col] = st
            # Relative variability against the response feature is diagnostic only.
            quality[suffix] = {
                "std": st["std"],
                "iqr": st["iqr"],
                "informative_variation": bool(st["std"] > 1e-5 and st["iqr"] > 1e-6),
            }
        stats["family_feature_quality"][fam] = quality

    out_path = BASE / args.calibration_tag / "calibration_stats.json"
    out_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)

    for tag in args.apply_tag:
        path = BASE / tag / "query_features.csv"
        frame = pd.read_csv(path)
        calibrated = calibrate_frame(frame, calibration, stats)
        calibrated.to_csv(BASE / tag / "query_features_calibrated.csv", index=False)
        print(
            json.dumps(
                {
                    "tag": tag,
                    "queries": len(calibrated),
                    "winner_counts": calibrated["family_calibrated_winner"].value_counts().to_dict(),
                    "margin_median": float(calibrated["family_calibrated_margin"].median()),
                    "entropy_median": float(calibrated["family_calibrated_entropy"].median()),
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
