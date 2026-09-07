from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = ROOT / "projects/active/terpene_screening/CATALYST_COMPREHENSIVE_DIRECTIONAL_CENTER_V1.json"
DEFAULT_EVAL = ROOT / "results/comprehensive_directional_center_v1/dev_eval"
DEFAULT_DIFFICULTY = ROOT / "results/comprehensive_directional_center_v1/difficulty"
DEFAULT_OUTPUT = ROOT / "results/comprehensive_directional_center_v1/development_gate.json"

ROW_METRICS = {
    "mrr": "reciprocal_rank",
    "map": "average_precision",
    "macro_roc_auc": "roc_auc",
    "ndcg_at_10": "ndcg_at_10",
    "hit_at_10": "hit_at_10",
    "hit_at_20": "hit_at_20",
    "hit_at_50": "hit_at_50",
}

SYSTEMS = {
    "r2e_safety": ("reaction_to_enzyme", "baseline_bundle"),
    "r2e_candidate": ("reaction_to_enzyme", "candidate_bundle"),
    "e2r_baseline": ("enzyme_to_reaction", "baseline_bundle"),
    "e2r_candidate": ("enzyme_to_reaction", "candidate_bundle"),
}



def _resolve_query_metrics(fold_dir: Path) -> Path:
    direct = fold_dir / "query_metrics.csv"
    if direct.is_file():
        return direct
    matches = sorted(fold_dir.glob("*/query_metrics.csv"))
    if len(matches) != 1:
        raise ValueError(f"Expected one query_metrics.csv under {fold_dir}, found {matches}")
    return matches[0]

def _load_query_metrics(path: Path, direction: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["direction"] == direction].copy()
    if frame.empty:
        raise ValueError(f"No {direction} rows in {path}")
    if frame["query_id"].duplicated().any():
        raise ValueError(f"Duplicate {direction} query IDs in {path}")
    return frame.sort_values("query_id").reset_index(drop=True)


def _aggregate(frame: pd.DataFrame) -> dict[str, float | int]:
    out: dict[str, float | int] = {"query_count": int(len(frame))}
    for name, column in ROW_METRICS.items():
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        out[name] = float(np.mean(finite)) if len(finite) else float("nan")
    ranks = pd.to_numeric(frame["best_positive_rank"], errors="coerce").to_numpy(dtype=np.float64)
    ranks = ranks[np.isfinite(ranks)]
    out["median_best_positive_rank"] = float(np.median(ranks)) if len(ranks) else float("nan")
    return out


def _assert_same_queries(a: pd.DataFrame, b: pd.DataFrame, label: str) -> None:
    qa = a["query_id"].astype(str).tolist()
    qb = b["query_id"].astype(str).tolist()
    if qa != qb:
        raise ValueError(f"Paired support mismatch for {label}: {len(qa)} vs {len(qb)} queries")


def _delta(candidate: dict[str, float | int], baseline: dict[str, float | int]) -> dict[str, float]:
    keys = set(candidate) & set(baseline)
    return {
        key: float(candidate[key]) - float(baseline[key])
        for key in sorted(keys)
        if key != "query_count"
    }


def _ge(candidate: dict[str, float | int], baseline: dict[str, float | int], key: str, tol: float = 0.0) -> bool:
    return float(candidate[key]) + tol >= float(baseline[key])


def _fold_stability(
    candidate: dict[int, dict[str, float | int]],
    baseline: dict[int, dict[str, float | int]],
) -> dict[str, Any]:
    per_fold = {}
    no_large_regression = True
    improvements = {"mrr": 0, "map": 0}
    for fold in sorted(candidate):
        deltas = {
            key: float(candidate[fold][key]) - float(baseline[fold][key])
            for key in ("mrr", "map")
        }
        per_fold[str(fold)] = deltas
        if any(delta < -0.005 - 1e-12 for delta in deltas.values()):
            no_large_regression = False
        for key, delta in deltas.items():
            if delta > 0:
                improvements[key] += 1
    improve_two_of_three = max(improvements.values()) >= 2
    return {
        "per_fold_deltas": per_fold,
        "no_mrr_or_map_regression_gt_0p005": no_large_regression,
        "improvement_fold_counts": improvements,
        "improves_mrr_or_map_on_at_least_2_of_3_folds": improve_two_of_three,
        "pass": no_large_regression and improve_two_of_three,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the pre-frozen comprehensive directional-center V1 development gate.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--difficulty-root", type=Path, default=DEFAULT_DIFFICULTY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "frozen_before_any_new_split_performance_materialization":
        raise ValueError("Unexpected preregistration status")
    folds = [int(v) for v in protocol["development"]["development_folds"]]
    if folds != [0, 1, 2]:
        raise ValueError(f"Unexpected development folds: {folds}")

    frames: dict[str, dict[int, pd.DataFrame]] = {name: {} for name in SYSTEMS}
    fold_metrics: dict[str, dict[int, dict[str, float | int]]] = {name: {} for name in SYSTEMS}
    for name, (direction, dirname) in SYSTEMS.items():
        for fold in folds:
            path = _resolve_query_metrics(args.eval_root / dirname / f"fold{fold}")
            frame = _load_query_metrics(path, direction)
            frames[name][fold] = frame
            fold_metrics[name][fold] = _aggregate(frame)

    for fold in folds:
        _assert_same_queries(frames["r2e_safety"][fold], frames["r2e_candidate"][fold], f"R2E fold{fold}")
        _assert_same_queries(frames["e2r_baseline"][fold], frames["e2r_candidate"][fold], f"E2R fold{fold}")

    pooled_frames = {
        name: pd.concat([frames[name][fold] for fold in folds], ignore_index=True)
        for name in SYSTEMS
    }
    pooled = {name: _aggregate(frame) for name, frame in pooled_frames.items()}

    hard_rows: dict[str, list[pd.DataFrame]] = {"r2e_safety": [], "r2e_candidate": []}
    hard_counts: dict[str, int] = {}
    for fold in folds:
        difficulty_dirs = sorted(args.difficulty_root.glob(f"*fold{fold}"))
        if len(difficulty_dirs) != 1:
            raise ValueError(f"Expected one difficulty directory for fold{fold}, found {difficulty_dirs}")
        reaction = pd.read_csv(difficulty_dirs[0] / "reaction_slices.csv", dtype={"reaction_id": str})
        hard_ids = set(
            reaction.loc[pd.to_numeric(reaction["max_train_drfp_tanimoto"], errors="coerce") < 0.3, "reaction_id"].astype(str)
        )
        hard_counts[str(fold)] = len(hard_ids)
        for name in hard_rows:
            frame = frames[name][fold]
            selected = frame.loc[frame["query_id"].astype(str).isin(hard_ids)].copy()
            if len(selected) != len(hard_ids):
                missing = sorted(hard_ids - set(selected["query_id"].astype(str)))[:10]
                raise ValueError(f"Hard R2E support missing in {name} fold{fold}: {missing}")
            hard_rows[name].append(selected)
    hard = {name: _aggregate(pd.concat(parts, ignore_index=True)) for name, parts in hard_rows.items()}

    r2e_floor_checks = {
        key: _ge(pooled["r2e_candidate"], pooled["r2e_safety"], key)
        for key in ("mrr", "map", "ndcg_at_10", "hit_at_10", "hit_at_50")
    }
    r2e_hard_checks = {
        "mrr": _ge(hard["r2e_candidate"], hard["r2e_safety"], "mrr"),
        "map": _ge(hard["r2e_candidate"], hard["r2e_safety"], "map"),
        "hit_at_10_floor_minus_0p005": _ge(hard["r2e_candidate"], hard["r2e_safety"], "hit_at_10", tol=0.005),
        "median_best_positive_rank": float(hard["r2e_candidate"]["median_best_positive_rank"])
        <= float(hard["r2e_safety"]["median_best_positive_rank"]),
    }
    e2r_floor_checks = {
        key: _ge(pooled["e2r_candidate"], pooled["e2r_baseline"], key)
        for key in ("mrr", "map", "macro_roc_auc", "ndcg_at_10", "hit_at_10")
    }
    e2r_floor_checks.update(
        {
            "hit_at_20_floor_minus_0p005": _ge(pooled["e2r_candidate"], pooled["e2r_baseline"], "hit_at_20", tol=0.005),
            "hit_at_50_floor_minus_0p005": _ge(pooled["e2r_candidate"], pooled["e2r_baseline"], "hit_at_50", tol=0.005),
        }
    )

    r2e_stability = _fold_stability(fold_metrics["r2e_candidate"], fold_metrics["r2e_safety"])
    e2r_stability = _fold_stability(fold_metrics["e2r_candidate"], fold_metrics["e2r_baseline"])
    material_deltas = {
        "r2e": _delta(pooled["r2e_candidate"], pooled["r2e_safety"]),
        "e2r": _delta(pooled["e2r_candidate"], pooled["e2r_baseline"]),
    }
    material_gain = any(
        material_deltas[direction][key] >= threshold - 1e-12
        for direction in ("r2e", "e2r")
        for key, threshold in (("mrr", 0.01), ("map", 0.01), ("hit_at_10", 0.02))
    )
    pass_gate = (
        all(r2e_floor_checks.values())
        and all(r2e_hard_checks.values())
        and all(e2r_floor_checks.values())
        and r2e_stability["pass"]
        and e2r_stability["pass"]
        and material_gain
    )

    result = {
        "protocol": str(args.protocol.resolve()),
        "protocol_status": protocol["status"],
        "development_folds": folds,
        "selection_uses_external_or_revealed_outer": False,
        "pooled_metrics": pooled,
        "fold_metrics": {name: {str(k): v for k, v in values.items()} for name, values in fold_metrics.items()},
        "paired_deltas": {
            "r2e_candidate_minus_safety": _delta(pooled["r2e_candidate"], pooled["r2e_safety"]),
            "e2r_candidate_minus_baseline": _delta(pooled["e2r_candidate"], pooled["e2r_baseline"]),
        },
        "hard_reaction_similarity_lt_0p3": {
            "query_counts_by_fold": hard_counts,
            "metrics": hard,
            "candidate_minus_safety": _delta(hard["r2e_candidate"], hard["r2e_safety"]),
        },
        "checks": {
            "r2e_full_candidate": r2e_floor_checks,
            "r2e_hard_reaction_similarity_lt_0p3": r2e_hard_checks,
            "e2r_full_candidate": e2r_floor_checks,
            "r2e_fold_stability": r2e_stability,
            "e2r_fold_stability": e2r_stability,
            "material_gain": material_gain,
        },
        "development_gate_pass": pass_gate,
        "next_action": "run_frozen_confirmation_once" if pass_gate else "reject_family_no_retuning_on_this_development_split",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
