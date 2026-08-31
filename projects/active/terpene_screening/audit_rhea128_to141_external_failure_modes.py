from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CELL = "rhea128_to141_sprot_strict_double_cold_v2"
DEFAULT_ROOT = ROOT / "results/rhea128_to141_external_v2"
DEFAULT_DIFFICULTY = DEFAULT_ROOT / "posthoc_difficulty" / CELL
DEFAULT_CENTER_AUDIT = ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1/audit.csv"
DEFAULT_OUTPUT = ROOT / "projects/active/terpene_screening/CLEANROOM_R2E_RHEA128_TO141_EXTERNAL_V2_FAILURE_AUDIT.json"

METRICS = {
    "mrr": "reciprocal_rank",
    "map": "average_precision",
    "macro_roc_auc": "roc_auc",
    "ndcg_at_10": "ndcg_at_10",
    "hit_at_10": "hit_at_10",
    "hit_at_20": "hit_at_20",
    "hit_at_50": "hit_at_50",
    "best_positive_rank_fraction": "best_positive_rank_fraction",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _r2e(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"query_id": str}).fillna("")
    frame = frame[frame.direction.eq("reaction_to_enzyme")].copy()
    return frame.sort_values("query_id").reset_index(drop=True)


def merge_query_audit(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    reaction_slices: pd.DataFrame,
    pair_slices: pd.DataFrame,
    center_audit: pd.DataFrame,
) -> pd.DataFrame:
    if not baseline.query_id.equals(candidate.query_id):
        raise ValueError("baseline/candidate query IDs differ")
    keep = ["query_id", *METRICS.values(), "best_positive_rank"]
    joined = baseline[keep].merge(candidate[keep], on="query_id", suffixes=("_base", "_candidate"), validate="one_to_one")
    rs = reaction_slices.rename(columns={"reaction_id": "query_id"})[
        ["query_id", "max_train_drfp_tanimoto", "reaction_similarity_bucket"]
    ].copy()
    rs["max_train_drfp_tanimoto"] = pd.to_numeric(rs.max_train_drfp_tanimoto, errors="coerce").fillna(0.0)
    joined = joined.merge(rs, on="query_id", how="left", validate="one_to_one")
    if joined.reaction_similarity_bucket.isna().any():
        raise ValueError("missing reaction difficulty labels")

    ps = pair_slices.copy()
    ps["mmseqs_fident_num"] = pd.to_numeric(ps.mmseqs_fident, errors="coerce")
    protein = ps.groupby("reaction_id", sort=True).agg(
        n_positives=("protein_id", "nunique"),
        n_no_hit_positives=("protein_identity_bucket", lambda x: int((x == "no_hit").sum())),
        max_positive_mmseqs_fident=("mmseqs_fident_num", "max"),
        mean_positive_mmseqs_fident=("mmseqs_fident_num", "mean"),
    ).reset_index().rename(columns={"reaction_id": "query_id"})
    protein["no_hit_positive_fraction"] = protein.n_no_hit_positives / protein.n_positives
    protein["max_positive_identity_bucket"] = pd.cut(
        protein.max_positive_mmseqs_fident.fillna(-1),
        bins=[-2, -0.5, 0.2, 0.4, 0.6, 0.8, 2],
        labels=["all_no_hit", "lt20", "20_40", "40_60", "60_80", "ge80"],
        right=False,
    ).astype(str)
    joined = joined.merge(protein, on="query_id", how="left", validate="one_to_one")

    ca = center_audit.rename(columns={"reaction_id": "query_id"})[["query_id", "status", "warning", "feature_nonzero"]].copy()
    ca["center_status"] = np.where(ca.status.eq("valid"), "valid", "zero_fallback")
    joined = joined.merge(ca[["query_id", "center_status", "warning", "feature_nonzero"]], on="query_id", how="left", validate="one_to_one")
    joined["center_status"] = joined.center_status.fillna("missing_audit")
    joined["positive_count_bucket"] = pd.cut(
        joined.n_positives,
        bins=[0, 1, 2, 5, 10, np.inf],
        labels=["1", "2", "3-5", "6-10", ">10"],
        include_lowest=True,
    ).astype(str)
    return joined


def aggregate(group: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {"n_queries": int(len(group))}
    if not len(group):
        return out
    out.update({
        "mean_positive_count": float(group.n_positives.mean()),
        "mean_no_hit_positive_fraction": float(group.no_hit_positive_fraction.mean()),
        "median_best_positive_rank_base": float(group.best_positive_rank_base.median()),
        "median_best_positive_rank_candidate": float(group.best_positive_rank_candidate.median()),
    })
    for name, col in METRICS.items():
        base = pd.to_numeric(group[f"{col}_base"], errors="coerce").astype(float)
        cand = pd.to_numeric(group[f"{col}_candidate"], errors="coerce").astype(float)
        out[name] = {
            "base": float(base.mean()),
            "candidate": float(cand.mean()),
            "delta": float((cand - base).mean()),
            "positive_query_fraction": float(((cand - base) > 0).mean()),
            "negative_query_fraction": float(((cand - base) < 0).mean()),
            "zero_query_fraction": float(((cand - base) == 0).mean()),
        }
    return out


def paired_bootstrap(joined: pd.DataFrame, *, replicates: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(joined), size=(replicates, len(joined)))
    result: dict[str, object] = {}
    for name, col in METRICS.items():
        delta = pd.to_numeric(joined[f"{col}_candidate"], errors="coerce").to_numpy(float) - pd.to_numeric(joined[f"{col}_base"], errors="coerce").to_numpy(float)
        draws = delta[indices].mean(axis=1)
        result[name] = {
            "delta": float(delta.mean()),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "bootstrap_probability_delta_gt_zero": float((draws > 0).mean()),
        }
    rank_improvement = pd.to_numeric(joined.best_positive_rank_base, errors="coerce").to_numpy(float) - pd.to_numeric(joined.best_positive_rank_candidate, errors="coerce").to_numpy(float)
    draws = rank_improvement[indices].mean(axis=1)
    result["mean_best_positive_rank_improvement"] = {
        "delta": float(rank_improvement.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "bootstrap_probability_delta_gt_zero": float((draws > 0).mean()),
    }
    return result


def group_report(joined: pd.DataFrame, column: str, order: list[str]) -> dict[str, object]:
    return {value: aggregate(joined[joined[column].astype(str).eq(value)]) for value in order if (joined[column].astype(str) == value).any()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Descriptive-only failure audit for the frozen Rhea128→141 external reveal.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--difficulty-dir", type=Path, default=DEFAULT_DIFFICULTY)
    parser.add_argument("--center-audit", type=Path, default=DEFAULT_CENTER_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    root = args.root.resolve()
    base_path = root / "eval_base" / CELL / "query_metrics.csv"
    candidate_path = root / "eval_candidate" / CELL / "query_metrics.csv"
    result_path = root / "external_result.json"
    manifest_path = root / CELL / "manifest.json"
    external = json.loads(result_path.read_text())
    if external.get("status") != "failed_fresh_external_snapshot_no_retuning" or external.get("model_selection_allowed_after_external_reveal") is not False:
        raise ValueError("external result is not the frozen failed/no-retuning reveal")
    baseline = _r2e(base_path)
    candidate = _r2e(candidate_path)
    reaction_slices = pd.read_csv(args.difficulty_dir.resolve() / "reaction_slices.csv", dtype=str).fillna("")
    pair_slices = pd.read_csv(args.difficulty_dir.resolve() / "pair_slices.csv", dtype=str).fillna("")
    center_audit = pd.read_csv(args.center_audit.resolve(), dtype={"reaction_id": str}).fillna("")
    joined = merge_query_audit(baseline, candidate, reaction_slices, pair_slices, center_audit)
    if len(joined) != 208:
        raise ValueError(f"expected 208 frozen queries, got {len(joined)}")
    out = {
        "name": "cleanroom_r2e_rhea128_to141_external_v2_failure_audit",
        "status": "posthoc_external_descriptive_only",
        "posthoc_external_descriptive_only": True,
        "model_selection_allowed": False,
        "no_router_threshold_or_hyperparameter_selection": True,
        "frozen_external_result": str(result_path.relative_to(ROOT)),
        "frozen_external_result_sha256": sha256_file(result_path),
        "baseline_query_metrics_sha256": sha256_file(base_path),
        "candidate_query_metrics_sha256": sha256_file(candidate_path),
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "difficulty_summary_sha256": sha256_file(args.difficulty_dir.resolve() / "summary.json"),
        "n_queries": int(len(joined)),
        "bootstrap": {"replicates": int(args.bootstrap_replicates), "seed": int(args.seed), "metrics": paired_bootstrap(joined, replicates=args.bootstrap_replicates, seed=args.seed)},
        "overall": aggregate(joined),
        "by_reaction_similarity": group_report(joined, "reaction_similarity_bucket", ["lt0p3", "0p3_0p5", "0p5_0p7", "0p7_0p9", "ge0p9"]),
        "by_positive_count": group_report(joined, "positive_count_bucket", ["1", "2", "3-5", "6-10", ">10"]),
        "by_reaction_center_status": group_report(joined, "center_status", ["valid", "zero_fallback", "missing_audit"]),
        "by_max_positive_protein_identity": group_report(joined, "max_positive_identity_bucket", ["all_no_hit", "lt20", "20_40", "40_60", "60_80", "ge80"]),
        "audit_counts": {
            "reaction_similarity_bucket": joined.reaction_similarity_bucket.value_counts().to_dict(),
            "positive_count_bucket": joined.positive_count_bucket.value_counts().to_dict(),
            "reaction_center_status": joined.center_status.value_counts().to_dict(),
            "max_positive_identity_bucket": joined.max_positive_identity_bucket.value_counts().to_dict(),
            "queries_with_any_no_hit_positive": int((joined.n_no_hit_positives > 0).sum()),
            "queries_with_all_no_hit_positives": int((joined.n_no_hit_positives == joined.n_positives).sum()),
        },
        "interpretation_guard": "These revealed-target strata may explain transfer failure but may not define a router, threshold, feature gate, loss, candidate family, or hyperparameter. Any next model must be selected only on a separately frozen internal development protocol.",
    }
    args.output.resolve().write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
