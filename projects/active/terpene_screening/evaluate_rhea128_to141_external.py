from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "projects/active/terpene_screening/CLEANROOM_R2E_RHEA128_TO141_EXTERNAL_V1.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    def mean(column: str) -> float:
        return float(pd.to_numeric(frame[column], errors="coerce").mean())
    return {
        "n_queries": int(len(frame)),
        "candidate_count_unique": sorted(map(int, pd.to_numeric(frame["candidate_count"], errors="raise").unique())),
        "mrr": mean("reciprocal_rank"),
        "map": mean("average_precision"),
        "macro_roc_auc": float(pd.to_numeric(frame["roc_auc"], errors="coerce").dropna().mean()),
        "ndcg_at_10": mean("ndcg_at_10"),
        "hit_at_10": mean("hit_at_10"),
        "hit_at_20": mean("hit_at_20"),
        "hit_at_50": mean("hit_at_50"),
        "median_best_positive_rank": float(pd.to_numeric(frame["best_positive_rank"], errors="coerce").dropna().median()),
        "mean_best_positive_rank_fraction": mean("best_positive_rank_fraction"),
    }


def delta(base: dict[str, object], candidate: dict[str, object]) -> dict[str, float]:
    keys = [
        "mrr", "map", "macro_roc_auc", "ndcg_at_10", "hit_at_10", "hit_at_20", "hit_at_50",
        "median_best_positive_rank", "mean_best_positive_rank_fraction",
    ]
    return {key: float(candidate[key]) - float(base[key]) for key in keys}


def evaluate(
    manifest: dict[str, object],
    base: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    min_queries: int,
    min_pairs: int,
    candidate_count: int,
) -> dict[str, object]:
    audit = dict(manifest.get("audit") or {})
    if int(audit.get("exact_train_test_pair_overlap", -1)) != 0:
        raise ValueError("External benchmark has train/test pair overlap")
    if int(audit.get("train_test_protein_overlap", -1)) != 0:
        raise ValueError("External benchmark has train/test protein overlap")
    if int(audit.get("train_test_reaction_overlap", -1)) != 0:
        raise ValueError("External benchmark has train/test reaction overlap")
    n_queries = int(audit.get("test_query_reactions", 0))
    n_pairs = int(audit.get("test_pairs", 0))
    powered = n_queries >= int(min_queries) and n_pairs >= int(min_pairs)

    base = base[base["direction"].eq("reaction_to_enzyme")].copy()
    candidate = candidate[candidate["direction"].eq("reaction_to_enzyme")].copy()
    if base["query_id"].duplicated().any() or candidate["query_id"].duplicated().any():
        raise ValueError("R2E query IDs must be unique")
    if set(base["query_id"]) != set(candidate["query_id"]):
        raise ValueError("Baseline/candidate query support differs")
    if len(base) != n_queries:
        raise ValueError(f"Metric query count {len(base)} != benchmark query count {n_queries}")
    base = base.sort_values("query_id").reset_index(drop=True)
    candidate = candidate.sort_values("query_id").reset_index(drop=True)
    if not base["query_id"].equals(candidate["query_id"]):
        raise ValueError("Paired query order differs")
    for name, frame in (("baseline", base), ("candidate", candidate)):
        counts = set(map(int, pd.to_numeric(frame["candidate_count"], errors="raise").unique()))
        if counts != {int(candidate_count)}:
            raise ValueError(f"{name} candidate universe mismatch: {counts}")

    baseline = summarize(base)
    residual = summarize(candidate)
    checks = {
        "mrr_strict_improve": float(residual["mrr"]) > float(baseline["mrr"]),
        "map_strict_improve": float(residual["map"]) > float(baseline["map"]),
        "ndcg10_no_regress": float(residual["ndcg_at_10"]) >= float(baseline["ndcg_at_10"]),
        "macro_auc_no_regress": float(residual["macro_roc_auc"]) >= float(baseline["macro_roc_auc"]),
        "hit10_no_regress": float(residual["hit_at_10"]) >= float(baseline["hit_at_10"]),
        "hit50_no_regress": float(residual["hit_at_50"]) >= float(baseline["hit_at_50"]),
        "median_rank_decrease": float(residual["median_best_positive_rank"]) < float(baseline["median_best_positive_rank"]),
    }
    if not powered:
        status = "underpowered_external_descriptive"
        passed = False
        decision = "underpowered_no_external_promotion"
    else:
        passed = all(checks.values())
        status = "passed_fresh_external_snapshot" if passed else "failed_fresh_external_snapshot_no_retuning"
        decision = status
    return {
        "minimum_support": {
            "required_query_reactions": int(min_queries), "observed_query_reactions": n_queries,
            "required_test_pairs": int(min_pairs), "observed_test_pairs": n_pairs,
            "met": powered,
        },
        "baseline": baseline,
        "candidate": residual,
        "delta": delta(baseline, residual),
        "checks": checks,
        "pass": passed,
        "status": status,
        "decision": decision,
        "model_selection_allowed_after_external_reveal": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen Rhea release128→141 external evaluation rule.")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cell = str(protocol["benchmark_cell"])
    manifest_path = args.benchmark_root.resolve() / cell / "manifest.json"
    base_path = args.baseline_root.resolve() / cell / "query_metrics.csv"
    candidate_path = args.candidate_root.resolve() / cell / "query_metrics.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("name") != cell:
        raise ValueError("Benchmark manifest cell differs from frozen protocol")
    result = evaluate(
        manifest,
        pd.read_csv(base_path, dtype={"query_id": str}),
        pd.read_csv(candidate_path, dtype={"query_id": str}),
        min_queries=int(protocol["minimum_support_rule"]["min_query_reactions"]),
        min_pairs=int(protocol["minimum_support_rule"]["min_test_pairs"]),
        candidate_count=int(protocol["fixed_support"]["expected_protein_candidates"]),
    )
    result.update({
        "protocol": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL), "benchmark_cell": cell,
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "baseline_query_metrics_sha256": sha256_file(base_path),
        "candidate_query_metrics_sha256": sha256_file(candidate_path),
        "fresh_external_snapshot": True,
    })
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
