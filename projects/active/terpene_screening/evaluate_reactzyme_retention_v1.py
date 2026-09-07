from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = ROOT / "projects/active/terpene_screening/REACTZYME_NATIVE_SUPPORT_ADAPTATION_V1.json"
DEFAULT_BASELINE = ROOT / "results/cleanroom_internal_full_candidate_rdkitplus_v1"
DEFAULT_DIFFICULTY = ROOT / "results/cleanroom_internal_full_candidate_difficulty_v1"
DEFAULT_CANDIDATE = ROOT / "results/reactzyme_native_support_adaptation_v1/eval"
DEFAULT_OUTPUT = ROOT / "results/reactzyme_native_support_adaptation_v1/retention_selection.json"
MEAN_COLUMNS = {
    "mrr": "reciprocal_rank",
    "map": "average_precision",
    "macro_roc_auc": "roc_auc",
    "ndcg_at_10": "ndcg_at_10",
    "hit_at_10": "hit_at_10",
    "hit_at_20": "hit_at_20",
    "hit_at_50": "hit_at_50",
}


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        raise ValueError("Cannot summarize an empty query frame")
    out = {name: float(pd.to_numeric(frame[column], errors="raise").mean()) for name, column in MEAN_COLUMNS.items()}
    out["median_best_positive_rank"] = float(pd.to_numeric(frame["best_positive_rank"], errors="raise").median())
    out["n_queries"] = int(len(frame))
    return out


def compare_frames(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, object]:
    if baseline["query_id"].duplicated().any() or candidate["query_id"].duplicated().any():
        raise ValueError("query_id must be unique within a direction/scope")
    base_ids = set(baseline["query_id"].astype(str))
    cand_ids = set(candidate["query_id"].astype(str))
    if base_ids != cand_ids:
        raise RuntimeError(f"Query support mismatch: baseline={len(base_ids)} candidate={len(cand_ids)}")
    base = summarize(baseline)
    cand = summarize(candidate)
    delta = {key: float(cand[key] - base[key]) for key in MEAN_COLUMNS}
    delta["median_best_positive_rank"] = float(cand["median_best_positive_rank"] - base["median_best_positive_rank"])
    return {"baseline": base, "candidate": cand, "delta": delta}


def scope_pass(comparison: dict[str, object], max_drop: dict[str, float], median_relative: float, median_floor: float) -> tuple[bool, list[str]]:
    delta = comparison["delta"]
    base = comparison["baseline"]
    failures = []
    for metric, allowed in max_drop.items():
        if float(delta[metric]) < -float(allowed) - 1e-12:
            failures.append(f"{metric}_drop_exceeds_{allowed}")
    allowed_rank_increase = max(float(median_floor), float(base["median_best_positive_rank"]) * float(median_relative))
    if float(delta["median_best_positive_rank"]) > allowed_rank_increase + 1e-12:
        failures.append(f"median_rank_increase_exceeds_{allowed_rank_increase}")
    return not failures, failures


def load_direction(path: Path, direction: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"query_id": str})
    return frame.loc[frame["direction"].astype(str).eq(direction)].copy()


def pooled_policy(policy: str, protocol: dict, baseline_root: Path, candidate_root: Path, difficulty_root: Path) -> dict[str, object]:
    pooled = {"r2e_all": [], "e2r_all": [], "r2e_lt0p3": [], "e2r_no_hit": []}
    fold_checks = []
    for fold in (0, 1, 2):
        cell = f"clean2023_internal_double_cold_fold{fold}"
        base = baseline_root / cell / "query_metrics.csv"
        cand = candidate_root / policy / f"fold{fold}" / "query_metrics.csv"
        if not cand.is_file():
            raise FileNotFoundError(cand)
        br = load_direction(base, "reaction_to_enzyme")
        cr = load_direction(cand, "reaction_to_enzyme")
        be = load_direction(base, "enzyme_to_reaction")
        ce = load_direction(cand, "enzyme_to_reaction")
        fold_r = compare_frames(br, cr)
        fold_e = compare_frames(be, ce)
        fold_failures = []
        for label, cmp in (("r2e", fold_r), ("e2r", fold_e)):
            for metric in ("mrr", "map"):
                if float(cmp["delta"][metric]) < -0.005 - 1e-12:
                    fold_failures.append(f"{label}_{metric}_drop_gt_0.005")
        fold_checks.append({"fold": fold, "r2e": fold_r, "e2r": fold_e, "pass": not fold_failures, "failures": fold_failures})
        br["fold"] = cr["fold"] = be["fold"] = ce["fold"] = fold
        pooled["r2e_all"].append((br, cr))
        pooled["e2r_all"].append((be, ce))
        reaction_slices = pd.read_csv(difficulty_root / cell / "reaction_slices.csv", dtype={"reaction_id": str})
        protein_slices = pd.read_csv(difficulty_root / cell / "protein_slices.csv", dtype={"protein_id": str})
        low_ids = set(reaction_slices.loc[reaction_slices["reaction_similarity_bucket"].eq("lt0p3"), "reaction_id"].astype(str))
        nohit_ids = set(protein_slices.loc[protein_slices["protein_identity_bucket"].eq("no_hit"), "protein_id"].astype(str))
        pooled["r2e_lt0p3"].append((br.loc[br.query_id.isin(low_ids)], cr.loc[cr.query_id.isin(low_ids)]))
        pooled["e2r_no_hit"].append((be.loc[be.query_id.isin(nohit_ids)], ce.loc[ce.query_id.isin(nohit_ids)]))

    comparisons = {}
    for scope, pairs in pooled.items():
        b = pd.concat([x[0] for x in pairs], ignore_index=True)
        c = pd.concat([x[1] for x in pairs], ignore_index=True)
        # fold-qualified IDs prevent accidental collision checks across distinct dev folds.
        b["query_id"] = b["fold"].astype(str) + ":" + b["query_id"].astype(str)
        c["query_id"] = c["fold"].astype(str) + ":" + c["query_id"].astype(str)
        comparisons[scope] = compare_frames(b, c)

    expected = protocol["retention_evaluation"]["expected_pooled_support"]
    for scope, count in expected.items():
        if int(comparisons[scope]["baseline"]["n_queries"]) != int(count):
            raise RuntimeError(f"Unexpected pooled support for {scope}: {comparisons[scope]['baseline']['n_queries']} != {count}")

    evaluation = protocol["retention_evaluation"]
    failures = []
    for scope in ("r2e_all", "e2r_all"):
        ok, why = scope_pass(
            comparisons[scope], evaluation["all_query_max_absolute_drop"],
            evaluation["all_query_median_rank_guard"]["max_relative_increase"],
            evaluation["all_query_median_rank_guard"]["absolute_increase_floor"],
        )
        if not ok:
            failures.extend(f"{scope}:{x}" for x in why)
    for scope in ("r2e_lt0p3", "e2r_no_hit"):
        spec = evaluation["hard_slice_guards"][scope]
        ok, why = scope_pass(
            comparisons[scope], spec["max_absolute_drop"],
            spec["median_rank_max_relative_increase"], spec["median_rank_absolute_increase_floor"],
        )
        if not ok:
            failures.extend(f"{scope}:{x}" for x in why)
    for check in fold_checks:
        if not check["pass"]:
            failures.extend(f"fold{check['fold']}:{x}" for x in check["failures"])
    return {"policy": policy, "pass": not failures, "failures": failures, "pooled": comparisons, "fold_checks": fold_checks}


def select_policy(results: dict[str, dict[str, object]]) -> str | None:
    if results["union_safe_max"]["pass"]:
        return "union_safe_max"
    if results["enzyme_safe"]["pass"]:
        return "enzyme_safe"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--difficulty-root", type=Path, default=DEFAULT_DIFFICULTY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    results = {
        policy: pooled_policy(policy, protocol, args.baseline_root, args.candidate_root, args.difficulty_root)
        for policy in ("union_safe_max", "enzyme_safe")
    }
    selected = select_policy(results)
    payload = {
        "status": "selected" if selected else "rejected_both_safe_policies",
        "selected_policy": selected,
        "selection_priority": ["union_safe_max", "enzyme_safe", "unchanged_current_model"],
        "external_metrics_used": False,
        "retuning_allowed": False,
        "policies": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
