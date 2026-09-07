from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_ROOT = ROOT / "results/cleanroom_internal_functional_prototype_residual_v1"
DEFAULT_DIFFICULTY_ROOT = ROOT / "results/cleanroom_internal_full_candidate_difficulty_v1"
DEFAULT_PROTOCOL = ROOT / "projects/active/terpene_screening/CLEANROOM_R2E_FUNCTIONAL_PROTOTYPE_RESIDUAL_V1.json"

MEAN_METRICS = {
    "mrr": "reciprocal_rank",
    "map": "average_precision",
    "ndcg_at_10": "ndcg_at_10",
    "macro_roc_auc": "roc_auc",
    "hit_at_10": "hit_at_10",
    "hit_at_20": "hit_at_20",
    "hit_at_50": "hit_at_50",
}
MEDIAN_METRICS = {"median_best_positive_rank": "best_positive_rank"}
TAG_RE = re.compile(r"^scale_(?P<scale>[0-9]+p[0-9]+)__margin_(?P<margin>[0-9]+(?:p[0-9]+)?)_query_metrics\.csv$")
EPS = 1e-12


def _decode(token: str) -> float:
    return float(token.replace("p", "."))


def _metric_summary(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        raise ValueError("cannot summarize empty query set")
    out = {name: float(pd.to_numeric(df[col], errors="raise").mean()) for name, col in MEAN_METRICS.items()}
    out.update({name: float(pd.to_numeric(df[col], errors="raise").median()) for name, col in MEDIAN_METRICS.items()})
    return out


def _delta(candidate: dict[str, float], coarse: dict[str, float]) -> dict[str, float]:
    return {k: float(candidate[k] - coarse[k]) for k in candidate}


def candidate_passes(record: dict) -> tuple[bool, list[str]]:
    p = record["pooled_primary_delta"]
    a = record["pooled_all_delta"]
    reasons: list[str] = []
    strict_pos = {"mrr", "map"}
    nonreg_primary = {"ndcg_at_10", "macro_roc_auc", "hit_at_10", "hit_at_50"}
    nonreg_all = {"mrr", "map", "ndcg_at_10", "hit_at_10", "hit_at_20", "hit_at_50"}
    for k in strict_pos:
        if not p[k] > EPS:
            reasons.append(f"primary_{k}_not_strictly_improved")
    for k in nonreg_primary:
        if p[k] < -EPS:
            reasons.append(f"primary_{k}_regressed")
    if not p["median_best_positive_rank"] < -EPS:
        reasons.append("primary_median_best_positive_rank_not_decreased")
    for k in nonreg_all:
        if a[k] < -EPS:
            reasons.append(f"all_{k}_regressed")

    stable = 0
    for fold in record["folds"]:
        d = fold["primary_delta"]
        if d["mrr"] > EPS and d["map"] > EPS:
            stable += 1
        if d["mrr"] < -0.005 - EPS:
            reasons.append(f"fold{fold['fold']}_primary_mrr_regressed_gt_0p005")
        if d["map"] < -0.005 - EPS:
            reasons.append(f"fold{fold['fold']}_primary_map_regressed_gt_0p005")
    if stable < 2:
        reasons.append("primary_mrr_and_map_not_improved_in_2_of_3_folds")
    return not reasons, reasons


def _candidate_sort_key(record: dict) -> tuple:
    p = record["pooled_primary_delta"]
    a = record["pooled_all_delta"]
    return (
        -p["hit_at_10"],
        -p["mrr"],
        -a["mrr"],
        record["residual_scale"],
        -record["confidence_margin"],
    )


def _load_fold(result_root: Path, difficulty_root: Path, fold: int, candidate_file: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fd = result_root / f"fold{fold}"
    coarse = pd.read_csv(fd / "coarse_query_metrics.csv", dtype={"query_id": str})
    cand = pd.read_csv(fd / candidate_file, dtype={"query_id": str})
    slices = pd.read_csv(
        difficulty_root / f"clean2023_internal_double_cold_fold{fold}" / "reaction_slices.csv",
        dtype={"reaction_id": str},
    )
    for name, df in (("coarse", coarse), ("candidate", cand)):
        if "direction" in df.columns:
            df.drop(df.index[~df["direction"].eq("reaction_to_enzyme")], inplace=True)
        if df["query_id"].duplicated().any():
            raise ValueError(f"fold{fold} {name}: duplicate query_id")
    if set(coarse.query_id) != set(cand.query_id):
        raise ValueError(f"fold{fold}: candidate/coarse query set mismatch")
    if slices["reaction_id"].duplicated().any():
        raise ValueError(f"fold{fold}: duplicate reaction_id in difficulty slices")
    bucket = slices.set_index("reaction_id")["reaction_similarity_bucket"]
    coarse = coarse.copy(); cand = cand.copy()
    coarse["reaction_similarity_bucket"] = coarse.query_id.map(bucket)
    cand["reaction_similarity_bucket"] = cand.query_id.map(bucket)
    if coarse["reaction_similarity_bucket"].isna().any() or cand["reaction_similarity_bucket"].isna().any():
        raise ValueError(f"fold{fold}: missing reaction difficulty bucket")
    coarse["fold"] = fold; cand["fold"] = fold
    return coarse, cand, slices


def select(result_root: Path, difficulty_root: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    folds = list(protocol["development"]["folds"])
    if folds != [0, 1, 2]:
        raise ValueError(f"unexpected frozen folds: {folds}")
    if protocol["selection"]["primary_slice"] != "reaction_similarity_bucket == lt0p3":
        raise ValueError("selector only implements the frozen lt0p3 primary slice")

    files = sorted(p.name for p in (result_root / "fold0").glob("scale_*__margin_*_query_metrics.csv"))
    expected = len(protocol["grid"]["residual_scales"]) * len(protocol["grid"]["confidence_margins"])
    if len(files) != expected:
        raise ValueError(f"candidate file count {len(files)} != frozen grid size {expected}")
    records = []
    pooled_coarse_cache: dict[int, pd.DataFrame] = {}
    for filename in files:
        m = TAG_RE.match(filename)
        if not m:
            raise ValueError(f"unparseable candidate filename: {filename}")
        scale = _decode(m.group("scale")); margin = _decode(m.group("margin"))
        if scale not in protocol["grid"]["residual_scales"] or margin not in protocol["grid"]["confidence_margins"]:
            raise ValueError(f"candidate outside frozen grid: {filename}")
        fold_rows = []
        cand_all = []
        coarse_all = []
        for fold in folds:
            coarse, cand, _ = _load_fold(result_root, difficulty_root, fold, filename)
            if fold not in pooled_coarse_cache:
                pooled_coarse_cache[fold] = coarse
            elif not pooled_coarse_cache[fold].equals(coarse):
                raise ValueError(f"fold{fold}: coarse source changed across candidates")
            cpri = coarse[coarse.reaction_similarity_bucket.eq("lt0p3")]
            rpri = cand[cand.reaction_similarity_bucket.eq("lt0p3")]
            if set(cpri.query_id) != set(rpri.query_id):
                raise ValueError(f"fold{fold}: primary query mismatch")
            cs = _metric_summary(cpri); rs = _metric_summary(rpri)
            fold_rows.append({
                "fold": fold,
                "primary_query_count": int(len(cpri)),
                "primary_coarse": cs,
                "primary_candidate": rs,
                "primary_delta": _delta(rs, cs),
            })
            coarse_all.append(coarse); cand_all.append(cand)
        pc = pd.concat(coarse_all, ignore_index=True)
        pr = pd.concat(cand_all, ignore_index=True)
        pc_primary = pc[pc.reaction_similarity_bucket.eq("lt0p3")]
        pr_primary = pr[pr.reaction_similarity_bucket.eq("lt0p3")]
        coarse_primary = _metric_summary(pc_primary); cand_primary = _metric_summary(pr_primary)
        coarse_all_s = _metric_summary(pc); cand_all_s = _metric_summary(pr)
        rec = {
            "candidate_file": filename,
            "residual_scale": scale,
            "confidence_margin": margin,
            "pooled_primary_query_count": int(len(pc_primary)),
            "pooled_all_query_count": int(len(pc)),
            "pooled_primary_coarse": coarse_primary,
            "pooled_primary_candidate": cand_primary,
            "pooled_primary_delta": _delta(cand_primary, coarse_primary),
            "pooled_all_coarse": coarse_all_s,
            "pooled_all_candidate": cand_all_s,
            "pooled_all_delta": _delta(cand_all_s, coarse_all_s),
            "folds": fold_rows,
        }
        passed, reasons = candidate_passes(rec)
        rec["passed"] = passed; rec["failure_reasons"] = reasons
        records.append(rec)

    # Ensure every candidate used exactly the same frozen coarse rows and primary support.
    primary_counts = {r["pooled_primary_query_count"] for r in records}
    all_counts = {r["pooled_all_query_count"] for r in records}
    if len(primary_counts) != 1 or len(all_counts) != 1:
        raise ValueError("candidate support mismatch")
    passing = sorted((r for r in records if r["passed"]), key=_candidate_sort_key)
    selected = None
    if passing:
        r = passing[0]
        selected = {
            "candidate_file": r["candidate_file"],
            "residual_scale": r["residual_scale"],
            "confidence_margin": r["confidence_margin"],
            "ranking_key": {
                "pooled_lt0p3_hit_at_10_delta": r["pooled_primary_delta"]["hit_at_10"],
                "pooled_lt0p3_mrr_delta": r["pooled_primary_delta"]["mrr"],
                "pooled_all_mrr_delta": r["pooled_all_delta"]["mrr"],
            },
        }
    return {
        "name": "cleanroom_r2e_functional_prototype_residual_v1_selection",
        "status": "passed_development_gate" if selected else "rejected_no_candidate_passed",
        "target_outer_labels_used": False,
        "target_benchmark_identity_used": False,
        "protocol": str(protocol_path.relative_to(ROOT)),
        "folds": folds,
        "primary_slice": "lt0p3",
        "candidate_count": len(records),
        "passing_candidate_count": len(passing),
        "selected": selected,
        "candidates": records,
        "future_confirmation": protocol["future_confirmation"],
        "post_selection_policy": "If selected, only the already-frozen confirmation salt/dev_fold may be materialized. If none passes, reject V1 with no retuning from these results.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    ap.add_argument("--difficulty-root", type=Path, default=DEFAULT_DIFFICULTY_ROOT)
    ap.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    ap.add_argument("--output", type=Path, default=DEFAULT_RESULT_ROOT / "selection.json")
    args = ap.parse_args()
    out = select(args.result_root, args.difficulty_root, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": out["status"], "candidate_count": out["candidate_count"], "passing_candidate_count": out["passing_candidate_count"], "selected": out["selected"]}, indent=2))


if __name__ == "__main__":
    main()
