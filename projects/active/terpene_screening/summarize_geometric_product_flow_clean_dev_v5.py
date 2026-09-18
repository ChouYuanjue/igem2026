from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from projects.active.terpene_screening import run_r2e_lambdarank_fusion_v1 as base
from projects.active.terpene_screening.broad_rhea_metrics import (
    DEFAULT_BUDGETS,
    DEFAULT_TOP_PERCENTS,
    summarize_query_metrics,
)

V5 = ROOT / "results/geometric_product_flow_clean_dev_v5"
V1 = ROOT / "results/geometric_product_flow_clean_dev_v1"
BIME = ROOT / "results/bime_rank_unified_v1/r2e_clipzyme_expert_v1"
SEED = 20260917
BOOTSTRAPS = 20000

METRIC_COLUMNS = {
    "mrr": "reciprocal_rank",
    "map": "average_precision",
    "macro_roc_auc": "roc_auc",
    "ndcg_at_10": "ndcg_at_10",
    "hit_at_10": "hit_at_10",
    "hit_at_20": "hit_at_20",
    "hit_at_50": "hit_at_50",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metric_map(frame: pd.DataFrame) -> dict[str, float]:
    m = summarize_query_metrics(frame, budgets=DEFAULT_BUDGETS, top_percents=DEFAULT_TOP_PERCENTS)
    keep = [
        "mrr", "map", "macro_roc_auc", "ndcg_at_10", "hit_at_10", "hit_at_20",
        "hit_at_50", "median_best_positive_rank",
    ]
    return {k: float(m[k]) for k in keep}


def bootstrap(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, object]:
    aa = a.sort_values(["fold", "query_id"]).reset_index(drop=True)
    bb = b.sort_values(["fold", "query_id"]).reset_index(drop=True)
    if not aa[["fold", "query_id"]].equals(bb[["fold", "query_id"]]):
        raise RuntimeError("paired query keys differ")
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(aa), size=(BOOTSTRAPS, len(aa)), dtype=np.int32)
    out: dict[str, object] = {}
    for name, column in METRIC_COLUMNS.items():
        delta = aa[column].to_numpy(np.float64) - bb[column].to_numpy(np.float64)
        sampled = delta[indices].mean(axis=1)
        out[name] = {
            "delta": float(delta.mean()),
            "ci95": [float(x) for x in np.quantile(sampled, [0.025, 0.975])],
            "p_delta_gt_0": float(np.mean(sampled > 0)),
            "improve": int(np.sum(delta > 1e-12)),
            "tie": int(np.sum(np.abs(delta) <= 1e-12)),
            "worse": int(np.sum(delta < -1e-12)),
        }
    return out


def main() -> None:
    frames = []
    convergence = []
    folds = []
    leakage = []
    for fold in range(3):
        path = V5 / f"fold{fold}" / "full_full"
        summary = json.loads((path / "summary.json").read_text())
        q = pd.read_csv(path / "flow_query_metrics.csv")
        q.insert(0, "fold", fold) if "fold" not in q else None
        d = pd.read_csv(path / "diagnostics.csv")
        frames.append(q)
        converged = d.final_relative_change.to_numpy(np.float64) <= 1e-5 + 1e-12
        convergence.append({
            "fold": fold,
            "queries": int(len(d)),
            "converged": int(converged.sum()),
            "hit_max_steps": int((d.flow_steps >= int(summary["max_steps"])).sum()),
            "median_steps": float(d.flow_steps.median()),
            "p90_steps": float(d.flow_steps.quantile(0.9)),
            "max_final_relative_change": float(d.final_relative_change.max()),
        })
        folds.append({"fold": fold, "metrics": metric_map(q)})

        split = base.DEV_ROOT / "baseline_base" / f"fold{fold}"
        tr = pd.read_csv(split / "training_pairs.csv", dtype=str).fillna("")
        dv = pd.read_csv(split / "dev_pairs.csv", dtype=str).fillna("")
        tr_pairs = set(zip(tr.reaction_id, tr.protein_id))
        dv_pairs = set(zip(dv.reaction_id, dv.protein_id))
        leakage.append({
            "fold": fold,
            "train_reactions": int(tr.reaction_id.nunique()),
            "dev_reactions": int(dv.reaction_id.nunique()),
            "reaction_overlap": int(len(set(tr.reaction_id) & set(dv.reaction_id))),
            "pair_overlap": int(len(tr_pairs & dv_pairs)),
        })

    v5 = pd.concat(frames, ignore_index=True).sort_values(["fold", "query_id"]).reset_index(drop=True)
    v1 = pd.read_csv(V1 / "development_oof_query_metrics.csv").sort_values(["fold", "query_id"]).reset_index(drop=True)
    bime = pd.read_csv(BIME / "development_oof_query_metrics.csv").sort_values(["fold", "query_id"]).reset_index(drop=True)
    if len(v5) != 1903:
        raise RuntimeError(f"expected 1903 v5 queries, got {len(v5)}")
    if not v5[["fold", "query_id"]].equals(v1[["fold", "query_id"]]):
        raise RuntimeError("v5/v1 query keys differ")
    if not v5[["fold", "query_id"]].equals(bime[["fold", "query_id"]]):
        raise RuntimeError("v5/BiME query keys differ")
    if any(x["reaction_overlap"] or x["pair_overlap"] for x in leakage):
        raise RuntimeError("train/dev leakage detected")

    v5.to_csv(V5 / "development_oof_query_metrics.csv", index=False)
    result = {
        "method": "dual_axis_intrinsic_product_geometry_charbonnier_bregman_field_v5",
        "mathematical_object": "one scalar compatibility field F(r,e) on product manifold M_R x M_E; positives are an empirical measure; inference is intrinsic-geometry-constrained variational extension",
        "selection_scope": "clean-dev only; one-time external retention was not rerun or used for v5 selection",
        "query_count": int(len(v5)),
        "candidate_count": 185918,
        "pooled_metrics": metric_map(v5),
        "v1_metrics": metric_map(v1),
        "bime_metrics": metric_map(bime),
        "delta_v5_vs_v1": {k: metric_map(v5)[k] - metric_map(v1)[k] for k in metric_map(v5)},
        "delta_v5_vs_bime": {k: metric_map(v5)[k] - metric_map(bime)[k] for k in metric_map(v5)},
        "folds": folds,
        "convergence": convergence,
        "leakage_audit": leakage,
        "paired_bootstrap_v5_vs_v1": bootstrap(v5, v1),
        "paired_bootstrap_v5_vs_bime": bootstrap(v5, bime),
        "provenance_sha256": {
            "evaluator": sha256(ROOT / "projects/active/terpene_screening/evaluate_geometric_product_flow_clean_dev_v5.py"),
            "multiscale_geometry": sha256(ROOT / "projects/active/terpene_screening/multiscale_geometry.py"),
            "product_field": sha256(ROOT / "projects/active/terpene_screening/geometric_product_field.py"),
            "pair_measure": sha256(ROOT / "projects/active/terpene_screening/geometric_pair_measure.py"),
            "v1_oof": sha256(V1 / "development_oof_query_metrics.csv"),
            "bime_oof": sha256(BIME / "development_oof_query_metrics.csv"),
        },
    }
    (V5 / "development_result.json").write_text(json.dumps(result, indent=2) + "\n")
    (V5 / "paired_bootstrap_vs_v1.json").write_text(json.dumps(result["paired_bootstrap_v5_vs_v1"], indent=2) + "\n")
    (V5 / "paired_bootstrap_vs_bime.json").write_text(json.dumps(result["paired_bootstrap_v5_vs_bime"], indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
