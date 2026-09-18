from __future__ import annotations

"""Matched-protocol internal development evaluator for the unified geometry mainline.

BiME-Rank is used only as the comparison target.  This evaluator intentionally does
not call its hard 0.9 router, LambdaRank prefix readout, or availability expert gate.
It builds a smooth compatibility potential from the same underlying representations
and ranks the complete 185,918-protein support directly.

The output also stores a top-k local chart per query.  Those charts are numerical
restrictions of the product geometry for later nonlinear anchored-flow experiments;
they are not a separate benchmark or a routing decision.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening import run_bime_r2e_clipzyme_expert_v1 as clip
from projects.active.terpene_screening import run_r2e_lambdarank_fusion_v1 as base
from projects.active.terpene_screening.broad_rhea_metrics import (
    DEFAULT_BUDGETS,
    DEFAULT_TOP_PERCENTS,
    evaluate_full_candidate_ranks,
    summarize_query_metrics,
)

OUT = ROOT / "results/geometric_compatibility_clean_dev_v1"
FOLDS = (0, 1, 2)
DEFAULT_CHART_K = 512


def _logmeanexp_two(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.logaddexp(a, b) - math.log(2.0)


def _metric_map(frame: pd.DataFrame) -> dict[str, float]:
    m = summarize_query_metrics(frame, budgets=DEFAULT_BUDGETS, top_percents=DEFAULT_TOP_PERCENTS)
    return {
        "mrr": float(m["mrr"]),
        "map": float(m["map"]),
        "macro_roc_auc": float(m["macro_roc_auc"]),
        "ndcg_at_10": float(m["ndcg_at_10"]),
        "hit_at_10": float(m["hit_at_10"]),
        "hit_at_20": float(m["hit_at_20"]),
        "hit_at_50": float(m["hit_at_50"]),
        "median_best_positive_rank": float(m["median_best_positive_rank"]),
    }


def evaluate_fold(fold: int, device_name: str, chart_k: int) -> dict[str, object]:
    device = torch.device(device_name)
    ids, _, reaction_ids, pe0, pe1, re0, re1 = base._load_fold_embeddings(fold, device)
    if len(ids) != 185918:
        raise RuntimeError("clean-dev candidate universe drifted")
    lex = base._lexical_rank(ids)
    pindex = {p: i for i, p in enumerate(ids)}
    rindex = {r: i for i, r in enumerate(reaction_ids)}

    clip_pt, clip_rows, _clip_lookup, clip_rmat, clip_ridx = clip._load_clip_assets(ids, device)
    clip_lex = lex[clip_rows]

    dev = pd.read_csv(base.DEV_ROOT / "baseline_base" / f"fold{fold}" / "dev_pairs.csv", dtype=str).fillna("")
    query_ids = sorted(dev.reaction_id.astype(str).unique())
    positives = dev.groupby("reaction_id").protein_id.apply(lambda x: sorted(set(map(str, x)))).to_dict()
    qrows = [rindex[q] for q in query_ids]

    records: list[dict[str, object]] = []
    chart_query: list[str] = []
    chart_ptr = [0]
    chart_rows: list[np.ndarray] = []
    chart_scores: list[np.ndarray] = []
    chart_primary_z: list[np.ndarray] = []
    chart_secondary_z: list[np.ndarray] = []
    chart_structure_z: list[np.ndarray] = []
    chart_structure_available: list[np.ndarray] = []

    batch = 16
    for st in range(0, len(query_ids), batch):
        stop = min(st + batch, len(query_ids))
        qt = torch.as_tensor(qrows[st:stop], dtype=torch.long, device=device)
        with torch.no_grad():
            s0b = (re0[qt] @ pe0.T).float().cpu().numpy().astype(np.float32, copy=False)
            s1b = (re1[qt] @ pe1.T).float().cpu().numpy().astype(np.float32, copy=False)

        clip_local: list[int] = []
        clip_queries: list[np.ndarray] = []
        for j, q in enumerate(query_ids[st:stop]):
            if q in clip_ridx:
                clip_local.append(j)
                clip_queries.append(np.asarray(clip_rmat[clip_ridx[q]], dtype=np.float32))
        clip_scores: dict[int, np.ndarray] = {}
        if clip_queries:
            cqt = torch.as_tensor(np.stack(clip_queries), dtype=torch.float32, device=device)
            with torch.no_grad():
                cb = (cqt @ clip_pt.T).float().cpu().numpy()
            for k, j in enumerate(clip_local):
                clip_scores[j] = cb[k].astype(np.float32, copy=False)

        for j, q in enumerate(query_ids[st:stop]):
            s0 = s0b[j].astype(np.float64, copy=False)
            s1 = s1b[j].astype(np.float64, copy=False)
            pz = (s0 - float(s0.mean())) / max(float(s0.std()), 1e-6)
            sz = (s1 - float(s1.mean())) / max(float(s1.std()), 1e-6)
            potential = _logmeanexp_two(pz, sz)
            structure_full = np.zeros(len(ids), dtype=np.float64)
            structure_available = np.zeros(len(ids), dtype=bool)
            if j in clip_scores:
                cs = clip_scores[j].astype(np.float64, copy=False)
                cz = (cs - float(cs.mean())) / max(float(cs.std()), 1e-6)
                structure_full[clip_rows] = cz
                structure_available[clip_rows] = True
                # Equal-weight log-mean-exp on the observed fibre only.  Availability
                # itself contributes no positive or negative evidence.
                local = np.logaddexp(np.logaddexp(pz[clip_rows], sz[clip_rows]), cz) - math.log(3.0)
                potential[clip_rows] = local

            order = np.lexsort((lex, -potential)).astype(np.int32)
            inv = np.empty(len(ids), dtype=np.int32)
            inv[order] = np.arange(1, len(ids) + 1, dtype=np.int32)
            prows = np.asarray([pindex[p] for p in positives[q]], dtype=np.int32)
            records.append({"fold": fold, "query_id": q, **evaluate_full_candidate_ranks(inv[prows], len(ids))})

            kk = min(int(chart_k), len(ids))
            top = order[:kk]
            chart_query.append(q)
            chart_rows.append(top)
            chart_scores.append(potential[top].astype(np.float32))
            chart_primary_z.append(pz[top].astype(np.float32))
            chart_secondary_z.append(sz[top].astype(np.float32))
            chart_structure_z.append(structure_full[top].astype(np.float32))
            chart_structure_available.append(structure_available[top])
            chart_ptr.append(chart_ptr[-1] + kk)
        print(f"geometry-base fold={fold} {stop}/{len(query_ids)}", flush=True)

    out = OUT / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / "query_metrics.csv", index=False)
    pd.DataFrame({"query_id": chart_query}).to_csv(out / "queries.csv", index=False)
    np.savez_compressed(
        out / "local_charts.npz",
        query_ptr=np.asarray(chart_ptr, dtype=np.int64),
        candidate_rows=np.concatenate(chart_rows).astype(np.int32),
        potential=np.concatenate(chart_scores).astype(np.float32),
        primary_z=np.concatenate(chart_primary_z).astype(np.float32),
        secondary_z=np.concatenate(chart_secondary_z).astype(np.float32),
        structure_z=np.concatenate(chart_structure_z).astype(np.float32),
        structure_available=np.concatenate(chart_structure_available).astype(bool),
    )
    summary = {
        "method": "availability_neutral_logmeanexp_compatibility",
        "role": "geometry-mainline smooth base field; no BiME hard router or LambdaRank readout",
        "fold": fold,
        "queries": len(query_ids),
        "candidate_count": len(ids),
        "chart_k": int(chart_k),
        "metrics": _metric_map(frame),
        "labels_used_for_scoring": False,
        "external_metrics_used": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, choices=FOLDS, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chart-k", type=int, default=DEFAULT_CHART_K)
    args = ap.parse_args()
    print(json.dumps(evaluate_fold(args.fold, args.device, args.chart_k), indent=2), flush=True)


if __name__ == "__main__":
    main()
