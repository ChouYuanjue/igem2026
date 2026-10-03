from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.bridge.runtime.ranking_metrics import (
    evaluate_full_candidate_ranks,
    summarize_query_metrics,
)


CACHE_ROOT = (
    ROOT.parent
    / "igem2026/results/bime_rank_unified_v1/r2e_seed_context_v1/prepared"
)
OUT = ROOT / "results/fibre_evidence_fusion_seed_v1"
FOLDS = (0, 1, 2)
PREFIX_K = 100

FEATURES = [
    "base_log_rank_fraction",
    "base_reciprocal_rank",
    "base_top10",
    "base_top50",
    "base_top100",
    "seed_raw_score",
    "seed_query_zscore",
    "seed_log_rank_fraction",
    "seed_reciprocal_rank",
    "seed_top10",
    "seed_top50",
    "seed_top100",
    "best_log_rank",
    "rank_gap_seed_minus_base",
    "both_top10",
    "both_top50",
    "both_top100",
]
IDX = {name: i for i, name in enumerate(FEATURES)}


def load_fold(fold: int) -> dict[str, object]:
    path = CACHE_ROOT / f"fold{fold}"
    with np.load(path / "cache.npz") as z:
        arrays = {name: z[name] for name in z.files}
    trials = pd.read_csv(path / "trials.csv", dtype=str)["trial_id"].astype(str).tolist()
    if arrays["X"].shape[1] != len(FEATURES):
        raise RuntimeError("seed-context feature contract drifted")
    return {"arrays": arrays, "trials": trials}


def _scores(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # The frozen zero-shot BIME order is encoded by base rank.  Smaller
    # log-rank fraction is better, so negate it to obtain a ranking score.
    core = -x[:, IDX["base_log_rank_fraction"]].astype(np.float64, copy=False)
    seed = x[:, IDX["seed_query_zscore"]].astype(np.float64, copy=False)
    return core, seed


def training_differences(cache: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = cache["arrays"]
    ptr = z["query_ptr"]
    d_core: list[np.ndarray] = []
    d_seed: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    for i in range(len(ptr) - 1):
        start, stop = int(ptr[i]), int(ptr[i + 1])
        x = z["X"][start:stop]
        y = z["labels"][start:stop] > 0
        pos = np.flatnonzero(y)
        neg = np.flatnonzero(~y)
        if len(pos) == 0 or len(neg) == 0:
            continue
        core, seed = _scores(x)
        dc = (core[pos, None] - core[None, neg]).reshape(-1)
        ds = (seed[pos, None] - seed[None, neg]).reshape(-1)
        w = np.full(len(dc), 1.0 / (len(pos) * len(neg)), dtype=np.float64)
        d_core.append(dc)
        d_seed.append(ds)
        weights.append(w)

    return np.concatenate(d_core), np.concatenate(d_seed), np.concatenate(weights)


def fit_strength(caches: list[dict[str, object]]) -> float:
    parts = [training_differences(cache) for cache in caches]
    dc = np.concatenate([part[0] for part in parts])
    ds = np.concatenate([part[1] for part in parts])
    w = np.concatenate([part[2] for part in parts])
    w = w / w.sum()

    def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        alpha = float(value[0])
        margin = dc + alpha * ds
        loss = np.logaddexp(0.0, -margin)
        value_out = float(np.sum(w * loss))
        grad = -np.sum(w * ds / (1.0 + np.exp(np.clip(margin, -60.0, 60.0))))
        return value_out, np.asarray([grad], dtype=np.float64)

    result = minimize(
        lambda x: objective(x)[0],
        x0=np.asarray([0.1], dtype=np.float64),
        jac=lambda x: objective(x)[1],
        bounds=[(0.0, None)],
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 200},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.x[0])


def rerank(cache: dict[str, object], alpha: float) -> pd.DataFrame:
    z = cache["arrays"]
    ptr = z["query_ptr"]
    pptr = z["pos_ptr"]
    lex = z["lexical_rank"]
    candidate_count = len(lex) - 1
    rows_out: list[dict[str, object]] = []

    for i, trial in enumerate(cache["trials"]):
        start, stop = int(ptr[i]), int(ptr[i + 1])
        x = z["X"][start:stop]
        rows = z["candidate_rows"][start:stop]
        fallback = z["fallback_ranks"][start:stop]
        core, seed = _scores(x)
        fused = core + alpha * seed

        order = np.lexsort((lex[rows], -fused))
        take = order[: min(PREFIX_K, len(order))]
        selected = rows[take]
        selected_fallback = fallback[take]
        selected_position = {int(row): rank + 1 for rank, row in enumerate(selected)}

        pstart, pstop = int(pptr[i]), int(pptr[i + 1])
        positive_rows = z["positive_rows"][pstart:pstop]
        positive_fallback = z["positive_fallback_ranks"][pstart:pstop]
        new_ranks: list[int] = []
        for row, fallback_rank in zip(
            positive_rows,
            positive_fallback,
            strict=True,
        ):
            row = int(row)
            fallback_rank = int(fallback_rank)
            if row in selected_position:
                new_rank = selected_position[row]
            else:
                removed_before = int(np.count_nonzero(selected_fallback < fallback_rank))
                new_rank = len(selected) + fallback_rank - removed_before
            new_ranks.append(new_rank)

        query = str(trial).split("|seed=", 1)[0]
        seed_id = str(trial).split("|seed=", 1)[1].split("|rep=", 1)[0]
        rows_out.append(
            {
                "trial_id": str(trial),
                "query_id": query,
                "seed_id": seed_id,
                "seed_strength": alpha,
                **evaluate_full_candidate_ranks(
                    np.asarray(new_ranks, dtype=np.int64),
                    candidate_count,
                ),
            }
        )
    return pd.DataFrame(rows_out)


def baseline(cache: dict[str, object]) -> pd.DataFrame:
    z = cache["arrays"]
    pptr = z["pos_ptr"]
    candidate_count = len(z["lexical_rank"]) - 1
    rows: list[dict[str, object]] = []
    for i, trial in enumerate(cache["trials"]):
        start, stop = int(pptr[i]), int(pptr[i + 1])
        ranks = z["positive_fallback_ranks"][start:stop]
        query = str(trial).split("|seed=", 1)[0]
        seed_id = str(trial).split("|seed=", 1)[1].split("|rep=", 1)[0]
        rows.append(
            {
                "trial_id": str(trial),
                "query_id": query,
                "seed_id": seed_id,
                **evaluate_full_candidate_ranks(ranks, candidate_count),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    caches = {fold: load_fold(fold) for fold in FOLDS}
    OUT.mkdir(parents=True, exist_ok=True)
    candidates: list[pd.DataFrame] = []
    baselines: list[pd.DataFrame] = []
    fold_result: dict[str, object] = {}

    for holdout in FOLDS:
        alpha = fit_strength([caches[f] for f in FOLDS if f != holdout])
        candidate = rerank(caches[holdout], alpha)
        candidate["fold"] = holdout
        base = baseline(caches[holdout])
        base["fold"] = holdout
        candidates.append(candidate)
        baselines.append(base)
        fold_result[str(holdout)] = {
            "seed_strength": alpha,
            "baseline": summarize_query_metrics(base),
            "candidate": summarize_query_metrics(candidate),
        }
        print(
            json.dumps(
                {
                    "holdout": holdout,
                    "seed_strength": alpha,
                    "baseline_mrr": fold_result[str(holdout)]["baseline"]["mrr"],
                    "candidate_mrr": fold_result[str(holdout)]["candidate"]["mrr"],
                }
            ),
            flush=True,
        )

    candidate_all = pd.concat(candidates, ignore_index=True)
    baseline_all = pd.concat(baselines, ignore_index=True)
    candidate_all.to_csv(OUT / "oof_query_metrics.csv", index=False)
    baseline_all.to_csv(OUT / "baseline_query_metrics.csv", index=False)

    base_summary = summarize_query_metrics(baseline_all)
    candidate_summary = summarize_query_metrics(candidate_all)
    keys = ("mrr", "map", "ndcg_at_10", "hit_at_10", "hit_at_20", "hit_at_50")
    delta = {
        key: float(candidate_summary[key] - base_summary[key])
        for key in keys
    }
    final_alpha = fit_strength([caches[f] for f in FOLDS])

    result = {
        "schema": "fibre-anchored-seed-evidence-v1",
        "status": "internal_crossfit_only",
        "question": (
            "Can one known-positive enzyme be attached to the frozen zero-shot "
            "R2E ranking as a single non-negative context-evidence residual?"
        ),
        "design": {
            "core": "frozen zero-shot structural BiME order with the seed masked",
            "evidence": "ESM-C cosine-to-known-positive, query-standardized",
            "coefficient": "one non-negative scalar fit by pairwise logistic ranking loss",
            "candidate_pool": "existing union of top-100 zero-shot and seed-context candidates",
            "prefix_reranked": PREFIX_K,
            "tail": "exact frozen zero-shot order",
            "selection": "three-fold OOF; no strict-temporal/external labels used",
        },
        "folds": fold_result,
        "pooled": {
            "baseline": base_summary,
            "candidate": candidate_summary,
            "delta": delta,
        },
        "final_internal_seed_strength": final_alpha,
    }
    (OUT / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
