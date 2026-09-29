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

from projects.active.fibre.runtime.ranking_metrics import (
    evaluate_full_candidate_ranks,
    summarize_query_metrics,
)


CACHE_ROOT = (
    ROOT.parent
    / "igem2026"
    / "results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/prepared"
)
OUT = ROOT / "results/fibre_evidence_fusion_structure_v1"
FOLDS = (0, 1, 2)
PREFIX_K = 100

BASE_FEATURES = [
    "primary_raw_score",
    "secondary_raw_score",
    "primary_query_zscore",
    "secondary_query_zscore",
    "primary_log_rank_fraction",
    "secondary_log_rank_fraction",
    "primary_reciprocal_rank",
    "secondary_reciprocal_rank",
    "zscore_difference",
    "log_rank_difference",
    "best_log_rank",
    "worst_log_rank",
    "fallback_log_rank_fraction",
    "alternate_log_rank_fraction",
    "fallback_zscore",
    "alternate_zscore",
    "primary_top10",
    "secondary_top10",
    "primary_top50",
    "secondary_top50",
    "primary_top200",
    "secondary_top200",
    "max_train_binary_drfp_tanimoto",
    "low_similarity_router_flag",
]
EXTRA_FEATURES = [
    "clip_raw_score",
    "clip_query_zscore",
    "clip_log_rank_fraction",
    "clip_reciprocal_rank",
    "clip_candidate_supported",
    "clip_query_supported",
    "clip_top10",
    "clip_top50",
    "clip_top100",
    "clip_z_minus_fallback",
    "clip_logrank_minus_fallback",
    "top10_votes3",
    "top50_votes3",
    "top100_votes3",
    "best3_log_rank",
    "best3_zscore",
]
FEATURES = [*BASE_FEATURES, *EXTRA_FEATURES]
IDX = {name: i for i, name in enumerate(FEATURES)}


def load_fold(fold: int) -> dict[str, object]:
    path = CACHE_ROOT / f"fold{fold}"
    with np.load(path / "cache.npz") as z:
        arrays = {name: z[name] for name in z.files}
    queries = (
        pd.read_csv(path / "queries.csv", dtype=str)["query_id"].astype(str).tolist()
    )
    if arrays["X"].shape[1] != len(FEATURES):
        raise RuntimeError("prepared feature contract drifted")
    return {"arrays": arrays, "queries": queries}


def training_differences(cache: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = cache["arrays"]
    ptr = z["query_ptr"]
    d_core: list[np.ndarray] = []
    d_struct: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    for qi in range(len(ptr) - 1):
        start, stop = int(ptr[qi]), int(ptr[qi + 1])
        x = z["X"][start:stop]
        y = z["labels"][start:stop] > 0
        pos = np.flatnonzero(y)
        neg = np.flatnonzero(~y)
        if len(pos) == 0 or len(neg) == 0:
            continue

        core = x[:, IDX["fallback_zscore"]].astype(np.float64, copy=False)
        struct = x[:, IDX["clip_query_zscore"]].astype(np.float64, copy=False)
        available = (
            x[:, IDX["clip_candidate_supported"]] > 0.5
        ) & (
            x[:, IDX["clip_query_supported"]] > 0.5
        )
        struct = np.where(available, struct, 0.0)

        dc = (core[pos, None] - core[None, neg]).reshape(-1)
        ds = (struct[pos, None] - struct[None, neg]).reshape(-1)
        # Give every query equal total weight, independent of how many positives
        # or shortlist negatives it happens to contain.
        w = np.full(
            len(dc),
            1.0 / (len(pos) * len(neg)),
            dtype=np.float64,
        )
        d_core.append(dc)
        d_struct.append(ds)
        weights.append(w)

    return np.concatenate(d_core), np.concatenate(d_struct), np.concatenate(weights)


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
        # d/d alpha log(1+exp(-m)) = -ds * sigmoid(-m)
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


def reranked_positive_ranks(
    cache: dict[str, object],
    alpha: float,
) -> pd.DataFrame:
    z = cache["arrays"]
    ptr = z["query_ptr"]
    pos_ptr = z["pos_ptr"]
    lex = z["lexical_rank"]
    rows_out: list[dict[str, object]] = []

    for qi, query in enumerate(cache["queries"]):
        start, stop = int(ptr[qi]), int(ptr[qi + 1])
        x = z["X"][start:stop]
        candidate_rows = z["candidate_rows"][start:stop]
        fallback_ranks = z["fallback_ranks"][start:stop]

        core = x[:, IDX["fallback_zscore"]].astype(np.float64, copy=False)
        struct = x[:, IDX["clip_query_zscore"]].astype(np.float64, copy=False)
        available = (
            x[:, IDX["clip_candidate_supported"]] > 0.5
        ) & (
            x[:, IDX["clip_query_supported"]] > 0.5
        )
        fused = core + alpha * np.where(available, struct, 0.0)

        order = np.lexsort((lex[candidate_rows], -fused))
        take = order[: min(PREFIX_K, len(order))]
        promoted_rows = candidate_rows[take]
        promoted_fallback = fallback_ranks[take]
        promoted_position = {
            int(row): rank + 1 for rank, row in enumerate(promoted_rows)
        }

        pstart, pstop = int(pos_ptr[qi]), int(pos_ptr[qi + 1])
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
            if row in promoted_position:
                new_rank = promoted_position[row]
            else:
                removed_before = int(np.count_nonzero(promoted_fallback < fallback_rank))
                new_rank = len(promoted_rows) + fallback_rank - removed_before
            new_ranks.append(new_rank)

        metrics = evaluate_full_candidate_ranks(
            np.asarray(new_ranks, dtype=np.int64),
            len(lex),
        )
        rows_out.append(
            {
                "query_id": str(query),
                "structure_strength": alpha,
                **metrics,
            }
        )
    return pd.DataFrame(rows_out)


def baseline_metrics(cache: dict[str, object]) -> pd.DataFrame:
    z = cache["arrays"]
    pos_ptr = z["pos_ptr"]
    rows: list[dict[str, object]] = []
    for qi, query in enumerate(cache["queries"]):
        start, stop = int(pos_ptr[qi]), int(pos_ptr[qi + 1])
        ranks = z["positive_fallback_ranks"][start:stop]
        rows.append(
            {
                "query_id": str(query),
                **evaluate_full_candidate_ranks(ranks, len(z["lexical_rank"])),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    caches = {fold: load_fold(fold) for fold in FOLDS}
    OUT.mkdir(parents=True, exist_ok=True)

    oof_rows: list[pd.DataFrame] = []
    baseline_rows: list[pd.DataFrame] = []
    fold_results: dict[str, object] = {}
    for holdout in FOLDS:
        alpha = fit_strength([caches[f] for f in FOLDS if f != holdout])
        candidate = reranked_positive_ranks(caches[holdout], alpha)
        candidate["fold"] = holdout
        baseline = baseline_metrics(caches[holdout])
        baseline["fold"] = holdout
        oof_rows.append(candidate)
        baseline_rows.append(baseline)
        fold_results[str(holdout)] = {
            "structure_strength": alpha,
            "baseline": summarize_query_metrics(baseline),
            "candidate": summarize_query_metrics(candidate),
        }
        print(
            json.dumps(
                {
                    "holdout": holdout,
                    "structure_strength": alpha,
                    "baseline_mrr": fold_results[str(holdout)]["baseline"]["mrr"],
                    "candidate_mrr": fold_results[str(holdout)]["candidate"]["mrr"],
                }
            ),
            flush=True,
        )

    oof = pd.concat(oof_rows, ignore_index=True)
    baseline = pd.concat(baseline_rows, ignore_index=True)
    oof.to_csv(OUT / "oof_query_metrics.csv", index=False)
    baseline.to_csv(OUT / "baseline_query_metrics.csv", index=False)

    baseline_summary = summarize_query_metrics(baseline)
    candidate_summary = summarize_query_metrics(oof)
    keys = ("mrr", "map", "ndcg_at_10", "hit_at_10", "hit_at_20", "hit_at_50")
    delta = {
        key: float(candidate_summary[key] - baseline_summary[key])
        for key in keys
    }

    final_alpha = fit_strength([caches[f] for f in FOLDS])
    result = {
        "schema": "fibre-anchored-structure-evidence-v1",
        "status": "internal_crossfit_only",
        "question": (
            "Can a frozen current R2E retrieval order accept CLIPZyme structural "
            "evidence through one non-negative anchored residual coefficient, "
            "without a learned tree router or softmax competition?"
        ),
        "design": {
            "core": "current clean-development R2E fallback/router score",
            "evidence": "CLIPZyme query-standardized structural score",
            "missing": "exact zero contribution; base and other evidence never renormalise",
            "coefficient": "one non-negative scalar fit by pairwise logistic ranking loss",
            "candidate_pool": "existing union of top-100 primary, secondary and CLIPZyme candidates",
            "prefix_reranked": PREFIX_K,
            "tail": "exact frozen fallback order",
            "selection": "three-fold OOF; no strict-temporal/external labels used",
        },
        "folds": fold_results,
        "pooled": {
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "delta": delta,
        },
        "final_internal_structure_strength": final_alpha,
    }
    (OUT / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
