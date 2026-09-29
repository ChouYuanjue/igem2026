from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.fibre.kernel.evidence_fusion import (  # noqa: E402
    AnchoredEvidenceFusion,
    EvidenceChannelSpec,
)
from projects.active.fibre.runtime.ranking_metrics import (  # noqa: E402
    evaluate_full_candidate_ranks,
    summarize_query_metrics,
)


ASSET_ROOT = Path("/home/s241850073/igem2026")
BASE_CACHE_ROOT = ASSET_ROOT / "results/r2e_lambdarank_fusion_v1/prepared"
CLIP_CACHE_ROOT = (
    ASSET_ROOT / "results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/prepared"
)
OUT = ROOT / "results/fibre_evidence_structure_r2e_v2"
FOLDS = (0, 1, 2)
POOL_K = 100
PREFIX_K = 100
SEED = 20261002

BASE_FEATURE_NAMES = [
    "primary_raw_score", "secondary_raw_score",
    "primary_query_zscore", "secondary_query_zscore",
    "primary_log_rank_fraction", "secondary_log_rank_fraction",
    "primary_reciprocal_rank", "secondary_reciprocal_rank",
    "zscore_difference", "log_rank_difference", "best_log_rank", "worst_log_rank",
    "fallback_log_rank_fraction", "alternate_log_rank_fraction",
    "fallback_zscore", "alternate_zscore",
    "primary_top10", "secondary_top10", "primary_top50", "secondary_top50",
    "primary_top200", "secondary_top200",
    "max_train_binary_drfp_tanimoto", "low_similarity_router_flag",
]
EXTRA_FEATURE_NAMES = [
    "clip_raw_score", "clip_query_zscore", "clip_log_rank_fraction",
    "clip_reciprocal_rank", "clip_candidate_supported", "clip_query_supported",
    "clip_top10", "clip_top50", "clip_top100", "clip_z_minus_fallback",
    "clip_logrank_minus_fallback", "top10_votes3", "top50_votes3",
    "top100_votes3", "best3_log_rank", "best3_zscore",
]
FEATURE_NAMES = BASE_FEATURE_NAMES + EXTRA_FEATURE_NAMES
IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def stable_seed(text: str) -> int:
    return (
        int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")
        % (2**31 - 1)
    )


def load_cache(root: Path, fold: int) -> dict[str, object]:
    p = root / f"fold{fold}"
    with np.load(p / "cache.npz") as z:
        arrays = {name: z[name] for name in z.files}
    queries = pd.read_csv(p / "queries.csv", dtype=str)["query_id"].astype(str).tolist()
    return {"z": arrays, "queries": queries}


def filtered_base(cache: dict[str, object]) -> dict[str, object]:
    z = cache["z"]
    ptr = z["query_ptr"]
    xs, ys, rows, qptr = [], [], [], [0]
    for qi in range(len(ptr) - 1):
        a, b = int(ptr[qi]), int(ptr[qi + 1])
        keep = (
            (z["primary_ranks"][a:b] <= POOL_K)
            | (z["secondary_ranks"][a:b] <= POOL_K)
        )
        xs.append(z["X"][a:b][keep])
        ys.append(z["labels"][a:b][keep])
        rows.append(z["candidate_rows"][a:b][keep])
        qptr.append(qptr[-1] + int(keep.sum()))
    return {
        "X": np.concatenate(xs),
        "y": np.concatenate(ys),
        "rows": np.concatenate(rows),
        "query_ptr": np.asarray(qptr, dtype=np.int64),
        "queries": cache["queries"],
    }


def training_matrix(caches: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    xs, ys, groups = [], [], []
    hard_col = BASE_FEATURE_NAMES.index("best_log_rank")
    for cache in caches:
        f = filtered_base(cache)
        ptr = f["query_ptr"]
        for qi, query_id in enumerate(f["queries"]):
            a, b = int(ptr[qi]), int(ptr[qi + 1])
            x = f["X"][a:b]
            y = f["y"][a:b]
            rows = f["rows"][a:b]
            positives = np.flatnonzero(y > 0)
            negatives = np.flatnonzero(y == 0)
            if len(positives) == 0:
                continue
            hard_order = negatives[np.lexsort((rows[negatives], x[negatives, hard_col]))]
            hard = hard_order[: min(128, len(hard_order))]
            remaining = hard_order[len(hard):]
            if len(remaining):
                rng = np.random.default_rng(
                    stable_seed(f"base-train|{SEED}|{query_id}")
                )
                random = rng.choice(
                    remaining, size=min(32, len(remaining)), replace=False
                ).astype(np.int64)
            else:
                random = np.empty(0, dtype=np.int64)
            keep = np.concatenate([positives, hard, random])
            xs.append(x[keep])
            ys.append(y[keep].astype(np.float32))
            groups.append(len(keep))
    return np.concatenate(xs), np.concatenate(ys), groups


def train_base_ranker(caches: list[dict[str, object]], holdout: int) -> xgb.Booster:
    x, y, groups = training_matrix(caches)
    dm = xgb.DMatrix(x, label=y)
    dm.set_group(groups)
    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "tree_method": "hist",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_depth": 2,
        "eta": 0.12,
        "min_child_weight": 5.0,
        "lambda": 1.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambdarank_pair_method": "topk",
        "lambdarank_num_pair_per_sample": 20,
        "seed": stable_seed(f"base-ranker|{SEED}|{holdout}"),
        "verbosity": 0,
    }
    return xgb.train(params, dm, num_boost_round=80)


def zscore_by_query(values: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    for qi in range(len(ptr) - 1):
        a, b = int(ptr[qi]), int(ptr[qi + 1])
        local = values[a:b].astype(np.float64)
        std = max(float(local.std()), 1e-6)
        out[a:b] = ((local - float(local.mean())) / std).astype(np.float32)
    return out


def build_core_oof(
    base_caches: dict[int, dict[str, object]],
    clip_caches: dict[int, dict[str, object]],
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for holdout in FOLDS:
        model = train_base_ranker(
            [base_caches[f] for f in FOLDS if f != holdout],
            holdout,
        )
        z = clip_caches[holdout]["z"]
        raw = model.predict(xgb.DMatrix(z["X"][:, : len(BASE_FEATURE_NAMES)]))
        out[holdout] = zscore_by_query(raw.astype(np.float32), z["query_ptr"])
        print(
            json.dumps(
                {
                    "stage": "base_oof",
                    "holdout": holdout,
                    "rows": int(len(raw)),
                }
            ),
            flush=True,
        )
    return out


def structure_available(x: np.ndarray) -> np.ndarray:
    return (
        (x[:, IDX["clip_candidate_supported"]] > 0.5)
        & (x[:, IDX["clip_query_supported"]] > 0.5)
    )


def pair_differences(
    caches: list[tuple[dict[str, object], np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor]:
    core_diffs, structure_diffs = [], []
    for cache, core_all in caches:
        z = cache["z"]
        ptr = z["query_ptr"]
        for qi, query_id in enumerate(cache["queries"]):
            a, b = int(ptr[qi]), int(ptr[qi + 1])
            x = z["X"][a:b]
            y = z["labels"][a:b]
            fallback = z["fallback_ranks"][a:b]
            positives = np.flatnonzero(y > 0)
            negatives = np.flatnonzero(y == 0)
            if len(positives) == 0 or len(negatives) == 0:
                continue
            hard = negatives[
                np.lexsort((z["candidate_rows"][a:b][negatives], fallback[negatives]))
            ][: min(96, len(negatives))]
            remaining = np.setdiff1d(negatives, hard, assume_unique=False)
            if len(remaining):
                rng = np.random.default_rng(
                    stable_seed(f"evidence-train|{SEED}|{query_id}")
                )
                random = rng.choice(
                    remaining, size=min(32, len(remaining)), replace=False
                ).astype(np.int64)
            else:
                random = np.empty(0, dtype=np.int64)
            neg = np.concatenate([hard, random])
            core = core_all[a:b]
            clip = (
                x[:, IDX["clip_query_zscore"]].astype(np.float32, copy=False)
                * structure_available(x).astype(np.float32)
            )
            for p in positives:
                core_diffs.append(core[p] - core[neg])
                structure_diffs.append(clip[p] - clip[neg])
    return (
        torch.as_tensor(np.concatenate(core_diffs), dtype=torch.float32),
        torch.as_tensor(np.concatenate(structure_diffs), dtype=torch.float32),
    )


def train_evidence(
    caches: list[tuple[dict[str, object], np.ndarray]],
) -> AnchoredEvidenceFusion:
    core_diff, structure_diff = pair_differences(caches)
    model = AnchoredEvidenceFusion(
        (EvidenceChannelSpec("clipzyme_structure", "structural"),),
        initial_strength=0.05,
        initial_quality_slope=0.05,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    evidence = structure_diff[:, None]
    available = torch.ones_like(evidence, dtype=torch.bool)
    quality = torch.zeros_like(evidence)
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        fused, _ = model(core_diff, evidence, available, quality)
        weight = model.channel_weights(quality[:1])[0, 0]
        loss = F.softplus(-fused).mean() + 1e-3 * weight.square()
        loss.backward()
        optimizer.step()
        model.project_nonnegative_()
    return model


def evaluate(
    cache: dict[str, object],
    core_all: np.ndarray,
    *,
    fold: int,
    evidence: AnchoredEvidenceFusion | None,
) -> pd.DataFrame:
    z = cache["z"]
    ptr = z["query_ptr"]
    pptr = z["pos_ptr"]
    lex = z["lexical_rank"]
    rows = []
    for qi, query_id in enumerate(cache["queries"]):
        a, b = int(ptr[qi]), int(ptr[qi + 1])
        pa, pb = int(pptr[qi]), int(pptr[qi + 1])
        candidate_rows = z["candidate_rows"][a:b]
        fallback = z["fallback_ranks"][a:b]
        core = core_all[a:b]
        x = z["X"][a:b]
        clip = x[:, IDX["clip_query_zscore"]].astype(np.float32, copy=False)
        clip_ok = structure_available(x)
        if evidence is None:
            local = core
            strength = 0.0
        else:
            with torch.no_grad():
                fused, diag = evidence(
                    torch.as_tensor(core),
                    torch.as_tensor(clip[:, None]),
                    torch.as_tensor(clip_ok[:, None]),
                )
            local = fused.numpy()
            strength = float(diag["channel_weights"][0, 0])
        order = np.lexsort((lex[candidate_rows], -local))
        take = order[: min(PREFIX_K, len(order))]
        selected_rows = candidate_rows[take]
        selected_fallback = fallback[take]
        selected_position = {
            int(row): rank + 1 for rank, row in enumerate(selected_rows)
        }
        ranks = []
        for row, old_rank in zip(
            z["positive_rows"][pa:pb],
            z["positive_fallback_ranks"][pa:pb],
            strict=True,
        ):
            row = int(row)
            old_rank = int(old_rank)
            if row in selected_position:
                rank = selected_position[row]
            else:
                removed_before = int(np.count_nonzero(selected_fallback < old_rank))
                rank = len(selected_rows) + old_rank - removed_before
            ranks.append(rank)
        metrics = evaluate_full_candidate_ranks(
            np.asarray(ranks, dtype=np.int64),
            len(lex),
        )
        rows.append(
            {
                "fold": fold,
                "query_id": query_id,
                "structure_supported_fraction": float(clip_ok.mean()),
                "structure_strength": strength,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def compact(summary: dict[str, object]) -> dict[str, float]:
    keys = (
        "mrr",
        "map",
        "ndcg_at_10",
        "hit_at_10",
        "hit_at_20",
        "hit_at_50",
        "median_best_positive_rank",
    )
    return {key: float(summary[key]) for key in keys}


def main() -> None:
    base_caches = {f: load_cache(BASE_CACHE_ROOT, f) for f in FOLDS}
    clip_caches = {f: load_cache(CLIP_CACHE_ROOT, f) for f in FOLDS}
    core_oof = build_core_oof(base_caches, clip_caches)

    base_frames, fused_frames = [], []
    learned = {}
    for holdout in FOLDS:
        model = train_evidence(
            [(clip_caches[f], core_oof[f]) for f in FOLDS if f != holdout]
        )
        base_frames.append(
            evaluate(clip_caches[holdout], core_oof[holdout], fold=holdout, evidence=None)
        )
        fused_frames.append(
            evaluate(
                clip_caches[holdout],
                core_oof[holdout],
                fold=holdout,
                evidence=model,
            )
        )
        learned[str(holdout)] = {
            "structure_strength": float(model.strength.detach()[0])
        }
        print(json.dumps({"holdout": holdout, **learned[str(holdout)]}), flush=True)

    base_frame = pd.concat(base_frames, ignore_index=True)
    fused_frame = pd.concat(fused_frames, ignore_index=True)
    base_summary = summarize_query_metrics(base_frame)
    fused_summary = summarize_query_metrics(fused_frame)
    final_model = train_evidence(
        [(clip_caches[f], core_oof[f]) for f in FOLDS]
    )
    final_structure_strength = float(final_model.strength.detach()[0])
    historical = json.loads(
        (
            ASSET_ROOT
            / "results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/development_result.json"
        ).read_text()
    )

    payload = {
        "schema": "fibre-anchored-evidence-r2e-v2-development",
        "status": "crossfit_complete",
        "method": "frozen learned core plus anchored additive structure evidence",
        "core": "OOF LambdaMART over ESM-C and EnzGFM candidate evidence",
        "scientific_evidence": "CLIPZyme structure",
        "trainable_evidence_parameters_per_fit": 1,
        "external_or_strict_temporal_metrics_used": False,
        "learned_parameters": learned,
        "final_structure_strength_fit_on_all_internal_oof": final_structure_strength,
        "oof_core": compact(base_summary),
        "oof_core_plus_structure": compact(fused_summary),
        "delta": {
            key: float(fused_summary[key]) - float(base_summary[key])
            for key in ("mrr", "map", "ndcg_at_10", "hit_at_10", "hit_at_20", "hit_at_50")
        },
        "historical_same_fold_reference": {
            "lambda_without_clip": historical["old"],
            "lambda_with_clip": historical["new"],
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    base_frame.to_csv(OUT / "core_oof_query_metrics.csv", index=False)
    fused_frame.to_csv(OUT / "core_plus_structure_oof_query_metrics.csv", index=False)
    (OUT / "development_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
