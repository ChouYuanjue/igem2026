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
CACHE_ROOT = (
    ASSET_ROOT
    / "results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/prepared"
)
OUT = ROOT / "results/fibre_evidence_structure_r2e_v1"
FOLDS = (0, 1, 2)
PREFIX_K = 100
HARD_NEGATIVES = 96
RANDOM_NEGATIVES = 32
TRAIN_STEPS = 400
LEARNING_RATE = 0.05
L2 = 1e-3
SEED = 20261001

BASE_FEATURE_NAMES = [
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
EXTRA_FEATURE_NAMES = [
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
FEATURE_NAMES = BASE_FEATURE_NAMES + EXTRA_FEATURE_NAMES
IDX = {name: index for index, name in enumerate(FEATURE_NAMES)}


def stable_seed(text: str) -> int:
    return (
        int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")
        % (2**31 - 1)
    )


def load_fold(fold: int) -> dict[str, object]:
    root = CACHE_ROOT / f"fold{fold}"
    with np.load(root / "cache.npz") as z:
        arrays = {name: z[name] for name in z.files}
    queries = (
        pd.read_csv(root / "queries.csv", dtype=str)["query_id"].astype(str).tolist()
    )
    return {"z": arrays, "queries": queries}


def structure_available(x: np.ndarray) -> np.ndarray:
    return (
        (x[:, IDX["clip_candidate_supported"]] > 0.5)
        & (x[:, IDX["clip_query_supported"]] > 0.5)
    )


def build_pair_differences(
    caches: list[dict[str, object]],
) -> tuple[torch.Tensor, torch.Tensor]:
    base_diff: list[np.ndarray] = []
    structure_diff: list[np.ndarray] = []
    for cache in caches:
        z = cache["z"]
        ptr = z["query_ptr"]
        for query_index, query_id in enumerate(cache["queries"]):
            a, b = int(ptr[query_index]), int(ptr[query_index + 1])
            x = z["X"][a:b]
            y = z["labels"][a:b]
            fallback_rank = z["fallback_ranks"][a:b]
            positives = np.flatnonzero(y > 0)
            negatives = np.flatnonzero(y == 0)
            if len(positives) == 0 or len(negatives) == 0:
                continue
            hard = negatives[
                np.lexsort((z["candidate_rows"][a:b][negatives], fallback_rank[negatives]))
            ][: min(HARD_NEGATIVES, len(negatives))]
            remaining = np.setdiff1d(negatives, hard, assume_unique=False)
            if len(remaining):
                rng = np.random.default_rng(
                    stable_seed(f"fibre-evidence|{SEED}|{query_id}")
                )
                random = rng.choice(
                    remaining,
                    size=min(RANDOM_NEGATIVES, len(remaining)),
                    replace=False,
                ).astype(np.int64)
            else:
                random = np.empty(0, dtype=np.int64)
            selected_negatives = np.concatenate([hard, random])

            core = x[:, IDX["fallback_zscore"]].astype(np.float32, copy=False)
            clip = x[:, IDX["clip_query_zscore"]].astype(np.float32, copy=False)
            clip = clip * structure_available(x).astype(np.float32)

            for p in positives:
                base_diff.append(core[p] - core[selected_negatives])
                structure_diff.append(clip[p] - clip[selected_negatives])

    return (
        torch.as_tensor(np.concatenate(base_diff), dtype=torch.float32),
        torch.as_tensor(np.concatenate(structure_diff), dtype=torch.float32),
    )


def train_fusion(caches: list[dict[str, object]]) -> AnchoredEvidenceFusion:
    core_diff, structure_diff = build_pair_differences(caches)
    model = AnchoredEvidenceFusion(
        (EvidenceChannelSpec("clipzyme_structure", "structural"),),
        initial_strength=0.05,
        initial_quality_slope=0.05,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    evidence = structure_diff[:, None]
    available = torch.ones_like(evidence, dtype=torch.bool)
    quality = torch.zeros_like(evidence)
    for _ in range(TRAIN_STEPS):
        optimizer.zero_grad(set_to_none=True)
        fused_diff, _ = model(core_diff, evidence, available, quality)
        weight = model.channel_weights(quality[:1])[0, 0]
        loss = F.softplus(-fused_diff).mean() + L2 * weight.square()
        loss.backward()
        optimizer.step()
        model.project_nonnegative_()
    return model


def query_metrics_from_cache(
    cache: dict[str, object],
    model: AnchoredEvidenceFusion | None,
    *,
    fold: int,
) -> pd.DataFrame:
    z = cache["z"]
    ptr = z["query_ptr"]
    pos_ptr = z["pos_ptr"]
    lex = z["lexical_rank"]
    rows: list[dict[str, object]] = []
    for qi, query_id in enumerate(cache["queries"]):
        a, b = int(ptr[qi]), int(ptr[qi + 1])
        pa, pb = int(pos_ptr[qi]), int(pos_ptr[qi + 1])
        candidate_rows = z["candidate_rows"][a:b]
        fallback_ranks = z["fallback_ranks"][a:b]
        core = z["X"][a:b, IDX["fallback_zscore"]].astype(np.float32, copy=False)
        clip = z["X"][a:b, IDX["clip_query_zscore"]].astype(np.float32, copy=False)
        clip_ok = structure_available(z["X"][a:b])

        if model is None:
            local_score = core
            structure_weight = 0.0
        else:
            with torch.no_grad():
                fused, diag = model(
                    torch.as_tensor(core),
                    torch.as_tensor(clip[:, None]),
                    torch.as_tensor(clip_ok[:, None]),
                )
            local_score = fused.numpy()
            structure_weight = float(diag["channel_weights"][0, 0])

        local_order = np.lexsort((lex[candidate_rows], -local_score))
        take = local_order[: min(PREFIX_K, len(local_order))]
        selected_rows = candidate_rows[take]
        selected_fallback = fallback_ranks[take]
        selected_position = {
            int(row): position + 1 for position, row in enumerate(selected_rows)
        }

        positive_rows = z["positive_rows"][pa:pb]
        positive_fallback = z["positive_fallback_ranks"][pa:pb]
        new_ranks: list[int] = []
        for row, fallback_rank in zip(positive_rows, positive_fallback, strict=True):
            row = int(row)
            fallback_rank = int(fallback_rank)
            if row in selected_position:
                rank = selected_position[row]
            else:
                removed_before = int(np.count_nonzero(selected_fallback < fallback_rank))
                rank = len(selected_rows) + fallback_rank - removed_before
            new_ranks.append(rank)
        metrics = evaluate_full_candidate_ranks(
            np.asarray(new_ranks, dtype=np.int64),
            len(lex),
        )
        rows.append(
            {
                "fold": fold,
                "query_id": query_id,
                "structure_supported_fraction": float(clip_ok.mean()),
                "structure_weight": structure_weight,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def selected_metrics(summary: dict[str, object]) -> dict[str, float]:
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
    caches = {fold: load_fold(fold) for fold in FOLDS}
    baseline_frames = [
        query_metrics_from_cache(caches[fold], None, fold=fold) for fold in FOLDS
    ]
    baseline = pd.concat(baseline_frames, ignore_index=True)
    oof_frames: list[pd.DataFrame] = []
    parameters: dict[str, object] = {}

    for holdout in FOLDS:
        model = train_fusion([caches[f] for f in FOLDS if f != holdout])
        frame = query_metrics_from_cache(caches[holdout], model, fold=holdout)
        oof_frames.append(frame)
        parameters[str(holdout)] = {
            "structure_strength": float(model.strength.detach()[0]),
        }
        print(
            json.dumps(
                {
                    "holdout": holdout,
                    **parameters[str(holdout)],
                    "queries": len(frame),
                }
            ),
            flush=True,
        )

    oof = pd.concat(oof_frames, ignore_index=True)
    baseline_summary = summarize_query_metrics(baseline)
    candidate_summary = summarize_query_metrics(oof)

    current_result = json.loads(
        (
            ASSET_ROOT
            / "results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/development_result.json"
        ).read_text()
    )
    payload = {
        "schema": "fibre-anchored-evidence-r2e-v1-development",
        "status": "crossfit_complete",
        "method": "anchored additive scientific evidence",
        "scientific_evidence": ["CLIPZyme structure"],
        "core": "existing similarity-conditioned ESM-C/EnzGFM fallback ranking",
        "fusion": (
            "core z-score with fixed coefficient 1 plus a non-negative learned "
            "structure z-score correction; unsupported structure contributes exactly 0"
        ),
        "training": {
            "folds": list(FOLDS),
            "pairwise_loss": "logistic",
            "hard_negatives_per_query": HARD_NEGATIVES,
            "random_negatives_per_query": RANDOM_NEGATIVES,
            "train_steps": TRAIN_STEPS,
            "learning_rate": LEARNING_RATE,
            "l2": L2,
            "trainable_parameters_per_fit": 1,
            "external_or_strict_temporal_metrics_used": False,
        },
        "crossfit_parameters": parameters,
        "baseline_fallback": selected_metrics(baseline_summary),
        "anchored_evidence": selected_metrics(candidate_summary),
        "delta_vs_fallback": {
            key: float(candidate_summary[key]) - float(baseline_summary[key])
            for key in (
                "mrr",
                "map",
                "ndcg_at_10",
                "hit_at_10",
                "hit_at_20",
                "hit_at_50",
            )
        },
        "historical_lambdamart_same_folds": {
            "without_clipzyme": current_result["old"],
            "with_clipzyme": current_result["new"],
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(OUT / "fallback_query_metrics.csv", index=False)
    oof.to_csv(OUT / "anchored_evidence_oof_query_metrics.csv", index=False)
    (OUT / "development_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
