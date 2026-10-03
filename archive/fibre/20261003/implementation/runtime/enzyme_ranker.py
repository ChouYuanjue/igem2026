from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import xgboost as xgb

FEATURE_NAMES = [
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

@dataclass(frozen=True)
class FusionRuntimeResult:
    full_order: np.ndarray
    priority_scores: np.ndarray
    primary_ranks: np.ndarray
    secondary_ranks: np.ndarray
    fallback_ranks: np.ndarray
    learned_rows: np.ndarray
    learned_scores: np.ndarray
    fallback_is_secondary: bool
    union_size: int
    prefix_size: int


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lexical_rank(candidate_ids: list[str]) -> np.ndarray:
    values = np.asarray(candidate_ids, dtype=object)
    order = np.argsort(values, kind="stable")
    rank = np.empty(len(values), dtype=np.int32)
    rank[order] = np.arange(len(values), dtype=np.int32)
    return rank


def full_order(scores: np.ndarray, lex_rank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((lex_rank, -values)).astype(np.int32)
    inverse = np.empty(len(order), dtype=np.int32)
    inverse[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return order, inverse


def build_features(primary_scores: np.ndarray, secondary_scores: np.ndarray, rows: np.ndarray,
                   primary_ranks: np.ndarray, secondary_ranks: np.ndarray,
                   fallback_is_secondary: bool, similarity: float) -> np.ndarray:
    n = len(primary_scores)
    p_mean = float(primary_scores.mean()); p_std = max(float(primary_scores.std()), 1e-6)
    s_mean = float(secondary_scores.mean()); s_std = max(float(secondary_scores.std()), 1e-6)
    pz = (primary_scores[rows] - p_mean) / p_std
    sz = (secondary_scores[rows] - s_mean) / s_std
    pr = primary_ranks[rows].astype(np.float64); sr = secondary_ranks[rows].astype(np.float64)
    denom = math.log1p(n); plog = np.log1p(pr) / denom; slog = np.log1p(sr) / denom
    if fallback_is_secondary:
        flog, alog, fz, az = slog, plog, sz, pz
    else:
        flog, alog, fz, az = plog, slog, pz, sz
    result = np.column_stack([
        primary_scores[rows], secondary_scores[rows], pz, sz, plog, slog, 1.0/pr, 1.0/sr,
        pz-sz, plog-slog, np.minimum(plog,slog), np.maximum(plog,slog), flog, alog, fz, az,
        (pr<=10).astype(np.float32), (sr<=10).astype(np.float32),
        (pr<=50).astype(np.float32), (sr<=50).astype(np.float32),
        (pr<=200).astype(np.float32), (sr<=200).astype(np.float32),
        np.full(len(rows),float(similarity),dtype=np.float32),
        np.full(len(rows),float(fallback_is_secondary),dtype=np.float32),
    ]).astype(np.float32,copy=False)
    if result.shape[1] != len(FEATURE_NAMES): raise AssertionError((result.shape,len(FEATURE_NAMES)))
    return result


@lru_cache(maxsize=4)
def load_ranker(bundle: str, expected_sha256: str) -> tuple[xgb.Booster, dict[str, object]]:
    directory = Path(bundle).resolve(); ranker_path = directory / "ranker.json"; config_path = directory / "config.json"
    if sha256_file(ranker_path) != str(expected_sha256): raise RuntimeError("Frozen R2E LambdaRank runtime hash mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(config.get("config_id")) != "cfg_07_392fe119": raise RuntimeError("Unexpected R2E LambdaRank runtime configuration")
    booster = xgb.Booster(); booster.load_model(ranker_path)
    return booster, config


def fuse_r2e_scores(primary_scores: np.ndarray, secondary_scores: np.ndarray, candidate_ids: list[str], *,
                    similarity: float, threshold: float, ranker_bundle: Path, ranker_sha256: str,
                    expected_pool_k: int = 100, expected_prefix_k: int = 100) -> FusionRuntimeResult:
    primary_scores = np.asarray(primary_scores,dtype=np.float32); secondary_scores = np.asarray(secondary_scores,dtype=np.float32)
    if primary_scores.shape != secondary_scores.shape or primary_scores.shape != (len(candidate_ids),):
        raise ValueError("R2E LambdaRank source scores/candidate IDs do not align")
    lex = lexical_rank(candidate_ids); p_order,p_inv = full_order(primary_scores,lex); s_order,s_inv = full_order(secondary_scores,lex)
    booster,config = load_ranker(str(ranker_bundle.resolve()),str(ranker_sha256))
    pool_k=int(config["pool_k"]); prefix_k=int(config["prefix_k"])
    if pool_k != int(expected_pool_k) or prefix_k != int(expected_prefix_k):
        raise RuntimeError("R2E LambdaRank runtime pool/prefix differs from confirmed configuration")
    fallback_is_secondary = float(similarity) < float(threshold)
    fallback_order = s_order if fallback_is_secondary else p_order; fallback_ranks = s_inv if fallback_is_secondary else p_inv
    union = np.unique(np.concatenate([p_order[:pool_k],s_order[:pool_k]])).astype(np.int32)
    features = build_features(primary_scores,secondary_scores,union,p_inv,s_inv,fallback_is_secondary,float(similarity))
    predictions = booster.predict(xgb.DMatrix(features)); learned_idx = np.lexsort((lex[union],-predictions))
    learned_rows = union[learned_idx]; learned_scores = np.asarray(predictions[learned_idx],dtype=np.float32)
    selected = learned_rows[:min(prefix_k,len(learned_rows))]
    selected_mask = np.zeros(len(candidate_ids),dtype=bool); selected_mask[selected]=True
    tail = fallback_order[~selected_mask[fallback_order]]; order = np.concatenate([selected,tail]).astype(np.int32,copy=False)
    if len(order)!=len(candidate_ids) or len(np.unique(order))!=len(candidate_ids): raise AssertionError("R2E LambdaRank did not construct a permutation")
    priority=np.empty(len(candidate_ids),dtype=np.float32)
    if len(candidate_ids)==1: priority[order]=1.0
    else: priority[order]=1.0-np.arange(len(candidate_ids),dtype=np.float32)/float(len(candidate_ids)-1)
    return FusionRuntimeResult(order,priority,p_inv,s_inv,fallback_ranks,learned_rows,learned_scores,fallback_is_secondary,int(len(union)),int(len(selected)))
