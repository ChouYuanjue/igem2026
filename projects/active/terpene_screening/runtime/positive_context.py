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
    "base_log_rank_fraction", "base_reciprocal_rank", "base_top10", "base_top50", "base_top100",
    "seed_raw_score", "seed_query_zscore", "seed_log_rank_fraction", "seed_reciprocal_rank",
    "seed_top10", "seed_top50", "seed_top100", "best_log_rank", "rank_gap_seed_minus_base",
    "both_top10", "both_top50", "both_top100",
]


@dataclass(frozen=True)
class SeedContextRuntimeResult:
    full_order: np.ndarray
    priority_scores: np.ndarray
    union_size: int
    prefix_size: int
    seed_count: int


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lexical_rank(candidate_ids: list[str]) -> np.ndarray:
    values = np.asarray(candidate_ids, dtype=object)
    order = np.argsort(values, kind="stable")
    rank = np.empty(len(values), dtype=np.int32)
    rank[order] = np.arange(len(values), dtype=np.int32)
    return rank


def masked_order(scores: np.ndarray, lex: np.ndarray, masked_rows: set[int]) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((lex, -np.asarray(scores, dtype=np.float64))).astype(np.int32)
    if masked_rows:
        keep = np.fromiter((int(x) not in masked_rows for x in order), dtype=bool, count=len(order))
        order = order[keep]
    inv = np.full(len(scores), len(order) + 1, dtype=np.int32)
    inv[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return order, inv


def context_features(rows: np.ndarray, base_inv: np.ndarray, seed_scores: np.ndarray, seed_inv: np.ndarray, n: int) -> np.ndarray:
    br = base_inv[rows].astype(np.float64); sr = seed_inv[rows].astype(np.float64)
    denom = math.log1p(n)
    blog = (np.log1p(br) / denom).astype(np.float32); slog = (np.log1p(sr) / denom).astype(np.float32)
    brr = (1.0 / br).astype(np.float32); srr = (1.0 / sr).astype(np.float32)
    mean = float(seed_scores.mean()); std = max(float(seed_scores.std()), 1e-6)
    sz = ((seed_scores[rows] - mean) / std).astype(np.float32)
    b10=(br<=10).astype(np.float32); b50=(br<=50).astype(np.float32); b100=(br<=100).astype(np.float32)
    s10=(sr<=10).astype(np.float32); s50=(sr<=50).astype(np.float32); s100=(sr<=100).astype(np.float32)
    x=np.column_stack([
        blog,brr,b10,b50,b100,seed_scores[rows].astype(np.float32),sz,slog,srr,s10,s50,s100,
        np.minimum(blog,slog),(slog-blog).astype(np.float32),b10*s10,b50*s50,b100*s100,
    ]).astype(np.float32)
    if x.shape[1] != len(FEATURE_NAMES): raise AssertionError((x.shape, len(FEATURE_NAMES)))
    return x


@lru_cache(maxsize=4)
def load_ranker(bundle: str, expected_sha256: str) -> tuple[xgb.Booster, dict[str, object]]:
    directory=Path(bundle).resolve(); ranker_path=directory/'ranker.json'; config_path=directory/'config.json'
    if sha256_file(ranker_path) != str(expected_sha256): raise RuntimeError('Frozen R2E seed-context ranker hash mismatch')
    cfg=json.loads(config_path.read_text())
    if cfg.get('external_metrics_used') is not False: raise RuntimeError('R2E seed-context ranker is not internal-only selected')
    if list(cfg.get('feature_names') or []) != FEATURE_NAMES: raise RuntimeError('R2E seed-context feature contract mismatch')
    booster=xgb.Booster(); booster.load_model(ranker_path)
    return booster,cfg


def fuse_r2e_seed_context(
    base_order: np.ndarray,
    seed_scores: np.ndarray,
    candidate_ids: list[str],
    *,
    seed_ids: list[str],
    ranker_bundle: Path,
    ranker_sha256: str,
    expected_pool_k: int = 100,
    expected_prefix_k: int = 100,
) -> SeedContextRuntimeResult:
    n=len(candidate_ids); base_order=np.asarray(base_order,dtype=np.int32); seed_scores=np.asarray(seed_scores,dtype=np.float32)
    if base_order.shape != (n,) or seed_scores.shape != (n,) or len(np.unique(base_order)) != n:
        raise ValueError('R2E seed-context inputs do not align')
    index={v:i for i,v in enumerate(candidate_ids)}; effective=[str(v) for v in seed_ids if str(v) in index]
    if not effective: raise ValueError('R2E seed-context requires at least one seed in candidate universe')
    masked={index[v] for v in effective}; lex=lexical_rank(candidate_ids)
    base_masked=base_order[np.fromiter((int(x) not in masked for x in base_order),dtype=bool,count=len(base_order))]
    base_inv=np.full(n,len(base_masked)+1,dtype=np.int32); base_inv[base_masked]=np.arange(1,len(base_masked)+1,dtype=np.int32)
    seed_order,seed_inv=masked_order(seed_scores,lex,masked)
    booster,cfg=load_ranker(str(ranker_bundle.resolve()),str(ranker_sha256))
    fixed=dict(cfg.get('fixed_config') or {}); pool_k=int(fixed.get('pool_k',expected_pool_k)); prefix_k=int(fixed.get('prefix_k',expected_prefix_k))
    if pool_k != int(expected_pool_k) or prefix_k != int(expected_prefix_k): raise RuntimeError('R2E seed-context pool/prefix contract mismatch')
    union=np.unique(np.concatenate([base_masked[:pool_k],seed_order[:pool_k]])).astype(np.int32)
    X=context_features(union,base_inv,seed_scores,seed_inv,max(len(base_masked),1))
    pred=booster.predict(xgb.DMatrix(X)); local=np.lexsort((lex[union],-pred))[:min(prefix_k,len(union))]
    selected=union[local]; chosen=np.zeros(n,dtype=bool); chosen[selected]=True
    tail=base_masked[~chosen[base_masked]]; effective_order=np.concatenate([selected,tail]).astype(np.int32,copy=False)
    # Put masked seeds at the end only to maintain a full permutation. The caller still applies the authoritative output mask.
    masked_tail=np.asarray(sorted(masked,key=lambda r:int(lex[r])),dtype=np.int32)
    full=np.concatenate([effective_order,masked_tail]).astype(np.int32,copy=False)
    if len(full)!=n or len(np.unique(full))!=n: raise AssertionError('R2E seed-context did not construct a full permutation')
    priority=np.empty(n,dtype=np.float32)
    priority[full]=1.0 if n==1 else 1.0-np.arange(n,dtype=np.float32)/float(n-1)
    return SeedContextRuntimeResult(full,priority,int(len(union)),int(len(selected)),len(effective))
