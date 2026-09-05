from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from projects.active.terpene_screening.r2e_lambdarank_runtime import (
    build_features as build_base_features,
    full_order,
    fuse_r2e_scores as fuse_base_r2e_scores,
    lexical_rank,
    sha256_file,
)

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


@dataclass(frozen=True)
class BiMER2ERuntimeResult:
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
    structure_expert_applied: bool
    structure_expert_name: str | None
    structure_supported_candidates: int
    structure_query_supported: bool


@dataclass(frozen=True)
class _ClipProteinAsset:
    embeddings_path: Path
    protein_ids: tuple[str, ...]
    candidate_rows: np.ndarray
    manifest_sha256: str


@dataclass(frozen=True)
class _ClipReactionAsset:
    embeddings_path: Path
    row_by_reaction: dict[str, int]
    manifest_sha256: str


@lru_cache(maxsize=4)
def _load_ranker(bundle: str, expected_sha256: str) -> tuple[xgb.Booster, dict[str, object]]:
    directory = Path(bundle).resolve()
    ranker_path = directory / "ranker.json"
    config_path = directory / "config.json"
    if sha256_file(ranker_path) != str(expected_sha256):
        raise RuntimeError("BiME-Rank R2E structural ranker hash mismatch")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("ranker_sha256") and str(cfg["ranker_sha256"]) != str(expected_sha256):
        raise RuntimeError("BiME-Rank R2E structural config/ranker hash mismatch")
    names = list(cfg.get("feature_names") or [])
    if not names or names[-len(EXTRA_FEATURE_NAMES):] != EXTRA_FEATURE_NAMES:
        raise RuntimeError("Unexpected BiME-Rank R2E structural feature schema")
    booster = xgb.Booster()
    booster.load_model(ranker_path)
    return booster, cfg


@lru_cache(maxsize=4)
def _load_clip_protein_asset(root: str, expected_manifest_sha256: str | None) -> _ClipProteinAsset:
    directory = Path(root).resolve()
    manifest_path = directory / "manifest.json"
    actual = sha256_file(manifest_path)
    if expected_manifest_sha256 and actual != str(expected_manifest_sha256):
        raise RuntimeError("CLIPZyme R2E protein asset manifest hash mismatch")
    entries = pd.read_csv(directory / "entries.csv", dtype=str).fillna("")
    rows = entries["candidate_row"].astype(int).to_numpy(np.int32)
    ids = tuple(entries["protein_id"].astype(str))
    embeddings_path = directory / "embeddings.npy"
    matrix = np.load(embeddings_path, mmap_mode="r")
    if len(matrix) != len(rows) or matrix.shape[1] != 1280:
        raise RuntimeError("CLIPZyme R2E protein asset shape mismatch")
    return _ClipProteinAsset(embeddings_path, ids, rows, actual)


@lru_cache(maxsize=4)
def _load_clip_reaction_asset(root: str, expected_manifest_sha256: str | None) -> _ClipReactionAsset:
    directory = Path(root).resolve()
    manifest_path = directory / "manifest.json"
    actual = sha256_file(manifest_path)
    if expected_manifest_sha256 and actual != str(expected_manifest_sha256):
        raise RuntimeError("CLIPZyme reaction asset manifest hash mismatch")
    entries = pd.read_csv(directory / "entries.csv", dtype=str).fillna("")
    supported = entries[entries["clipzyme_supported"].str.lower().eq("true")]
    row_by_reaction = {
        str(r): int(row)
        for r, row in supported[["reaction_id", "row"]].itertuples(index=False)
    }
    embeddings_path = directory / "embeddings.npy"
    matrix = np.load(embeddings_path, mmap_mode="r")
    if matrix.shape[1] != 1280:
        raise RuntimeError("CLIPZyme reaction asset dimension mismatch")
    return _ClipReactionAsset(embeddings_path, row_by_reaction, actual)


@lru_cache(maxsize=4)
def _protein_tensor(embeddings_path: str, device_name: str) -> torch.Tensor:
    matrix = np.load(Path(embeddings_path), mmap_mode="r")
    # Copy before torch conversion: mmap is read-only and torch warns about non-writable arrays.
    return torch.as_tensor(np.array(matrix, dtype=np.float32, copy=True), device=torch.device(device_name))


def _structural_features(
    base_x: np.ndarray,
    rows: np.ndarray,
    clip_scores_supported: np.ndarray,
    clip_inverse_supported: np.ndarray,
    support_lookup: np.ndarray,
    candidate_count: int,
) -> np.ndarray:
    """Append exactly the feature family used by the admitted internal R2E experiment."""
    pos = support_lookup[rows]
    candidate_supported = pos >= 0
    raw = np.zeros(len(rows), dtype=np.float32)
    z = np.zeros(len(rows), dtype=np.float32)
    cr = np.full(len(rows), candidate_count + 1, dtype=np.float64)

    vals = np.asarray(clip_scores_supported, dtype=np.float32)
    mean = float(vals.mean())
    std = max(float(vals.std()), 1e-6)
    raw[candidate_supported] = vals[pos[candidate_supported]]
    z[candidate_supported] = (vals[pos[candidate_supported]] - mean) / std
    cr[candidate_supported] = clip_inverse_supported[pos[candidate_supported]].astype(np.float64)

    clip_denom = math.log1p(max(int(np.count_nonzero(support_lookup >= 0)), 1))
    clog = np.ones(len(rows), dtype=np.float32)
    rr = np.zeros(len(rows), dtype=np.float32)
    clog[candidate_supported] = (np.log1p(cr[candidate_supported]) / clip_denom).astype(np.float32)
    rr[candidate_supported] = (1.0 / cr[candidate_supported]).astype(np.float32)
    c10 = (candidate_supported & (cr <= 10)).astype(np.float32)
    c50 = (candidate_supported & (cr <= 50)).astype(np.float32)
    c100 = (candidate_supported & (cr <= 100)).astype(np.float32)

    # Base feature column positions are frozen by r2e_lambdarank_runtime.py.
    fallback_log = base_x[:, 12]
    fallback_z = base_x[:, 14]
    pz, sz = base_x[:, 2], base_x[:, 3]
    plog, slog = base_x[:, 4], base_x[:, 5]
    p10, s10 = base_x[:, 16], base_x[:, 17]
    p50, s50 = base_x[:, 18], base_x[:, 19]
    nlog100 = math.log1p(100.0) / math.log1p(candidate_count)
    p100 = (plog <= nlog100 + 1e-12).astype(np.float32)
    s100 = (slog <= nlog100 + 1e-12).astype(np.float32)
    best3_log = np.minimum(plog, slog)
    best3_log[candidate_supported] = np.minimum(best3_log[candidate_supported], clog[candidate_supported])
    best3_z = np.maximum(pz, sz)
    best3_z[candidate_supported] = np.maximum(best3_z[candidate_supported], z[candidate_supported])

    extra = np.column_stack([
        raw,
        z,
        clog,
        rr,
        candidate_supported.astype(np.float32),
        np.ones(len(rows), dtype=np.float32),
        c10,
        c50,
        c100,
        z - fallback_z,
        clog - fallback_log,
        p10 + s10 + c10,
        p50 + s50 + c50,
        p100 + s100 + c100,
        best3_log,
        best3_z,
    ]).astype(np.float32, copy=False)
    return np.concatenate([base_x, extra], axis=1).astype(np.float32, copy=False)


def _wrap_base(base_result, supported_candidates: int = 0) -> BiMER2ERuntimeResult:
    return BiMER2ERuntimeResult(
        full_order=base_result.full_order,
        priority_scores=base_result.priority_scores,
        primary_ranks=base_result.primary_ranks,
        secondary_ranks=base_result.secondary_ranks,
        fallback_ranks=base_result.fallback_ranks,
        learned_rows=base_result.learned_rows,
        learned_scores=base_result.learned_scores,
        fallback_is_secondary=base_result.fallback_is_secondary,
        union_size=base_result.union_size,
        prefix_size=base_result.prefix_size,
        structure_expert_applied=False,
        structure_expert_name=None,
        structure_supported_candidates=int(supported_candidates),
        structure_query_supported=False,
    )


def fuse_bime_r2e_scores(
    primary_scores: np.ndarray,
    secondary_scores: np.ndarray,
    candidate_ids: list[str],
    *,
    reaction_id: str | None,
    similarity: float,
    threshold: float,
    base_ranker_bundle: Path,
    base_ranker_sha256: str,
    structural_ranker_bundle: Path,
    structural_ranker_sha256: str,
    clip_protein_asset: Path,
    clip_reaction_asset: Path,
    clip_protein_manifest_sha256: str | None = None,
    clip_reaction_manifest_sha256: str | None = None,
    device: str = "cuda",
    expected_pool_k: int = 100,
    expected_prefix_k: int = 100,
) -> BiMER2ERuntimeResult:
    """Availability-aware BiME-Rank R2E fusion.

    Registered reaction queries with a valid CLIPZyme reaction embedding use the admitted
    structural expert. Any query outside that native support follows the previously frozen
    R2E LambdaRank path exactly; structural absence is never encoded as a low score.
    """
    primary_scores = np.asarray(primary_scores, dtype=np.float32)
    secondary_scores = np.asarray(secondary_scores, dtype=np.float32)
    if primary_scores.shape != secondary_scores.shape or primary_scores.shape != (len(candidate_ids),):
        raise ValueError("BiME-Rank R2E source scores/candidate IDs do not align")

    p_asset = _load_clip_protein_asset(str(clip_protein_asset), clip_protein_manifest_sha256)
    supported_candidates = len(p_asset.candidate_rows)
    r_asset = _load_clip_reaction_asset(str(clip_reaction_asset), clip_reaction_manifest_sha256)
    query_supported = bool(reaction_id and str(reaction_id) in r_asset.row_by_reaction)
    if not query_supported:
        base_result = fuse_base_r2e_scores(
            primary_scores,
            secondary_scores,
            candidate_ids,
            similarity=similarity,
            threshold=threshold,
            ranker_bundle=base_ranker_bundle,
            ranker_sha256=base_ranker_sha256,
            expected_pool_k=expected_pool_k,
            expected_prefix_k=expected_prefix_k,
        )
        return _wrap_base(base_result, supported_candidates)

    if np.any(p_asset.candidate_rows < 0) or np.any(p_asset.candidate_rows >= len(candidate_ids)):
        raise RuntimeError("CLIPZyme protein candidate rows outside active candidate universe")
    aligned = [candidate_ids[int(row)] for row in p_asset.candidate_rows]
    if tuple(aligned) != p_asset.protein_ids:
        raise RuntimeError("CLIPZyme protein asset does not align to active candidate IDs")

    structural_ranker, structural_cfg = _load_ranker(
        str(structural_ranker_bundle.resolve()), str(structural_ranker_sha256)
    )
    fixed = dict(structural_cfg.get("fixed_config") or {})
    pool_k = int(fixed.get("pool_k", expected_pool_k))
    prefix_k = int(fixed.get("prefix_k", expected_prefix_k))
    if pool_k != int(expected_pool_k) or prefix_k != int(expected_prefix_k):
        raise RuntimeError("BiME-Rank structural R2E pool/prefix differs from admitted configuration")

    lex = lexical_rank(candidate_ids)
    p_order, p_inv = full_order(primary_scores, lex)
    s_order, s_inv = full_order(secondary_scores, lex)
    fallback_is_secondary = float(similarity) < float(threshold)
    fallback_order = s_order if fallback_is_secondary else p_order
    fallback_ranks = s_inv if fallback_is_secondary else p_inv

    reaction_matrix = np.load(r_asset.embeddings_path, mmap_mode="r")
    qvec = np.array(reaction_matrix[r_asset.row_by_reaction[str(reaction_id)]], dtype=np.float32, copy=True)
    if not np.isfinite(qvec).all():
        raise RuntimeError("Supported CLIPZyme reaction embedding is non-finite")
    ptorch = _protein_tensor(str(p_asset.embeddings_path), str(device))
    qt = torch.as_tensor(qvec, dtype=torch.float32, device=torch.device(device))
    with torch.no_grad():
        clip_scores = (ptorch @ qt).float().cpu().numpy().astype(np.float32, copy=False)
    clip_lex = lex[p_asset.candidate_rows]
    clip_order_local = np.lexsort((clip_lex, -clip_scores)).astype(np.int32)
    clip_inv = np.empty(len(clip_order_local), dtype=np.int32)
    clip_inv[clip_order_local] = np.arange(1, len(clip_order_local) + 1, dtype=np.int32)
    clip_top = p_asset.candidate_rows[clip_order_local[:pool_k]]

    union = np.unique(np.concatenate([p_order[:pool_k], s_order[:pool_k], clip_top])).astype(np.int32)
    base_x = build_base_features(
        primary_scores,
        secondary_scores,
        union,
        p_inv,
        s_inv,
        fallback_is_secondary,
        float(similarity),
    )
    support_lookup = np.full(len(candidate_ids), -1, dtype=np.int32)
    support_lookup[p_asset.candidate_rows] = np.arange(len(p_asset.candidate_rows), dtype=np.int32)
    X = _structural_features(base_x, union, clip_scores, clip_inv, support_lookup, len(candidate_ids))
    expected_features = list(structural_cfg.get("feature_names") or [])
    if X.shape[1] != len(expected_features):
        raise RuntimeError("BiME-Rank R2E structural feature count mismatch")
    predictions = structural_ranker.predict(xgb.DMatrix(X))
    learned_idx = np.lexsort((lex[union], -predictions))
    learned_rows = union[learned_idx]
    learned_scores = np.asarray(predictions[learned_idx], dtype=np.float32)
    selected = learned_rows[: min(prefix_k, len(learned_rows))]
    selected_mask = np.zeros(len(candidate_ids), dtype=bool)
    selected_mask[selected] = True
    tail = fallback_order[~selected_mask[fallback_order]]
    order = np.concatenate([selected, tail]).astype(np.int32, copy=False)
    if len(order) != len(candidate_ids) or len(np.unique(order)) != len(candidate_ids):
        raise AssertionError("BiME-Rank structural R2E did not construct a full permutation")
    priority = np.empty(len(candidate_ids), dtype=np.float32)
    if len(candidate_ids) == 1:
        priority[order] = 1.0
    else:
        priority[order] = 1.0 - np.arange(len(candidate_ids), dtype=np.float32) / float(len(candidate_ids) - 1)
    return BiMER2ERuntimeResult(
        full_order=order,
        priority_scores=priority,
        primary_ranks=p_inv,
        secondary_ranks=s_inv,
        fallback_ranks=fallback_ranks,
        learned_rows=learned_rows,
        learned_scores=learned_scores,
        fallback_is_secondary=fallback_is_secondary,
        union_size=int(len(union)),
        prefix_size=int(len(selected)),
        structure_expert_applied=True,
        structure_expert_name="CLIPZyme",
        structure_supported_candidates=int(supported_candidates),
        structure_query_supported=True,
    )
