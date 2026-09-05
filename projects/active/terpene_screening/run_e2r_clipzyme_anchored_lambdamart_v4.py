from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from projects.active.terpene_screening import run_unified_safe_system_e2r_anchored_lambdamart_v3 as v3
from projects.active.terpene_screening.broad_rhea_metrics import evaluate_full_candidate_ranks

ROOT = v3.ROOT
OUT = ROOT / "results/unified_safe_system_v1/e2r_clipzyme_anchored_lambdamart_v4_dev"
CLIP_PROTEINS = ROOT / "external_models/clipzyme_audit/clipzyme_data/clipzyme_screening_set.p"
CLIP_REACTIONS = ROOT / "results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1"
FOLDS = (0, 1, 2)
MAX_POOL = 100
N_CANDIDATES = v3.N_CANDIDATES
V3_OOF = ROOT / "results/unified_safe_system_v1/e2r_anchored_lambdamart_v3_dev/anchored/selected_oof_query_metrics.csv"

# Keep the exact frozen V3 features first, then append CLIPZyme-aware features.
FEATURE_NAMES = [
    *v3.FEATURE_NAMES,
    "clip_raw",
    "clip_z",
    "clip_logrank",
    "clip_rr",
    "clip_candidate_supported",
    "clip_query_supported",
    "clip_top10",
    "clip_top50",
    "clip_top100",
    "clip_z_minus_base",
    "best5_z_minus_base",
    "top10_votes5",
    "top50_votes5",
    "top100_votes5",
    "z5_max",
]
BASELINE_ANCHOR_INDEX = FEATURE_NAMES.index("baseline_anchor")
assert FEATURE_NAMES[: len(v3.FEATURE_NAMES)] == v3.FEATURE_NAMES
METRICS = dict(v3.METRICS)


def _norm_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big") % (2**32)


def _clip_assets(candidate_ids: list[str], query_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return query vectors, query support, candidate vectors, candidate support.

    Candidate vectors are dense in the Catalyst reaction order; unsupported rows are zero and
    guarded by an explicit support mask. Query vectors use the public CLIPZyme screening
    embeddings; unsupported query IDs receive zero vectors and a false support flag.
    """
    clip = pickle.load(CLIP_PROTEINS.open("rb"))
    pids = [str(x) for x in clip["uniprots"]]
    pidx = {p: i for i, p in enumerate(pids)}
    h = clip["hiddens"]
    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
    h = np.asarray(h, dtype=np.float32)
    q_support = np.asarray([q in pidx for q in query_ids], dtype=bool)
    q = np.zeros((len(query_ids), h.shape[1]), dtype=np.float32)
    if q_support.any():
        rows = [pidx[q] for q in query_ids if q in pidx]
        q[q_support] = _norm_rows(h[rows])

    re = pd.read_csv(CLIP_REACTIONS / "entries.csv", dtype=str).fillna("")
    mat = np.load(CLIP_REACTIONS / "embeddings.npy", mmap_mode="r")
    supported = re[re["clipzyme_supported"].str.lower().eq("true")]
    ridx = {str(r): int(row) for r, row in supported[["reaction_id", "row"]].itertuples(index=False)}
    c_support = np.asarray([r in ridx for r in candidate_ids], dtype=bool)
    c = np.zeros((len(candidate_ids), mat.shape[1]), dtype=np.float32)
    if c_support.any():
        rows = [ridx[r] for r in candidate_ids if r in ridx]
        c[c_support] = _norm_rows(np.asarray(mat[rows], dtype=np.float32))
    if q.shape[1] != c.shape[1]:
        raise ValueError((q.shape, c.shape))
    return q, q_support, c, c_support


def _clip_rank_and_features(scores: np.ndarray, candidate_support: np.ndarray, query_supported: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    n = len(scores)
    ranks = np.full(n, n + 1, dtype=np.int32)
    if not query_supported or not candidate_support.any():
        return ranks, np.empty(0, dtype=np.int32), np.zeros(n, dtype=np.float32), 0.0, 1.0
    rows = np.flatnonzero(candidate_support)
    local = np.asarray(scores[rows], dtype=np.float32)
    order_local = np.argsort(-local, kind="stable")
    inv = np.empty(len(rows), dtype=np.int32)
    inv[order_local] = np.arange(1, len(rows) + 1, dtype=np.int32)
    ranks[rows] = inv
    top = rows[order_local[: min(MAX_POOL, len(rows))]].astype(np.int32)
    mean = float(local.mean())
    std = max(float(local.std()), 1e-6)
    z = np.zeros(n, dtype=np.float32)
    z[rows] = (local - mean) / std
    return ranks, top, z, mean, std


def feature_matrix_v4(
    current_scores: np.ndarray,
    rows: np.ndarray,
    current_ranks: np.ndarray,
    clip_scores: np.ndarray,
    clip_ranks: np.ndarray,
    clip_z: np.ndarray,
    clip_candidate_support: np.ndarray,
    clip_query_supported: bool,
) -> np.ndarray:
    base = v3.feature_matrix(current_scores, rows, current_ranks)
    # Recompute current z values only for cross-expert interaction terms.
    means = current_scores.mean(1, keepdims=True)
    std = np.maximum(current_scores.std(1, keepdims=True), 1e-6)
    z4 = (current_scores - means) / std
    zr = z4[:, rows].T.astype(np.float32)
    cr = clip_ranks[rows].astype(np.float32)
    cs = clip_candidate_support[rows] & bool(clip_query_supported)
    craw = np.where(cs, clip_scores[rows], 0.0).astype(np.float32)
    cz = np.where(cs, clip_z[rows], 0.0).astype(np.float32)
    clog = np.ones(len(rows), dtype=np.float32)
    good = cs & (cr <= N_CANDIDATES)
    clog[good] = np.log1p(cr[good]) / math.log1p(max(int(clip_candidate_support.sum()), 1))
    crr = np.zeros(len(rows), dtype=np.float32)
    crr[good] = 1.0 / cr[good]
    c10 = (good & (cr <= 10)).astype(np.float32)
    c50 = (good & (cr <= 50)).astype(np.float32)
    c100 = (good & (cr <= 100)).astype(np.float32)
    current_rr = current_ranks[rows]
    v10_4 = (current_rr <= 10).sum(1).astype(np.float32)
    v50_4 = (current_rr <= 50).sum(1).astype(np.float32)
    v100_4 = (current_rr <= 100).sum(1).astype(np.float32)
    base_z = zr[:, 0]
    best5 = np.maximum(zr.max(1), cz)
    extra = np.column_stack(
        [
            craw,
            cz,
            clog,
            crr,
            cs.astype(np.float32),
            np.full(len(rows), float(bool(clip_query_supported)), dtype=np.float32),
            c10,
            c50,
            c100,
            cz - base_z,
            best5 - base_z,
            v10_4 + c10,
            v50_4 + c50,
            v100_4 + c100,
            best5,
        ]
    ).astype(np.float32, copy=False)
    out = np.concatenate([base, extra], axis=1).astype(np.float32, copy=False)
    if out.shape[1] != len(FEATURE_NAMES):
        raise AssertionError((out.shape, len(FEATURE_NAMES)))
    return out


def prepare_fold(fold: int) -> None:
    dev, candidate_ids, query_ids, emb = v3.load_fold_embeddings(fold)
    positives = dev.groupby("protein_id").reaction_id.apply(set).to_dict()
    ridx = {r: i for i, r in enumerate(candidate_ids)}
    clip_q, clip_q_support, clip_c, clip_c_support = _clip_assets(candidate_ids, query_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_rt = {n: torch.from_numpy(emb[n][1]).to(device) for n in v3.NAMES}
    clip_ct = torch.from_numpy(clip_c).to(device)

    X = []
    rows_all = []
    ranks_all = []
    labels = []
    offsets = [0]
    pos_rows = []
    pos_base_ranks = []
    pos_offsets = [0]
    baseline = []
    audits = []

    with torch.no_grad():
        for st in range(0, len(query_ids), 64):
            stop = min(st + 64, len(query_ids))
            sc = {
                n: (torch.from_numpy(emb[n][0][st:stop]).to(device) @ current_rt[n].T).cpu().numpy()
                for n in v3.NAMES
            }
            cq = torch.from_numpy(clip_q[st:stop]).to(device)
            csc = (cq @ clip_ct.T).cpu().numpy()
            for j, q in enumerate(query_ids[st:stop]):
                S = np.stack([sc[n][j] for n in v3.NAMES]).astype(np.float32)
                ranks4 = np.stack([v3.full_ranks(S[e]) for e in range(4)], axis=1)
                cscore = np.asarray(csc[j], dtype=np.float32)
                cranks, ctop, cz, _, _ = _clip_rank_and_features(cscore, clip_c_support, bool(clip_q_support[st + j]))
                current_top = [set(map(int, v3.top_rows(S[e], MAX_POOL))) for e in range(4)]
                union_set = set().union(*current_top)
                if len(ctop):
                    union_set.update(map(int, ctop))
                union = np.asarray(sorted(union_set), dtype=np.int32)
                ranks5 = np.column_stack([ranks4[union], cranks[union]]).astype(np.int32)
                feat = feature_matrix_v4(S, union, ranks4, cscore, cranks, cz, clip_c_support, bool(clip_q_support[st + j]))
                p = np.asarray(sorted({ridx[r] for r in positives[q]}), dtype=np.int32)
                pset = set(map(int, p))
                y = np.asarray([int(int(r) in pset) for r in union], dtype=np.uint8)
                br = ranks4[p, 0].astype(np.int32)
                X.append(feat)
                rows_all.append(union)
                ranks_all.append(ranks5)
                labels.append(y)
                offsets.append(offsets[-1] + len(union))
                pos_rows.append(p)
                pos_base_ranks.append(br)
                pos_offsets.append(pos_offsets[-1] + len(p))
                baseline.append({"query_id": q, **evaluate_full_candidate_ranks(br, N_CANDIDATES)})
                audits.append(
                    {
                        "fold": fold,
                        "query_id": q,
                        "union_size": len(union),
                        "positive_count": len(p),
                        "positive_in_union": int(y.sum()),
                        "baseline_best_rank": int(br.min()),
                        "clip_query_supported": bool(clip_q_support[st + j]),
                        "clip_positive_supported": int(sum(clip_c_support[x] for x in p)),
                        "clip_positive_top100": int(sum(cranks[x] <= 100 for x in p)),
                    }
                )
            print(f"prepare-v4 fold={fold} {stop}/{len(query_ids)}", flush=True)

    out = OUT / "prepared" / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", np.concatenate(X))
    np.save(out / "rows.npy", np.concatenate(rows_all))
    np.save(out / "ranks.npy", np.concatenate(ranks_all))
    np.save(out / "labels.npy", np.concatenate(labels))
    np.save(out / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(out / "positive_rows.npy", np.concatenate(pos_rows))
    np.save(out / "positive_base_ranks.npy", np.concatenate(pos_base_ranks))
    np.save(out / "positive_offsets.npy", np.asarray(pos_offsets, dtype=np.int64))
    pd.DataFrame({"query_id": query_ids}).to_csv(out / "queries.csv", index=False)
    pd.DataFrame(baseline).to_csv(out / "baseline_query_metrics.csv", index=False)
    adf = pd.DataFrame(audits)
    adf.to_csv(out / "audit.csv", index=False)
    (out / "candidate_reactions.txt").write_text("\n".join(candidate_ids) + "\n")
    (out / "feature_names.json").write_text(json.dumps(FEATURE_NAMES, indent=2) + "\n")
    summary = {
        "fold": fold,
        "queries": len(query_ids),
        "candidate_count": len(candidate_ids),
        "rows": int(offsets[-1]),
        "mean_union_size": float(adf.union_size.mean()),
        "query_positive_in_union_fraction": float((adf.positive_in_union > 0).mean()),
        "clip_query_supported_fraction": float(adf.clip_query_supported.mean()),
        "clip_positive_top100_queries": int((adf.clip_positive_top100 > 0).sum()),
        "dev_pairs": len(dev),
        "external_evaluation_metrics_used": False,
        "external_model_used_as_frozen_expert": "official CLIPZyme embeddings only",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def load_cache(fold: int) -> dict[str, object]:
    p = OUT / "prepared" / f"fold{fold}"
    return {
        "X": np.load(p / "X.npy", mmap_mode="r"),
        "rows": np.load(p / "rows.npy", mmap_mode="r"),
        "ranks": np.load(p / "ranks.npy", mmap_mode="r"),
        "labels": np.load(p / "labels.npy", mmap_mode="r"),
        "offsets": np.load(p / "offsets.npy"),
        "pos_rows": np.load(p / "positive_rows.npy", mmap_mode="r"),
        "pos_base_ranks": np.load(p / "positive_base_ranks.npy", mmap_mode="r"),
        "pos_offsets": np.load(p / "positive_offsets.npy"),
        "queries": pd.read_csv(p / "queries.csv", dtype=str).query_id.astype(str).tolist(),
        "baseline": pd.read_csv(p / "baseline_query_metrics.csv"),
    }


def sampled_training(cache: dict[str, object], fold: int):
    xs = []
    ys = []
    groups = []
    hard_col = FEATURE_NAMES.index("z5_max")
    for qi, q in enumerate(cache["queries"]):
        a, b = map(int, cache["offsets"][qi : qi + 2])
        y = np.asarray(cache["labels"][a:b], dtype=np.uint8)
        pos = np.flatnonzero(y > 0)
        if not len(pos):
            continue
        neg = np.flatnonzero(y == 0)
        X = np.asarray(cache["X"][a:b], dtype=np.float32)
        rows = np.asarray(cache["rows"][a:b])
        hard = neg[np.lexsort((rows[neg], -X[neg, hard_col]))][:96]
        rem = np.setdiff1d(neg, hard, assume_unique=False)
        rng = np.random.default_rng(_stable_seed(f"e2r-v4|{fold}|{q}"))
        rnd = rng.choice(rem, size=min(32, len(rem)), replace=False) if len(rem) else np.empty(0, dtype=np.int64)
        keep = np.concatenate([pos, hard, rnd])
        xs.append(X[keep])
        ys.append(y[keep].astype(np.float32))
        groups.append(len(keep))
    return np.concatenate(xs), np.concatenate(ys), groups


def ranker_configs() -> list[dict[str, object]]:
    # Freeze the ranker hyperparameters to the incumbent V3 winner.  The only learned
    # intervention under study is the fifth expert and its derived features; this also
    # avoids a second broad HPO sweep on the same clean-dev folds.
    return [dict(x) for x in v3.ranker_configs() if str(x["id"]) == "ndcg_d3_e010"]


def structure_configs() -> list[dict[str, int]]:
    return v3.structure_configs()


def train_model(train_X, train_y, groups, cfg, seed: int):
    d = xgb.DMatrix(train_X, label=train_y)
    d.set_group(groups)
    mono = [0] * len(FEATURE_NAMES)
    mono[BASELINE_ANCHOR_INDEX] = 1
    params = {
        "objective": cfg["objective"],
        "tree_method": "hist",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_depth": int(cfg["max_depth"]),
        "eta": float(cfg["learning_rate"]),
        "min_child_weight": 5.0,
        "lambda": 5.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambdarank_pair_method": "topk",
        "lambdarank_num_pair_per_sample": int(cfg["lambdarank_pairs"]),
        "monotone_constraints": "(" + ",".join(map(str, mono)) + ")",
        "seed": int(seed),
        "nthread": 16,
        "verbosity": 0,
    }
    return xgb.train(params, d, num_boost_round=int(cfg["rounds"]))


def anchored_order_v4(
    current_scores: np.ndarray,
    clip_scores: np.ndarray,
    clip_candidate_support: np.ndarray,
    clip_query_supported: bool,
    ranker: xgb.Booster,
    *,
    protected_prefix: int,
    pool_k: int,
    prefix_k: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Construct a full V4 order while preserving the EnzGFM anchor/tail contract."""
    S = np.asarray(current_scores, dtype=np.float32)
    if S.ndim != 2 or S.shape[0] != 4:
        raise ValueError(f"expected 4 incumbent expert score rows, got {S.shape}")
    if S.shape[1] != len(clip_scores) or len(clip_scores) != len(clip_candidate_support):
        raise ValueError("V4 expert/candidate dimensions do not align")
    ranks4 = np.stack([v3.full_ranks(S[e]) for e in range(4)], axis=1)
    cranks, ctop, cz, _, _ = _clip_rank_and_features(
        np.asarray(clip_scores, dtype=np.float32),
        np.asarray(clip_candidate_support, dtype=bool),
        bool(clip_query_supported),
    )
    union_set = set().union(*(set(map(int, v3.top_rows(S[e], MAX_POOL))) for e in range(4)))
    if len(ctop):
        union_set.update(map(int, ctop))
    union = np.asarray(sorted(union_set), dtype=np.int32)
    ranks5 = np.column_stack([ranks4[union], cranks[union]]).astype(np.int32)
    X = feature_matrix_v4(
        S, union, ranks4, np.asarray(clip_scores, dtype=np.float32), cranks, cz,
        np.asarray(clip_candidate_support, dtype=bool), bool(clip_query_supported),
    )
    pred = ranker.predict(xgb.DMatrix(X))
    pool = (ranks5.min(1) <= int(pool_k)) & (ranks5[:, 0] > int(protected_prefix))
    local = np.flatnonzero(pool)
    local = local[np.lexsort((union[local], -pred[local]))]
    learned_slots = max(0, int(prefix_k) - int(protected_prefix))
    selected = union[local[:learned_slots]].astype(np.int32, copy=False)
    baseline_order = np.argsort(-S[0], kind="stable").astype(np.int32)
    protected = baseline_order[: int(protected_prefix)]
    blocked = np.zeros(S.shape[1], dtype=bool)
    blocked[protected] = True
    blocked[selected] = True
    tail = baseline_order[~blocked[baseline_order]]
    order = np.concatenate([protected, selected, tail]).astype(np.int32, copy=False)
    if len(order) != S.shape[1] or len(np.unique(order)) != len(order):
        raise AssertionError("V4 anchored order is not a full permutation")
    return order, {
        "union_size": int(len(union)),
        "pool_size": int(pool.sum()),
        "selected_size": int(len(selected)),
        "clip_query_supported": bool(clip_query_supported),
    }


def candidate_query_metrics(cache: dict, pred: np.ndarray, structure: dict) -> pd.DataFrame:
    A, K, P = structure["protected_prefix"], structure["pool_k"], structure["prefix_k"]
    rec = []
    for qi, q in enumerate(cache["queries"]):
        a, b = map(int, cache["offsets"][qi : qi + 2])
        rows = np.asarray(cache["rows"][a:b], dtype=np.int32)
        ranks = np.asarray(cache["ranks"][a:b], dtype=np.int32)
        local_pred = pred[a:b]
        pool = (ranks.min(1) <= K) & (ranks[:, 0] > A)
        cand = np.flatnonzero(pool)
        order = cand[np.lexsort((rows[cand], -local_pred[cand]))]
        L = min(P - A, len(order))
        selected_local = order[:L]
        selected_rows = rows[selected_local]
        selected_base = ranks[selected_local, 0]
        selected_pos = {int(r): A + i + 1 for i, r in enumerate(selected_rows)}
        pa, pb = map(int, cache["pos_offsets"][qi : qi + 2])
        prows = np.asarray(cache["pos_rows"][pa:pb], dtype=np.int32)
        pbase = np.asarray(cache["pos_base_ranks"][pa:pb], dtype=np.int32)
        new = []
        for r, br in zip(prows, pbase, strict=True):
            r = int(r)
            br = int(br)
            if br <= A:
                nr = br
            elif r in selected_pos:
                nr = selected_pos[r]
            else:
                nr = br + L - int(np.count_nonzero(selected_base < br))
            new.append(nr)
        rec.append(
            {
                "query_id": q,
                "selected_prefix_size": L,
                "pool_candidates": int(pool.sum()),
                **evaluate_full_candidate_ranks(np.asarray(new, dtype=np.int64), N_CANDIDATES),
            }
        )
    return pd.DataFrame(rec)


def metric_map(frame: pd.DataFrame):
    return {k: float(frame[c].mean()) for k, c in METRICS.items()}


def gm_ratio(c, b):
    return float(math.exp(np.mean([math.log(max(c[k], 1e-12) / max(b[k], 1e-12)) for k in ["mrr", "map", "ndcg10", "hit10"]])))


def run_search() -> None:
    caches = {f: load_cache(f) for f in FOLDS}
    sampled = {f: sampled_training(caches[f], f) for f in FOLDS}
    incumbent_all = pd.read_csv(V3_OOF, dtype={"query_id": str})
    incumbent = {}
    for f in FOLDS:
        local = incumbent_all[incumbent_all["fold"].astype(int).eq(f)].copy().reset_index(drop=True)
        if local.query_id.astype(str).tolist() != caches[f]["queries"]:
            raise RuntimeError(f"V3 OOF query order differs from V4 fold {f}")
        incumbent[f] = local
    models = OUT / "search_models"
    models.mkdir(parents=True, exist_ok=True)
    rows = []
    for ci, cfg in enumerate(ranker_configs()):
        preds = {}
        for hold in FOLDS:
            xs, ys, groups = [], [], []
            for f in FOLDS:
                if f == hold:
                    continue
                X, y, g = sampled[f]
                xs.append(X)
                ys.append(y)
                groups.extend(g)
            booster = train_model(np.concatenate(xs), np.concatenate(ys), groups, cfg, 20260904 + ci * 10 + hold)
            preds[hold] = booster.predict(xgb.DMatrix(np.asarray(caches[hold]["X"], dtype=np.float32)))
            md = models / str(cfg["id"])
            md.mkdir(parents=True, exist_ok=True)
            booster.save_model(md / f"fold{hold}.json")
        for structure in structure_configs():
            fold_deltas = {}
            raw_fold_deltas = {}
            cand_frames = []
            incumbent_frames = []
            raw_base_frames = []
            for f in FOLDS:
                cand = candidate_query_metrics(caches[f], preds[f], structure)
                raw_base = caches[f]["baseline"]
                current = incumbent[f]
                assert cand.query_id.tolist() == raw_base.query_id.tolist() == current.query_id.astype(str).tolist()
                cm = metric_map(cand)
                im = metric_map(current)
                bm = metric_map(raw_base)
                fold_deltas[str(f)] = {k: cm[k] - im[k] for k in METRICS}
                raw_fold_deltas[str(f)] = {k: cm[k] - bm[k] for k in METRICS}
                cand_frames.append(cand)
                incumbent_frames.append(current)
                raw_base_frames.append(raw_base)
            C = pd.concat(cand_frames, ignore_index=True)
            I = pd.concat(incumbent_frames, ignore_index=True)
            B = pd.concat(raw_base_frames, ignore_index=True)
            cm = metric_map(C)
            im = metric_map(I)
            bm = metric_map(B)
            delta = {k: cm[k] - im[k] for k in METRICS}
            raw_delta = {k: cm[k] - bm[k] for k in METRICS}
            pooled = all(delta[k] >= -1e-12 for k in METRICS)
            safe = all(
                fd["mrr"] >= -0.001
                and fd["map"] >= -0.001
                and fd["auc"] >= -0.001
                and fd["ndcg10"] >= -0.002
                and fd["hit10"] >= -0.005
                and fd["hit20"] >= -0.005
                and fd["hit50"] >= -0.005
                for fd in fold_deltas.values()
            )
            two = all(sum(fd[k] > 0 for fd in fold_deltas.values()) >= 2 for k in ["mrr", "map", "hit10"])
            material = delta["mrr"] >= 0.001 or delta["map"] >= 0.001 or delta["hit10"] >= 0.005
            raw_safe = all(raw_delta[k] >= -1e-12 for k in METRICS)
            feasible = pooled and safe and two and material and raw_safe
            rows.append(
                {
                    "ranker_id": cfg["id"],
                    "ranker_max_depth": int(cfg["max_depth"]),
                    "ranker_rounds": int(cfg["rounds"]),
                    **structure,
                    **{f"raw_baseline_{k}": v for k, v in bm.items()},
                    **{f"incumbent_v3_{k}": v for k, v in im.items()},
                    **{f"candidate_{k}": v for k, v in cm.items()},
                    **{f"delta_vs_v3_{k}": v for k, v in delta.items()},
                    **{f"delta_vs_raw_{k}": v for k, v in raw_delta.items()},
                    "primary_geomean_ratio_vs_v3": gm_ratio(cm, im),
                    "pooled_no_regression_vs_v3": pooled,
                    "fold_safe_vs_v3": safe,
                    "two_of_three_positive_vs_v3": two,
                    "material_gate_vs_v3": material,
                    "raw_baseline_no_regression": raw_safe,
                    "feasible": feasible,
                    "fold_deltas_vs_v3_json": json.dumps(fold_deltas, sort_keys=True),
                    "fold_deltas_vs_raw_json": json.dumps(raw_fold_deltas, sort_keys=True),
                }
            )
        print("searched-v4 ranker", cfg["id"], flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "search_results.csv", index=False)
    feasible = frame[frame.feasible.astype(bool)].copy()
    if feasible.empty:
        result = {
            "status": "rejected_no_feasible_material_config",
            "configuration_count": len(frame),
            "feasible_count": 0,
            "external_evaluation_metrics_used": False,
        }
        (OUT / "selection_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    feasible = feasible.sort_values(
        ["primary_geomean_ratio_vs_v3", "delta_vs_v3_hit20", "delta_vs_v3_hit50", "protected_prefix", "prefix_k", "pool_k", "ranker_max_depth", "ranker_rounds", "ranker_id"],
        ascending=[False, False, False, False, True, True, True, True, True],
        kind="stable",
    )
    best = feasible.iloc[0].to_dict()
    selected = {
        "ranker_id": best["ranker_id"],
        "protected_prefix": int(best["protected_prefix"]),
        "pool_k": int(best["pool_k"]),
        "prefix_k": int(best["prefix_k"]),
    }
    cfg = next(x for x in ranker_configs() if x["id"] == selected["ranker_id"])
    selected["ranker_config"] = cfg
    result = {
        "status": "selected_clean_dev_only",
        "configuration_count": len(frame),
        "feasible_count": len(feasible),
        "selected_config": selected,
        "selected_summary": {k: (bool(v) if isinstance(v, (np.bool_, bool)) else float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in best.items() if k not in {"fold_deltas_vs_v3_json", "fold_deltas_vs_raw_json"}},
        "fold_deltas_vs_v3": json.loads(best["fold_deltas_vs_v3_json"]),
        "fold_deltas_vs_raw": json.loads(best["fold_deltas_vs_raw_json"]),
        "external_evaluation_metrics_used": False,
        "test_or_common_support_metrics_used_for_selection": False,
        "incumbent_comparator": "frozen E2R Anchored LambdaMART V3 selected OOF",
        "frozen_external_model": "official CLIPZyme",
    }
    (OUT / "selection_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def fit_selected() -> None:
    result = json.loads((OUT / "selection_result.json").read_text())
    if result.get("status") != "selected_clean_dev_only":
        raise RuntimeError("No selected clean-dev configuration")
    selected = result["selected_config"]
    cfg = selected["ranker_config"]
    caches = {f: load_cache(f) for f in FOLDS}
    sampled = {f: sampled_training(caches[f], f) for f in FOLDS}
    xs, ys, groups = [], [], []
    for f in FOLDS:
        X, y, g = sampled[f]
        xs.append(X)
        ys.append(y)
        groups.extend(g)
    booster = train_model(np.concatenate(xs), np.concatenate(ys), groups, cfg, _stable_seed("e2r-v4-final-clean-dev"))
    out = OUT / "selected"
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(out / "ranker.json")
    payload = {
        "selected_config": selected,
        "feature_names": FEATURE_NAMES,
        "training_folds": list(FOLDS),
        "external_evaluation_metrics_used": False,
        "ranker_sha256": hashlib.sha256((out / "ranker.json").read_bytes()).hexdigest(),
    }
    (out / "config.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "search", "fit-selected"])
    ap.add_argument("--fold", type=int, choices=FOLDS)
    args = ap.parse_args()
    if args.stage == "prepare":
        if args.fold is None:
            raise ValueError("--fold required")
        prepare_fold(args.fold)
    elif args.stage == "search":
        run_search()
    else:
        fit_selected()


if __name__ == "__main__":
    main()
