from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex
from reproducibility.bime_rank.scripts import evaluate_bridge_reciprocal_relation_context_v4 as base_eval

TRAIN = base_eval.TRAIN
BENCH = base_eval.BENCH
POOL_CELL = base_eval.POOL_CELL
OUT = ROOT / "results/bridge_adaptive_relation_context_v8"
ASSET = OUT / "adaptive_relation_gate.validation.pkl"

ALPHAS = np.asarray(
    [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0],
    dtype=np.float64,
)
FEATURE_NAMES = [
    "log_query_degree",
    "query_neighbor_coherence",
    "context_coverage",
    "base_margin_1_2",
    "base_top10_spread",
    "base_top100_spread",
    "context_margin_1_2",
    "context_top10_mean",
    "context_top100_mean",
    "base_context_corr",
    "top10_overlap",
    "top100_overlap",
    "component_agreement",
    "context_top20_broad_logrank_mean",
]


def _metric(ranks: np.ndarray) -> dict[str, float]:
    r = np.asarray(ranks, dtype=np.int64)
    return {
        "queries": int(len(r)),
        "mrr": float(np.mean(1.0 / r)),
        "hit10": float(np.mean(r <= 10)),
        "hit100": float(np.mean(r <= 100)),
        "hit1000": float(np.mean(r <= 1000)),
        "median_rank": float(np.median(r)),
    }


def _best_rank(score: np.ndarray, target_rows: np.ndarray, lexical: np.ndarray) -> int:
    values = score[target_rows]
    best_value = float(values.max())
    tied_targets = target_rows[values == best_value]
    best_target = int(tied_targets[np.argmin(lexical[tied_targets])])
    return int(
        np.count_nonzero(score > best_value)
        + np.count_nonzero((score == best_value) & (lexical < lexical[best_target]))
        + 1
    )


def _safe_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        a = a[mask]
        b = b[mask]
    if len(a) < 3:
        return 0.0
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _top_rows(values: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), len(values))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    rows = np.argpartition(-values, k - 1)[:k]
    return rows[np.argsort(-values[rows], kind="stable")]


def _features(
    base: np.ndarray,
    context: np.ndarray,
    comp_a: np.ndarray,
    comp_b: np.ndarray,
    available: np.ndarray,
    both_available: np.ndarray,
    query_degree: int,
    query_neighbor_coherence: float,
) -> np.ndarray:
    order100 = _top_rows(base, 100)
    order10 = order100[: min(10, len(order100))]
    ctx100 = _top_rows(context, 100)
    ctx10 = ctx100[: min(10, len(ctx100))]
    ctx20 = ctx100[: min(20, len(ctx100))]

    base_sorted = np.sort(base[order100])[::-1]
    ctx_sorted = np.sort(context[available])[::-1] if np.any(available) else np.asarray([0.0])

    broad_rank = np.empty(len(base), dtype=np.int64)
    broad_order = np.argsort(-base, kind="stable")
    broad_rank[broad_order] = np.arange(1, len(base) + 1)
    ctx20_logrank = (
        float(np.mean(np.log1p(broad_rank[ctx20]))) if len(ctx20) else math.log1p(len(base))
    )

    return np.asarray(
        [
            math.log1p(max(int(query_degree), 0)),
            float(query_neighbor_coherence),
            float(np.mean(available)),
            float(base_sorted[0] - base_sorted[1]) if len(base_sorted) > 1 else 0.0,
            float(base_sorted[0] - base_sorted[min(9, len(base_sorted) - 1)]),
            float(base_sorted[0] - base_sorted[-1]),
            float(ctx_sorted[0] - ctx_sorted[1]) if len(ctx_sorted) > 1 else 0.0,
            float(np.mean(ctx_sorted[: min(10, len(ctx_sorted))])),
            float(np.mean(ctx_sorted[: min(100, len(ctx_sorted))])),
            _safe_corr(base, context, available),
            float(len(set(order10.tolist()) & set(ctx10.tolist())) / max(1, len(order10))),
            float(len(set(order100.tolist()) & set(ctx100.tolist())) / max(1, len(order100))),
            _safe_corr(comp_a, comp_b, both_available),
            ctx20_logrank,
        ],
        dtype=np.float64,
    )


def _validation_relations(train: pd.DataFrame) -> pd.DataFrame:
    newer = pd.concat(
        [
            pd.read_csv(POOL_CELL / "train_pairs.csv", dtype=str),
            pd.read_csv(POOL_CELL / "test_pairs.csv", dtype=str),
        ],
        ignore_index=True,
    ).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    old = set(zip(train.protein_id.astype(str), train.reaction_id.astype(str)))
    frozen_outer: set[tuple[str, str]] = set()
    for directory in BENCH.iterdir():
        path = directory / "test_pairs.csv"
        if path.exists():
            frame = pd.read_csv(path, dtype=str).fillna("")
            frozen_outer |= set(zip(frame.protein_id.astype(str), frame.reaction_id.astype(str)))
    pairs = (set(zip(newer.protein_id.astype(str), newer.reaction_id.astype(str))) - old) - frozen_outer
    return pd.DataFrame(sorted(pairs), columns=["protein_id", "reaction_id"])


def _query_neighbor_coherence(
    rows: list[int],
    candidate_embeddings: torch.Tensor,
) -> float:
    if not rows:
        return 0.0
    with torch.no_grad():
        x = candidate_embeddings.index_select(
            0, torch.as_tensor(rows, dtype=torch.long, device=candidate_embeddings.device)
        )
        mean = x.mean(0)
        return float(mean.norm().item())


def _collect_direction(
    direction: str,
    index: FibreCandidateIndex,
    frame: pd.DataFrame,
    train: pd.DataFrame,
    pproto: torch.Tensor,
    pmask: torch.Tensor,
    rproto: torch.Tensor,
    rmask: torch.Tensor,
    batch_size: int,
) -> pd.DataFrame:
    P = index.protein_embeddings
    R = index.reaction_embeddings
    if direction == "r2e":
        targets = frame.groupby("reaction_id").protein_id.apply(lambda s: set(map(str, s))).to_dict()
        qids = sorted(targets)
        qindex = index.reaction_index
        cindex = index.protein_index
        candidate_ids = index.protein_ids
        candidate_embeddings = P
        known = train.groupby("reaction_id").protein_id.apply(lambda s: set(map(str, s))).to_dict()
    else:
        targets = frame.groupby("protein_id").reaction_id.apply(lambda s: set(map(str, s))).to_dict()
        qids = sorted(targets)
        qindex = index.protein_index
        cindex = index.reaction_index
        candidate_ids = index.reaction_ids
        candidate_embeddings = R
        known = train.groupby("protein_id").reaction_id.apply(lambda s: set(map(str, s))).to_dict()

    lexical = np.empty(len(candidate_ids), dtype=np.int64)
    lexical_order = np.argsort(np.asarray(candidate_ids, dtype=object), kind="stable")
    lexical[lexical_order] = np.arange(len(lexical_order))
    records: list[dict[str, object]] = []
    for start in range(0, len(qids), batch_size):
        qs = [q for q in qids[start : start + batch_size] if q in qindex]
        if not qs:
            continue
        qi = torch.as_tensor([qindex[q] for q in qs], dtype=torch.long, device=index.device)
        with torch.no_grad():
            if direction == "r2e":
                qemb = R.index_select(0, qi)
                raw_base = qemb @ P.T
                raw_a = rproto.index_select(0, qi) @ P.T
                av_a = rmask.index_select(0, qi)[:, None].expand_as(raw_a)
                raw_b = qemb @ pproto.T
                av_b = pmask[None, :].expand_as(raw_b)
            else:
                qemb = P.index_select(0, qi)
                raw_base = qemb @ R.T
                raw_a = pproto.index_select(0, qi) @ R.T
                av_a = pmask.index_select(0, qi)[:, None].expand_as(raw_a)
                raw_b = qemb @ rproto.T
                av_b = rmask[None, :].expand_as(raw_b)

            base_z = base_eval.z_masked(raw_base, torch.ones_like(raw_base, dtype=torch.bool))
            za = base_eval.z_masked(raw_a, av_a)
            zb = base_eval.z_masked(raw_b, av_b)
            den = av_a.float() + av_b.float()
            context = (za + zb) / den.clamp_min(1.0)
            context[den == 0] = 0

        B = base_z.cpu().numpy().astype(np.float64, copy=False)
        C = context.cpu().numpy().astype(np.float64, copy=False)
        A = za.cpu().numpy().astype(np.float64, copy=False)
        ZB = zb.cpu().numpy().astype(np.float64, copy=False)
        AV = (den > 0).cpu().numpy()
        BOTH = (av_a & av_b).cpu().numpy()

        for j, query_id in enumerate(qs):
            known_ids = [x for x in known.get(query_id, set()) if x in cindex]
            seed_rows = [cindex[x] for x in known_ids]
            target_rows = np.asarray(
                [cindex[x] for x in targets[query_id] if x in cindex], dtype=np.int64
            )
            if not len(target_rows):
                continue

            base = B[j].copy()
            ctx = C[j].copy()
            if seed_rows:
                base[np.asarray(seed_rows, dtype=np.int64)] = -np.inf
                ctx[np.asarray(seed_rows, dtype=np.int64)] = 0.0

            finite = np.isfinite(base)
            work_base = base.copy()
            if not np.all(finite):
                floor = float(np.min(work_base[finite]) - 100.0) if np.any(finite) else -100.0
                work_base[~finite] = floor

            ranks = []
            for alpha in ALPHAS:
                score = work_base + float(alpha) * ctx
                ranks.append(_best_rank(score, target_rows, lexical))
            ranks_arr = np.asarray(ranks, dtype=np.int64)
            best_idx = int(np.argmin(ranks_arr))
            best_alpha = float(ALPHAS[best_idx])
            base_rank = int(ranks_arr[0])

            coherence = _query_neighbor_coherence(seed_rows, candidate_embeddings)
            x = _features(
                work_base,
                ctx,
                A[j],
                ZB[j],
                AV[j] & finite,
                BOTH[j] & finite,
                len(seed_rows),
                coherence,
            )
            row: dict[str, object] = {
                "direction": direction,
                "query_id": query_id,
                "query_degree": len(seed_rows),
                "base_rank": base_rank,
                "oracle_alpha": best_alpha,
                "oracle_rank": int(ranks_arr[best_idx]),
                "oracle_log_gain": float(math.log(base_rank / max(int(ranks_arr[best_idx]), 1))),
            }
            row.update({name: float(value) for name, value in zip(FEATURE_NAMES, x, strict=True)})
            for alpha, rank in zip(ALPHAS, ranks_arr, strict=True):
                row[f"rank_a_{alpha:g}"] = int(rank)
            records.append(row)

        print(direction, min(start + batch_size, len(qids)), "/", len(qids), flush=True)
    return pd.DataFrame(records)


def _rank_for_predicted_alpha(frame: pd.DataFrame, predicted_alpha: np.ndarray) -> np.ndarray:
    grid = ALPHAS
    nearest = np.abs(predicted_alpha[:, None] - grid[None, :]).argmin(1)
    cols = [f"rank_a_{a:g}" for a in grid]
    matrix = frame[cols].to_numpy(np.int64)
    return matrix[np.arange(len(frame)), nearest]


def _cv_fit(frame: pd.DataFrame, random_state: int = 20261006) -> tuple[dict, object]:
    X = frame[FEATURE_NAMES].to_numpy(np.float64)
    oracle_alpha = frame["oracle_alpha"].to_numpy(np.float64)
    helpful = (oracle_alpha > 0).astype(np.int64)
    strength_target = np.log1p(oracle_alpha)
    base_ranks = frame["base_rank"].to_numpy(np.int64)

    candidates = [
        {"max_iter": 60, "learning_rate": 0.05, "max_leaf_nodes": 7, "l2_regularization": 1.0, "min_samples_leaf": 30},
        {"max_iter": 100, "learning_rate": 0.04, "max_leaf_nodes": 7, "l2_regularization": 2.0, "min_samples_leaf": 40},
        {"max_iter": 120, "learning_rate": 0.03, "max_leaf_nodes": 15, "l2_regularization": 4.0, "min_samples_leaf": 50},
    ]
    permission_powers = (1.0, 2.0, 4.0, 8.0, 12.0)
    n_splits = 5 if len(frame) >= 1000 else 4
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    summaries = []
    best = None
    base_metrics = _metric(base_ranks)

    for params in candidates:
        oof_probability = np.zeros(len(frame), dtype=np.float64)
        oof_strength = np.zeros(len(frame), dtype=np.float64)
        for train_idx, valid_idx in kf.split(X):
            scaler = StandardScaler().fit(X[train_idx])
            xt = scaler.transform(X[train_idx])
            xv = scaler.transform(X[valid_idx])

            permission = HistGradientBoostingClassifier(random_state=random_state, **params)
            permission.fit(xt, helpful[train_idx])
            oof_probability[valid_idx] = permission.predict_proba(xv)[:, 1]

            pos = train_idx[helpful[train_idx] > 0]
            strength = HistGradientBoostingRegressor(random_state=random_state, **params)
            strength.fit(scaler.transform(X[pos]), strength_target[pos])
            oof_strength[valid_idx] = np.clip(
                np.expm1(strength.predict(xv)), 0.0, float(ALPHAS[-1])
            )

        for power in permission_powers:
            oof_alpha = np.clip(
                np.power(oof_probability, power) * oof_strength,
                0.0,
                float(ALPHAS[-1]),
            )
            oof_ranks = _rank_for_predicted_alpha(frame, oof_alpha)
            metrics = _metric(oof_ranks)
            safe = (
                metrics["mrr"] >= base_metrics["mrr"]
                and metrics["hit10"] >= base_metrics["hit10"]
                and metrics["hit100"] >= base_metrics["hit100"]
                and metrics["hit1000"] >= base_metrics["hit1000"]
                and metrics["median_rank"] <= base_metrics["median_rank"]
            )
            item = {
                "params": params,
                "permission_power": power,
                "safe": bool(safe),
                "base": base_metrics,
                "oof": metrics,
                "mean_permission_probability": float(np.mean(oof_probability)),
                "mean_alpha": float(np.mean(oof_alpha)),
                "median_alpha": float(np.median(oof_alpha)),
                "zero_or_near_zero_fraction": float(np.mean(oof_alpha < 0.05)),
            }
            summaries.append(item)
            key = (
                int(safe),
                metrics["mrr"],
                metrics["hit10"],
                metrics["hit100"],
                metrics["hit1000"],
                -metrics["median_rank"],
            )
            if best is None or key > best[0]:
                best = (key, params, power)

    assert best is not None
    selected_params = best[1]
    selected_power = float(best[2])
    scaler = StandardScaler().fit(X)
    xs = scaler.transform(X)

    permission = HistGradientBoostingClassifier(random_state=random_state, **selected_params)
    permission.fit(xs, helpful)
    positive_rows = np.flatnonzero(helpful > 0)
    strength = HistGradientBoostingRegressor(random_state=random_state, **selected_params)
    strength.fit(xs[positive_rows], strength_target[positive_rows])

    selected_item = next(
        item
        for item in summaries
        if item["params"] == selected_params and item["permission_power"] == selected_power
    )
    asset = {
        "schema": "bridge-adaptive-relation-gate-v2",
        "direction": str(frame.direction.iloc[0]),
        "feature_names": FEATURE_NAMES,
        "alpha_support": ALPHAS.tolist(),
        "target": "validation-oracle alpha minimizing best-positive rank; ties choose smallest alpha",
        "permission_semantics": "continuous probability that nonzero relation authority improves the validation query",
        "strength_semantics": "conditional relation strength among validation queries where relation evidence helps",
        "combination": "alpha(q) = p_help(q)^permission_power * conditional_strength(q)",
        "scaler": scaler,
        "permission_model": permission,
        "strength_model": strength,
        "permission_power": selected_power,
        "selected_params": selected_params,
    }
    return {
        "cv_candidates": summaries,
        "selected_params": selected_params,
        "selected_permission_power": selected_power,
        "selected_safe": bool(selected_item["safe"]),
        "selected_oof": selected_item["oof"],
        "selected_mean_alpha": selected_item["mean_alpha"],
        "selected_median_alpha": selected_item["median_alpha"],
        "selected_zero_or_near_zero_fraction": selected_item["zero_or_near_zero_fraction"],
    }, asset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN, dtype=str).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    validation = _validation_relations(train)

    if args.reuse_features and (OUT / "validation_r2e.csv.gz").exists() and (OUT / "validation_e2r.csv.gz").exists():
        frames = {
            "r2e": pd.read_csv(OUT / "validation_r2e.csv.gz"),
            "e2r": pd.read_csv(OUT / "validation_e2r.csv.gz"),
        }
    else:
        index = FibreCandidateIndex(device=args.device)
        pproto, pmask, rproto, rmask = base_eval.build_prototypes(index, train)
        frames = {
            "r2e": _collect_direction("r2e", index, validation, train, pproto, pmask, rproto, rmask, 24),
            "e2r": _collect_direction("e2r", index, validation, train, pproto, pmask, rproto, rmask, 192),
        }
        for direction, frame in frames.items():
            frame.to_csv(OUT / f"validation_{direction}.csv.gz", index=False)

    assets = {}
    report = {
        "schema": "bridge-adaptive-relation-context-validation-v8",
        "protocol": {
            "fit_surface": "firewalled source-expansion validation relations only",
            "test_metrics_used": False,
            "outer_test_evaluated": False,
            "context_source": "clean2023 exact relation graph only",
            "pair_context": "reciprocal train-neighborhood prototypes with missing-neutral terms",
            "authority": "query-conditioned continuous alpha; no fixed protected prefix and no fixed 6-100 band",
            "candidate_scope": "full ranking surface during validation; production integration will retain the existing BRIDGE candidate universe",
            "selection": "5-fold within-validation cross-validation; aggregate MRR/Hit@10/100/1000/median non-regression required when possible",
        },
        "directions": {},
    }

    for direction, frame in frames.items():
        cv, asset = _cv_fit(frame)
        assets[direction] = asset
        oracle = _metric(frame["oracle_rank"].to_numpy(np.int64))
        base = _metric(frame["base_rank"].to_numpy(np.int64))
        report["directions"][direction] = {
            "queries": int(len(frame)),
            "base": base,
            "oracle_upper_bound": oracle,
            "oracle_nonzero_alpha_fraction": float(np.mean(frame.oracle_alpha.to_numpy(float) > 0)),
            "oracle_alpha_median": float(np.median(frame.oracle_alpha.to_numpy(float))),
            "oracle_alpha_mean": float(np.mean(frame.oracle_alpha.to_numpy(float))),
            **cv,
        }

    with open(ASSET, "wb") as handle:
        pickle.dump({"schema": "bridge-adaptive-relation-gate-bundle-v2", "directions": assets}, handle)
    (OUT / "validation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
