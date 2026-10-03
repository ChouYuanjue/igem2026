from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex

CACHE_ROOT = ROOT / "results/fibre_dynamic_router_v4/prepared"
OUT = ROOT / "results/fibre_dynamic_router_v4"
TRAIN_FOLDS = (0, 1)
VAL_FOLD = 2
WEIGHTS = np.asarray([0.0, 0.10, 0.20, 0.35, 0.50, 0.75], dtype=np.float32)
PCA_DIM = 16

BASE_FEATURE_NAMES = [
    "expert_available_fraction",
    "expert_score_std",
    "expert_top1_margin",
    "expert_core_corr",
    "top20_overlap",
    "core_top1_margin",
    "core_score_std",
    "pocket_support_fraction",
]


def _load_fold(fold: int):
    rows = np.load(CACHE_ROOT / f"fold{fold}" / "cache.npy", allow_pickle=True)
    feat = pd.read_csv(CACHE_ROOT / f"fold{fold}" / "query_features.csv")
    if len(rows) != len(feat):
        raise ValueError((fold, len(rows), len(feat)))
    core = np.stack([r["core"] for r in rows]).astype(np.float32)
    functional = np.stack([r["functional"] for r in rows]).astype(np.float32)
    geometry = np.stack([r["geometry"] for r in rows]).astype(np.float32)
    candidates = np.stack([r["candidate_rows"] for r in rows]).astype(np.int32)
    pocket = np.stack([r["pocket"] for r in rows]).astype(bool)
    pos = np.zeros_like(core, dtype=bool)
    for i, r in enumerate(rows):
        positives = set(map(int, r["positive_rows"]))
        pos[i] = np.asarray([int(x) in positives for x in candidates[i]], dtype=bool)
    cols = (
        [f"functional_{n}" for n in BASE_FEATURE_NAMES]
        + [f"geometry_{n}" for n in BASE_FEATURE_NAMES]
    )
    base_x = feat[cols].to_numpy(np.float32)
    return {
        "query_ids": feat["query_id"].astype(str).tolist(),
        "base_x": base_x,
        "core": core,
        "functional": functional,
        "geometry": geometry,
        "positive": pos,
        "pocket": pocket,
    }


def _concat(parts):
    return {
        "query_ids": sum((p["query_ids"] for p in parts), []),
        "base_x": np.concatenate([p["base_x"] for p in parts]),
        "core": np.concatenate([p["core"] for p in parts]),
        "functional": np.concatenate([p["functional"] for p in parts]),
        "geometry": np.concatenate([p["geometry"] for p in parts]),
        "positive": np.concatenate([p["positive"] for p in parts]),
        "pocket": np.concatenate([p["pocket"] for p in parts]),
    }


def _reaction_features(query_ids: list[str], index: FibreCandidateIndex) -> np.ndarray:
    rows = [index.reaction_index[q] for q in query_ids]
    return index.reaction_embeddings.detach().cpu().numpy()[rows].astype(np.float32)


def _extra_pocket_features(pocket: np.ndarray) -> np.ndarray:
    return np.column_stack([
        pocket[:, :20].mean(axis=1),
        pocket[:, :100].mean(axis=1),
        pocket[:, :500].mean(axis=1),
        pocket.mean(axis=1),
    ]).astype(np.float32)


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)) / np.clip(x.std(axis=1, keepdims=True), 1e-6, None)


def _rr_hit(scores: np.ndarray, pos: np.ndarray) -> tuple[float, int]:
    p = np.flatnonzero(pos)
    if len(p) == 0:
        return 0.0, 0
    order = np.argsort(-scores, kind="stable")
    inv = np.empty(len(order), dtype=np.int32)
    inv[order] = np.arange(1, len(order) + 1)
    rank = int(inv[p].min())
    return 1.0 / rank, int(rank <= 10)


def _oracle_targets(data):
    core = _z(data["core"].astype(np.float64))
    f = data["functional"].astype(np.float64)
    g = data["geometry"].astype(np.float64)
    pos = data["positive"]
    wf = np.zeros(len(core), dtype=np.float32)
    wg = np.zeros(len(core), dtype=np.float32)
    base_rr = np.zeros(len(core), dtype=np.float32)
    oracle_rr = np.zeros(len(core), dtype=np.float32)
    oracle_hit = np.zeros(len(core), dtype=np.int8)
    eligible = pos.any(axis=1)

    for i in range(len(core)):
        if not eligible[i]:
            continue
        br, _ = _rr_hit(core[i], pos[i]); base_rr[i] = br
        best = (br, 0, 0.0, 0.0)  # rr, hit10, -l1, -maxw via sort below
        best_pair = (0.0, 0.0)
        for a in WEIGHTS:
            for b in WEIGHTS:
                score = core[i] + float(a) * f[i] + float(b) * g[i]
                rr, hit = _rr_hit(score, pos[i])
                key = (rr, hit, -float(a+b), -float(max(a,b)))
                if key > best:
                    best = key
                    best_pair = (float(a), float(b))
        oracle_rr[i] = best[0]
        oracle_hit[i] = best[1]
        wf[i], wg[i] = best_pair
    return {
        "functional_weight": wf,
        "geometry_weight": wg,
        "base_rr": base_rr,
        "oracle_rr": oracle_rr,
        "oracle_hit10": oracle_hit,
        "eligible": eligible,
    }


def _fit_models(X: np.ndarray, targets: dict[str, np.ndarray]):
    models = {}
    for fam in ("functional", "geometry"):
        y = targets[f"{fam}_weight"]
        eligible = targets["eligible"]
        use = (y > 0).astype(int)
        cls = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=2.0, random_state=20261002,
        )
        cls.fit(X[eligible], use[eligible])
        reg = HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=2.0, random_state=20261002,
        )
        mask = eligible & (y > 0)
        reg.fit(X[mask], y[mask])
        models[fam] = (cls, reg)
    return models


def _predict(models, X, thresholds, scales):
    out = {}
    for fam in ("functional", "geometry"):
        cls, reg = models[fam]
        p = cls.predict_proba(X)[:, 1]
        strength = np.clip(reg.predict(X), 0.0, float(WEIGHTS.max()))
        w = np.where(p >= thresholds[fam], strength * scales[fam], 0.0)
        out[fam] = {"prob": p, "weight": np.clip(w, 0.0, float(WEIGHTS.max()))}
    return out


def _evaluate(data, pred):
    core = _z(data["core"].astype(np.float64))
    fused = (
        core
        + pred["functional"]["weight"][:, None] * data["functional"]
        + pred["geometry"]["weight"][:, None] * data["geometry"]
    )
    rows = []
    for i, q in enumerate(data["query_ids"]):
        br, bh = _rr_hit(core[i], data["positive"][i])
        fr, fh = _rr_hit(fused[i], data["positive"][i])
        rows.append({
            "query_id": q,
            "eligible": bool(data["positive"][i].any()),
            "core_rr_top1000": br,
            "fused_rr_top1000": fr,
            "core_hit10": bh,
            "fused_hit10": fh,
            "functional_prob": float(pred["functional"]["prob"][i]),
            "functional_weight": float(pred["functional"]["weight"][i]),
            "geometry_prob": float(pred["geometry"]["prob"][i]),
            "geometry_weight": float(pred["geometry"]["weight"][i]),
        })
    frame = pd.DataFrame(rows)
    e = frame[frame.eligible]
    return frame, {
        "queries": int(len(frame)),
        "eligible_queries": int(len(e)),
        "core_mrr_top1000": float(e.core_rr_top1000.mean()),
        "fused_mrr_top1000": float(e.fused_rr_top1000.mean()),
        "delta_mrr_top1000": float((e.fused_rr_top1000-e.core_rr_top1000).mean()),
        "core_hit10": float(e.core_hit10.mean()),
        "fused_hit10": float(e.fused_hit10.mean()),
        "delta_hit10": float((e.fused_hit10-e.core_hit10).mean()),
        "functional_active_fraction": float((frame.functional_weight > 1e-8).mean()),
        "functional_weight_mean_active": float(frame.loc[frame.functional_weight > 1e-8, "functional_weight"].mean()) if (frame.functional_weight > 1e-8).any() else 0.0,
        "geometry_active_fraction": float((frame.geometry_weight > 1e-8).mean()),
        "geometry_weight_mean_active": float(frame.loc[frame.geometry_weight > 1e-8, "geometry_weight"].mean()) if (frame.geometry_weight > 1e-8).any() else 0.0,
        "both_active_fraction": float(((frame.functional_weight > 1e-8) & (frame.geometry_weight > 1e-8)).mean()),
        "neither_active_fraction": float(((frame.functional_weight <= 1e-8) & (frame.geometry_weight <= 1e-8)).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    train = _concat([_load_fold(f) for f in TRAIN_FOLDS])
    val = _load_fold(VAL_FOLD)
    index = FibreCandidateIndex(device=args.device)
    train_r = _reaction_features(train["query_ids"], index)
    val_r = _reaction_features(val["query_ids"], index)

    pca = PCA(n_components=PCA_DIM, whiten=True, random_state=20261002)
    train_pca = pca.fit_transform(train_r).astype(np.float32)
    val_pca = pca.transform(val_r).astype(np.float32)
    train_X = np.column_stack([train["base_x"], _extra_pocket_features(train["pocket"]), train_pca]).astype(np.float32)
    val_X = np.column_stack([val["base_x"], _extra_pocket_features(val["pocket"]), val_pca]).astype(np.float32)

    targets = _oracle_targets(train)
    val_targets = _oracle_targets(val)
    models = _fit_models(train_X, targets)

    # Threshold/scale selection uses validation only.  No expert inference is rerun.
    trials = []
    for tf in (0.35, 0.45, 0.55, 0.65):
        for tg in (0.35, 0.45, 0.55, 0.65):
            for sf in (0.75, 1.0):
                for sg in (0.75, 1.0):
                    pred = _predict(
                        models, val_X,
                        {"functional": tf, "geometry": tg},
                        {"functional": sf, "geometry": sg},
                    )
                    _, summary = _evaluate(val, pred)
                    trials.append({
                        "functional_threshold": tf, "geometry_threshold": tg,
                        "functional_scale": sf, "geometry_scale": sg,
                        **summary,
                    })
    table = pd.DataFrame(trials)
    # Prefer MRR then Hit@10, then fewer active experts for the same utility.
    best = table.sort_values(
        ["delta_mrr_top1000", "delta_hit10", "both_active_fraction", "functional_active_fraction", "geometry_active_fraction"],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).iloc[0]
    thresholds = {"functional": float(best.functional_threshold), "geometry": float(best.geometry_threshold)}
    scales = {"functional": float(best.functional_scale), "geometry": float(best.geometry_scale)}
    pred = _predict(models, val_X, thresholds, scales)
    qmetrics, summary = _evaluate(val, pred)

    # Diagnostics for whether the applicability models actually learned something.
    applicability = {}
    for fam in ("functional", "geometry"):
        y = (val_targets[f"{fam}_weight"] > 0).astype(int)
        prob = pred[fam]["prob"]
        applicability[fam] = {
            "oracle_active_fraction": float(y.mean()),
            "validation_auc": float(roc_auc_score(y, prob)) if len(set(y)) > 1 else None,
            "predicted_active_fraction": float((pred[fam]["weight"] > 0).mean()),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "validation_trials.csv", index=False)
    qmetrics.to_csv(OUT / "validation_query_metrics.csv", index=False)
    with open(OUT / "router.pkl", "wb") as f:
        pickle.dump({
            "pca": pca, "models": models, "thresholds": thresholds, "scales": scales,
            "base_feature_names": BASE_FEATURE_NAMES,
            "pca_dim": PCA_DIM,
            "weights_grid": WEIGHTS,
        }, f)
    result = {
        "schema": "fibre-applicability-strength-router-v4",
        "status": "development_selected",
        "expert_families": {
            "functional_evolutionary": ["EnzGFM", "seed/homology when supplied"],
            "structural_mechanistic": ["CLIPZyme", "reaction-center", "cached pocket support"],
            "domain_specialist": ["P450/Phosphatase/Terpene fine-tuned CAGE", "TPS specialist"],
        },
        "method": {
            "stage1": "predict whether an expert family should speak at all",
            "stage2": "predict query-specific residual strength only when applicable",
            "training_target": "per-query oracle (functional_weight, geometry_weight) from a frozen 6x6 nonnegative weight grid; ties choose less expert mass",
            "router_inputs": "expert support/confidence/disagreement + Broad confidence + cached pocket coverage at top20/top100/top500/top1000 + train-only PCA of Broad reaction embedding",
            "final_score": "z(Broad) + w_functional(q)*z(functional) + w_geometry(q)*z(geometry)",
            "domain_specialist": "separate hard applicability gate; never trained from held-out family test labels",
        },
        "split": {
            "train": "internal strict double-cold fold0+fold1",
            "validation": "internal strict double-cold fold2",
            "outer_test": "the full previously frozen R2E benchmark suite; still untouched",
        },
        "selected_thresholds": thresholds,
        "selected_scales": scales,
        "applicability": applicability,
        "validation": summary,
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
