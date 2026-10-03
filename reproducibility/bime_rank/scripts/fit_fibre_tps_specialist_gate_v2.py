from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from projects.active.bridge.model.assets import ROOT

TPS = ROOT / "results/fibre_tps_specialist_response_v2"
V4 = ROOT / "results/fibre_dynamic_router_v4"
OUT = ROOT / "results/fibre_tps_specialist_gate_v2"
TRAIN_FOLDS = (0, 1)
VAL_FOLD = 2
WEIGHTS = np.asarray([0.0, 0.05, 0.10, 0.20, 0.35, 0.50], dtype=np.float64)

FEATURES = [
    "tps_ref_max_cosine",
    "tps_ref_top5_mean",
    "tps_ref_top20_mean",
    "tps_ref_margin_1_2",
    "tps_ref_softmax_entropy",
    "tps_score_mean",
    "tps_score_std",
    "tps_score_top1_margin",
    "tps_score_top20_mean",
]


def z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / max(float(x.std()), 1e-8)


def best_rank(scores: np.ndarray, positive_local: np.ndarray) -> int:
    if not len(positive_local):
        return 0
    order = np.argsort(-scores, kind="stable")
    inverse = np.empty(len(order), dtype=np.int32)
    inverse[order] = np.arange(1, len(order) + 1)
    return int(inverse[positive_local].min())


def load_fold(fold: int) -> pd.DataFrame:
    q = pd.read_csv(TPS / f"fold{fold}/query_features.csv")
    cache = np.load(TPS / f"fold{fold}/score_cache.npy", allow_pickle=True)
    v4 = np.load(V4 / f"prepared/fold{fold}/cache.npy", allow_pickle=True)
    by_q = {str(r["query_id"]): r for r in v4}
    rows = []
    for rec in cache:
        query_id = str(rec["query_id"])
        base = by_q[query_id]
        candidate_rows = np.asarray(rec["candidate_rows"], dtype=np.int64)
        if not np.array_equal(
            candidate_rows, np.asarray(base["candidate_rows"], dtype=np.int64)
        ):
            raise RuntimeError(f"TPS/V4 candidate drift: {query_id}")
        positives = set(map(int, rec["positive_rows"]))
        local_pos = np.asarray(
            [i for i, row in enumerate(candidate_rows) if int(row) in positives],
            dtype=np.int64,
        )
        core = z(np.asarray(base["core"], dtype=np.float64))
        tps = z(np.asarray(rec["tps_score"], dtype=np.float64))
        base_rank = best_rank(core, local_pos)
        base_rr = 0.0 if base_rank <= 0 else 1.0 / base_rank

        best_weight = 0.0
        best_tps_rank = base_rank
        best_rr = base_rr
        for weight in WEIGHTS:
            rank = best_rank(core + float(weight) * tps, local_pos)
            rr = 0.0 if rank <= 0 else 1.0 / rank
            if (rr, -float(weight)) > (best_rr, -best_weight):
                best_weight = float(weight)
                best_tps_rank = rank
                best_rr = rr

        rows.append(
            {
                "query_id": query_id,
                "base_rank": int(base_rank),
                "oracle_tps_weight": best_weight,
                "oracle_tps_rank": int(best_tps_rank),
                "oracle_delta_rr": float(best_rr - base_rr),
            }
        )
    out = q.merge(
        pd.DataFrame(rows), on="query_id", how="inner", validate="one_to_one"
    )
    out["fold"] = fold
    return out


def main() -> None:
    train = pd.concat([load_fold(f) for f in TRAIN_FOLDS], ignore_index=True)
    val = load_fold(VAL_FOLD)
    OUT.mkdir(parents=True, exist_ok=True)

    reference = pd.read_csv(
        ROOT / "results/fibre_application/tps_adapted_coordinate/reaction_tps_adapted.csv",
        dtype=str,
    ).fillna("")
    reference.pop("reaction_id")
    rz = reference.to_numpy(np.float32)
    rz /= np.maximum(np.linalg.norm(rz, axis=1, keepdims=True), 1e-8)
    sim = rz @ rz.T
    np.fill_diagonal(sim, -np.inf)
    ref_nn = sim.max(axis=1)

    threshold_trials = []
    for quantile in (0.90, 0.95, 0.975, 0.98):
        threshold = float(train["tps_ref_max_cosine"].quantile(quantile))
        threshold_trials.append(
            {
                "broad_train_quantile": quantile,
                "threshold": threshold,
                "train_activation": float(
                    (train.tps_ref_max_cosine >= threshold).mean()
                ),
                "val_activation": float(
                    (val.tps_ref_max_cosine >= threshold).mean()
                ),
                "tps_reference_loo_retention": float((ref_nn >= threshold).mean()),
            }
        )
    candidates = [
        row for row in threshold_trials if row["train_activation"] <= 0.025
    ]
    if not candidates:
        candidates = threshold_trials
    selected_threshold = max(
        candidates,
        key=lambda row: (
            row["tps_reference_loo_retention"],
            -row["train_activation"],
        ),
    )
    threshold = float(selected_threshold["threshold"])

    semantic_train = (
        train.tps_ref_max_cosine.to_numpy(float) >= threshold
    )
    target = train.oracle_tps_weight.to_numpy(float)
    strength_mask = semantic_train & (train.base_rank.to_numpy(int) > 0)
    if int(strength_mask.sum()) < 10:
        raise RuntimeError(
            f"too few semantically eligible TPS train queries: {strength_mask.sum()}"
        )
    reg = HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.05,
        max_leaf_nodes=12,
        l2_regularization=3.0,
        random_state=20261002,
    )
    reg.fit(
        train.loc[strength_mask, FEATURES].to_numpy(float),
        target[strength_mask],
    )

    v4_cache = np.load(
        V4 / f"prepared/fold{VAL_FOLD}/cache.npy", allow_pickle=True
    )
    tps_cache = np.load(
        TPS / f"fold{VAL_FOLD}/score_cache.npy", allow_pickle=True
    )
    v4_by_q = {str(r["query_id"]): r for r in v4_cache}
    tps_by_q = {str(r["query_id"]): r for r in tps_cache}

    trials = []
    for scale in (0.25, 0.5, 0.75, 1.0):
        raw = np.clip(
            reg.predict(val[FEATURES].to_numpy(float)),
            0.0,
            WEIGHTS.max(),
        )
        active = val.tps_ref_max_cosine.to_numpy(float) >= threshold
        weights = np.where(active, raw * scale, 0.0)
        base_rr = []
        fused_rr = []
        base_h10 = []
        fused_h10 = []
        improved = worsened = tied = 0
        for i, query_id in enumerate(val.query_id.astype(str)):
            base = v4_by_q[query_id]
            specialist = tps_by_q[query_id]
            candidate_rows = np.asarray(
                base["candidate_rows"], dtype=np.int64
            )
            positives = set(map(int, base["positive_rows"]))
            local = np.asarray(
                [
                    j
                    for j, row in enumerate(candidate_rows)
                    if int(row) in positives
                ],
                dtype=np.int64,
            )
            core = z(np.asarray(base["core"], dtype=np.float64))
            score = z(
                np.asarray(specialist["tps_score"], dtype=np.float64)
            )
            br = best_rank(core, local)
            fr = best_rank(core + float(weights[i]) * score, local)
            brr = 0.0 if br <= 0 else 1.0 / br
            frr = 0.0 if fr <= 0 else 1.0 / fr
            base_rr.append(brr)
            fused_rr.append(frr)
            base_h10.append(int(br > 0 and br <= 10))
            fused_h10.append(int(fr > 0 and fr <= 10))
            if frr > brr:
                improved += 1
            elif frr < brr:
                worsened += 1
            else:
                tied += 1
        trials.append(
            {
                "scale": scale,
                "active_fraction": float(active.mean()),
                "active_count": int(active.sum()),
                "mean_active_weight": (
                    float(weights[active].mean()) if active.any() else 0.0
                ),
                "base_mrr": float(np.mean(base_rr)),
                "fused_mrr": float(np.mean(fused_rr)),
                "delta_mrr": float(
                    np.mean(np.asarray(fused_rr) - np.asarray(base_rr))
                ),
                "base_hit10": float(np.mean(base_h10)),
                "fused_hit10": float(np.mean(fused_h10)),
                "delta_hit10": float(
                    np.mean(
                        np.asarray(fused_h10) - np.asarray(base_h10)
                    )
                ),
                "improved": improved,
                "worsened": worsened,
                "tied": tied,
            }
        )
    selected = max(
        trials,
        key=lambda row: (
            row["delta_mrr"],
            row["delta_hit10"],
            -row["worsened"],
            -row["scale"],
        ),
    )

    with open(OUT / "gate.pkl", "wb") as handle:
        pickle.dump(
            {
                "regressor": reg,
                "features": FEATURES,
                "semantic_threshold": threshold,
                "strength_scale": float(selected["scale"]),
                "weights_grid": WEIGHTS,
            },
            handle,
        )
    train.to_csv(OUT / "train_oracle.csv", index=False)
    val.to_csv(OUT / "validation_oracle.csv", index=False)
    result = {
        "schema": "fibre-tps-specialist-gate-v2",
        "status": (
            "development_selected"
            if selected["delta_mrr"] > 0
            else "not_promoted"
        ),
        "semantic_gate": {
            "feature": "tps_ref_max_cosine",
            "threshold": threshold,
            "selection": selected_threshold,
            "all_threshold_trials": threshold_trials,
            "principle": (
                "TPS pair score cannot activate the specialist. The query must "
                "first lie inside a conservative historical TPS reaction manifold."
            ),
        },
        "strength": {
            "model": "HistGradientBoostingRegressor",
            "features": FEATURES,
            "selected_scale": float(selected["scale"]),
            "trained_only_on_semantically_eligible_train_queries": True,
            "eligible_train_queries": int(strength_mask.sum()),
        },
        "validation": selected,
        "validation_trials": trials,
        "external_or_outer_labels_used": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
