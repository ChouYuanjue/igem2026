from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex
from reproducibility.bime_rank.scripts.fit_fibre_dynamic_router_v4 import (
    BASE_FEATURE_NAMES,
    PCA_DIM,
    TRAIN_FOLDS,
    VAL_FOLD,
    WEIGHTS,
    _concat,
    _evaluate,
    _extra_pocket_features,
    _fit_models,
    _load_fold,
    _oracle_targets,
    _predict,
    _reaction_features,
)

REACTION_ROOT = ROOT / "results/enzymecage_reaction_family_response_v1"
OUT = ROOT / "results/fibre_dynamic_router_v6_reaction_family_guidance"
FAMILIES = ("p450", "phosphatase", "terpene")
RAW_SUFFIXES = (
    "reaction_pre_cosine_shift",
    "reaction_pre_norm_log_ratio",
    "reaction_post_cosine_shift",
    "reaction_post_norm_log_ratio",
    "reaction_attention_entropy_delta",
    "reaction_attention_peak_delta",
)


def raw_feature_names() -> list[str]:
    names = [f"{fam}_{suffix}" for fam in FAMILIES for suffix in RAW_SUFFIXES]
    names += ["reaction_family_margin", "reaction_family_entropy"]
    names += [f"reaction_family_winner_{fam}" for fam in FAMILIES]
    names += ["reaction_family_probe_available"]
    return names


def load_raw_covariates(fold: int, query_ids: list[str]) -> np.ndarray:
    p = REACTION_ROOT / f"internal_fold{fold}/query_features.csv"
    frame = pd.read_csv(p, dtype=str).fillna("")
    frame = frame.set_index("reaction_id", drop=False)
    rows = []
    for q in map(str, query_ids):
        if q not in frame.index:
            vals = [0.0] * (len(FAMILIES) * len(RAW_SUFFIXES) + 2 + len(FAMILIES))
            vals.append(0.0)
            rows.append(vals)
            continue
        r = frame.loc[q]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        vals = [float(r[f"{fam}_{suffix}"]) for fam in FAMILIES for suffix in RAW_SUFFIXES]
        vals += [float(r["reaction_family_margin"]), float(r["reaction_family_entropy"])]
        winner = str(r["reaction_family_winner"])
        vals += [1.0 if winner == fam else 0.0 for fam in FAMILIES]
        vals.append(1.0)
        rows.append(vals)
    x = np.asarray(rows, dtype=np.float32)
    if x.shape != (len(query_ids), len(raw_feature_names())):
        raise RuntimeError((x.shape, len(query_ids), len(raw_feature_names())))
    if not np.isfinite(x).all():
        raise RuntimeError("non-finite reaction-family covariates")
    return x


def normalize_train_val(train: np.ndarray, val: np.ndarray):
    continuous = len(FAMILIES) * len(RAW_SUFFIXES) + 2
    mean = train[:, :continuous].mean(0, keepdims=True)
    std = train[:, :continuous].std(0, keepdims=True)
    std = np.maximum(std, 1e-6)
    a = train.copy()
    b = val.copy()
    a[:, :continuous] = (a[:, :continuous] - mean) / std
    b[:, :continuous] = (b[:, :continuous] - mean) / std
    return a, b, mean.astype(np.float32), std.astype(np.float32)


def select(train_X, val_X, train, val):
    targets = _oracle_targets(train)
    val_targets = _oracle_targets(val)
    fitted = _fit_models(train_X, targets)
    trials = []
    for tf in (0.35, 0.45, 0.55, 0.65):
        for tg in (0.35, 0.45, 0.55, 0.65):
            for sf in (0.75, 1.0):
                for sg in (0.75, 1.0):
                    pred = _predict(
                        fitted,
                        val_X,
                        {"functional": tf, "geometry": tg},
                        {"functional": sf, "geometry": sg},
                    )
                    _, summary = _evaluate(val, pred)
                    trials.append(
                        {
                            "functional_threshold": tf,
                            "geometry_threshold": tg,
                            "functional_scale": sf,
                            "geometry_scale": sg,
                            **summary,
                        }
                    )
    table = pd.DataFrame(trials)
    best = table.sort_values(
        [
            "delta_mrr_top1000",
            "delta_hit10",
            "both_active_fraction",
            "functional_active_fraction",
            "geometry_active_fraction",
        ],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).iloc[0]
    thresholds = {
        "functional": float(best.functional_threshold),
        "geometry": float(best.geometry_threshold),
    }
    scales = {
        "functional": float(best.functional_scale),
        "geometry": float(best.geometry_scale),
    }
    pred = _predict(fitted, val_X, thresholds, scales)
    qmetrics, summary = _evaluate(val, pred)
    applicability = {}
    for family in ("functional", "geometry"):
        y = (val_targets[f"{family}_weight"] > 0).astype(int)
        prob = pred[family]["prob"]
        applicability[family] = {
            "oracle_active_fraction": float(y.mean()),
            "validation_auc": float(roc_auc_score(y, prob)) if len(set(y)) > 1 else None,
            "predicted_active_fraction": float((pred[family]["weight"] > 0).mean()),
        }
    return fitted, thresholds, scales, table, qmetrics, summary, applicability


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    train_parts = [_load_fold(fold) for fold in TRAIN_FOLDS]
    val = _load_fold(VAL_FOLD)
    train = _concat(train_parts)

    reaction_train_raw = np.concatenate(
        [load_raw_covariates(fold, train_parts[i]["query_ids"]) for i, fold in enumerate(TRAIN_FOLDS)],
        axis=0,
    )
    reaction_val_raw = load_raw_covariates(VAL_FOLD, val["query_ids"])
    reaction_train, reaction_val, family_mean, family_std = normalize_train_val(
        reaction_train_raw, reaction_val_raw
    )

    index = FibreCandidateIndex(device=args.device)
    train_r = _reaction_features(train["query_ids"], index)
    val_r = _reaction_features(val["query_ids"], index)
    pca = PCA(n_components=PCA_DIM, whiten=True, random_state=20261002)
    train_pca = pca.fit_transform(train_r).astype(np.float32)
    val_pca = pca.transform(val_r).astype(np.float32)

    train_X = np.column_stack([
        train["base_x"],
        _extra_pocket_features(train["pocket"]),
        train_pca,
        reaction_train,
    ]).astype(np.float32)
    val_X = np.column_stack([
        val["base_x"],
        _extra_pocket_features(val["pocket"]),
        val_pca,
        reaction_val,
    ]).astype(np.float32)

    models, thresholds, scales, table, qmetrics, summary, applicability = select(
        train_X, val_X, train, val
    )

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "validation_trials.csv", index=False)
    qmetrics.to_csv(OUT / "validation_query_metrics.csv", index=False)
    with open(OUT / "router.pkl", "wb") as f:
        pickle.dump(
            {
                "pca": pca,
                "models": models,
                "thresholds": thresholds,
                "scales": scales,
                "base_feature_names": BASE_FEATURE_NAMES,
                "reaction_family_feature_names": raw_feature_names(),
                "reaction_family_mean": family_mean,
                "reaction_family_std": family_std,
                "pca_dim": PCA_DIM,
                "weights_grid": WEIGHTS,
            },
            f,
        )

    v4 = json.loads((ROOT / "results/fibre_dynamic_router_v4/summary.json").read_text())
    v5_path = ROOT / "results/fibre_dynamic_router_v5_family_guidance/summary.json"
    v5 = json.loads(v5_path.read_text()) if v5_path.exists() else None
    result = {
        "schema": "fibre-dynamic-router-v6-reaction-family-guidance",
        "status": "validation_only",
        "change_from_v4": (
            "append candidate-independent generic-to-family EnzymeCAGE reaction-branch "
            "adaptation covariates; keep V4 scores, targets, folds, PCA policy, model "
            "classes and validation grid unchanged"
        ),
        "reaction_family_guidance": {
            "source": "generic + P450/phosphatase/terpene EnzymeCAGE checkpoints",
            "branch": "molecule_encoder + reaction_cross_attn only",
            "candidate_protein_used": False,
            "external_family_labels_used_for_fit": False,
            "feature_names": raw_feature_names(),
            "train_probe_available_fraction": float(reaction_train_raw[:, -1].mean()),
            "validation_probe_available_fraction": float(reaction_val_raw[:, -1].mean()),
        },
        "split": {
            "train": "same V4 internal strict double-cold fold0+fold1",
            "validation": "same V4 internal strict double-cold fold2",
            "outer_test": "untouched; run only if reaction-only guidance is competitive",
        },
        "selected_thresholds": thresholds,
        "selected_scales": scales,
        "applicability": applicability,
        "validation": summary,
        "v4_validation": v4["validation"],
        "v5_pair_aggregated_validation": (None if v5 is None else v5["validation"]),
        "comparison": {
            "fused_mrr_vs_v4": float(summary["fused_mrr_top1000"] - v4["validation"]["fused_mrr_top1000"]),
            "fused_hit10_vs_v4": float(summary["fused_hit10"] - v4["validation"]["fused_hit10"]),
            "fused_mrr_vs_v5_pair": (
                None if v5 is None else float(
                    summary["fused_mrr_top1000"]
                    - v5["validation"]["fused_mrr_top1000"]
                )
            ),
            "fused_hit10_vs_v5_pair": (
                None if v5 is None else float(
                    summary["fused_hit10"]
                    - v5["validation"]["fused_hit10"]
                )
            ),
        },
        "outer_test_run": False,
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
