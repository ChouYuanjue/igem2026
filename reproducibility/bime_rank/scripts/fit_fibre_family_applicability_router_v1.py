from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support

from projects.active.fibre.model.assets import ROOT

REACTION_ROOT = ROOT / "results/enzymecage_reaction_family_response_v1"
ORACLE_ROOT = ROOT / "results/fibre_cage_family_specialist_gate_v1"
OUT = ROOT / "results/fibre_family_applicability_router_v1"
FAMILIES = ("p450", "phosphatase", "terpene")
TRAIN_FOLDS = (0, 1)
VAL_FOLD = 2

FEATURES = [
    *(f"{fam}_reaction_pre_cosine_shift" for fam in FAMILIES),
    *(f"{fam}_reaction_pre_norm_log_ratio" for fam in FAMILIES),
    *(f"{fam}_reaction_post_cosine_shift" for fam in FAMILIES),
    *(f"{fam}_reaction_post_norm_log_ratio" for fam in FAMILIES),
    *(f"{fam}_reaction_attention_entropy_delta" for fam in FAMILIES),
    *(f"{fam}_reaction_attention_peak_delta" for fam in FAMILIES),
    "reaction_family_margin",
    "reaction_family_entropy",
]


def load_reaction_features(fold: int) -> pd.DataFrame:
    d = pd.read_csv(
        REACTION_ROOT / f"internal_fold{fold}/query_features.csv",
        dtype={"reaction_id": str},
    ).fillna("")
    d = d.drop_duplicates("reaction_id", keep="first")
    for c in FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    return d


def semantic_labels(path: Path) -> dict[str, str]:
    d = pd.read_csv(path, dtype={"query_id": str}).fillna("")
    # Oracle rows only exist after the old strict semantic family gate, and
    # each query has at most one family because the gate used a winner rule.
    labels = {}
    for q, g in d.groupby("query_id", sort=False):
        fams = list(dict.fromkeys(g["family"].astype(str)))
        if len(fams) != 1:
            raise RuntimeError(f"non-unique family label for {q}: {fams}")
        labels[str(q)] = fams[0]
    return labels


def assemble(folds: tuple[int, ...], labels: dict[str, str]) -> pd.DataFrame:
    parts = []
    for f in folds:
        d = load_reaction_features(f)
        d["fold"] = f
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out["target"] = out["reaction_id"].map(labels).fillna("none")
    return out


def metrics(y_true, y_pred):
    result = {}
    for label in ("none", *FAMILIES):
        yt = np.asarray(y_true) == label
        yp = np.asarray(y_pred) == label
        tp = int((yt & yp).sum())
        fp = int((~yt & yp).sum())
        fn = int((yt & ~yp).sum())
        result[label] = {
            "support": int(yt.sum()),
            "predicted": int(yp.sum()),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
        }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_labels = semantic_labels(ORACLE_ROOT / "train_oracle.csv")
    val_labels = semantic_labels(ORACLE_ROOT / "validation_oracle.csv")
    train = assemble(TRAIN_FOLDS, train_labels)
    val = assemble((VAL_FOLD,), val_labels)

    x_train = train[FEATURES].to_numpy(np.float32)
    x_val = val[FEATURES].to_numpy(np.float32)
    y_train = train["target"].to_numpy(str)
    y_val = val["target"].to_numpy(str)

    model = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=3.0,
        random_state=20261002,
    )
    model.fit(x_train, y_train)

    probs = model.predict_proba(x_val)
    classes = list(model.classes_)
    # Select only a global confidence threshold; family identity remains the
    # model argmax.  Threshold is selected on validation semantic labels, not
    # on seven-cell performance.
    trials = []
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
        raw_idx = probs.argmax(1)
        raw_cls = np.asarray([classes[i] for i in raw_idx], dtype=object)
        raw_p = probs.max(1)
        pred = np.where(raw_p >= threshold, raw_cls, "none")
        m = metrics(y_val, pred)
        fam_true = np.asarray(y_val) != "none"
        fam_pred = np.asarray(pred) != "none"
        tp = int((fam_true & fam_pred & (np.asarray(y_val) == np.asarray(pred))).sum())
        fp = int((fam_pred & ((np.asarray(y_val) != np.asarray(pred)) | ~fam_true)).sum())
        fn = int((fam_true & (np.asarray(y_val) != np.asarray(pred))).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        trials.append(
            {
                "threshold": threshold,
                "precision": float(precision),
                "recall": float(recall),
                "predicted_specialist_queries": int(fam_pred.sum()),
                "per_class": m,
            }
        )

    # Conservative selector: maximize precision first, then recall, while
    # requiring at least one predicted specialist query.
    viable = [x for x in trials if x["predicted_specialist_queries"] > 0]
    selected = max(viable, key=lambda x: (x["precision"], x["recall"], x["threshold"]))
    threshold = float(selected["threshold"])

    with open(OUT / "router.pkl", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "features": FEATURES,
                "threshold": threshold,
                "classes": classes,
            },
            f,
        )
    pd.DataFrame(trials).to_json(
        OUT / "validation_trials.json", orient="records", indent=2
    )
    result = {
        "schema": "fibre-family-applicability-router-v1",
        "status": "development_selected",
        "role": (
            "choose at most one CAGE family specialist from reaction-only "
            "features; prediction below confidence threshold means no CAGE "
            "family specialist"
        ),
        "train_folds": list(TRAIN_FOLDS),
        "validation_fold": VAL_FOLD,
        "features": FEATURES,
        "selected_threshold": threshold,
        "selected_validation": selected,
        "seven_cell_labels_used": False,
        "official_family_test_labels_used": False,
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
