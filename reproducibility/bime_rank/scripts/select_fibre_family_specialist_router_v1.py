from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex
from reproducibility.bime_rank.scripts.fit_fibre_dynamic_router_v4 import (
    _extra_pocket_features,
    _load_fold,
    _predict,
    _reaction_features,
)
from reproducibility.bime_rank.scripts.fit_fibre_family_applicability_router_v1 import (
    FEATURES as APPLICABILITY_FEATURES,
    load_reaction_features,
)

FOLD = 2
FAMILIES = ("p450", "phosphatase", "terpene")
V4 = ROOT / "results/fibre_dynamic_router_v4"
PAIR_ROOT = ROOT / "results/enzymecage_family_response_v1"
GATE_ROOT = ROOT / "results/fibre_cage_family_specialist_gate_v1"
APP_ROOT = ROOT / "results/fibre_family_applicability_router_v1"
OUT = ROOT / "results/fibre_integrated_family_specialist_validation_v1"
ENTRIES = ROOT / "data/catalyst_candidate_universes/general_merged/proteins/entries.csv"

SELECTED_SCALE = {
    "p450": 0.5,
    "phosphatase": 0.25,
    "terpene": 0.75,
}

def z(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if mask is None:
        mask = np.ones(len(x), dtype=bool)
    out = np.zeros(len(x), dtype=float)
    if mask.any():
        v = x[mask]
        out[mask] = (v - v.mean()) / max(float(v.std()), 1e-8)
    return out

def best_rank(scores: np.ndarray, pos: np.ndarray) -> int:
    p = np.flatnonzero(pos)
    if not len(p):
        return 0
    order = np.argsort(-scores, kind="stable")
    inv = np.empty(len(order), dtype=np.int32)
    inv[order] = np.arange(1, len(order) + 1)
    return int(inv[p].min())

def protein_row_map() -> dict[str, int]:
    d = pd.read_csv(ENTRIES, dtype=str).fillna("")
    d["row"] = pd.to_numeric(d["row"]).astype(int)
    return dict(zip(d["Entry"].astype(str), d["row"].astype(int)))

def v4_baseline(index, router):
    data = _load_fold(FOLD)
    raw = np.load(V4 / f"prepared/fold{FOLD}/cache.npy", allow_pickle=True)
    rx = _reaction_features(data["query_ids"], index)
    X = np.column_stack(
        [
            data["base_x"],
            _extra_pocket_features(data["pocket"]),
            router["pca"].transform(rx).astype(np.float32),
        ]
    ).astype(np.float32)
    pred = _predict(router["models"], X, router["thresholds"], router["scales"])
    out = {}
    for i, q in enumerate(data["query_ids"]):
        core = z(data["core"][i])
        score = (
            core
            + float(pred["functional"]["weight"][i]) * np.asarray(data["functional"][i], float)
            + float(pred["geometry"]["weight"][i]) * np.asarray(data["geometry"][i], float)
        )
        out[str(q)] = {
            "score": score,
            "candidate_rows": np.asarray(raw[i]["candidate_rows"], dtype=np.int64),
            "positive": np.asarray(data["positive"][i], dtype=bool),
        }
    return out

def family_feature_columns(fam: str, suffixes) -> list[str]:
    return [f"{fam}_{s}" for s in suffixes] + [
        "family_calibrated_max_percentile",
        "family_calibrated_margin",
        "family_calibrated_entropy",
        "candidate_count",
    ]

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(V4 / "router.pkl", "rb") as f:
        v4_router = pickle.load(f)
    with open(APP_ROOT / "router.pkl", "rb") as f:
        app = pickle.load(f)
    with open(GATE_ROOT / "gates.pkl", "rb") as f:
        strength = pickle.load(f)

    index = FibreCandidateIndex(device="cuda")
    base = v4_baseline(index, v4_router)
    row_map = protein_row_map()

    reaction = load_reaction_features(FOLD).set_index("reaction_id", drop=False)
    pair = pd.read_csv(PAIR_ROOT / f"internal_fold{FOLD}/pair_features.csv.gz", dtype=str).fillna("")
    qf = pd.read_csv(PAIR_ROOT / f"internal_fold{FOLD}/query_features_calibrated.csv")
    qf = qf.set_index("CANO_RXN_SMILES", drop=False)
    pair_groups = {str(s): g for s, g in pair.groupby("CANO_RXN_SMILES", sort=False)}

    # reaction-id -> smiles
    rxmeta = pd.read_csv(
        ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv",
        dtype=str,
    ).fillna("")
    smiles = dict(zip(rxmeta.reaction_id.astype(str), rxmeta.reaction_smiles.astype(str)))

    trials = []
    query_outputs = {}
    for threshold in (0.50, 0.60, 0.70):
        rows = []
        for q, b in base.items():
            score = b["score"].copy()
            br = best_rank(score, b["positive"])
            active = "none"
            weight = 0.0

            if q in reaction.index:
                rr = reaction.loc[q]
                if isinstance(rr, pd.DataFrame):
                    rr = rr.iloc[0]
                X = rr[APPLICABILITY_FEATURES].to_numpy(dtype=float)[None, :]
                prob = app["model"].predict_proba(X)[0]
                cls = app["classes"][int(np.argmax(prob))]
                confidence = float(np.max(prob))
                if cls in FAMILIES and confidence >= threshold:
                    rxn = smiles.get(q, "")
                    if rxn in qf.index and rxn in pair_groups:
                        qr = qf.loc[rxn]
                        if isinstance(qr, pd.DataFrame):
                            qr = qr.iloc[0]
                        pg = pair_groups[rxn]
                        local_by_global = {
                            int(r): i for i, r in enumerate(b["candidate_rows"])
                        }
                        residual = np.zeros(len(score), dtype=float)
                        avail = np.zeros(len(score), dtype=bool)
                        col = f"{cls}_normalized_logit_response"
                        for pid, rawv in pg[["protein_id", col]].itertuples(index=False):
                            gr = row_map.get(str(pid))
                            if gr is None:
                                continue
                            loc = local_by_global.get(int(gr))
                            if loc is None:
                                continue
                            residual[loc] = float(rawv)
                            avail[loc] = True
                        residual = z(residual, avail)
                        cols = family_feature_columns(cls, strength["feature_suffixes"])
                        feat = [float(qr[c]) for c in cols] + [float(avail.mean())]
                        raw_w = float(
                            np.clip(
                                strength["models"][cls].predict(
                                    np.asarray(feat, dtype=float)[None, :]
                                )[0],
                                0.0,
                                0.50,
                            )
                        )
                        weight = raw_w * SELECTED_SCALE[cls]
                        if weight > 0:
                            score += weight * residual
                            active = cls

            fr = best_rank(score, b["positive"])
            rows.append(
                {
                    "query_id": q,
                    "base_rank": br,
                    "final_rank": fr,
                    "active_family": active,
                    "weight": weight,
                }
            )
        d = pd.DataFrame(rows)
        e = d[d.base_rank > 0].copy()
        delta = 1.0 / e.final_rank - 1.0 / e.base_rank
        trial = {
            "threshold": threshold,
            "queries": int(len(d)),
            "active_queries": int(d.active_family.ne("none").sum()),
            "active_by_family": {
                fam: int(d.active_family.eq(fam).sum()) for fam in FAMILIES
            },
            "base_mrr": float((1.0 / e.base_rank).mean()),
            "final_mrr": float((1.0 / e.final_rank).mean()),
            "delta_mrr": float(delta.mean()),
            "base_hit10": float((e.base_rank <= 10).mean()),
            "final_hit10": float((e.final_rank <= 10).mean()),
            "delta_hit10": float(
                (e.final_rank <= 10).mean() - (e.base_rank <= 10).mean()
            ),
            "improved": int((delta > 0).sum()),
            "worsened": int((delta < 0).sum()),
            "tied": int((delta == 0).sum()),
        }
        trials.append(trial)
        query_outputs[threshold] = d

    # Select by ranking quality, then fewer harmed queries, then more active coverage.
    selected = max(
        trials,
        key=lambda x: (
            x["delta_mrr"],
            x["delta_hit10"],
            -x["worsened"],
            x["active_queries"],
        ),
    )
    threshold = float(selected["threshold"])
    with open(APP_ROOT / "router.pkl", "rb") as f:
        bundle = pickle.load(f)
    bundle["threshold"] = threshold
    bundle["threshold_selection"] = "validation ranking impact after actual family residual correction"
    with open(APP_ROOT / "router.pkl", "wb") as f:
        pickle.dump(bundle, f)

    query_outputs[threshold].to_csv(OUT / "validation_query_metrics.csv", index=False)
    result = {
        "schema": "fibre-integrated-family-specialist-validation-v1",
        "status": "development_selected",
        "baseline": "general query-conditioned router",
        "specialists": list(FAMILIES),
        "threshold_trials": trials,
        "selected": selected,
        "seven_cell_labels_used": False,
        "official_family_test_labels_used": False,
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
