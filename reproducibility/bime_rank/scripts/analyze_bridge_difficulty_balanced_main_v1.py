from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.bridge.model.assets import ROOT

TRAIN = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
R2E = ROOT / "results/bridge_current_edgewise_r2e_v1/edge_metrics.csv.gz"
E2R = ROOT / "results/bridge_current_edgewise_e2r_v1/edge_metrics.csv.gz"
OUT = ROOT / "results/bridge_difficulty_balanced_main_v1"

NOVELTY_ORDER = ("both_seen", "protein_cold", "reaction_cold", "double_cold")
METRICS = ("mrr", "hit10", "hit100", "hit1000")


def metric_values(ranks: np.ndarray) -> np.ndarray:
    r = ranks.astype(np.float64, copy=False)
    return np.asarray([
        np.mean(1.0 / r),
        np.mean(r <= 10),
        np.mean(r <= 100),
        np.mean(r <= 1000),
    ], dtype=np.float64)


def degree_bucket(x: int) -> str:
    x = int(x)
    if x <= 0:
        return "0"
    if x == 1:
        return "1"
    if x <= 3:
        return "2-3"
    if x <= 7:
        return "4-7"
    if x <= 15:
        return "8-15"
    return "16+"


def annotate(frame: pd.DataFrame, pdeg: dict[str, int], rdeg: dict[str, int]) -> pd.DataFrame:
    f = frame.copy()
    f["protein_degree"] = f["protein_id"].map(pdeg).fillna(0).astype(int)
    f["reaction_degree"] = f["reaction_id"].map(rdeg).fillna(0).astype(int)

    both_seen = f["protein_seen"] & f["reaction_seen"]
    protein_cold = (~f["protein_seen"]) & f["reaction_seen"]
    reaction_cold = f["protein_seen"] & (~f["reaction_seen"])
    f["novelty"] = "double_cold"
    f.loc[both_seen, "novelty"] = "both_seen"
    f.loc[protein_cold, "novelty"] = "protein_cold"
    f.loc[reaction_cold, "novelty"] = "reaction_cold"

    strata = []
    for row in f.itertuples(index=False):
        if row.novelty == "protein_cold":
            db = degree_bucket(row.reaction_degree)
        elif row.novelty == "reaction_cold":
            db = degree_bucket(row.protein_degree)
        elif row.novelty == "both_seen":
            db = "1" if min(row.protein_degree, row.reaction_degree) <= 1 else "2+"
        else:
            db = "0"
        strata.append(f"{row.novelty}|degree={db}")
    f["difficulty_stratum"] = strata
    return f


def summarize(frame: pd.DataFrame, rank_col: str) -> dict:
    raw = metric_values(frame[rank_col].to_numpy(np.int64))
    by_novelty = {}
    novelty_vectors = []
    for novelty in NOVELTY_ORDER:
        group = frame[frame["novelty"] == novelty]
        cells = []
        for stratum, cell in group.groupby("difficulty_stratum", sort=True):
            cells.append(metric_values(cell[rank_col].to_numpy(np.int64)))
        if not cells:
            continue
        vec = np.mean(cells, axis=0)
        novelty_vectors.append(vec)
        by_novelty[novelty] = dict(zip(METRICS, map(float, vec), strict=True))
    macro = np.mean(novelty_vectors, axis=0)
    return {
        "micro": dict(zip(METRICS, map(float, raw), strict=True)),
        "difficulty_macro": dict(zip(METRICS, map(float, macro), strict=True)),
        "novelty_degree_macro": by_novelty,
    }


def bootstrap(frame: pd.DataFrame, seed: int, repeats: int) -> dict:
    rng = np.random.default_rng(seed)
    strata = {
        key: cell.index.to_numpy(np.int64)
        for key, cell in frame.groupby("difficulty_stratum", sort=True)
    }
    novelty_strata = {
        novelty: [
            key for key in strata
            if key.startswith(novelty + "|")
        ]
        for novelty in NOVELTY_ORDER
    }
    broad = frame["broad_rank"].to_numpy(np.int64)
    full = frame["full_rank"].to_numpy(np.int64)
    deltas = np.empty((repeats, len(METRICS)), dtype=np.float64)
    fulls = np.empty_like(deltas)

    for rep in range(repeats):
        novelty_b = []
        novelty_f = []
        for novelty in NOVELTY_ORDER:
            cell_b = []
            cell_f = []
            for key in novelty_strata[novelty]:
                idx = strata[key]
                sample = rng.choice(idx, size=len(idx), replace=True)
                cell_b.append(metric_values(broad[sample]))
                cell_f.append(metric_values(full[sample]))
            novelty_b.append(np.mean(cell_b, axis=0))
            novelty_f.append(np.mean(cell_f, axis=0))
        b = np.mean(novelty_b, axis=0)
        f = np.mean(novelty_f, axis=0)
        fulls[rep] = f
        deltas[rep] = f - b

    out = {}
    for i, metric in enumerate(METRICS):
        out[metric] = {
            "full_ci95": [
                float(np.quantile(fulls[:, i], 0.025)),
                float(np.quantile(fulls[:, i], 0.975)),
            ],
            "delta_ci95": [
                float(np.quantile(deltas[:, i], 0.025)),
                float(np.quantile(deltas[:, i], 0.975)),
            ],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-repeats", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN, dtype=str).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    pdeg = train.groupby("protein_id").size().astype(int).to_dict()
    rdeg = train.groupby("reaction_id").size().astype(int).to_dict()

    result = {
        "schema": "bridge-difficulty-balanced-main-v1",
        "protocol": {
            "test_edges": 23773,
            "all_edges_retained": True,
            "relation_unseen_relative_to_clean2023": True,
            "candidate_universe": {
                "r2e": 185918,
                "e2r": 11081,
            },
            "balancing": {
                "novelty_groups_equal_weight": list(NOVELTY_ORDER),
                "single_cold_degree_bins_equal_weight": ["1", "2-3", "4-7", "8-15", "16+"],
                "both_seen_degree_bins_equal_weight": ["min_degree=1", "min_degree>=2"],
                "double_cold": "single degree-0 cell",
                "sampling": "none; all edges retained, balance is macro weighting only",
            },
            "episodic_runtime_support": "none",
            "bootstrap": {
                "type": "stratified edge bootstrap within frozen difficulty cells",
                "repeats": args.bootstrap_repeats,
                "seed": args.seed,
            },
        },
        "directions": {},
    }

    for direction, path in (("r2e", R2E), ("e2r", E2R)):
        f = pd.read_csv(path, dtype={"protein_id": str, "reaction_id": str})
        f = annotate(f, pdeg, rdeg)
        if len(f) != 23773:
            raise RuntimeError(f"{direction}: expected 23773 edges, got {len(f)}")
        f.to_csv(OUT / f"{direction}_annotated_edges.csv.gz", index=False)

        counts = (
            f.groupby(["novelty", "difficulty_stratum"], sort=True)
            .size()
            .reset_index(name="edges")
        )
        counts.to_csv(OUT / f"{direction}_strata_counts.csv", index=False)

        broad = summarize(f, "broad_rank")
        full = summarize(f, "full_rank")
        delta = {
            metric: float(full["difficulty_macro"][metric] - broad["difficulty_macro"][metric])
            for metric in METRICS
        }
        candidate = f["candidate_count_filtered"].astype(int)
        result["directions"][direction] = {
            "edges": int(len(f)),
            "queries": int(f["reaction_id" if direction == "r2e" else "protein_id"].nunique()),
            "candidate_count_filtered": {
                "min": int(candidate.min()),
                "median": float(candidate.median()),
                "max": int(candidate.max()),
            },
            "broad": broad,
            "full": full,
            "difficulty_macro_delta": delta,
            "bootstrap": bootstrap(f, args.seed + (0 if direction == "r2e" else 1), args.bootstrap_repeats),
            "strata": counts.to_dict("records"),
        }

    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
