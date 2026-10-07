from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.bridge.model.assets import ROOT

TRAIN = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
R2E_EDGE = ROOT / "results/bridge_current_edgewise_r2e_v1/edge_metrics.csv.gz"
E2R_EDGE = ROOT / "results/bridge_current_edgewise_e2r_v1/edge_metrics.csv.gz"
R2E_ABL = ROOT / "results/bridge_r2e_four_group_ablation_v1/edge_metrics.csv.gz"
E2R_ABL = ROOT / "results/bridge_e2r_four_group_ablation_v1/edge_metrics.csv.gz"
CAGE_EDGE = ROOT / "results/bridge_layered_v4_cage_edgewise"
CAGE_GATE_BRIDGE = ROOT / "results/bridge_layered_v4_cage_gate_bridge"
OUT = ROOT / "results/bridge_difficulty_standardized_v2"
RECORD = ROOT / "reproducibility/bime_rank/records/BRIDGE_DIFFICULTY_STANDARDIZED_V2_RESULT.json"

KEYS = ["protein_id", "reaction_id"]
NOVELTY_ORDER = ("both_seen", "protein_cold", "reaction_cold", "double_cold")
METRICS = ("mrr", "hit3", "hit10", "hit100")
BASE_RANK_COLUMNS = {
    "Broad Retrieval": "broad_rank",
    "BRIDGE": "ablation_full_rank",
    "BRIDGE - Functional": "minus_functional_rank",
    "BRIDGE - Structure/Mechanism": "minus_structure_mechanism_rank",
    "BRIDGE - Long-term Relation Context": "minus_relational_memory_rank",
    "BRIDGE - Family/Domain": "minus_family_domain_rank",
}
METHOD_ORDER = (
    "EnzymeCAGE",
    "Broad Retrieval",
    "Broad Retrieval + CAGE Reranking",
    "EnzymeCAGE Gate + BRIDGE Reranking",
    "BRIDGE",
    "BRIDGE - Functional",
    "BRIDGE - Structure/Mechanism",
    "BRIDGE - Long-term Relation Context",
    "BRIDGE - Family/Domain",
)


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

    novelty = []
    strata = []
    for row in f.itertuples(index=False):
        pdg = int(row.protein_degree)
        rdg = int(row.reaction_degree)
        if pdg > 0 and rdg > 0:
            n = "both_seen"
            s = "both_seen|min_degree=1" if min(pdg, rdg) <= 1 else "both_seen|min_degree=2+"
        elif pdg == 0 and rdg > 0:
            n = "protein_cold"
            s = f"protein_cold|seen_degree={degree_bucket(rdg)}"
        elif pdg > 0 and rdg == 0:
            n = "reaction_cold"
            s = f"reaction_cold|seen_degree={degree_bucket(pdg)}"
        else:
            n = "double_cold"
            s = "double_cold|degree=0"
        novelty.append(n)
        strata.append(s)

    f["novelty"] = novelty
    f["difficulty_stratum"] = strata
    return f


def harmonic_table(nmax: int) -> np.ndarray:
    h = np.zeros(nmax + 1, dtype=np.float64)
    if nmax:
        h[1:] = np.cumsum(1.0 / np.arange(1, nmax + 1, dtype=np.float64))
    return h


def expected_values(candidate_n: np.ndarray, harmonic: np.ndarray) -> dict[str, np.ndarray]:
    n = candidate_n.astype(np.int64, copy=False)
    if np.any(n <= 0):
        raise ValueError("candidate_count_filtered must be positive")
    out = {
        "mrr": harmonic[n] / n,
        "hit3": np.minimum(3, n) / n,
        "hit10": np.minimum(10, n) / n,
        "hit100": np.minimum(100, n) / n,
    }
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def observed_values(rank: np.ndarray) -> dict[str, np.ndarray]:
    r = rank.astype(np.float64, copy=False)
    return {
        "mrr": 1.0 / r,
        "hit3": (r <= 3).astype(np.float64),
        "hit10": (r <= 10).astype(np.float64),
        "hit100": (r <= 100).astype(np.float64),
    }


def adjusted(obs: float, exp: float) -> float:
    denom = 1.0 - exp
    if denom <= 0:
        raise ValueError(f"non-positive adjusted-metric denominator: exp={exp}")
    return float((obs - exp) / denom)


def summarize_method(frame: pd.DataFrame, rank_col: str, harmonic: np.ndarray) -> dict:
    ranks = pd.to_numeric(frame[rank_col], errors="raise").to_numpy(np.int64)
    candidate_n = pd.to_numeric(frame["candidate_count_filtered"], errors="raise").to_numpy(np.int64)
    obs = observed_values(ranks)
    exp = expected_values(candidate_n, harmonic)

    direct = {m: float(obs[m].mean()) for m in METRICS}
    random_expectation = {m: float(exp[m].mean()) for m in METRICS}
    global_adjusted = {m: adjusted(direct[m], random_expectation[m]) for m in METRICS}

    cell_rows = []
    novelty_vectors: dict[str, list[dict[str, float]]] = {n: [] for n in NOVELTY_ORDER}

    for (novelty, stratum), cell in frame.groupby(["novelty", "difficulty_stratum"], sort=True):
        idx = cell.index.to_numpy(np.int64)
        row: dict[str, object] = {
            "novelty": str(novelty),
            "stratum": str(stratum),
            "edges": int(len(idx)),
        }
        adjusted_cell: dict[str, float] = {}
        for metric in METRICS:
            o = float(obs[metric][idx].mean())
            e = float(exp[metric][idx].mean())
            a = adjusted(o, e)
            row[f"{metric}_raw"] = o
            row[f"{metric}_random"] = e
            row[f"{metric}_adjusted"] = a
            adjusted_cell[metric] = a
        cell_rows.append(row)
        novelty_vectors[str(novelty)].append(adjusted_cell)

    novelty_macro = {}
    novelty_vectors_ordered = []
    for novelty in NOVELTY_ORDER:
        cells = novelty_vectors[novelty]
        if not cells:
            raise RuntimeError(f"missing novelty group: {novelty}")
        vec = {
            metric: float(np.mean([cell[metric] for cell in cells]))
            for metric in METRICS
        }
        novelty_macro[novelty] = vec
        novelty_vectors_ordered.append(vec)

    difficulty_standardized = {
        metric: float(np.mean([vec[metric] for vec in novelty_vectors_ordered]))
        for metric in METRICS
    }
    return {
        "direct_raw": direct,
        "global_chance_adjusted": global_adjusted,
        "difficulty_standardized": difficulty_standardized,
        "random_expectation": random_expectation,
        "novelty_macro_adjusted": novelty_macro,
        "cells": cell_rows,
    }


def _merge_optional_rank_file(
    frame: pd.DataFrame,
    path,
    columns: list[str],
) -> tuple[pd.DataFrame, bool]:
    if not path.exists():
        return frame, False
    extra = pd.read_csv(
        path,
        dtype={"protein_id": str, "reaction_id": str},
    ).fillna("")
    if len(extra) != 23773:
        raise RuntimeError(f"{path}: expected 23773 held-out edges, got {len(extra)}")
    if "candidate_count_filtered" in extra.columns:
        audit = frame[KEYS + ["candidate_count_filtered"]].merge(
            extra[KEYS + ["candidate_count_filtered"]],
            on=KEYS,
            how="outer",
            validate="one_to_one",
            suffixes=("_base", "_extra"),
            indicator=True,
        )
        if not audit["_merge"].eq("both").all():
            raise RuntimeError(f"{path}: edge keys diverge from base evaluation")
        left = pd.to_numeric(audit["candidate_count_filtered_base"], errors="raise")
        right = pd.to_numeric(audit["candidate_count_filtered_extra"], errors="raise")
        if not left.eq(right).all():
            return frame, False
    missing = [col for col in columns if col not in extra.columns]
    if missing:
        raise RuntimeError(f"{path}: missing rank columns {missing}")
    return (
        frame.merge(
            extra[KEYS + columns],
            on=KEYS,
            validate="one_to_one",
        ),
        True,
    )


def load_direction(direction: str) -> tuple[pd.DataFrame, dict[str, str]]:
    edge_path = R2E_EDGE if direction == "r2e" else E2R_EDGE
    abl_path = R2E_ABL if direction == "r2e" else E2R_ABL
    edge = pd.read_csv(edge_path, dtype={"protein_id": str, "reaction_id": str}).fillna("")
    abl = pd.read_csv(abl_path, dtype={"protein_id": str, "reaction_id": str}).fillna("")
    abl_cols = [
        "full_rank",
        "minus_functional_rank",
        "minus_structure_mechanism_rank",
        "minus_relational_memory_rank",
        "minus_family_domain_rank",
    ]
    abl = abl[KEYS + abl_cols].rename(columns={"full_rank": "ablation_full_rank"})
    frame = edge.merge(abl, on=KEYS, validate="one_to_one")
    if len(frame) != 23773:
        raise RuntimeError(f"{direction}: expected 23773 held-out edges, got {len(frame)}")
    rank_columns = dict(BASE_RANK_COLUMNS)

    cage_path = CAGE_EDGE / f"{direction}_edge_metrics.csv.gz"
    if cage_path.exists():
        frame, merged = _merge_optional_rank_file(
            frame,
            cage_path,
            ["enzymecage_rank", "broad_cage_rank"],
        )
        if merged:
            rank_columns["EnzymeCAGE"] = "enzymecage_rank"
            rank_columns["Broad Retrieval + CAGE Reranking"] = "broad_cage_rank"

    gate_bridge_path = CAGE_GATE_BRIDGE / f"{direction}_edge_metrics.csv.gz"
    if gate_bridge_path.exists():
        frame, merged = _merge_optional_rank_file(
            frame,
            gate_bridge_path,
            ["cage_gate_bridge_rank"],
        )
        if merged:
            rank_columns["EnzymeCAGE Gate + BRIDGE Reranking"] = "cage_gate_bridge_rank"

    ordered = {
        name: rank_columns[name]
        for name in METHOD_ORDER
        if name in rank_columns
    }
    return frame.reset_index(drop=True), ordered


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN, dtype=str).fillna("").drop_duplicates(KEYS)
    pdeg = train.groupby("protein_id").size().astype(int).to_dict()
    rdeg = train.groupby("reaction_id").size().astype(int).to_dict()

    result: dict[str, object] = {
        "schema": "bridge-difficulty-standardized-v2",
        "status": "completed",
        "protocol": {
            "mother_benchmark": "bridge_relation_unseen_max_v3",
            "held_out_edges": 23773,
            "evaluation_unit": "filtered held-out relation edge",
            "candidate_space": "native full type-valid candidate universe after filtered-known-positive removal",
            "chance_adjustment": {
                "framework": "Hoyt et al. rank-metric adjustment against analytical random-ranking expectation",
                "mrr_random_per_task": "H(N_i)/N_i",
                "hitk_random_per_task": "min(K,N_i)/N_i",
                "adjusted_metric": "(M - E0[M]) / (1 - E0[M])",
                "monte_carlo": False,
            },
            "difficulty_standardization": {
                "framework": "direct standardization over frozen graph-degree strata",
                "novelty_groups_equal_weight": list(NOVELTY_ORDER),
                "protein_cold_seen_reaction_degree_bins_equal_weight": ["1", "2-3", "4-7", "8-15", "16+"],
                "reaction_cold_seen_protein_degree_bins_equal_weight": ["1", "2-3", "4-7", "8-15", "16+"],
                "both_seen_bins_equal_weight": ["min_degree=1", "min_degree=2+"],
                "double_cold": "single degree-0 stratum",
                "formula": "1/4 * sum_n [ 1/|S_n| * sum_s adjusted_metric_s ]",
            },
            "important": "Point estimates are deterministic; no random sampling is used.",
        },
        "directions": {},
    }

    all_frames = {}
    all_rank_columns = {}
    nmax = 0
    for direction in ("r2e", "e2r"):
        raw, rank_columns = load_direction(direction)
        f = annotate(raw, pdeg, rdeg)
        all_frames[direction] = f
        all_rank_columns[direction] = rank_columns
        nmax = max(nmax, int(pd.to_numeric(f["candidate_count_filtered"]).max()))
    harmonic = harmonic_table(nmax)

    for direction, frame in all_frames.items():
        dres: dict[str, object] = {
            "edges": int(len(frame)),
            "queries": int(frame["reaction_id" if direction == "r2e" else "protein_id"].nunique()),
            "strata_counts": (
                frame.groupby(["novelty", "difficulty_stratum"], sort=True)
                .size().reset_index(name="edges").to_dict("records")
            ),
            "methods": {},
        }
        for name, rank_col in all_rank_columns[direction].items():
            dres["methods"][name] = summarize_method(frame, rank_col, harmonic)
        result["directions"][direction] = dres
        frame.to_csv(OUT / f"{direction}_annotated_edges.csv.gz", index=False)

    text = json.dumps(result, indent=2) + "\n"
    (OUT / "summary.json").write_text(text)
    RECORD.write_text(text)

    for direction in ("r2e", "e2r"):
        print(f"\n## {direction.upper()}")
        print("method\traw_mrr\tbalanced_mrr\traw_h3\tbalanced_h3\traw_h10\tbalanced_h10\traw_h100\tbalanced_h100")
        methods = result["directions"][direction]["methods"]
        for name in all_rank_columns[direction]:
            m = methods[name]
            r = m["direct_raw"]
            b = m["difficulty_standardized"]
            print(
                f"{name}\t{r['mrr']:.9f}\t{b['mrr']:.9f}"
                f"\t{r['hit3']:.9f}\t{b['hit3']:.9f}"
                f"\t{r['hit10']:.9f}\t{b['hit10']:.9f}"
                f"\t{r['hit100']:.9f}\t{b['hit100']:.9f}"
            )


if __name__ == "__main__":
    main()
