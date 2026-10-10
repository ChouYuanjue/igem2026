from __future__ import annotations

import json

import numpy as np
import pandas as pd

from projects.active.bridge.model.assets import ROOT

TRAIN = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
R2E_EDGE = ROOT / "results/bridge_current_edgewise_r2e_v1/edge_metrics.csv.gz"
E2R_EDGE = ROOT / "results/bridge_current_edgewise_e2r_v1/edge_metrics.csv.gz"
R2E_ABL = ROOT / "results/bridge_r2e_four_group_ablation_v1/edge_metrics.csv.gz"
E2R_ABL = ROOT / "results/bridge_e2r_four_group_ablation_v1/edge_metrics.csv.gz"
CAGE_DIR = ROOT / "results/bridge_layered_v4_cage_edge_metrics"
GATE_BRIDGE_DIR = ROOT / "results/bridge_layered_v4_cage_gate_bridge"
OUT = ROOT / "results/bridge_difficulty_standardized_v3"
RECORD = ROOT / "reproducibility/bime_rank/records/BRIDGE_DIFFICULTY_STANDARDIZED_V3_RESULT.json"

KEYS = ["protein_id", "reaction_id"]
NOVELTY_ORDER = ("both_seen", "protein_cold", "reaction_cold", "double_cold")
METRICS = ("mrr", "hit3", "hit10", "hit100")
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
        raise ValueError("candidate count must be positive")
    return {
        "mrr": harmonic[n] / n,
        "hit3": np.minimum(3, n) / n,
        "hit10": np.minimum(10, n) / n,
        "hit100": np.minimum(100, n) / n,
    }


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
        raise ValueError(f"non-positive adjusted denominator exp={exp}")
    return float((obs - exp) / denom)


def summarize_method(
    frame: pd.DataFrame,
    rank_col: str,
    candidate_col: str,
    harmonic: np.ndarray,
) -> dict[str, object]:
    ranks = pd.to_numeric(frame[rank_col], errors="raise").to_numpy(np.int64)
    candidate_n = pd.to_numeric(frame[candidate_col], errors="raise").to_numpy(np.int64)
    if np.any(ranks < 1) or np.any(ranks > candidate_n):
        bad = np.where((ranks < 1) | (ranks > candidate_n))[0][:10].tolist()
        raise RuntimeError(
            f"{rank_col}: ranks outside 1..N at rows {bad}"
        )

    obs = observed_values(ranks)
    exp = expected_values(candidate_n, harmonic)
    direct = {m: float(obs[m].mean()) for m in METRICS}
    random_expectation = {m: float(exp[m].mean()) for m in METRICS}
    global_adjusted = {
        m: adjusted(direct[m], random_expectation[m])
        for m in METRICS
    }

    cells: list[dict[str, object]] = []
    novelty_cells: dict[str, list[dict[str, float]]] = {
        n: [] for n in NOVELTY_ORDER
    }
    for (novelty, stratum), cell in frame.groupby(
        ["novelty", "difficulty_stratum"],
        sort=True,
    ):
        idx = cell.index.to_numpy(np.int64)
        adjusted_cell: dict[str, float] = {}
        row: dict[str, object] = {
            "novelty": str(novelty),
            "stratum": str(stratum),
            "edges": int(len(idx)),
        }
        for metric in METRICS:
            raw = float(obs[metric][idx].mean())
            rnd = float(exp[metric][idx].mean())
            adj = adjusted(raw, rnd)
            row[f"{metric}_raw"] = raw
            row[f"{metric}_random"] = rnd
            row[f"{metric}_adjusted"] = adj
            adjusted_cell[metric] = adj
        cells.append(row)
        novelty_cells[str(novelty)].append(adjusted_cell)

    novelty_macro: dict[str, dict[str, float]] = {}
    for novelty in NOVELTY_ORDER:
        parts = novelty_cells[novelty]
        if not parts:
            raise RuntimeError(f"missing novelty group {novelty}")
        novelty_macro[novelty] = {
            metric: float(np.mean([part[metric] for part in parts]))
            for metric in METRICS
        }

    balanced = {
        metric: float(
            np.mean([
                novelty_macro[novelty][metric]
                for novelty in NOVELTY_ORDER
            ])
        )
        for metric in METRICS
    }
    return {
        "direct_raw": direct,
        "global_chance_adjusted": global_adjusted,
        "difficulty_standardized": balanced,
        "random_expectation": random_expectation,
        "novelty_macro_adjusted": novelty_macro,
        "cells": cells,
        "candidate_count": {
            "min": int(candidate_n.min()),
            "median": float(np.median(candidate_n)),
            "max": int(candidate_n.max()),
        },
    }


def load_direction(direction: str) -> tuple[pd.DataFrame, dict[str, tuple[str, str]], list[str]]:
    edge_path = R2E_EDGE if direction == "r2e" else E2R_EDGE
    abl_path = R2E_ABL if direction == "r2e" else E2R_ABL

    edge = pd.read_csv(
        edge_path,
        dtype={"protein_id": str, "reaction_id": str},
    ).fillna("")
    abl = pd.read_csv(
        abl_path,
        dtype={"protein_id": str, "reaction_id": str},
    ).fillna("")
    abl_cols = [
        "minus_functional_rank",
        "minus_structure_mechanism_rank",
        "minus_relational_memory_rank",
        "minus_family_domain_rank",
    ]
    frame = edge.merge(
        abl[KEYS + abl_cols],
        on=KEYS,
        validate="one_to_one",
    )
    if len(frame) != 23773:
        raise RuntimeError(f"{direction}: expected 23773 base edges, got {len(frame)}")

    specs: dict[str, tuple[str, str]] = {
        "Broad Retrieval": ("broad_rank", "candidate_count_filtered"),
        "BRIDGE": ("full_rank", "candidate_count_filtered"),
        "BRIDGE - Functional": (
            "minus_functional_rank",
            "candidate_count_filtered",
        ),
        "BRIDGE - Structure/Mechanism": (
            "minus_structure_mechanism_rank",
            "candidate_count_filtered",
        ),
        "BRIDGE - Long-term Relation Context": (
            "minus_relational_memory_rank",
            "candidate_count_filtered",
        ),
        "BRIDGE - Family/Domain": (
            "minus_family_domain_rank",
            "candidate_count_filtered",
        ),
    }
    missing: list[str] = []

    cage_path = CAGE_DIR / f"{direction}_edge_metrics.csv.gz"
    if cage_path.exists():
        cage = pd.read_csv(
            cage_path,
            dtype={"protein_id": str, "reaction_id": str},
        ).fillna("")
        if len(cage) != 23773 or cage.duplicated(KEYS).any():
            raise RuntimeError(f"{direction}: invalid CAGE edge metrics")
        frame = frame.merge(cage, on=KEYS, validate="one_to_one")
        specs["EnzymeCAGE"] = (
            "enzymecage_rank",
            "enzymecage_candidate_count_filtered",
        )
        specs["Broad Retrieval + CAGE Reranking"] = (
            "broad_cage_rank",
            "broad_cage_candidate_count_filtered",
        )
    else:
        missing.extend([
            "EnzymeCAGE",
            "Broad Retrieval + CAGE Reranking",
        ])

    gate_path = GATE_BRIDGE_DIR / f"{direction}_edge_metrics.csv.gz"
    if gate_path.exists():
        gate = pd.read_csv(
            gate_path,
            dtype={"protein_id": str, "reaction_id": str},
        ).fillna("")
        if len(gate) != 23773 or gate.duplicated(KEYS).any():
            raise RuntimeError(f"{direction}: invalid CAGE Gate+BRIDGE edge metrics")
        gate = gate.rename(
            columns={
                "candidate_count_filtered":
                    "cage_gate_bridge_candidate_count_filtered"
            }
        )
        frame = frame.merge(gate, on=KEYS, validate="one_to_one")
        specs["EnzymeCAGE Gate + BRIDGE Reranking"] = (
            "cage_gate_bridge_rank",
            "cage_gate_bridge_candidate_count_filtered",
        )
    else:
        missing.append("EnzymeCAGE Gate + BRIDGE Reranking")

    return frame.reset_index(drop=True), specs, missing


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN, dtype=str).fillna("").drop_duplicates(KEYS)
    pdeg = train.groupby("protein_id").size().astype(int).to_dict()
    rdeg = train.groupby("reaction_id").size().astype(int).to_dict()

    loaded = {}
    nmax = 0
    for direction in ("r2e", "e2r"):
        frame, specs, missing = load_direction(direction)
        frame = annotate(frame, pdeg, rdeg)
        loaded[direction] = (frame, specs, missing)
        for _, candidate_col in specs.values():
            nmax = max(
                nmax,
                int(pd.to_numeric(frame[candidate_col], errors="raise").max()),
            )
    harmonic = harmonic_table(nmax)

    any_missing = any(
        bool(missing)
        for _, _, missing in loaded.values()
    )
    result: dict[str, object] = {
        "schema": "bridge-difficulty-standardized-v3",
        "status": "partial" if any_missing else "completed",
        "protocol": {
            "mother_benchmark": "bridge_relation_unseen_max_v3",
            "held_out_edges": 23773,
            "evaluation_unit": "one filtered held-out relation edge",
            "chance_adjustment": {
                "framework": (
                    "Hoyt et al. analytical random-ranking adjustment"
                ),
                "mrr_random_per_task": "H(N_i)/N_i",
                "hitk_random_per_task": "min(K,N_i)/N_i",
                "adjusted_metric": "(M - E0[M]) / (1 - E0[M])",
                "monte_carlo": False,
            },
            "difficulty_standardization": {
                "framework": (
                    "direct standardization over frozen graph-degree strata"
                ),
                "novelty_groups_equal_weight": list(NOVELTY_ORDER),
                "single_cold_seen_side_degree_bins_equal_weight": [
                    "1", "2-3", "4-7", "8-15", "16+"
                ],
                "both_seen_bins_equal_weight": [
                    "min_degree=1", "min_degree=2+"
                ],
                "double_cold": "single degree-0 stratum",
                "formula": (
                    "1/4 * sum_n [ 1/|S_n| * "
                    "sum_s adjusted_metric_s ]"
                ),
            },
            "cage_completion": {
                "rule": (
                    "ranked/scored gate candidates precede all unreturned "
                    "eligible candidates; an unreturned target receives "
                    "pessimistic last filtered rank"
                ),
                "r2e_native_candidate_universe": (
                    "current protein universe plus native CAGE-only logical "
                    "identities; Broad+CAGE remains on the current universe"
                ),
                "e2r_candidate_universe": (
                    "shared current reaction universe"
                ),
            },
        },
        "directions": {},
    }

    for direction in ("r2e", "e2r"):
        frame, specs, missing = loaded[direction]
        methods = {}
        for method in METHOD_ORDER:
            if method not in specs:
                continue
            rank_col, candidate_col = specs[method]
            methods[method] = summarize_method(
                frame,
                rank_col,
                candidate_col,
                harmonic,
            )
        result["directions"][direction] = {
            "edges": int(len(frame)),
            "queries": int(
                frame[
                    "reaction_id" if direction == "r2e" else "protein_id"
                ].nunique()
            ),
            "methods": methods,
            "missing_methods": missing,
        }

    text = json.dumps(result, indent=2) + "\n"
    (OUT / "summary.json").write_text(text)
    RECORD.write_text(text)

    for direction in ("r2e", "e2r"):
        print(f"\n## {direction.upper()}")
        d = result["directions"][direction]
        for method in METHOD_ORDER:
            if method not in d["methods"]:
                continue
            m = d["methods"][method]
            raw = m["direct_raw"]
            bal = m["difficulty_standardized"]
            print(
                f"{method}\t"
                f"raw_mrr={raw['mrr']:.9f}\tbal_mrr={bal['mrr']:.9f}\t"
                f"raw_h3={raw['hit3']:.9f}\tbal_h3={bal['hit3']:.9f}\t"
                f"raw_h10={raw['hit10']:.9f}\tbal_h10={bal['hit10']:.9f}\t"
                f"raw_h100={raw['hit100']:.9f}\tbal_h100={bal['hit100']:.9f}"
            )
        if d["missing_methods"]:
            print("missing:", ", ".join(d["missing_methods"]))


if __name__ == "__main__":
    main()
