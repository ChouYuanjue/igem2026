from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parents[3]
TARGETS = ROOT / "results/bridge_rhea_longitudinal_growth_v4/targets.csv"
R2E = ROOT / "results/bridge_rhea_sprot_temporal_r2e_v3/edge_metrics.csv.gz"
E2R = ROOT / "results/bridge_rhea_sprot_temporal_e2r_v3/edge_metrics.csv.gz"
BROAD_R2E = ROOT / "results/bridge_rhea_sprot_temporal_broad_v3/r2e_edges.csv.gz"
BROAD_E2R = ROOT / "results/bridge_rhea_sprot_temporal_broad_v3/e2r_edges.csv.gz"
OUT = ROOT / "reproducibility/bime_rank/records/BRIDGE_CANDIDATE_DIFFICULTY_DECOMPOSITION_V2_RESULT.json"

FULL_POOL = {"r2e": 185918, "e2r": 11081}
# Number of entities in the current deployment universe that were absent from
# official Rhea release 128. These are fixed before post-cutoff evaluation.
COLD_POOL_AT_CUTOFF = {"r2e": 7998, "e2r": 515}


def logcomb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -np.inf
    return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


def expected_rr(full_n: int, full_rank: int, sample_n: int) -> float:
    if sample_n >= full_n:
        return 1.0 / full_rank
    population_negatives = full_n - 1
    better_negatives = full_rank - 1
    sampled_negatives = sample_n - 1
    scale = full_n / sample_n
    lb = (
        logcomb(population_negatives - better_negatives, sampled_negatives + 1)
        - logcomb(population_negatives, sampled_negatives)
    )
    correction = 0.0 if not np.isfinite(lb) else float(np.exp(lb))
    return max(0.0, min(1.0, (scale - correction) / full_rank))


def expected_hit(full_n: int, full_rank: int, sample_n: int, k: int) -> float:
    if sample_n >= full_n:
        return float(full_rank <= k)
    population_negatives = full_n - 1
    better_negatives = full_rank - 1
    sampled_negatives = sample_n - 1
    return float(
        hypergeom.cdf(
            k - 1,
            population_negatives,
            better_negatives,
            sampled_negatives,
        )
    )


def summarize(frame: pd.DataFrame, rank_col: str, sample_n: int) -> dict[str, float | int]:
    rr = []
    h10 = []
    h100 = []
    for full_n, rank in zip(
        frame.candidate_count_filtered.astype(int),
        frame[rank_col].astype(int),
        strict=True,
    ):
        n = min(int(sample_n), int(full_n))
        rr.append(expected_rr(int(full_n), int(rank), n))
        h10.append(expected_hit(int(full_n), int(rank), n, 10))
        h100.append(expected_hit(int(full_n), int(rank), n, 100))
    return {
        "edges": int(len(frame)),
        "candidate_pool_size": int(sample_n),
        "expected_mrr": float(np.mean(rr)),
        "expected_hit10": float(np.mean(h10)),
        "expected_hit100": float(np.mean(h100)),
    }


def load_direction(direction: str) -> pd.DataFrame:
    target = pd.read_csv(TARGETS, dtype=str).fillna("")
    if direction == "r2e":
        score = pd.read_csv(R2E, dtype=str)[
            ["protein_id", "reaction_id", "broad_rank", "full_rank"]
        ]
        counts = pd.read_csv(BROAD_R2E, dtype=str)[
            ["protein_id", "reaction_id", "candidate_count_filtered"]
        ]
    elif direction == "e2r":
        score = pd.read_csv(E2R, dtype=str)[
            ["protein_id", "reaction_id", "full_rank"]
        ]
        broad = pd.read_csv(BROAD_E2R, dtype=str)[
            ["protein_id", "reaction_id", "rank", "candidate_count_filtered"]
        ].rename(columns={"rank": "broad_rank"})
        score = score.merge(
            broad[["protein_id", "reaction_id", "broad_rank"]],
            on=["protein_id", "reaction_id"],
            validate="one_to_one",
        )
        counts = broad[
            ["protein_id", "reaction_id", "candidate_count_filtered"]
        ]
    else:
        raise ValueError(direction)

    frame = (
        score.merge(
            counts,
            on=["protein_id", "reaction_id"],
            validate="one_to_one",
        )
        .merge(
            target[
                [
                    "protein_id",
                    "reaction_id",
                    "protein_present_previous",
                    "reaction_present_previous",
                ]
            ],
            on=["protein_id", "reaction_id"],
            validate="one_to_one",
        )
    )
    for col in ["broad_rank", "full_rank", "candidate_count_filtered"]:
        frame[col] = frame[col].astype(int)
    p_warm = frame.protein_present_previous.str.lower().eq("true")
    r_warm = frame.reaction_present_previous.str.lower().eq("true")
    frame["endpoint_anchor"] = np.where(
        p_warm & r_warm,
        "warm_warm",
        np.where((~p_warm) & (~r_warm), "cold_cold", "mixed"),
    )
    return frame


def main() -> None:
    result: dict[str, object] = {
        "schema": "bridge-candidate-difficulty-decomposition-v2",
        "status": "completed",
        "protocol": {
            "targets": "persistent post-release128 relation-unseen temporal edges",
            "headline_policy": (
                "warm/cold never changes the candidate universe in the main benchmark; "
                "all exposure groups rank against the complete deployment candidate universe"
            ),
            "controlled_anchor_states": ["warm_warm", "cold_cold"],
            "controlled_pool_conditions": {
                "full": FULL_POOL,
                "cold_sized_at_cutoff": COLD_POOL_AT_CUTOFF,
            },
            "cold_pool_definition": (
                "number of current deployment-universe entities absent from official Rhea release128: "
                "7998 proteins and 515 reactions"
            ),
            "pool_control": (
                "exact expectation under uniform sampling without replacement from each edge's full "
                "filtered candidate universe; target is always included; this is not top-k model truncation"
            ),
            "interpretation": {
                "same_state_across_pool_sizes": "isolates candidate-count difficulty",
                "warm_vs_cold_at_same_pool_size": "isolates endpoint-exposure contrast from candidate count",
                "main_full_pool": "retains real deployment difficulty and remains the headline result",
            },
        },
        "directions": {},
    }

    for direction in ("r2e", "e2r"):
        frame = load_direction(direction)
        direction_result: dict[str, object] = {
            "full_candidate_universe": FULL_POOL[direction],
            "cold_sized_control": COLD_POOL_AT_CUTOFF[direction],
            "anchors": {},
        }
        for anchor in ("warm_warm", "cold_cold"):
            z = frame[frame.endpoint_anchor.eq(anchor)].copy()
            direction_result["anchors"][anchor] = {
                "edges": int(len(z)),
                "broad": {
                    "full": summarize(z, "broad_rank", FULL_POOL[direction]),
                    "cold_sized_control": summarize(
                        z, "broad_rank", COLD_POOL_AT_CUTOFF[direction]
                    ),
                },
                "bridge": {
                    "full": summarize(z, "full_rank", FULL_POOL[direction]),
                    "cold_sized_control": summarize(
                        z, "full_rank", COLD_POOL_AT_CUTOFF[direction]
                    ),
                },
            }

        ww = direction_result["anchors"]["warm_warm"]["bridge"]
        cc = direction_result["anchors"]["cold_cold"]["bridge"]
        direction_result["bridge_effect_decomposition"] = {
            "novelty_effect_hit10_full": (
                cc["full"]["expected_hit10"] - ww["full"]["expected_hit10"]
            ),
            "novelty_effect_hit10_cold_sized": (
                cc["cold_sized_control"]["expected_hit10"]
                - ww["cold_sized_control"]["expected_hit10"]
            ),
            "pool_size_effect_hit10_warm_warm": (
                ww["cold_sized_control"]["expected_hit10"]
                - ww["full"]["expected_hit10"]
            ),
            "pool_size_effect_hit10_cold_cold": (
                cc["cold_sized_control"]["expected_hit10"]
                - cc["full"]["expected_hit10"]
            ),
            "novelty_effect_hit100_full": (
                cc["full"]["expected_hit100"] - ww["full"]["expected_hit100"]
            ),
            "novelty_effect_hit100_cold_sized": (
                cc["cold_sized_control"]["expected_hit100"]
                - ww["cold_sized_control"]["expected_hit100"]
            ),
        }
        result["directions"][direction] = direction_result

    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
