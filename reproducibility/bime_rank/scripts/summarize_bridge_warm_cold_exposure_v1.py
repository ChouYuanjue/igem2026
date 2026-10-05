from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.bime_rank.scripts.build_bridge_rhea_longitudinal_growth_v4 import (
    build_alias_map,
    load_release,
)
DEFAULT_RELEASE_ROOT = Path("/tmp/bridge_rhea_hist")
TARGETS = ROOT / "results/bridge_rhea_longitudinal_growth_v4/targets.csv"
R2E = ROOT / "results/bridge_rhea_sprot_temporal_r2e_v3/edge_metrics.csv.gz"
E2R = ROOT / "results/bridge_rhea_sprot_temporal_e2r_v3/edge_metrics.csv.gz"
BROAD_E2R = ROOT / "results/bridge_rhea_sprot_temporal_broad_v3/e2r_edges.csv.gz"
OUT = ROOT / "reproducibility/bime_rank/records/BRIDGE_WARM_COLD_EXPOSURE_DIAGNOSTIC_V1_RESULT.json"

RELEASES = tuple(range(128, 143))
DEGREE_BINS = (
    ("1", 1, 1),
    ("2-3", 2, 3),
    ("4-10", 4, 10),
    ("11-50", 11, 50),
    ("51+", 51, 10**12),
)


def rank_metrics(frame: pd.DataFrame, col: str) -> dict[str, float | int]:
    r = frame[col].astype(int).to_numpy()
    return {
        "edges": int(len(r)),
        "mrr": float((1.0 / r).mean()),
        "hit10": float((r <= 10).mean()),
        "hit100": float((r <= 100).mean()),
        "hit1000": float((r <= 1000).mean()),
        "median_rank": float(np.median(r)),
    }


def dist(frame: pd.DataFrame, col: str) -> dict[str, float | int]:
    x = frame[col].astype(float).to_numpy()
    return {
        "median": float(np.median(x)),
        "mean": float(x.mean()),
        "p90": float(np.quantile(x, 0.9, method="nearest")),
        "max": float(x.max()),
    }


def state_name(q_warm: bool, c_warm: bool) -> str:
    return f"query_{'warm' if q_warm else 'cold'}__candidate_{'warm' if c_warm else 'cold'}"


def load_graph_metadata(release_root: Path) -> tuple[pd.DataFrame, dict[int, set[tuple[str, str]]]]:
    meta = pd.read_csv(
        ROOT / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv",
        dtype=str,
    ).fillna("")
    alias, _ = build_alias_map(meta)
    reaction_ids = set(
        pd.read_csv(
            ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv",
            dtype=str,
        ).reaction_id.astype(str)
    )
    graphs: dict[int, set[tuple[str, str]]] = {}
    for release in RELEASES:
        frame, _, _ = load_release(release, release_root, alias, reaction_ids)
        graphs[release] = set(map(tuple, frame[["protein_id", "reaction_id"]].to_numpy()))

    target = pd.read_csv(TARGETS, dtype=str).fillna("")
    target["first_observed_release"] = target.first_observed_release.astype(int)
    target["previous_release"] = target.previous_release.astype(int)

    pdeg = {
        n: Counter(protein for protein, _ in graph)
        for n, graph in graphs.items()
    }
    rdeg = {
        n: Counter(reaction for _, reaction in graph)
        for n, graph in graphs.items()
    }
    p_event = {}
    r_event = {}
    for release in RELEASES[1:]:
        added = graphs[release] - graphs[release - 1]
        p_event[release] = Counter(protein for protein, _ in added)
        r_event[release] = Counter(reaction for _, reaction in added)

    rows = []
    for row in target.itertuples(index=False):
        prev = int(row.previous_release)
        first = int(row.first_observed_release)
        rows.append(
            {
                "protein_id": row.protein_id,
                "reaction_id": row.reaction_id,
                "first_observed_release": first,
                "previous_release": prev,
                "insertion_orientation": row.insertion_orientation,
                "protein_prev_degree": int(pdeg[prev][row.protein_id]),
                "reaction_prev_degree": int(rdeg[prev][row.reaction_id]),
                "protein_final_degree": int(pdeg[142][row.protein_id]),
                "reaction_final_degree": int(rdeg[142][row.reaction_id]),
                "protein_event_edges": int(p_event[first][row.protein_id]),
                "reaction_event_edges": int(r_event[first][row.reaction_id]),
            }
        )
    return pd.DataFrame(rows), graphs


def prepare_scores(meta: pd.DataFrame, direction: str) -> pd.DataFrame:
    if direction == "r2e":
        score = pd.read_csv(R2E, dtype=str)[
            ["protein_id", "reaction_id", "broad_rank", "full_rank"]
        ].copy()
        query = "reaction"
        candidate = "protein"
    elif direction == "e2r":
        score = pd.read_csv(E2R, dtype=str)[
            ["protein_id", "reaction_id", "full_rank"]
        ].copy()
        broad = pd.read_csv(BROAD_E2R, dtype=str)[
            ["protein_id", "reaction_id", "rank"]
        ].rename(columns={"rank": "broad_rank"})
        score = score.merge(
            broad,
            on=["protein_id", "reaction_id"],
            validate="one_to_one",
        )
        query = "protein"
        candidate = "reaction"
    else:
        raise ValueError(direction)

    frame = score.merge(
        meta,
        on=["protein_id", "reaction_id"],
        validate="one_to_one",
    )
    frame["broad_rank"] = frame.broad_rank.astype(int)
    frame["full_rank"] = frame.full_rank.astype(int)
    frame["query_warm"] = frame[f"{query}_prev_degree"].astype(int) > 0
    frame["candidate_warm"] = frame[f"{candidate}_prev_degree"].astype(int) > 0
    frame["exposure_state"] = [
        state_name(q, c)
        for q, c in zip(frame.query_warm, frame.candidate_warm)
    ]
    frame["query_prev_degree"] = frame[f"{query}_prev_degree"].astype(int)
    frame["candidate_prev_degree"] = frame[f"{candidate}_prev_degree"].astype(int)
    frame["query_final_degree"] = frame[f"{query}_final_degree"].astype(int)
    frame["candidate_final_degree"] = frame[f"{candidate}_final_degree"].astype(int)
    frame["query_event_edges"] = frame[f"{query}_event_edges"].astype(int)
    return frame


def exposure_summary(frame: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {}
    for q_warm in (True, False):
        for c_warm in (True, False):
            key = state_name(q_warm, c_warm)
            z = frame[frame.exposure_state.eq(key)].copy()
            out[key] = {
                "edges": int(len(z)),
                "query_warm": q_warm,
                "candidate_warm": c_warm,
                "broad": rank_metrics(z, "broad_rank"),
                "bridge": rank_metrics(z, "full_rank"),
                "query_previous_degree": dist(z, "query_prev_degree"),
                "candidate_previous_degree": dist(z, "candidate_prev_degree"),
                "query_final_degree_posthoc": dist(z, "query_final_degree"),
                "query_insertion_event_edges_posthoc": dist(z, "query_event_edges"),
            }
    return out


def degree_diagnostic(frame: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {}
    for label, low, high in DEGREE_BINS:
        z = frame[
            frame.query_final_degree.astype(int).between(low, high, inclusive="both")
        ]
        if len(z) == 0:
            continue
        out[label] = {
            "edges": int(len(z)),
            "broad": rank_metrics(z, "broad_rank"),
            "bridge": rank_metrics(z, "full_rank"),
        }
    return out


def within_degree_candidate_contrast(frame: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {}
    for q_warm in (True, False):
        qname = "query_warm" if q_warm else "query_cold"
        out[qname] = {}
        qframe = frame[frame.query_warm.eq(q_warm)]
        for label, low, high in DEGREE_BINS:
            z = qframe[
                qframe.query_final_degree.astype(int).between(low, high, inclusive="both")
            ]
            cells = {}
            for c_warm in (True, False):
                c = z[z.candidate_warm.eq(c_warm)]
                if len(c) == 0:
                    continue
                cells["candidate_warm" if c_warm else "candidate_cold"] = {
                    "edges": int(len(c)),
                    "bridge": rank_metrics(c, "full_rank"),
                }
            if len(cells) == 2:
                out[qname][label] = cells
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    args = parser.parse_args()

    meta, _ = load_graph_metadata(args.release_root.resolve())
    r2e = prepare_scores(meta, "r2e")
    e2r = prepare_scores(meta, "e2r")

    result = {
        "schema": "bridge-warm-cold-exposure-diagnostic-v1",
        "status": "completed",
        "protocol": {
            "source": "persistent post-release128 relations first observed in official Rhea releases 129-142",
            "targets": int(len(meta)),
            "relation_status": "every evaluated pair is relation-unseen with respect to the frozen clean2023 training relation set",
            "candidate_universe": {
                "r2e_proteins": 185918,
                "e2r_reactions": 11081,
            },
            "definition": (
                "warm/cold is an endpoint-exposure state at the immediately preceding official release, "
                "not an ordinal difficulty label"
            ),
            "directional_axes": {
                "r2e": "query=reaction, candidate=protein",
                "e2r": "query=protein, candidate=reaction",
            },
            "four_cells": [
                "query_warm__candidate_warm",
                "query_warm__candidate_cold",
                "query_cold__candidate_warm",
                "query_cold__candidate_cold",
            ],
            "posthoc_degree_warning": (
                "release142 final degree and insertion-event bundle size are explanatory diagnostics only; "
                "they are not model inputs, split-selection variables, or tuning variables"
            ),
        },
        "r2e": {
            "exposure_matrix": exposure_summary(r2e),
            "posthoc_query_degree": degree_diagnostic(r2e),
            "posthoc_within_degree_candidate_contrast": within_degree_candidate_contrast(r2e),
        },
        "e2r": {
            "exposure_matrix": exposure_summary(e2r),
            "posthoc_query_degree": degree_diagnostic(e2r),
            "posthoc_within_degree_candidate_contrast": within_degree_candidate_contrast(e2r),
        },
        "interpretation": {
            "warm_cold_is_not_difficulty": (
                "raw performance is not monotone from warm/warm to cold/cold. "
                "The cells differ in endpoint exposure, query popularity, event-bundle size, and curation selection."
            ),
            "why_old_labels_mislead": (
                "a label such as new_protein means warm-query/cold-candidate for R2E but "
                "cold-query/warm-candidate for E2R; one protein-cold number therefore conflates different retrieval problems"
            ),
            "recommended_role": (
                "use the four directional exposure cells as the warm/cold performance view; "
                "retain natural temporal aggregation as the deployment headline and use degree/event analyses only to explain composition"
            ),
        },
        "reproduction": {
            "script": "reproducibility/bime_rank/scripts/summarize_bridge_warm_cold_exposure_v1.py",
            "targets": "results/bridge_rhea_longitudinal_growth_v4/targets.csv",
            "r2e_scores": "results/bridge_rhea_sprot_temporal_r2e_v3/edge_metrics.csv.gz",
            "e2r_scores": "results/bridge_rhea_sprot_temporal_e2r_v3/edge_metrics.csv.gz",
            "broad_e2r_scores": "results/bridge_rhea_sprot_temporal_broad_v3/e2r_edges.csv.gz",
        },
    }

    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
