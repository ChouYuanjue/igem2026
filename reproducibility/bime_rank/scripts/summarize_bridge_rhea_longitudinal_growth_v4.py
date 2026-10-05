from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "results/bridge_rhea_longitudinal_growth_v4"
R2E = ROOT / "results/bridge_rhea_sprot_temporal_r2e_v3/edge_metrics.csv.gz"
E2R = ROOT / "results/bridge_rhea_sprot_temporal_e2r_v3/edge_metrics.csv.gz"
BROAD_E2R = ROOT / "results/bridge_rhea_sprot_temporal_broad_v3/e2r_edges.csv.gz"
CAGE = ROOT / "results/bridge_rhea_sprot_temporal_cage_gate_v3/edge_gate_recall.csv.gz"
OUT = ROOT / "reproducibility/bime_rank/records/BRIDGE_RHEA_LONGITUDINAL_GROWTH_V4_RESULT.json"


def metrics(frame: pd.DataFrame, col: str) -> dict[str, float | int]:
    r = frame[col].astype(int).to_numpy()
    return {
        "edges": int(len(r)),
        "mrr": float((1 / r).mean()),
        "hit10": float((r <= 10).mean()),
        "hit100": float((r <= 100).mean()),
        "hit1000": float((r <= 1000).mean()),
        "median_rank": float(np.median(r)),
    }


def bootstrap_delta(frame: pd.DataFrame, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    broad = 1 / frame.broad_rank.astype(int).to_numpy()
    bridge = 1 / frame.full_rank.astype(int).to_numpy()
    delta = bridge - broad
    n = len(delta)
    samples = np.empty(10000, dtype=np.float64)
    for i in range(len(samples)):
        samples[i] = delta[rng.integers(0, n, n)].mean()
    return {
        "delta_mrr": float(delta.mean()),
        "bootstrap_replicates": 10000,
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "bootstrap_nonpositive_fraction": float((samples <= 0).mean()),
    }


def event_metrics(frame: pd.DataFrame, rank_col: str, query_col: str) -> dict[str, float | int]:
    z = frame[["first_observed_release", query_col, rank_col]].copy()
    z["rr"] = 1 / z[rank_col].astype(int)
    z["hit10"] = z[rank_col].astype(int) <= 10
    z["hit100"] = z[rank_col].astype(int) <= 100
    z["hit1000"] = z[rank_col].astype(int) <= 1000
    event = z.groupby(["first_observed_release", query_col], sort=False).agg(
        mrr=("rr", "mean"),
        hit10=("hit10", "mean"),
        hit100=("hit100", "mean"),
        hit1000=("hit1000", "mean"),
        edges=(rank_col, "size"),
    )
    return {
        "events": int(len(event)),
        "mrr_macro": float(event.mrr.mean()),
        "hit10_macro": float(event.hit10.mean()),
        "hit100_macro": float(event.hit100.mean()),
        "hit1000_macro": float(event.hit1000.mean()),
        "median_edges_per_event": float(event.edges.median()),
        "max_edges_per_event": int(event.edges.max()),
    }


def retrieval_block(frame: pd.DataFrame, query_col: str, seed: int) -> dict[str, object]:
    return {
        "edge_micro": {
            "broad": metrics(frame, "broad_rank"),
            "bridge": metrics(frame, "full_rank"),
            "paired_gain": bootstrap_delta(frame, seed),
        },
        "query_event_macro": {
            "broad": event_metrics(frame, "broad_rank", query_col),
            "bridge": event_metrics(frame, "full_rank", query_col),
        },
    }


def cage_block(frame: pd.DataFrame) -> dict[str, float | int]:
    event = frame.groupby(
        ["first_observed_release", "reaction_id"], sort=False
    ).cage_gate_hit
    return {
        "edges": int(len(frame)),
        "edge_recall": float(frame.cage_gate_hit.mean()),
        "r2e_query_events": int(event.ngroups),
        "event_macro_positive_recall": float(event.mean().mean()),
        "event_any_positive_hit": float(event.any().mean()),
    }


def main() -> None:
    bench = json.load(open(BENCH / "summary.json"))
    target = pd.read_csv(BENCH / "targets.csv", dtype=str).fillna("")
    target["first_observed_release"] = target.first_observed_release.astype(int)
    target["headline_growth"] = np.where(
        target.insertion_growth_class.eq("edge_completion"),
        "graph_completion",
        "graph_expansion",
    )

    r2e = pd.read_csv(R2E, dtype=str).merge(
        target,
        on=["protein_id", "reaction_id"],
        validate="one_to_one",
    )
    for col in ["broad_rank", "full_rank"]:
        r2e[col] = r2e[col].astype(int)

    e2r = pd.read_csv(E2R, dtype=str)
    broad_e2r = pd.read_csv(BROAD_E2R, dtype=str)[
        ["protein_id", "reaction_id", "rank"]
    ].rename(columns={"rank": "broad_rank"})
    e2r = e2r.merge(
        broad_e2r,
        on=["protein_id", "reaction_id"],
        validate="one_to_one",
    ).merge(
        target,
        on=["protein_id", "reaction_id"],
        validate="one_to_one",
    )
    for col in ["broad_rank", "full_rank"]:
        e2r[col] = e2r[col].astype(int)

    cage = pd.read_csv(CAGE, dtype=str)[
        ["protein_id", "reaction_id", "cage_gate_hit"]
    ]
    cage["cage_gate_hit"] = cage.cage_gate_hit.map(
        {"True": True, "False": False, "true": True, "false": False}
    ).fillna(False).astype(bool)
    cage = cage.merge(
        target,
        on=["protein_id", "reaction_id"],
        validate="one_to_one",
    )

    result: dict[str, object] = {
        "schema": "bridge-rhea-longitudinal-growth-v4-result",
        "status": "completed",
        "headline_protocol": {
            "historical_source": "official consecutive Rhea Swiss-Prot mappings, releases 116-142",
            "ranking_cutoff": 128,
            "ranking_final_release": 142,
            "ranking_targets": int(len(target)),
            "candidate_universe": {
                "r2e_proteins": 185918,
                "e2r_reactions": 11081,
            },
            "evaluation_unit": "persistent post-release128 future relation edge; filtered full-candidate rank",
            "growth_label": (
                "label each target at its first observed release relative to the immediately "
                "preceding official release, rather than relative to release128"
            ),
            "query_event_macro": (
                "within each first-observed release, edges sharing the retrieval query are one "
                "database-growth event; event means are macro-averaged"
            ),
            "pre_cutoff_role": (
                "releases 116-128 characterize graph-growth statistics only and are never used "
                "as future-label ranking tests"
            ),
            "selection_or_tuning_on_temporal_labels": False,
        },
        "longitudinal_graph_growth": bench["longitudinal_graph_growth"],
        "classification_correction": {
            "changed_edges": bench["persistent_post_cutoff_targets"]["classification_changed_edges"],
            "changed_fraction": bench["persistent_post_cutoff_targets"]["classification_changed_fraction"],
            "baseline_orientation_counts": bench["persistent_post_cutoff_targets"]["by_baseline_orientation"],
            "insertion_orientation_counts": bench["persistent_post_cutoff_targets"]["by_insertion_orientation"],
        },
        "headline": {},
        "orientation_diagnostic": {},
        "release_stability": {},
        "interpretation": {
            "graph_process": (
                "Rhea growth is a mixture of relation completion and entity-arrival expansion, "
                "with new proteins preferentially attaching to already high-degree reactions."
            ),
            "evaluation_reason": (
                "A single cold label cannot represent this process. Consecutive releases recover "
                "the mechanism at insertion time while full candidate universes keep candidate "
                "difficulty fixed."
            ),
            "cage_boundary": (
                "CAGE is evaluated through its published 2023 association-gated candidate "
                "generation. Its recall is therefore a property of its own retrieval domain; "
                "BRIDGE candidates or ranking are not injected into CAGE."
            ),
        },
        "reproduction": {
            "build": "reproducibility/bime_rank/scripts/build_bridge_rhea_longitudinal_growth_v4.py",
            "summarize": "reproducibility/bime_rank/scripts/summarize_bridge_rhea_longitudinal_growth_v4.py",
            "reused_r2e_scores": "results/bridge_rhea_sprot_temporal_r2e_v3/edge_metrics.csv.gz",
            "reused_e2r_scores": "results/bridge_rhea_sprot_temporal_e2r_v3/edge_metrics.csv.gz",
            "reused_broad_e2r_scores": "results/bridge_rhea_sprot_temporal_broad_v3/e2r_edges.csv.gz",
            "reused_cage_gate": "results/bridge_rhea_sprot_temporal_cage_gate_v3/edge_gate_recall.csv.gz",
            "reuse_reason": "the v4 persistent target edge set is exactly the same 5409 edges as v3; only the historical insertion labels and event aggregation change",
        },
    }

    seeds = {
        ("graph_completion", "r2e"): 202610051,
        ("graph_expansion", "r2e"): 202610052,
        ("graph_completion", "e2r"): 202610053,
        ("graph_expansion", "e2r"): 202610054,
    }
    for growth in ("graph_completion", "graph_expansion"):
        rz = r2e[r2e.headline_growth.eq(growth)].copy()
        ez = e2r[e2r.headline_growth.eq(growth)].copy()
        cz = cage[cage.headline_growth.eq(growth)].copy()
        rkeys = set(map(tuple, rz[["protein_id", "reaction_id"]].to_numpy()))
        ekeys = set(map(tuple, ez[["protein_id", "reaction_id"]].to_numpy()))
        ckeys = set(map(tuple, cz[["protein_id", "reaction_id"]].to_numpy()))
        if not (rkeys == ekeys == ckeys):
            raise AssertionError(f"{growth} edge-set mismatch")
        result["headline"][growth] = {
            "edges": int(len(rz)),
            "r2e": retrieval_block(rz, "reaction_id", seeds[(growth, "r2e")]),
            "e2r": retrieval_block(ez, "protein_id", seeds[(growth, "e2r")]),
            "cage_candidate_generation": cage_block(cz),
        }

    for orientation in ("both_old", "new_protein", "new_reaction", "both_new"):
        rz = r2e[r2e.insertion_orientation.eq(orientation)]
        ez = e2r[e2r.insertion_orientation.eq(orientation)]
        cz = cage[cage.insertion_orientation.eq(orientation)]
        result["orientation_diagnostic"][orientation] = {
            "edges": int(len(rz)),
            "r2e_broad": metrics(rz, "broad_rank"),
            "r2e_bridge": metrics(rz, "full_rank"),
            "e2r_broad": metrics(ez, "broad_rank"),
            "e2r_bridge": metrics(ez, "full_rank"),
            "cage_candidate_generation": cage_block(cz),
        }

    for release in sorted(target.first_observed_release.unique()):
        rz = r2e[r2e.first_observed_release.eq(release)]
        ez = e2r[e2r.first_observed_release.eq(release)]
        result["release_stability"][str(int(release))] = {
            "edges": int(len(rz)),
            "r2e_broad": metrics(rz, "broad_rank"),
            "r2e_bridge": metrics(rz, "full_rank"),
            "e2r_broad": metrics(ez, "broad_rank"),
            "e2r_bridge": metrics(ez, "full_rank"),
        }

    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
