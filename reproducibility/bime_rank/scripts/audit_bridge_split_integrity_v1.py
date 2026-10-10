from __future__ import annotations

"""Read-only provenance audit for the E2R BRIDGE relation-unseen benchmark.

This file deliberately does not alter model scores or existing evaluation tables.
"""

import json

import numpy as np
import pandas as pd

from projects.active.bridge.model.assets import ROOT
from reproducibility.bime_rank.scripts.analyze_bridge_difficulty_standardized_v3 import annotate
from reproducibility.bime_rank.scripts.evaluate_bridge_adaptive_relation_context_v8 import _validation_relations

TRAIN = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
OUTER = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
EDGES = ROOT / "results/bridge_e2r_four_group_ablation_v1/edge_metrics.csv.gz"
DEST = ROOT / "results/bridge_e2r_query_gate_v1"
METRICS = ("mrr", "hit3", "hit10", "hit100")


def balanced_raw(frame: pd.DataFrame, rank_column: str) -> dict[str, dict]:
    """Same frozen degree strata and novelty macro as v3, without per-K chance subtraction."""
    if frame.empty:
        raise ValueError("No edges")
    strata: dict[str, dict] = {}
    values = {}
    for novelty, subset in frame.groupby("novelty"):
        cells = []
        for difficulty, cell in subset.groupby("difficulty_stratum"):
            ranks = cell[rank_column].to_numpy(np.int64)
            row = {
                "edges": len(ranks),
                "mrr": float(np.mean(1.0 / ranks)),
                "hit3": float(np.mean(ranks <= 3)),
                "hit10": float(np.mean(ranks <= 10)),
                "hit100": float(np.mean(ranks <= 100)),
            }
            cells.append(row)
        values[novelty] = {
            key: float(np.mean([row[key] for row in cells])) for key in METRICS
        }
        strata[novelty] = {"edges": int(len(subset)), "cells": len(cells)}
    if len(values) != 4:
        raise RuntimeError(f"Need all four novelty classes, got {list(values)}")
    balanced = {
        key: float(np.mean([row[key] for row in values.values()])) for key in METRICS
    }
    direct_r = frame[rank_column].to_numpy(np.int64)
    direct = {
        "mrr": float(np.mean(1 / direct_r)),
        "hit3": float(np.mean(direct_r <= 3)),
        "hit10": float(np.mean(direct_r <= 10)),
        "hit100": float(np.mean(direct_r <= 100)),
    }
    for label, obj in (("balanced", balanced), ("direct", direct)):
        if not (0 <= obj["hit3"] <= obj["hit10"] <= obj["hit100"] <= 1):
            raise RuntimeError(f"Hit@K not monotone in {label}: {obj}")
    return {"direct": direct, "balanced": balanced, "strata": strata}


def main() -> None:
    train = pd.read_csv(TRAIN, dtype=str).drop_duplicates(["protein_id", "reaction_id"])
    validation = _validation_relations(train)
    all_outer = pd.read_csv(OUTER, dtype=str).drop_duplicates(["protein_id", "reaction_id"])
    dev_pairs = set(zip(validation.protein_id, validation.reaction_id))
    target_pairs = set(zip(all_outer.protein_id, all_outer.reaction_id))
    train_pairs = set(zip(train.protein_id, train.reaction_id))
    true_outer = target_pairs - dev_pairs
    if len(train_pairs & target_pairs):
        raise AssertionError("Target includes clean2023 training pair")
    frame = pd.read_csv(EDGES, dtype={"protein_id": str, "reaction_id": str})
    frame = annotate(frame, train.groupby("protein_id").size().to_dict(),
                     train.groupby("reaction_id").size().to_dict())
    frame["gate_training_pair"] = [
        (p, r) in dev_pairs for p, r in zip(frame.protein_id, frame.reaction_id)
    ]
    if set(zip(frame.protein_id, frame.reaction_id)) != target_pairs:
        raise AssertionError("Ablation E2R edges disagree with frozen target pairs")
    report: dict = {
        "schema": "bridge-e2r-split-integrity-v1",
        "dev_relations": len(dev_pairs),
        "reported_outer_relations": len(target_pairs),
        "exact_dev_outer_overlap": len(dev_pairs & target_pairs),
        "independent_of_existing_gate_dev": len(true_outer),
        "strict_claim": len(dev_pairs & target_pairs) == 0,
        "methods": {},
        "warnings": [
            "The 23,773 relation 'outer' is not independent of the frozen adaptive relation gate.",
            "Only pairs absent from the 16,108 gate-development set are eligible for gate-independent scoring.",
            "Other models' tuning histories must be audited separately before calling even the 7,665 fully independent.",
        ],
    }
    methods = {
        "Broad Retrieval": "broad_rank",
        "BRIDGE fixed experts": "full_rank",
        "w/o Functional": "minus_functional_rank",
        "w/o Structure": "minus_structure_mechanism_rank",
        "w/o Long-term Relation": "minus_relational_memory_rank",
    }
    for method, rank_col in methods.items():
        report["methods"][method] = {}
        for name, mask in [
            ("gate_dev", frame.gate_training_pair),
            ("candidate_independent_outer", ~frame.gate_training_pair),
            ("mixed_23773", pd.Series(np.ones(len(frame), dtype=bool))),
        ]:
            report["methods"][method][name] = balanced_raw(frame.loc[mask], rank_col)
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "split_integrity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "dev": len(dev_pairs),
        "mixed": len(target_pairs),
        "exact_overlap": len(dev_pairs & target_pairs),
        "candidate_independent_outer": len(true_outer),
        "E2R_Hit10": {
            m: {k: round(v["balanced"]["hit10"], 5) for k, v in periods.items()}
            for m, periods in report["methods"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
