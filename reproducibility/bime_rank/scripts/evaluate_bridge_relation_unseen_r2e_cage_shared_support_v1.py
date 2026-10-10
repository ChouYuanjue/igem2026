from __future__ import annotations

import json

import numpy as np
import pandas as pd

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.runtime.final_system import FinalBridgeRuntime
from reproducibility.bime_rank.scripts.evaluate_bridge_current_r2e_cage_shared_pool_v1 import (
    bridge_variant_orders,
    metrics_from_order,
)

BASE = ROOT / "results/bridge_relation_unseen_r2e_cage_shared_support_v1"
PAIRS = BASE / "pairs.csv"
MEMBERSHIP = BASE / "membership.csv.gz"
QUERY = BASE / "query_audit.csv"
CAGE = BASE / "cage_inference/pairs_epoch_19.csv"


def summarize(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    return {
        "queries": int(len(frame)),
        "mrr": float(frame[f"{prefix}_rr"].mean()),
        "map": float(frame[f"{prefix}_average_precision"].mean()),
        "hit3": float(frame[f"{prefix}_hit_at_3"].mean()),
        "hit10": float(frame[f"{prefix}_hit_at_10"].mean()),
        "hit20": float(frame[f"{prefix}_hit_at_20"].mean()),
        "macro_positive_recall": float(
            frame[f"{prefix}_positive_recall"].mean()
        ),
    }


def metrics_with_recall(
    ordered: list[str],
    positives: set[str],
    total_positives: int,
) -> dict[str, float]:
    m = metrics_from_order(ordered, positives, total_positives)
    captured = sum(pid in positives for pid in ordered)
    m["positive_recall"] = (
        float(captured / total_positives) if total_positives else 0.0
    )
    return m


def main() -> None:
    pairs = pd.read_csv(PAIRS, dtype=str).fillna("")
    pairs["Label"] = pd.to_numeric(
        pairs["Label"], errors="coerce"
    ).fillna(0).astype(int)
    membership = pd.read_csv(MEMBERSHIP, dtype=str).fillna("")
    for c in ("native_pool", "broad_pool", "broad_rank"):
        membership[c] = pd.to_numeric(
            membership[c], errors="coerce"
        ).fillna(0).astype(int)
    membership["broad_score"] = pd.to_numeric(
        membership["broad_score"], errors="coerce"
    )
    query = pd.read_csv(QUERY, dtype=str).fillna("")
    for c in (
        "scoreable_positive_count",
        "native_budget",
        "broad_budget",
        "native_positive_count",
        "broad_positive_count",
    ):
        query[c] = pd.to_numeric(query[c], errors="coerce").fillna(0).astype(int)
    query = query.set_index("reaction_id")

    cage = pd.read_csv(CAGE, dtype=str).fillna("")
    cage["pred_logit"] = pd.to_numeric(
        cage["pred_logit"], errors="coerce"
    ).astype(float)
    score = cage[
        ["reaction_id", "protein_id", "pred_logit"]
    ].drop_duplicates(["reaction_id", "protein_id"])
    membership = membership.merge(
        score,
        on=["reaction_id", "protein_id"],
        how="left",
        validate="one_to_one",
    )
    if membership.pred_logit.isna().any():
        raise RuntimeError("missing CAGE scores in strict R2E union")

    positives = (
        pairs[pairs.Label.eq(1)]
        .groupby("reaction_id")["protein_id"]
        .agg(lambda x: set(map(str, x)))
        .to_dict()
    )

    rt = FinalBridgeRuntime(device="cuda")
    records: list[dict[str, object]] = []
    for qi, rid in enumerate(sorted(query.index.astype(str)), 1):
        q = query.loc[rid]
        group = membership[membership.reaction_id.eq(rid)].copy()
        pos = positives.get(rid, set())
        total_pos = int(q.scoreable_positive_count)

        native = group[group.native_pool.eq(1)].copy()
        native_order = (
            native.sort_values(
                ["pred_logit", "protein_id"],
                ascending=[False, True],
                kind="stable",
            )
            .protein_id.astype(str)
            .tolist()
        )

        broad = group[group.broad_pool.eq(1)].copy()
        broad_order = (
            broad.sort_values(
                ["broad_score", "protein_id"],
                ascending=[False, True],
                kind="stable",
            )
            .protein_id.astype(str)
            .tolist()
        )
        broad_cage_order = (
            broad.sort_values(
                ["pred_logit", "protein_id"],
                ascending=[False, True],
                kind="stable",
            )
            .protein_id.astype(str)
            .tolist()
        )

        candidate_ids = broad_order
        variant_orders, audit = bridge_variant_orders(
            rt, rid, candidate_ids
        )

        orders = {
            "enzymecage": native_order,
            "broad": broad_order,
            "broad_cage": broad_cage_order,
            **variant_orders,
        }
        row: dict[str, object] = {
            "reaction_id": rid,
            "scoreable_positive_count": total_pos,
            "native_budget": int(q.native_budget),
            "broad_budget": int(q.broad_budget),
            **audit,
        }
        for name, ordered in orders.items():
            m = metrics_with_recall(
                ordered,
                pos,
                total_pos,
            )
            for key, value in m.items():
                row[f"{name}_{key}"] = value
        records.append(row)

        if qi % 50 == 0 or qi == len(query):
            print("strict-r2e-eval", qi, "/", len(query), flush=True)

    qf = pd.DataFrame(records)
    qf.to_csv(BASE / "query_metrics.csv", index=False)

    methods = (
        "enzymecage",
        "broad",
        "broad_cage",
        "bridge",
        "minus_functional",
        "minus_structure_mechanism",
        "minus_relational_memory",
        "minus_family_domain",
    )
    summary = {
        "schema": "bridge-relation-unseen-r2e-cage-shared-support-eval-v1",
        "status": "completed",
        "protocol": {
            "mother_benchmark": "bridge_relation_unseen_max_v3",
            "mother_edges": 23773,
            "mother_queries": 1923,
            "exact_pair_relation_unseen_relative_to_clean2023": True,
            "shared_scoreable_protein_universe": 7327,
            "candidate_budget": (
                "per query, Broad pool size equals the EnzymeCAGE native "
                "gate candidate count after both are restricted to the same "
                "CAGE-scoreable protein support and clean2023 known positives "
                "are filtered"
            ),
            "generalization_claim": (
                "reranking control only; full-space retrieval is reported "
                "separately on all strict queries"
            ),
            "cage_checkpoint": "generic pretrain seed42 epoch_19",
        },
        "queries": int(len(qf)),
        "strict_positive_edges_in_support": int(
            qf.scoreable_positive_count.sum()
        ),
        "metrics": {
            method: summarize(qf, method)
            for method in methods
        },
    }
    (BASE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
