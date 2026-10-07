from __future__ import annotations

import json

import numpy as np
import pandas as pd

from projects.active.bridge.model.assets import ROOT

BASE = ROOT / "results/bridge_layered_v4_e2r_cage"
MEMBERSHIP = BASE / "pair_membership.csv.gz"
SCORES = BASE / "cage_inference/pairs_epoch_19.csv"
TARGETS = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
BUDGETS = ROOT / "results/bridge_layered_v4_candidates/e2r_query_budget.csv"


def metrics(order: list[str], positives: set[str], total: int) -> dict[str, float]:
    ranks = [i + 1 for i, rid in enumerate(order) if rid in positives]
    if not ranks:
        return {
            "rr": 0.0,
            "hit3": 0.0,
            "hit10": 0.0,
            "hit100": 0.0,
            "positive_recall": 0.0,
        }
    best = min(ranks)
    return {
        "rr": 1.0 / best,
        "hit3": float(best <= 3),
        "hit10": float(best <= 10),
        "hit100": float(best <= 100),
        "positive_recall": float(len(ranks) / total) if total else 0.0,
    }


def summarize(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    return {
        "queries": int(len(frame)),
        "mrr": float(frame[f"{prefix}_rr"].mean()),
        "hit3": float(frame[f"{prefix}_hit3"].mean()),
        "hit10": float(frame[f"{prefix}_hit10"].mean()),
        "hit100": float(frame[f"{prefix}_hit100"].mean()),
        "macro_positive_recall": float(frame[f"{prefix}_positive_recall"].mean()),
        "query_hit": float((frame[f"{prefix}_positive_recall"] > 0).mean()),
    }


def main() -> None:
    membership = pd.read_csv(MEMBERSHIP, dtype=str).fillna("")
    for col in ("native_pool", "broad_pool", "cage_pair_supported"):
        membership[col] = membership[col].astype(str).str.lower().eq("true")
    membership["label"] = pd.to_numeric(membership.label, errors="coerce").fillna(0).astype(int)

    scored = pd.read_csv(SCORES, dtype=str).fillna("")
    scored["pred_logit"] = pd.to_numeric(scored.pred_logit, errors="raise")
    score_key = ["protein_id", "reaction_id"]
    if scored.duplicated(score_key).any():
        raise RuntimeError("CAGE scorer output contains duplicate E2R pairs")
    expected_score = membership.loc[
        membership.cage_pair_supported, score_key
    ].drop_duplicates()
    actual_score = scored[score_key].drop_duplicates()
    coverage = expected_score.merge(actual_score, on=score_key, how="outer", indicator=True)
    if not coverage._merge.eq("both").all():
        counts = coverage._merge.value_counts().to_dict()
        raise RuntimeError(f"incomplete E2R scorer pair coverage: {counts}")
    score = scored[["protein_id", "reaction_id", "pred_logit"]]
    membership = membership.merge(
        score,
        on=["protein_id", "reaction_id"],
        how="left",
        validate="one_to_one",
    )
    unexpected = membership[membership.pred_logit.notna() & ~membership.cage_pair_supported]
    if len(unexpected):
        raise RuntimeError("CAGE produced scores for pairs marked unsupported")

    targets = pd.read_csv(TARGETS, dtype=str).fillna("").drop_duplicates(
        ["protein_id", "reaction_id"]
    )
    positives = (
        targets.groupby("protein_id").reaction_id.agg(lambda x: set(map(str, x))).to_dict()
    )
    budgets = pd.read_csv(BUDGETS, dtype=str).fillna("")
    queries = budgets.protein_id.astype(str).tolist()

    groups = {q: g for q, g in membership.groupby("protein_id", sort=False)}
    records: list[dict[str, object]] = []
    for q in queries:
        pos = positives.get(q, set())
        total = len(pos)
        g = groups.get(q)
        row: dict[str, object] = {
            "protein_id": q,
            "positive_count": total,
            "effective_budget": int(
                budgets.loc[budgets.protein_id.eq(q), "effective_budget"].iloc[0]
            ),
        }
        if g is None:
            native_scored = pd.DataFrame(columns=membership.columns)
            broad_scored = pd.DataFrame(columns=membership.columns)
            native_gate = set()
            broad_gate = set()
        else:
            native_gate = set(g.loc[g.native_pool, "reaction_id"].astype(str))
            broad_gate = set(g.loc[g.broad_pool, "reaction_id"].astype(str))
            native_scored = g[g.native_pool & g.pred_logit.notna()].sort_values(
                ["pred_logit", "reaction_id"],
                ascending=[False, True],
                kind="stable",
            )
            broad_scored = g[g.broad_pool & g.pred_logit.notna()].sort_values(
                ["pred_logit", "reaction_id"],
                ascending=[False, True],
                kind="stable",
            )

        row["native_gate_positive_recall"] = (
            len(native_gate & pos) / total if total else 0.0
        )
        row["broad_gate_positive_recall"] = (
            len(broad_gate & pos) / total if total else 0.0
        )
        row["native_scored_candidates"] = int(len(native_scored))
        row["broad_scored_candidates"] = int(len(broad_scored))
        for prefix, frame in (
            ("cage", native_scored),
            ("broad_cage", broad_scored),
        ):
            order = frame.reaction_id.astype(str).tolist()
            for key, value in metrics(order, pos, total).items():
                row[f"{prefix}_{key}"] = value
        records.append(row)

    qf = pd.DataFrame(records)
    qf.to_csv(BASE / "query_metrics.csv", index=False)

    summary = {
        "schema": "bridge-layered-v4-e2r-cage-eval",
        "status": "completed",
        "protocol": {
            "mother_benchmark": "bridge_relation_unseen_max_v3",
            "mother_queries": int(len(queries)),
            "strict_relation_unseen_relative_to_clean2023": True,
            "matched_budget_only_for_cage_paths": True,
            "unsupported_query_or_pair_is_miss": True,
            "native_gate": "Top-10 similar clean2023 enzymes -> union known reactions",
            "broad_gate": "Broad top-Nq after same clean2023 mask",
            "cage_checkpoint": "generic pretrain seed42 epoch_19",
        },
        "metrics": {
            "enzymecage": summarize(qf, "cage"),
            "broad_cage": summarize(qf, "broad_cage"),
        },
        "support": {
            "scored_unique_pairs": int(len(actual_score)),
            "queries_with_any_native_cage_score": int((qf.native_scored_candidates > 0).sum()),
            "queries_with_any_broad_cage_score": int((qf.broad_scored_candidates > 0).sum()),
            "native_gate_macro_positive_recall_before_scorer_support": float(
                qf.native_gate_positive_recall.mean()
            ),
            "broad_gate_macro_positive_recall_before_scorer_support": float(
                qf.broad_gate_positive_recall.mean()
            ),
        },
    }
    (BASE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
