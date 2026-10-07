from __future__ import annotations

import json

import pandas as pd

from projects.active.bridge.model.assets import ROOT

BASE = ROOT / "results/bridge_layered_v4_r2e_cage"
MEMBERSHIP = BASE / "pair_membership.csv.gz"
SCORES = BASE / "cage_inference/pairs_epoch_19.csv"
TARGETS = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
BUDGETS = ROOT / "results/bridge_layered_v4_candidates/r2e_query_budget.csv"


def parse_targets(value: str) -> set[str]:
    return {x for x in str(value).split(";") if x}


def route_metrics(
    frame: pd.DataFrame,
    positives: set[str],
    total: int,
) -> dict[str, float]:
    if frame.empty:
        return {
            "rr": 0.0,
            "hit3": 0.0,
            "hit10": 0.0,
            "hit100": 0.0,
            "positive_recall": 0.0,
        }

    found: set[str] = set()
    best_rank: int | None = None
    for rank, rec in enumerate(frame.itertuples(index=False), 1):
        mapped = parse_targets(rec.target_protein_ids)
        hit = mapped & positives
        if hit:
            found |= hit
            if best_rank is None:
                best_rank = rank
    if best_rank is None:
        return {
            "rr": 0.0,
            "hit3": 0.0,
            "hit10": 0.0,
            "hit100": 0.0,
            "positive_recall": 0.0,
        }
    return {
        "rr": 1.0 / best_rank,
        "hit3": float(best_rank <= 3),
        "hit10": float(best_rank <= 10),
        "hit100": float(best_rank <= 100),
        "positive_recall": float(len(found) / total) if total else 0.0,
    }


def gate_recall(frame: pd.DataFrame, positives: set[str], total: int) -> float:
    found: set[str] = set()
    for value in frame.target_protein_ids.astype(str):
        found |= parse_targets(value) & positives
    return float(len(found) / total) if total else 0.0


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
    membership["cage_pair_supported"] = membership.cage_pair_supported.astype(
        str
    ).str.lower().eq("true")
    membership["label"] = pd.to_numeric(
        membership.label, errors="coerce"
    ).fillna(0).astype(int)

    scored = pd.read_csv(SCORES, dtype=str).fillna("")
    scored["pred_logit"] = pd.to_numeric(scored.pred_logit, errors="raise")
    score = scored[
        ["reaction_id", "UniprotID", "pred_logit"]
    ].drop_duplicates(["reaction_id", "UniprotID"])
    membership = membership.merge(
        score,
        left_on=["reaction_id", "score_uid"],
        right_on=["reaction_id", "UniprotID"],
        how="left",
        validate="many_to_one",
    )
    unexpected = membership[
        membership.pred_logit.notna() & ~membership.cage_pair_supported
    ]
    if len(unexpected):
        raise RuntimeError("CAGE produced scores for pairs marked unsupported")

    targets = pd.read_csv(TARGETS, dtype=str).fillna("").drop_duplicates(
        ["protein_id", "reaction_id"]
    )
    positives = (
        targets.groupby("reaction_id").protein_id.agg(
            lambda x: set(map(str, x))
        ).to_dict()
    )
    budgets = pd.read_csv(BUDGETS, dtype=str).fillna("")
    queries = budgets.reaction_id.astype(str).tolist()
    groups = {
        q: group for q, group in membership.groupby("reaction_id", sort=False)
    }

    records: list[dict[str, object]] = []
    for q in queries:
        pos = positives.get(q, set())
        total = len(pos)
        group = groups.get(q)
        row: dict[str, object] = {
            "reaction_id": q,
            "positive_count": total,
            "effective_budget": int(
                budgets.loc[
                    budgets.reaction_id.eq(q),
                    "effective_budget_post_clean2023_mask",
                ].iloc[0]
            ),
        }
        for route, prefix in (
            ("enzymecage", "cage"),
            ("broad_cage", "broad_cage"),
        ):
            if group is None:
                all_route = pd.DataFrame(columns=membership.columns)
                scored_route = all_route
            else:
                all_route = group[group.route.eq(route)].copy()
                scored_route = all_route[all_route.pred_logit.notna()].sort_values(
                    ["pred_logit", "logical_candidate_id"],
                    ascending=[False, True],
                    kind="stable",
                )
            row[f"{prefix}_gate_positive_recall"] = gate_recall(
                all_route, pos, total
            )
            row[f"{prefix}_scored_candidates"] = int(len(scored_route))
            for key, value in route_metrics(
                scored_route, pos, total
            ).items():
                row[f"{prefix}_{key}"] = value
        records.append(row)

    query_frame = pd.DataFrame(records)
    query_frame.to_csv(BASE / "query_metrics.csv", index=False)

    summary = {
        "schema": "bridge-layered-v4-r2e-cage-eval",
        "status": "completed",
        "protocol": {
            "mother_benchmark": "bridge_relation_unseen_max_v3",
            "mother_queries": int(len(queries)),
            "strict_relation_unseen_relative_to_clean2023": True,
            "known_relation_mask_precedes_budget": True,
            "matched_budget_only_for_cage_paths": True,
            "unsupported_query_or_pair_is_miss": True,
            "native_gate": "published EnzymeCAGE Top-10 similar-reaction retrieval",
            "broad_gate": "Broad top-Nq after the same clean2023 mask",
            "cage_checkpoint": "generic pretrain seed42 epoch_19",
        },
        "metrics": {
            "enzymecage": summarize(query_frame, "cage"),
            "broad_cage": summarize(query_frame, "broad_cage"),
        },
        "support": {
            "queries_with_any_native_cage_score": int(
                (query_frame.cage_scored_candidates > 0).sum()
            ),
            "queries_with_any_broad_cage_score": int(
                (query_frame.broad_cage_scored_candidates > 0).sum()
            ),
            "native_gate_macro_positive_recall_before_scorer_support": float(
                query_frame.cage_gate_positive_recall.mean()
            ),
            "broad_gate_macro_positive_recall_before_scorer_support": float(
                query_frame.broad_cage_gate_positive_recall.mean()
            ),
        },
    }
    (BASE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
