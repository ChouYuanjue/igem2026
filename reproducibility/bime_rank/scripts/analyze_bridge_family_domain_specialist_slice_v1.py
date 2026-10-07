from __future__ import annotations

import json

import pandas as pd

from projects.active.bridge.model.assets import ROOT

QUERY_AUDIT = ROOT / "results/bridge_r2e_four_group_ablation_v1/query_audit.csv.gz"
EDGE_METRICS = ROOT / "results/bridge_r2e_four_group_ablation_v1/edge_metrics.csv.gz"
FAMILY_APP = ROOT / "results/fibre_family_applicability_router_v1/full_outer_predictions.csv"
OUT = ROOT / "reproducibility/bime_rank/records/BRIDGE_FAMILY_DOMAIN_SPECIALIST_SLICE_V1_RESULT.json"
DOC = ROOT / "projects/active/bridge/docs/family_domain_specialist_slice_v1.md"

FAMILY_ROWS = (
    ("p450", "P450 family-CAGE"),
    ("phosphatase", "Phosphatase family-CAGE"),
    ("terpene", "Terpene family-CAGE"),
    ("tps", "TPS specialist"),
)


def metrics(ranks: pd.Series) -> dict[str, float]:
    r = pd.to_numeric(ranks, errors="raise")
    return {
        "mrr": float((1.0 / r).mean()),
        "hit3": float((r <= 3).mean()),
        "hit10": float((r <= 10).mean()),
        "hit100": float((r <= 100).mean()),
    }


def fmt_pair(before: float, after: float, percent: bool = False) -> str:
    if percent:
        return f"{100*before:.2f}% -> {100*after:.2f}%"
    return f"{before:.5f} -> {after:.5f}"


def main() -> None:
    qa = pd.read_csv(QUERY_AUDIT, dtype=str).fillna("")
    for col in ("family_weight", "tps_weight"):
        qa[col] = pd.to_numeric(qa[col], errors="coerce").fillna(0.0)

    app = (
        pd.read_csv(FAMILY_APP, dtype=str)
        .fillna("")
        .drop_duplicates("reaction_id")
        .set_index("reaction_id")
    )

    labels: dict[str, str] = {}
    for rec in qa[qa.family_weight > 0].itertuples(index=False):
        rid = str(rec.query_id)
        if rid not in app.index:
            raise RuntimeError(f"family-active query missing applicability label: {rid}")
        labels[rid] = str(app.loc[rid, "pred"])

    for rid in qa.loc[qa.tps_weight > 0, "query_id"].astype(str):
        if rid in labels:
            raise RuntimeError(f"query has both family-CAGE and TPS authority: {rid}")
        labels[rid] = "tps"

    edge = pd.read_csv(EDGE_METRICS, dtype=str).fillna("")
    for col in ("full_rank", "minus_family_domain_rank"):
        edge[col] = pd.to_numeric(edge[col], errors="raise")

    query = (
        edge.groupby("reaction_id")[["full_rank", "minus_family_domain_rank"]]
        .min()
        .sort_index()
    )

    rows: list[dict[str, object]] = []
    for key, display in FAMILY_ROWS:
        ids = sorted(rid for rid, label in labels.items() if label == key)
        if not ids:
            rows.append(
                {
                    "key": key,
                    "expert": display,
                    "activated_queries": 0,
                    "heldout_edges": 0,
                    "query_before": None,
                    "query_after": None,
                    "edge_before_mrr": None,
                    "edge_after_mrr": None,
                    "improved_edges": 0,
                    "worsened_edges": 0,
                    "same_edges": 0,
                }
            )
            continue

        qsub = query.loc[ids]
        esub = edge[edge.reaction_id.astype(str).isin(ids)].copy()
        before_q = metrics(qsub.minus_family_domain_rank)
        after_q = metrics(qsub.full_rank)
        before_e = metrics(esub.minus_family_domain_rank)
        after_e = metrics(esub.full_rank)
        rows.append(
            {
                "key": key,
                "expert": display,
                "activated_queries": len(ids),
                "heldout_edges": int(len(esub)),
                "query_ids": ids,
                "query_before": before_q,
                "query_after": after_q,
                "query_delta_mrr": after_q["mrr"] - before_q["mrr"],
                "edge_before_mrr": before_e["mrr"],
                "edge_after_mrr": after_e["mrr"],
                "improved_edges": int((esub.full_rank < esub.minus_family_domain_rank).sum()),
                "worsened_edges": int((esub.full_rank > esub.minus_family_domain_rank).sum()),
                "same_edges": int((esub.full_rank == esub.minus_family_domain_rank).sum()),
            }
        )

    all_ids = sorted(labels)
    qsub = query.loc[all_ids]
    esub = edge[edge.reaction_id.astype(str).isin(all_ids)].copy()
    before_q = metrics(qsub.minus_family_domain_rank)
    after_q = metrics(qsub.full_rank)
    pooled = {
        "expert": "All activated Family/Domain",
        "activated_queries": len(all_ids),
        "heldout_edges": int(len(esub)),
        "query_ids": all_ids,
        "query_before": before_q,
        "query_after": after_q,
        "query_delta_mrr": after_q["mrr"] - before_q["mrr"],
        "edge_before_mrr": metrics(esub.minus_family_domain_rank)["mrr"],
        "edge_after_mrr": metrics(esub.full_rank)["mrr"],
        "improved_edges": int((esub.full_rank < esub.minus_family_domain_rank).sum()),
        "worsened_edges": int((esub.full_rank > esub.minus_family_domain_rank).sum()),
        "same_edges": int((esub.full_rank == esub.minus_family_domain_rank).sum()),
    }

    result = {
        "schema": "bridge-family-domain-specialist-slice-v1",
        "status": "completed",
        "direction": "r2e",
        "evaluation_unit": "activated-query slice; query-macro best-positive rank",
        "before": "BRIDGE - Family/Domain",
        "after": "BRIDGE",
        "family_label_source": str(FAMILY_APP.relative_to(ROOT)),
        "ablation_source": str(EDGE_METRICS.relative_to(ROOT)),
        "rows": rows,
        "pooled": pooled,
        "notes": [
            "Only queries with non-zero production family_weight or tps_weight are included.",
            "Terpene family-CAGE has zero activated queries in this strict relation-unseen mother benchmark.",
            "TPS is reported separately from generic terpene family-CAGE because it is a distinct specialist route.",
            "Main score columns use query-macro best-positive rank, matching the direct R2E main-table semantics.",
            "Edge-change counts expose relation-level movement even when the best-positive query rank is unchanged.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# BRIDGE Family/Domain specialist slice",
        "",
        "口径：仅统计生产路径中 Family/Domain 专家实际激活的 R2E query；"
        "Before 为 BRIDGE - Family/Domain，After 为完整 BRIDGE。"
        "MRR/Hit 使用每个 query 的最佳 held-out positive rank，与 R2E direct 主表一致。",
        "",
        "| Expert slice | Activated q | Held-out edges | Query MRR Before -> After | Delta MRR | Hit@10 Before -> After | Hit@100 Before -> After | Edge rank changes (up/down) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["activated_queries"] == 0:
            lines.append(f"| {row['expert']} | 0 | 0 | -- | -- | -- | -- | 0 / 0 |")
            continue
        b = row["query_before"]
        a = row["query_after"]
        lines.append(
            f"| {row['expert']} | {row['activated_queries']} | {row['heldout_edges']} | "
            f"{fmt_pair(b['mrr'], a['mrr'])} | {row['query_delta_mrr']:+.5f} | "
            f"{fmt_pair(b['hit10'], a['hit10'], True)} | "
            f"{fmt_pair(b['hit100'], a['hit100'], True)} | "
            f"{row['improved_edges']} / {row['worsened_edges']} |"
        )

    b = pooled["query_before"]
    a = pooled["query_after"]
    lines += [
        f"| **{pooled['expert']}** | **{pooled['activated_queries']}** | **{pooled['heldout_edges']}** | "
        f"**{fmt_pair(b['mrr'], a['mrr'])}** | **{pooled['query_delta_mrr']:+.5f}** | "
        f"**{fmt_pair(b['hit10'], a['hit10'], True)}** | "
        f"**{fmt_pair(b['hit100'], a['hit100'], True)}** | "
        f"**{pooled['improved_edges']} / {pooled['worsened_edges']}** |",
        "",
        "说明：P450 的 query-level 最佳正例排名未变化，但 106 条 held-out 关系中有 1 条由 rank 16 变为 17；"
        "TPS specialist 在 13 条 held-out 关系中改善 7 条、无恶化，并使其 11 个激活 query 的 "
        "MRR 从 0.01047 提升到 0.02820。",
        "",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines))

    print(DOC)
    print(OUT)
    for row in rows:
        print(row["expert"], row["activated_queries"], row.get("query_delta_mrr"))
    print("pooled", pooled["query_delta_mrr"])


if __name__ == "__main__":
    main()
