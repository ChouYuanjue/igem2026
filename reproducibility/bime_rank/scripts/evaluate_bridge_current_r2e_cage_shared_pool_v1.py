from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.runtime.final_system import FinalBridgeRuntime, _z
from reproducibility.bime_rank.scripts.evaluate_fibre_cage_shared_pool_v1 import (
    DEFAULT_CAGE_PAIRS,
    DEFAULT_CAGE_QUERY_METRICS,
    DEFAULT_PROTEIN_SEQUENCES,
    exact_sequence_aliases,
    ranking_metrics,
    score_broad_matrix,
    summarize_rows,
)

OUT = ROOT / "results/bridge_current_r2e_cage_shared_pool_v1"


def metrics_from_order(
    ordered_ids: list[str],
    positive_ids: set[str],
    total_positives: int,
) -> dict[str, float]:
    ranks = [
        rank
        for rank, pid in enumerate(ordered_ids, 1)
        if pid in positive_ids
    ]
    if not ranks:
        return {
            "rr": 0.0,
            "average_precision": 0.0,
            "hit_at_3": 0.0,
            "hit_at_10": 0.0,
            "hit_at_20": 0.0,
        }
    best = min(ranks)
    precisions = [
        sum(other <= rank for other in ranks) / rank
        for rank in ranks
    ]
    return {
        "rr": 1.0 / best,
        "average_precision": float(
            sum(precisions) / total_positives
        ),
        "hit_at_3": float(best <= 3),
        "hit_at_10": float(best <= 10),
        "hit_at_20": float(best <= 20),
    }


def bridge_variant_orders(
    rt: FinalBridgeRuntime,
    reaction_id: str,
    candidate_ids: list[str],
) -> tuple[dict[str, list[str]], dict[str, float]]:
    qrow = rt.index.reaction_index[reaction_id]
    with torch.no_grad():
        broad_t = (
            rt.index.reaction_embeddings[qrow]
            @ rt.index.protein_embeddings.T
        ).float()
    broad = broad_t.cpu().numpy().astype(
        np.float64, copy=False
    )
    bstd = max(float(broad.std()), 1e-8)
    bz = (broad - float(broad.mean())) / bstd
    ctx, ca0, cb0, available, both = rt._relation_components(
        "r2e", reaction_id
    )
    rw, _, _ = rt._relation_authority(
        "r2e",
        reaction_id,
        bz,
        ctx,
        ca0,
        cb0,
        available,
        both,
    )

    rows = np.asarray(
        [rt.index.protein_index[p] for p in candidate_ids],
        dtype=np.int64,
    )
    core = broad[rows]
    core_std = max(float(core.std()), 1e-6)
    core_z = (core - float(core.mean())) / core_std
    relation_local = rw * (bstd / core_std) * ctx[rows]

    frows = rt._r2e_functional_p_row[rows]
    fr = rt.functional_r2e.r_index.get(reaction_id, -1)
    fa = frows >= 0
    fraw = np.zeros(len(rows), dtype=np.float64)
    if fr < 0:
        fa[:] = False
    elif fa.any():
        rr = torch.as_tensor(
            frows[fa], dtype=torch.long, device=rt.index.device
        )
        with torch.no_grad():
            fraw[fa] = (
                rt.functional_r2e.p.index_select(0, rr)
                @ rt.functional_r2e.r[fr]
            ).float().cpu().numpy()
    fz = _z(fraw, fa)

    crows = rt._clip_p_row[rows]
    cr = rt.clip.r_index.get(reaction_id, -1)
    cav = crows >= 0
    if cr >= 0 and bool(rt.clip.r_supported[cr]):
        cav &= rt.clip.p_supported[np.maximum(crows, 0)]
    else:
        cav[:] = False
    craw = np.zeros(len(rows), dtype=np.float64)
    if cav.any():
        rr = torch.as_tensor(
            crows[cav], dtype=torch.long, device=rt.index.device
        )
        with torch.no_grad():
            craw[cav] = (
                rt.clip.p_device.index_select(0, rr)
                @ rt.clip.r_device[cr]
            ).float().cpu().numpy()
    cz = _z(craw, cav)

    mrows = rt._mechanism_p_row[rows]
    mr = rt.mechanism.r_index.get(reaction_id, -1)
    mav = mrows >= 0
    mraw = np.zeros(len(rows), dtype=np.float64)
    if mr < 0:
        mav[:] = False
    elif mav.any():
        rr = torch.as_tensor(
            mrows[mav], dtype=torch.long, device=rt.index.device
        )
        with torch.no_grad():
            mraw[mav] = (
                rt.mechanism.p.index_select(0, rr)
                @ rt.mechanism.r[mr]
            ).float().cpu().numpy()
    mz = _z(mraw, mav)

    gav = cav | mav
    geometry = np.zeros(len(rows), dtype=np.float64)
    count = np.zeros(len(rows), dtype=np.int32)
    for part, mask in ((cz, cav), (mz, mav)):
        geometry[mask] += part[mask]
        count[mask] += 1
    geometry[gav] /= count[gav]

    fw, gw = rt._r2e_general_weights(
        reaction_id, core, fz, fa, geometry, gav
    )
    specialist, specialist_audit = rt._r2e_specialists(
        reaction_id, candidate_ids
    )

    components = {
        "core": core_z,
        "relation": relation_local,
        "functional": fw * fz,
        "structure": gw * geometry,
        "domain": specialist,
    }
    variant_scores = {
        "bridge": (
            components["core"]
            + components["relation"]
            + components["functional"]
            + components["structure"]
            + components["domain"]
        ),
        "minus_functional": (
            components["core"]
            + components["relation"]
            + components["structure"]
            + components["domain"]
        ),
        "minus_structure_mechanism": (
            components["core"]
            + components["relation"]
            + components["functional"]
            + components["domain"]
        ),
        "minus_relational_memory": (
            components["core"]
            + components["functional"]
            + components["structure"]
            + components["domain"]
        ),
        "minus_family_domain": (
            components["core"]
            + components["relation"]
            + components["functional"]
            + components["structure"]
        ),
    }

    orders: dict[str, list[str]] = {}
    for name, score in variant_scores.items():
        if name == "minus_structure_mechanism":
            order = np.argsort(-score, kind="stable")
        else:
            order = rt._pocket_reorder(
                reaction_id, candidate_ids, score
            )
        orders[name] = [
            candidate_ids[int(i)] for i in order
        ]
    audit = {
        "relational_memory_weight": float(rw),
        "functional_weight": float(fw),
        "geometry_weight": float(gw),
        "family_weight": float(
            specialist_audit.get("family_weight", 0.0)
        ),
        "tps_weight": float(
            specialist_audit.get("tps_weight", 0.0)
        ),
    }
    return orders, audit


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(
        DEFAULT_CAGE_PAIRS,
        dtype={"reaction_id": str, "uniprot_id": str},
    ).fillna("")
    pairs["label"] = pairs["label"].astype(int)
    pairs["pred_logit"] = pairs["pred_logit"].astype(float)

    qm = pd.read_csv(
        DEFAULT_CAGE_QUERY_METRICS,
        dtype={"reaction_id": str},
    ).fillna("")
    query_ids = qm["reaction_id"].astype(str).tolist()
    if len(query_ids) != 459:
        raise RuntimeError("expected frozen 459-query CAGE support")
    pairs = pairs[pairs["reaction_id"].isin(query_ids)].copy()

    rt = FinalBridgeRuntime(device="cuda")
    protein_alias = exact_sequence_aliases(
        pairs,
        rt.index,
        DEFAULT_PROTEIN_SEQUENCES,
    )
    scored = score_broad_matrix(
        pairs,
        query_ids,
        rt.index,
        protein_alias,
    )
    scored["broad_protein_id"] = scored["uniprot_id"].map(
        protein_alias
    )

    qm = qm.set_index("reaction_id")
    rows: list[dict[str, object]] = []

    for qi, (rid, group) in enumerate(
        scored.groupby("reaction_id", sort=False), 1
    ):
        q = qm.loc[str(rid)]
        budget = int(q["candidate_gate_size"])
        positive_count = int(group["label"].sum())

        broad_pool = group.sort_values(
            ["broad_score", "uniprot_id"],
            ascending=[False, True],
            kind="stable",
        ).head(budget).copy()

        broad_metrics = ranking_metrics(
            broad_pool,
            score_column="broad_score",
            total_positives=positive_count,
        )
        cage_metrics = ranking_metrics(
            broad_pool,
            score_column="pred_logit",
            total_positives=positive_count,
        )

        candidate_ids = (
            broad_pool["broad_protein_id"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        orders, audit = bridge_variant_orders(
            rt, str(rid), candidate_ids
        )
        positive_ids = set(
            broad_pool.loc[
                broad_pool["label"].eq(1),
                "broad_protein_id",
            ].astype(str)
        )

        row = {
            "reaction_id": str(rid),
            "candidate_budget": budget,
            "positive_count": positive_count,
            "broad_pool_positive_count": int(
                broad_pool["label"].sum()
            ),
            **audit,
        }
        for name, ordered_ids in orders.items():
            vm = metrics_from_order(
                ordered_ids,
                positive_ids,
                positive_count,
            )
            for key, value in vm.items():
                row[f"{name}_{key}"] = value
        row.update(
            {
                f"broad_{k}": v
                for k, v in broad_metrics.items()
            }
        )
        row.update(
            {
                f"cage_{k}": v
                for k, v in cage_metrics.items()
            }
        )
        rows.append(row)

        if qi % 50 == 0 or qi == len(query_ids):
            print("r2e-shared", qi, "/", len(query_ids), flush=True)

    qf = pd.DataFrame(rows)
    qf.to_csv(OUT / "query_metrics.csv", index=False)

    summary = {
        "schema": "bridge-current-r2e-cage-shared-pool-v1",
        "status": "completed",
        "protocol": {
            "queries": len(qf),
            "candidate_budget": (
                "per-query K equals archived EnzymeCAGE native "
                "candidate_gate_size"
            ),
            "candidate_generation": (
                "Broad ranks the complete CAGE-scored 1379-protein "
                "support and selects exactly K candidates"
            ),
            "broad": "Broad score ranks the same K candidates",
            "broad_plus_cage": (
                "EnzymeCAGE pred_logit reranks exactly the Broad K"
            ),
            "bridge": (
                "current FinalBridgeRuntime reranks exactly the same "
                "Broad K via explicit candidate_ids"
            ),
            "generalization_claim": False,
            "no_parameter_fitting": True,
            "no_model_retraining": True,
            "ablation_groups": {
                "functional": "EnzGFM functional compatibility",
                "structure_mechanism": (
                    "CLIPZyme + reaction-center/mechanism + pocket"
                ),
                "relational_memory": (
                    "clean2023 long-term relation memory"
                ),
                "family_domain": "family CAGE + TPS specialists",
            },
        },
        "broad": summarize_rows(qf, "broad"),
        "broad_plus_cage": summarize_rows(qf, "cage"),
        **{
            name: {
                "queries": int(len(qf)),
                "mrr": float(qf[f"{name}_rr"].mean()),
                "map": float(
                    qf[f"{name}_average_precision"].mean()
                ),
                "hit_at_3": float(
                    qf[f"{name}_hit_at_3"].mean()
                ),
                "hit_at_10": float(
                    qf[f"{name}_hit_at_10"].mean()
                ),
                "hit_at_20": float(
                    qf[f"{name}_hit_at_20"].mean()
                ),
                "query_positive_coverage": float(
                    qf[f"{name}_rr"].gt(0).mean()
                ),
            }
            for name in (
                "bridge",
                "minus_functional",
                "minus_structure_mechanism",
                "minus_relational_memory",
                "minus_family_domain",
            )
        },
        "broad_pool_query_positive_coverage": float(
            qf["broad_pool_positive_count"].gt(0).mean()
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
