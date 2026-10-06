from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import torch

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.runtime.final_system import FinalBridgeRuntime
from reproducibility.bime_rank.scripts.evaluate_fibre_cage_shared_pool_v1 import (
    exact_sequence_aliases,
)

PAIRS = (
    ROOT
    / "results/terpene_pure_cage_full_support_v1/pretrain/"
    "pure_cage_native_full_pairs_epoch_19.csv.gz"
)
PROTEIN_SEQUENCES = (
    ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
)
OUT = ROOT / "results/bridge_cage_e2r_fixed465_v1"
BUDGET = 155


def calibrated(
    raw: np.ndarray, available: np.ndarray, center: float, scale: float
) -> np.ndarray:
    out = np.zeros(len(raw), dtype=np.float64)
    if available.any():
        out[available] = (
            raw[available] - float(center)
        ) / max(float(scale), 1e-8)
    return out


def query_metrics(
    frame: pd.DataFrame,
    ordered_ids: list[str],
    positives: set[str],
) -> dict[str, float]:
    ranks = [
        i + 1 for i, rid in enumerate(ordered_ids)
        if rid in positives
    ]
    if not ranks:
        return {
            "rr": 0.0,
            "ap": 0.0,
            "hit3": 0.0,
            "hit10": 0.0,
            "hit20": 0.0,
            "positive_recall": 0.0,
        }
    best = min(ranks)
    precisions = [
        sum(r <= rank for r in ranks) / rank
        for rank in ranks
    ]
    return {
        "rr": 1.0 / best,
        "ap": float(sum(precisions) / len(positives)),
        "hit3": float(best <= 3),
        "hit10": float(best <= 10),
        "hit20": float(best <= 20),
        "positive_recall": float(len(ranks) / len(positives)),
    }


def summarize(rows: pd.DataFrame, prefix: str) -> dict[str, float]:
    return {
        "queries": int(len(rows)),
        "mrr": float(rows[f"{prefix}_rr"].mean()),
        "map": float(rows[f"{prefix}_ap"].mean()),
        "hit3": float(rows[f"{prefix}_hit3"].mean()),
        "hit10": float(rows[f"{prefix}_hit10"].mean()),
        "hit20": float(rows[f"{prefix}_hit20"].mean()),
        "macro_positive_recall": float(
            rows[f"{prefix}_positive_recall"].mean()
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(
        PAIRS,
        dtype={"reaction_id": str, "uniprot_id": str},
    ).fillna("")
    pairs["label"] = pd.to_numeric(
        pairs["label"], errors="coerce"
    ).fillna(0).astype(int)
    pairs["pred_logit"] = pd.to_numeric(
        pairs["pred_logit"], errors="coerce"
    ).astype(float)

    reactions = sorted(pairs["reaction_id"].astype(str).unique())
    proteins = sorted(pairs["uniprot_id"].astype(str).unique())
    if len(reactions) != 465 or len(proteins) != 1379:
        raise RuntimeError(
            f"frozen CAGE cartesian support drift: "
            f"{len(reactions)} reactions, {len(proteins)} proteins"
        )

    positive_by_protein = (
        pairs[pairs["label"].eq(1)]
        .groupby("uniprot_id")["reaction_id"]
        .agg(lambda x: set(map(str, x)))
        .to_dict()
    )
    query_ids = sorted(positive_by_protein)
    if len(query_ids) != 852:
        raise RuntimeError(
            f"expected 852 positive E2R queries, got {len(query_ids)}"
        )

    rt = FinalBridgeRuntime(device="cuda")
    missing_reactions = sorted(
        set(reactions) - set(rt.index.reaction_index)
    )
    if missing_reactions:
        raise RuntimeError(
            f"CAGE support reactions missing from BRIDGE: "
            f"{missing_reactions[:10]}"
        )

    protein_alias = exact_sequence_aliases(
        pairs[pairs["uniprot_id"].isin(query_ids)],
        rt.index,
        PROTEIN_SEQUENCES,
    )
    mapped_queries = [protein_alias[q] for q in query_ids]

    candidate_rows = np.asarray(
        [rt.index.reaction_index[r] for r in reactions],
        dtype=np.int64,
    )
    candidate_rows_t = torch.as_tensor(
        candidate_rows, dtype=torch.long, device=rt.index.device
    )
    reaction_lex = np.asarray(reactions, dtype=object)

    cage_score = (
        pairs[pairs["uniprot_id"].isin(query_ids)]
        .pivot(
            index="uniprot_id",
            columns="reaction_id",
            values="pred_logit",
        )
        .reindex(index=query_ids, columns=reactions)
    )
    if cage_score.isna().any().any():
        raise RuntimeError("CAGE full-cartesian score matrix has holes")
    cage_matrix = cage_score.to_numpy(np.float64)

    functional_member = rt.e2r_members["enzgfm_e2r"]
    clip_member = rt.e2r_members["clipzyme_structure"]
    frows_all = rt._e2r_functional_r_row
    crows_all = rt._clip_r_row

    records: list[dict[str, object]] = []

    for qi, (source_q, q) in enumerate(
        zip(query_ids, mapped_queries, strict=True), 1
    ):
        prow = rt.index.protein_index[q]
        with torch.no_grad():
            broad_t = (
                rt.index.protein_embeddings[prow]
                @ rt.index.reaction_embeddings.T
            ).float()
        broad = broad_t.cpu().numpy().astype(
            np.float64, copy=False
        )
        bstd = max(float(broad.std()), 1e-8)
        bz = (broad - float(broad.mean())) / bstd
        ctx, ca0, cb0, av, both = rt._relation_components(
            "e2r", q
        )
        rw, _, _ = rt._relation_authority(
            "e2r", q, bz, ctx, ca0, cb0, av, both
        )

        core_scale = float(rt.e2r_core_cal["scale"])
        core = (
            broad - float(rt.e2r_core_cal["center"])
        ) / core_scale
        relation_core = rw * (bstd / core_scale) * ctx

        fp = rt.functional_e2r.p_index.get(q, -1)
        frows = frows_all[candidate_rows]
        fa = frows >= 0
        fraw = np.zeros(len(reactions), dtype=np.float64)
        if fp < 0:
            fa[:] = False
        elif fa.any():
            rr = torch.as_tensor(
                frows[fa],
                dtype=torch.long,
                device=rt.index.device,
            )
            with torch.no_grad():
                fraw[fa] = (
                    rt.functional_e2r.r.index_select(0, rr)
                    @ rt.functional_e2r.p[fp]
                ).float().cpu().numpy()
        fcal = calibrated(
            fraw,
            fa,
            functional_member["calibration"]["score_center"],
            functional_member["calibration"]["score_scale"],
        )

        cp = rt.clip.p_index.get(q, -1)
        crows = crows_all[candidate_rows]
        cav = crows >= 0
        craw = np.zeros(len(reactions), dtype=np.float64)
        if cp < 0 or not bool(rt.clip.p_supported[cp]):
            cav[:] = False
        else:
            cav &= rt.clip.r_supported[np.maximum(crows, 0)]
        if cav.any():
            rr = torch.as_tensor(
                crows[cav],
                dtype=torch.long,
                device=rt.index.device,
            )
            with torch.no_grad():
                craw[cav] = (
                    rt.clip.r_device.index_select(0, rr)
                    @ rt.clip.p_device[cp]
                ).float().cpu().numpy()
        ccal = calibrated(
            craw,
            cav,
            clip_member["calibration"]["score_center"],
            clip_member["calibration"]["score_scale"],
        )

        core_subset = core[candidate_rows]
        relation_subset = relation_core[candidate_rows]
        functional_subset = float(functional_member["strength"]) * fcal
        structure_subset = float(clip_member["strength"]) * ccal
        full = (
            core_subset
            + relation_subset
            + functional_subset
            + structure_subset
        )
        variant_scores = {
            "bridge": full,
            "minus_functional": (
                core_subset + relation_subset + structure_subset
            ),
            "minus_structure_mechanism": (
                core_subset + relation_subset + functional_subset
            ),
            "minus_relational_memory": (
                core_subset + functional_subset + structure_subset
            ),
            # Current production E2R has no family/TPS specialist members.
            "minus_family_domain": full,
        }
        broad_subset = broad[candidate_rows]
        cage_subset = cage_matrix[qi - 1]

        def order(score: np.ndarray) -> list[str]:
            idx = np.lexsort(
                (reaction_lex, -score)
            )
            return [reactions[int(i)] for i in idx]

        broad_order_full = order(broad_subset)
        cage_order_full = order(cage_subset)
        variant_orders_full = {
            name: order(score)
            for name, score in variant_scores.items()
        }
        bridge_order_full = variant_orders_full["bridge"]

        broad_pool = broad_order_full[:BUDGET]
        cage_pool = cage_order_full[:BUDGET]
        bridge_pool = bridge_order_full[:BUDGET]

        cage_map = dict(zip(reactions, cage_subset, strict=True))
        broad_cage = sorted(
            broad_pool,
            key=lambda r: (-cage_map[r], r),
        )

        positives = positive_by_protein[source_q]
        methods = {
            "cage": cage_pool,
            "broad": broad_pool,
            "broad_cage": broad_cage,
            **{
                name: ordered[:BUDGET]
                for name, ordered in variant_orders_full.items()
            },
            "cage_full465": cage_order_full,
            "broad_full465": broad_order_full,
            "bridge_full465": bridge_order_full,
        }
        row: dict[str, object] = {
            "uniprot_id": source_q,
            "broad_protein_id": q,
            "positive_count": len(positives),
            "budget": BUDGET,
            "train_memory_weight": rw,
        }
        for name, ordered in methods.items():
            m = query_metrics(pairs, ordered, positives)
            for key, value in m.items():
                row[f"{name}_{key}"] = value
        records.append(row)

        if qi % 100 == 0 or qi == len(query_ids):
            print(
                "e2r-fixed465", qi, "/", len(query_ids),
                flush=True,
            )

    qf = pd.DataFrame(records)
    qf.to_csv(OUT / "query_metrics.csv", index=False)

    methods = [
        "cage",
        "broad",
        "broad_cage",
        "bridge",
        "minus_functional",
        "minus_structure_mechanism",
        "minus_relational_memory",
        "minus_family_domain",
        "cage_full465",
        "broad_full465",
        "bridge_full465",
    ]
    summary = {
        "schema": "bridge-cage-e2r-fixed465-v1",
        "status": "completed",
        "protocol": {
            "source": str(PAIRS.relative_to(ROOT)),
            "queries": len(query_ids),
            "positive_pairs": int(pairs["label"].sum()),
            "fixed_candidate_reactions": len(reactions),
            "budget": BUDGET,
            "budget_basis": (
                "rounded native EnzymeCAGE E2R median candidate "
                "support (155 reactions)"
            ),
            "cage": (
                "generic-pretrain pred_logit selects/ranks top155 "
                "from the fixed 465-reaction domain"
            ),
            "broad": (
                "Broad score selects/ranks top155 from the same "
                "465-reaction domain"
            ),
            "broad_plus_cage": (
                "Broad selects top155; EnzymeCAGE pred_logit "
                "reranks exactly those same 155 reactions"
            ),
            "bridge": (
                "current BRIDGE selects/ranks top155 from the same "
                "465-reaction domain"
            ),
            "no_parameter_fitting": True,
            "no_model_retraining": True,
            "generalization_claim": False,
            "note": (
                "This is a controlled low-cost E2R ranking "
                "benchmark on the EnzymeCAGE TPS support domain, "
                "not an 11,081-reaction open-world test."
            ),
            "ablation_groups": {
                "functional": "EnzGFM functional compatibility",
                "structure_mechanism": "CLIPZyme structural evidence",
                "relational_memory": "clean2023 long-term relation memory",
                "family_domain": (
                    "family/TPS specialists; absent from current E2R, "
                    "therefore identical to full BRIDGE"
                ),
            },
        },
        "metrics": {
            name: summarize(qf, name) for name in methods
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
