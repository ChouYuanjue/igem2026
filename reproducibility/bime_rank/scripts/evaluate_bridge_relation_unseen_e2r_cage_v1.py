from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.runtime.final_system import FinalBridgeRuntime

PAIR_INPUT = ROOT / "results/bridge_relation_unseen_e2r_cage_v1/pairs.csv"
CAGE_SCORE = (
    ROOT
    / "results/bridge_relation_unseen_e2r_cage_v1/cage_inference/"
    "pairs_epoch_19.csv"
)
TRAIN = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
OUT = ROOT / "results/bridge_relation_unseen_e2r_cage_v1"
BUDGETS = (50, 100, 200)


def calibrated(
    raw: np.ndarray,
    available: np.ndarray,
    center: float,
    scale: float,
) -> np.ndarray:
    out = np.zeros(len(raw), dtype=np.float64)
    if available.any():
        out[available] = (
            raw[available] - float(center)
        ) / max(float(scale), 1e-8)
    return out


def ordered_ids(
    ids: np.ndarray,
    score: np.ndarray,
) -> list[str]:
    order = np.lexsort((ids, -score))
    return [str(ids[int(i)]) for i in order]


def rank_metrics(
    ordered: list[str],
    positives: set[str],
) -> dict[str, float]:
    ranks = [i + 1 for i, rid in enumerate(ordered) if rid in positives]
    if not ranks:
        return {
            "rr": 0.0,
            "ap": 0.0,
            "hit10": 0.0,
            "hit100": 0.0,
            "hit1000": 0.0,
            "positive_recall": 0.0,
        }
    ranks.sort()
    best = ranks[0]
    precisions = [
        sum(other <= rank for other in ranks) / rank
        for rank in ranks
    ]
    return {
        "rr": 1.0 / best,
        "ap": float(sum(precisions) / len(positives)),
        "hit10": float(best <= 10),
        "hit100": float(best <= 100),
        "hit1000": float(best <= 1000),
        "positive_recall": float(len(ranks) / len(positives)),
    }


def summarize(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    return {
        "queries": int(len(frame)),
        "mrr": float(frame[f"{prefix}_rr"].mean()),
        "map": float(frame[f"{prefix}_ap"].mean()),
        "hit10": float(frame[f"{prefix}_hit10"].mean()),
        "hit100": float(frame[f"{prefix}_hit100"].mean()),
        "hit1000": float(frame[f"{prefix}_hit1000"].mean()),
        "macro_positive_recall": float(
            frame[f"{prefix}_positive_recall"].mean()
        ),
    }


def main() -> None:
    pairs = pd.read_csv(PAIR_INPUT, dtype=str).fillna("")
    pairs["Label"] = pd.to_numeric(
        pairs["Label"], errors="coerce"
    ).fillna(0).astype(int)
    if not CAGE_SCORE.exists():
        raise FileNotFoundError(CAGE_SCORE)
    scored = pd.read_csv(CAGE_SCORE, dtype=str).fillna("")
    scored["pred_logit"] = pd.to_numeric(
        scored["pred_logit"], errors="coerce"
    ).astype(float)
    keys = ["protein_id", "reaction_id"]
    keep = scored[keys + ["pred_logit"]].copy()
    pairs = pairs.merge(keep, on=keys, how="inner", validate="one_to_one")

    candidate_ids = sorted(pairs.reaction_id.astype(str).unique())
    query_ids = sorted(
        pairs.loc[pairs.Label.eq(1), "protein_id"].astype(str).unique()
    )
    if len(query_ids) == 0:
        raise RuntimeError("no positive strict E2R queries")

    positive = (
        pairs[pairs.Label.eq(1)]
        .groupby("protein_id")["reaction_id"]
        .agg(lambda x: set(map(str, x)))
        .to_dict()
    )
    cage = (
        pairs.pivot(
            index="protein_id",
            columns="reaction_id",
            values="pred_logit",
        )
        .reindex(index=query_ids, columns=candidate_ids)
    )
    if cage.isna().any().any():
        raise RuntimeError("CAGE score matrix has holes")
    cage_matrix = cage.to_numpy(np.float64)

    train = pd.read_csv(TRAIN, dtype=str).fillna("").drop_duplicates(keys)
    known = (
        train.groupby("protein_id")["reaction_id"]
        .agg(lambda x: set(map(str, x)))
        .to_dict()
    )
    train_pairs = set(map(tuple, train[keys].itertuples(index=False, name=None)))
    for q in query_ids:
        if any((q, rid) in train_pairs for rid in positive[q]):
            raise RuntimeError(f"strict positive leaks into clean2023: {q}")

    rt = FinalBridgeRuntime(device="cuda")
    ridx = rt.index.reaction_index
    candidate_rows = np.asarray([ridx[r] for r in candidate_ids], dtype=np.int64)
    candidate_rows_t = torch.as_tensor(
        candidate_rows, dtype=torch.long, device=rt.index.device
    )
    ids_np = np.asarray(candidate_ids, dtype=object)

    functional_member = rt.e2r_members["enzgfm_e2r"]
    clip_member = rt.e2r_members["clipzyme_structure"]
    frows_all = rt._e2r_functional_r_row
    crows_all = rt._clip_r_row
    frows = frows_all[candidate_rows]
    crows = crows_all[candidate_rows]

    records: list[dict[str, object]] = []
    for qi, q in enumerate(query_ids):
        if q not in rt.index.protein_index:
            continue
        prow = rt.index.protein_index[q]
        with torch.no_grad():
            broad_full = (
                rt.index.protein_embeddings[prow]
                @ rt.index.reaction_embeddings.T
            ).float().cpu().numpy().astype(np.float64, copy=False)

        bstd = max(float(broad_full.std()), 1e-8)
        bz = (broad_full - float(broad_full.mean())) / bstd
        ctx, ca0, cb0, av, both = rt._relation_components("e2r", q)
        rw, _, _ = rt._relation_authority(
            "e2r", q, bz, ctx, ca0, cb0, av, both
        )

        core_scale = float(rt.e2r_core_cal["scale"])
        core_full = (
            broad_full - float(rt.e2r_core_cal["center"])
        ) / core_scale
        relation_full = rw * (bstd / core_scale) * ctx

        fp = rt.functional_e2r.p_index.get(q, -1)
        fa = frows >= 0
        fraw = np.zeros(len(candidate_rows), dtype=np.float64)
        if fp < 0:
            fa = np.zeros_like(fa)
        elif fa.any():
            rows_t = torch.as_tensor(
                frows[fa], dtype=torch.long, device=rt.index.device
            )
            with torch.no_grad():
                fraw[fa] = (
                    rt.functional_e2r.r.index_select(0, rows_t)
                    @ rt.functional_e2r.p[fp]
                ).float().cpu().numpy()
        fcal = calibrated(
            fraw,
            fa,
            functional_member["calibration"]["score_center"],
            functional_member["calibration"]["score_scale"],
        )

        cp = rt.clip.p_index.get(q, -1)
        cav = crows >= 0
        craw = np.zeros(len(candidate_rows), dtype=np.float64)
        if cp < 0 or not bool(rt.clip.p_supported[cp]):
            cav = np.zeros_like(cav)
        else:
            cav &= rt.clip.r_supported[np.maximum(crows, 0)]
        if cav.any():
            rows_t = torch.as_tensor(
                crows[cav], dtype=torch.long, device=rt.index.device
            )
            with torch.no_grad():
                craw[cav] = (
                    rt.clip.r_device.index_select(0, rows_t)
                    @ rt.clip.p_device[cp]
                ).float().cpu().numpy()
        ccal = calibrated(
            craw,
            cav,
            clip_member["calibration"]["score_center"],
            clip_member["calibration"]["score_scale"],
        )

        core = core_full[candidate_rows]
        relation = relation_full[candidate_rows]
        functional = float(functional_member["strength"]) * fcal
        structure = float(clip_member["strength"]) * ccal
        variants = {
            "bridge": core + relation + functional + structure,
            "minus_functional": core + relation + structure,
            "minus_structure_mechanism": core + relation + functional,
            "minus_relational_memory": core + functional + structure,
            "minus_family_domain": core + relation + functional + structure,
        }

        allowed = np.ones(len(candidate_ids), dtype=bool)
        known_q = known.get(q, set())
        if known_q:
            allowed &= np.asarray(
                [rid not in known_q for rid in candidate_ids],
                dtype=bool,
            )

        positive_q = positive[q]
        if not positive_q <= set(np.asarray(candidate_ids)[allowed]):
            raise RuntimeError(f"positive outside allowed domain for {q}")

        broad_score = broad_full[candidate_rows].copy()
        cage_score = cage_matrix[qi].copy()
        broad_score[~allowed] = -np.inf
        cage_score[~allowed] = -np.inf
        for score in variants.values():
            score[~allowed] = -np.inf

        allowed_ids = ids_np[allowed]
        broad_allowed = broad_score[allowed]
        cage_allowed = cage_score[allowed]

        orders = {
            "cage": ordered_ids(allowed_ids, cage_allowed),
            "broad": ordered_ids(allowed_ids, broad_allowed),
        }
        for name, score in variants.items():
            orders[name] = ordered_ids(allowed_ids, score[allowed])

        cage_map = {
            rid: float(cage_score[i])
            for i, rid in enumerate(candidate_ids)
            if allowed[i]
        }
        full_order_names = [
            "cage",
            "broad",
            "bridge",
            "minus_functional",
            "minus_structure_mechanism",
            "minus_relational_memory",
            "minus_family_domain",
        ]
        for budget in BUDGETS:
            k = min(budget, len(allowed_ids))
            for name in full_order_names:
                orders[f"{name}_k{budget}"] = orders[name][:k]
            broad_top = orders["broad"][:k]
            orders[f"broad_cage_k{budget}"] = sorted(
                broad_top,
                key=lambda rid: (-cage_map[rid], rid),
            )

        row: dict[str, object] = {
            "protein_id": q,
            "candidate_count": int(allowed.sum()),
            "positive_count": len(positive_q),
            "relation_memory_weight": float(rw),
            "functional_support_fraction": float(fa.mean()),
            "structure_support_fraction": float(cav.mean()),
        }
        for name, order in orders.items():
            metrics = rank_metrics(order, positive_q)
            for metric, value in metrics.items():
                row[f"{name}_{metric}"] = value
        records.append(row)

        if (qi + 1) % 50 == 0 or qi + 1 == len(query_ids):
            print("strict-e2r-eval", qi + 1, "/", len(query_ids), flush=True)

    qf = pd.DataFrame(records)
    qf.to_csv(OUT / "query_metrics.csv", index=False)

    base_names = [
        "cage",
        "broad",
        "bridge",
        "minus_functional",
        "minus_structure_mechanism",
        "minus_relational_memory",
        "minus_family_domain",
    ]
    names = [
        *base_names,
        *[
            f"{name}_k{k}"
            for k in BUDGETS
            for name in base_names
        ],
        *[f"broad_cage_k{k}" for k in BUDGETS],
    ]
    summary = {
        "schema": "bridge-relation-unseen-e2r-cage-eval-v1",
        "status": "completed",
        "protocol": {
            "mother_benchmark": "bridge_relation_unseen_max_v3",
            "exact_pair_relation_unseen_relative_to_clean2023": True,
            "query_selection_uses_labels": False,
            "query_selection": (
                "pre-existing EnzymeCAGE protein feature availability; "
                "query must have at least one strict positive inside the "
                "prebuilt parser-compatible reaction domain"
            ),
            "candidate_reactions": int(len(candidate_ids)),
            "training_known_relations_filtered_per_query": True,
            "broad_cage_budgets": list(BUDGETS),
            "cage_checkpoint": "generic pretrain seed42 epoch_19",
        },
        "strict_positive_edges": int(sum(len(v) for v in positive.values())),
        "queries": int(len(qf)),
        "metrics": {name: summarize(qf, name) for name in names},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
