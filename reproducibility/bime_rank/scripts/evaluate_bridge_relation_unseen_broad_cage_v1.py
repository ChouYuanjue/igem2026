from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex

TARGETS = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
TRAIN = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
INFER = ROOT / "results/bridge_relation_unseen_broad_cage_v1/cage_inference/broad_top1000_cage_supported_pairs_epoch_19.csv"
OUT = ROOT / "results/bridge_relation_unseen_broad_cage_v1"


def filtered_ranks(raw: list[int]) -> list[int]:
    arr = np.asarray(raw, dtype=np.int64)
    return [int(v - np.count_nonzero(arr < v)) for v in arr]


def rank_score(score: np.ndarray, row: int, lex: np.ndarray) -> int:
    value = float(score[row])
    return int(
        np.count_nonzero(score > value)
        + np.count_nonzero((score == value) & (lex < lex[row]))
        + 1
    )


def main() -> None:
    targets = (
        pd.read_csv(TARGETS, dtype=str)
        .fillna("")
        .drop_duplicates(["protein_id", "reaction_id"])
    )
    train = (
        pd.read_csv(TRAIN, dtype=str)
        .fillna("")
        .drop_duplicates(["protein_id", "reaction_id"])
    )
    known = (
        train.groupby("reaction_id")["protein_id"]
        .apply(lambda x: set(map(str, x)))
        .to_dict()
    )
    positives = (
        targets.groupby("reaction_id")["protein_id"]
        .apply(lambda x: list(map(str, x)))
        .to_dict()
    )

    infer = pd.read_csv(
        INFER, dtype={"protein_id": str, "reaction_id": str}
    ).fillna("")
    required = {"protein_id", "reaction_id", "broad_rank", "pred_logit"}
    if not required <= set(infer.columns):
        raise RuntimeError(
            f"missing inference columns: {sorted(required-set(infer.columns))}"
        )
    if infer.duplicated(["reaction_id", "protein_id"]).any():
        raise RuntimeError("duplicate Broad+CAGE inference pair")

    # Parameter-free slot-preserving local rerank:
    # only CAGE-scoreable candidates exchange the Broad slots that they
    # themselves occupied. Unsupported candidates never move.
    hybrid_slot: dict[tuple[str, str], int] = {}
    for rid, group in infer.groupby("reaction_id", sort=False):
        slots = np.sort(group.broad_rank.astype(int).to_numpy())
        ranked = group.sort_values(
            ["pred_logit", "protein_id"],
            ascending=[False, True],
            kind="stable",
        )
        for slot, pid in zip(slots, ranked.protein_id.astype(str), strict=True):
            hybrid_slot[(str(rid), str(pid))] = int(slot)

    index = FibreCandidateIndex(device="cuda")
    lex = np.empty(len(index.protein_ids), dtype=np.int64)
    order = np.argsort(np.asarray(index.protein_ids, dtype=object), kind="stable")
    lex[order] = np.arange(len(order))

    rows: list[dict[str, object]] = []
    for qi, rid in enumerate(sorted(positives), 1):
        if rid not in index.reaction_index:
            continue
        target_ids = [pid for pid in positives[rid] if pid in index.protein_index]
        if not target_ids:
            continue

        qrow = index.reaction_index[rid]
        with torch.no_grad():
            score_t = (
                index.reaction_embeddings[qrow] @ index.protein_embeddings.T
            ).float()
        score = score_t.cpu().numpy().astype(np.float64, copy=False)
        score = score.copy()

        masked = [
            index.protein_index[pid]
            for pid in known.get(rid, set())
            if pid in index.protein_index
        ]
        if masked:
            score[np.asarray(masked, dtype=np.int64)] = -np.inf

        raw_broad: list[int] = []
        raw_hybrid: list[int] = []
        cage_applied: list[int] = []
        for pid in target_ids:
            row = index.protein_index[pid]
            br = rank_score(score, row, lex)
            raw_broad.append(br)
            slot = hybrid_slot.get((str(rid), str(pid)))
            if slot is None:
                raw_hybrid.append(br)
                cage_applied.append(0)
            else:
                raw_hybrid.append(int(slot))
                cage_applied.append(1)

        broad_f = filtered_ranks(raw_broad)
        hybrid_f = filtered_ranks(raw_hybrid)
        for pid, br, hr, ca in zip(
            target_ids, broad_f, hybrid_f, cage_applied, strict=True
        ):
            rows.append(
                {
                    "protein_id": pid,
                    "reaction_id": rid,
                    "broad_rank": int(br),
                    "broad_cage_rank": int(hr),
                    "cage_score_applied_to_target": int(ca),
                    "candidate_count_filtered": (
                        len(index.protein_ids)
                        - len(masked)
                        - len(target_ids)
                        + 1
                    ),
                }
            )
        if qi % 100 == 0 or qi == len(positives):
            print("broad-cage-eval", qi, "/", len(positives), flush=True)

    out = pd.DataFrame(rows)
    if len(out) != len(targets):
        raise RuntimeError(
            f"edge count drift: expected {len(targets)}, got {len(out)}"
        )
    out.to_csv(OUT / "edge_metrics.csv.gz", index=False)

    def met(col: str) -> dict:
        r = out[col].to_numpy(np.int64)
        return {
            "mrr": float((1.0 / r).mean()),
            "hit10": float((r <= 10).mean()),
            "hit100": float((r <= 100).mean()),
            "hit1000": float((r <= 1000).mean()),
        }

    summary = {
        "schema": "bridge-relation-unseen-broad-cage-v1",
        "status": "completed",
        "edges": int(len(out)),
        "queries": int(out.reaction_id.nunique()),
        "target_edges_with_cage_score": int(
            out.cage_score_applied_to_target.sum()
        ),
        "broad": met("broad_rank"),
        "broad_plus_cage": met("broad_cage_rank"),
        "fusion_policy": (
            "parameter-free slot-preserving local rerank: EnzymeCAGE "
            "pred_logit only reorders the Broad rank slots occupied by "
            "candidates with complete CAGE evidence; all unsupported "
            "candidates retain their Broad slots"
        ),
        "capability_boundary": (
            "This is a conservative materialized-evidence hybrid, not an "
            "upper bound on EnzymeCAGE after hypothetical full structural "
            "feature materialization."
        ),
        "edge_filtering": (
            "raw full-candidate ranks are computed first, then all other "
            "held-out positives for the same query are removed by rank "
            "correction, matching the BRIDGE main-table protocol"
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
