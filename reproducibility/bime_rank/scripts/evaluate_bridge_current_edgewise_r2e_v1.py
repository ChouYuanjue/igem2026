from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.runtime.final_system import FinalBridgeRuntime

TARGETS = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
TRAIN = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
OUT = ROOT / "results/bridge_current_edgewise_r2e_v1"


def filtered_ranks(raw: list[int]) -> list[int]:
    arr = np.asarray(raw, dtype=np.int64)
    return [int(v - np.count_nonzero(arr < v)) for v in arr]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(TARGETS, dtype=str).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    positives = targets.groupby("reaction_id")["protein_id"].apply(lambda x: list(map(str, x))).to_dict()
    train = pd.read_csv(TRAIN, dtype=str).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    known = train.groupby("reaction_id")["protein_id"].apply(lambda x: set(map(str, x))).to_dict()
    seen_p = set(train["protein_id"].astype(str))
    seen_r = set(train["reaction_id"].astype(str))

    rt = FinalBridgeRuntime(device=args.device)
    lex = rt._protein_lex
    rows = []
    qa = []

    for qi, q in enumerate(sorted(positives), start=1):
        if q not in rt.index.reaction_index:
            continue
        target_ids = [p for p in positives[q] if p in rt.index.protein_index]
        if not target_ids:
            continue

        qrow = rt.index.reaction_index[q]
        with torch.no_grad():
            broad_t = (rt.index.reaction_embeddings[qrow] @ rt.index.protein_embeddings.T).float()
        broad = broad_t.detach().cpu().numpy().astype(np.float64, copy=False)
        bstd = max(float(broad.std()), 1e-8)
        bz = (broad - float(broad.mean())) / bstd
        ctx, ca, cb, av, both = rt._relation_components("r2e", q)
        rw, rp, rs = rt._relation_authority("r2e", q, bz, ctx, ca, cb, av, both)
        retrieval = broad + bstd * rw * ctx

        broad_eval = broad.copy()
        retrieval_eval = retrieval.copy()
        masked = [rt.index.protein_index[p] for p in known.get(q, set()) if p in rt.index.protein_index]
        if masked:
            m = np.asarray(masked, dtype=np.int64)
            broad_eval[m] = -np.inf
            retrieval_eval[m] = -np.inf

        result = rt.rank_enzymes({
            "reaction_id": q,
            "top_k": 1000,
            "mask_clean2023": True,
        })
        top_map = {str(rec["candidate_id"]): int(rec["rank"]) for rec in result["candidates"]}

        def rank_score(score: np.ndarray, pid: str) -> int:
            row = rt.index.protein_index[pid]
            val = float(score[row])
            return int(
                np.count_nonzero(score > val)
                + np.count_nonzero((score == val) & (lex < lex[row]))
                + 1
            )

        broad_raw = [rank_score(broad_eval, p) for p in target_ids]
        final_raw = [
            top_map[p] if p in top_map else rank_score(retrieval_eval, p)
            for p in target_ids
        ]
        broad_f = filtered_ranks(broad_raw)
        final_f = filtered_ranks(final_raw)

        for p, br, fr in zip(target_ids, broad_f, final_f, strict=True):
            rows.append({
                "protein_id": p,
                "reaction_id": q,
                "broad_rank": br,
                "full_rank": fr,
                "protein_seen": p in seen_p,
                "reaction_seen": q in seen_r,
                "candidate_count_filtered": len(rt.index.protein_ids) - len(masked) - len(target_ids) + 1,
            })
        meta = result["query"]
        qa.append({
            "query_id": q,
            "target_relations": len(target_ids),
            "train_memory_weight": meta.get("train_memory_weight", 0.0),
            "functional_weight": meta.get("functional_weight", 0.0),
            "geometry_weight": meta.get("geometry_weight", 0.0),
            "family_active": meta.get("family_active", "none"),
            "family_weight": meta.get("family_weight", 0.0),
            "tps_active": meta.get("tps_active", 0),
            "tps_weight": meta.get("tps_weight", 0.0),
        })
        if qi % 50 == 0 or qi == len(positives):
            print("r2e", qi, "/", len(positives), flush=True)

    ef = pd.DataFrame(rows)
    qf = pd.DataFrame(qa)
    ef.to_csv(OUT / "edge_metrics.csv.gz", index=False)
    qf.to_csv(OUT / "query_audit.csv.gz", index=False)

    def met(df: pd.DataFrame, col: str) -> dict:
        r = df[col].to_numpy(np.int64)
        return {
            "edges": int(len(r)),
            "mrr": float((1.0 / r).mean()),
            "hit10": float((r <= 10).mean()),
            "hit100": float((r <= 100).mean()),
            "hit1000": float((r <= 1000).mean()),
            "median_rank": float(np.median(r)),
        }

    masks = {
        "both_seen_edge_unseen": ef.protein_seen & ef.reaction_seen,
        "protein_cold_only": (~ef.protein_seen) & ef.reaction_seen,
        "reaction_cold_only": ef.protein_seen & (~ef.reaction_seen),
        "double_cold": (~ef.protein_seen) & (~ef.reaction_seen),
    }
    summary = {
        "schema": "bridge-current-edgewise-r2e-v1",
        "protocol": {
            "unit": "one held-out relation edge",
            "current_runtime": rt.version,
            "episodic_support": "none",
            "long_term_memory": "enabled",
            "clean2023_known_relations_masked": True,
            "candidate_universe": len(rt.index.protein_ids),
            "edge_filtering": "all other held-out positives for the same query removed by rank correction",
        },
        "all": {"broad": met(ef, "broad_rank"), "full": met(ef, "full_rank")},
        "slices": {
            name: {"broad": met(ef[mask], "broad_rank"), "full": met(ef[mask], "full_rank")}
            for name, mask in masks.items()
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
