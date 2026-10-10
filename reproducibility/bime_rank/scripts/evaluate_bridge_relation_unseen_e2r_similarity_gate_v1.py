from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from projects.active.bridge.model.assets import ROOT

TARGETS = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
TRAIN = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
ENTRIES = ROOT / "data/catalyst_candidate_universes/general_merged/proteins/entries.csv"
EMBEDDINGS = ROOT / "data/catalyst_candidate_universes/general_merged/proteins/embeddings.npy"
OUT = ROOT / "results/bridge_relation_unseen_e2r_similarity_gate_v1"
TOPK_NEIGHBORS = 10
BATCH = 64


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
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

    entries = (
        pd.read_csv(ENTRIES, dtype={"row": int, "Entry": str})
        .sort_values("row")
        .reset_index(drop=True)
    )
    protein_ids = entries["Entry"].astype(str).tolist()
    pidx = {p: i for i, p in enumerate(protein_ids)}
    emb = np.load(EMBEDDINGS, mmap_mode="r")
    if emb.shape != (len(protein_ids), 1152):
        raise RuntimeError(f"unexpected protein embedding shape {emb.shape}")

    train_proteins = sorted(set(train["protein_id"].astype(str)) & set(pidx))
    train_rows = np.asarray([pidx[p] for p in train_proteins], dtype=np.int64)
    train_reactions = (
        train.groupby("protein_id")["reaction_id"]
        .agg(lambda x: sorted(set(map(str, x))))
        .to_dict()
    )
    target_reactions = (
        targets.groupby("protein_id")["reaction_id"]
        .agg(lambda x: set(map(str, x)))
        .to_dict()
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_tensor = torch.as_tensor(
        np.asarray(emb[train_rows], dtype=np.float32),
        device=device,
    )
    train_tensor = F.normalize(train_tensor, dim=1)

    query_ids = sorted(target_reactions)
    missing_queries = [q for q in query_ids if q not in pidx]
    if missing_queries:
        raise RuntimeError(f"missing query proteins: {missing_queries[:10]}")

    candidate_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []

    for start in range(0, len(query_ids), BATCH):
        batch_ids = query_ids[start : start + BATCH]
        qrows = np.asarray([pidx[q] for q in batch_ids], dtype=np.int64)
        q = torch.as_tensor(
            np.asarray(emb[qrows], dtype=np.float32),
            device=device,
        )
        q = F.normalize(q, dim=1)
        with torch.no_grad():
            sim = q @ train_tensor.T
            vals, inds = torch.topk(
                sim,
                k=min(TOPK_NEIGHBORS, len(train_proteins)),
                dim=1,
                largest=True,
                sorted=True,
            )
        vals = vals.cpu().numpy()
        inds = inds.cpu().numpy()

        for local, pid in enumerate(batch_ids):
            candidate_reactions: set[str] = set()
            neighbor_records = []
            for rank, (j, score) in enumerate(
                zip(inds[local], vals[local], strict=True), 1
            ):
                neighbor = train_proteins[int(j)]
                rxns = train_reactions.get(neighbor, [])
                candidate_reactions.update(rxns)
                neighbor_records.append(
                    {
                        "protein_id": pid,
                        "neighbor_rank": rank,
                        "neighbor_protein_id": neighbor,
                        "cosine_similarity": float(score),
                        "neighbor_reaction_count": len(rxns),
                    }
                )
            # Discovery evaluation must not spend candidate budget on
            # relations already known for this exact query enzyme in clean2023.
            candidate_reactions -= set(train_reactions.get(pid, []))
            candidate_rows.extend(
                {
                    "protein_id": pid,
                    "candidate_reaction_id": rid,
                }
                for rid in sorted(candidate_reactions)
            )
            positives = target_reactions[pid]
            hits = positives & candidate_reactions
            query_rows.append(
                {
                    "protein_id": pid,
                    "candidate_count": len(candidate_reactions),
                    "positive_count": len(positives),
                    "positive_hit_count": len(hits),
                    "query_hit": bool(hits),
                    "positive_recall": (
                        len(hits) / len(positives) if positives else 0.0
                    ),
                    "nearest_similarity": float(vals[local, 0]),
                    "top10_mean_similarity": float(vals[local].mean()),
                }
            )
            for rid in sorted(positives):
                edge_rows.append(
                    {
                        "protein_id": pid,
                        "reaction_id": rid,
                        "gate_hit": rid in candidate_reactions,
                    }
                )
            if start % 1024 == 0 and local == 0:
                print("e2r gate", start, "/", len(query_ids), flush=True)

    cf = pd.DataFrame(candidate_rows).drop_duplicates()
    qf = pd.DataFrame(query_rows)
    ef = pd.DataFrame(edge_rows)
    cf.to_csv(OUT / "candidates.csv.gz", index=False)
    qf.to_csv(OUT / "query_gate.csv", index=False)
    ef.to_csv(OUT / "edge_gate_recall.csv.gz", index=False)

    candidate = qf["candidate_count"].to_numpy(np.int64)
    summary = {
        "schema": "bridge-relation-unseen-e2r-similarity-gate-v1",
        "status": "completed",
        "protocol": {
            "mother_edges": int(len(targets)),
            "queries": int(len(query_ids)),
            "train_graph": "clean2023",
            "similarity_space": (
                "raw 1152-d general_merged protein ESM-C embeddings; "
                "L2-normalized cosine"
            ),
            "topk_similar_training_enzymes": TOPK_NEIGHBORS,
            "candidate_definition": (
                "union of clean2023 reactions associated with the Top-10 "
                "most similar training enzymes, after masking reactions "
                "already known for the exact query enzyme"
            ),
            "analogy": (
                "direction-symmetric counterpart of EnzymeCAGE R2E "
                "Top-10 similar-reaction retrieval"
            ),
            "labels_used_for_gate": False,
            "exact_relation_unseen_relative_to_clean2023": True,
        },
        "candidate_budget": {
            "mean": float(candidate.mean()),
            "median": float(np.median(candidate)),
            "p90": float(np.quantile(candidate, 0.90)),
            "p95": float(np.quantile(candidate, 0.95)),
            "p99": float(np.quantile(candidate, 0.99)),
            "min": int(candidate.min()),
            "max": int(candidate.max()),
        },
        "query_hit": float(qf["query_hit"].mean()),
        "macro_positive_recall": float(qf["positive_recall"].mean()),
        "edge_recall": float(ef["gate_hit"].mean()),
        "candidate_rows": int(len(cf)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
