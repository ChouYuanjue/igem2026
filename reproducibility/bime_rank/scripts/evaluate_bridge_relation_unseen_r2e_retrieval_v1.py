from __future__ import annotations

import json

import pandas as pd
import torch

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex

TARGETS = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
TRAIN = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
CAGE_GATE = ROOT / "results/bridge_relation_unseen_cage_gate_v1/gate_candidates.csv.gz"
CAGE_QUERY = ROOT / "results/bridge_relation_unseen_cage_gate_v1/query_gate.csv"
OUT = ROOT / "results/bridge_relation_unseen_r2e_retrieval_v1"


def summarize(qf: pd.DataFrame, prefix: str) -> dict[str, float]:
    return {
        "queries": int(len(qf)),
        "query_hit": float((qf[f"{prefix}_hits"] > 0).mean()),
        "macro_positive_recall": float(
            (
                qf[f"{prefix}_hits"]
                / qf["positive_count"].clip(lower=1)
            ).mean()
        ),
        "edge_recall": float(
            qf[f"{prefix}_hits"].sum()
            / qf["positive_count"].sum()
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(TARGETS, dtype=str).fillna("").drop_duplicates(
        ["protein_id", "reaction_id"]
    )
    train = pd.read_csv(TRAIN, dtype=str).fillna("").drop_duplicates(
        ["protein_id", "reaction_id"]
    )
    gate = pd.read_csv(CAGE_GATE, dtype=str).fillna("")
    gate_q = pd.read_csv(CAGE_QUERY, dtype=str).fillna("")
    gate_q["candidate_count"] = pd.to_numeric(
        gate_q["candidate_count"], errors="coerce"
    ).fillna(0).astype(int)

    positive = (
        targets.groupby("reaction_id")["protein_id"]
        .agg(lambda x: set(map(str, x)))
        .to_dict()
    )
    known = (
        train.groupby("reaction_id")["protein_id"]
        .agg(lambda x: set(map(str, x)))
        .to_dict()
    )
    budget = dict(
        zip(
            gate_q.reaction_id.astype(str),
            gate_q.candidate_count.astype(int),
        )
    )
    cage_uid = (
        gate.groupby("reaction_id")["candidate_uid"]
        .agg(lambda x: set(map(str, x)))
        .to_dict()
    )

    # Exact alias mapping lets author gate UIDs be evaluated against the
    # canonical general_merged protein IDs used by the benchmark.
    meta = pd.read_csv(
        ROOT / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv",
        dtype=str,
    ).fillna("")
    alias: dict[str, set[str]] = {}
    for row in meta[
        ["protein_id", "canonical_accession", "aliases"]
    ].itertuples(index=False):
        vals = {str(row.protein_id), str(row.canonical_accession)}
        vals |= {x for x in str(row.aliases).split(";") if x}
        alias[str(row.protein_id)] = {x for x in vals if x}

    index = FibreCandidateIndex(device="cuda")
    records: list[dict[str, object]] = []

    for qi, rid in enumerate(sorted(positive), 1):
        positives = positive[rid]
        cage_candidates = cage_uid.get(rid, set())
        cage_hits = sum(
            bool(alias.get(pid, {pid}) & cage_candidates)
            for pid in positives
        )

        k = int(budget.get(rid, 0))
        broad_hits = 0
        if k > 0 and rid in index.reaction_index:
            with torch.no_grad():
                score = (
                    index.reaction_embeddings[
                        index.reaction_index[rid]
                    ]
                    @ index.protein_embeddings.T
                ).float()
            for pid in known.get(rid, set()):
                row = index.protein_index.get(pid)
                if row is not None:
                    score[row] = -torch.inf
            k_eff = min(
                k, int(torch.isfinite(score).sum().item())
            )
            if k_eff > 0:
                _, inds = torch.topk(
                    score, k=k_eff, largest=True, sorted=False
                )
                broad_ids = {
                    index.protein_ids[int(i)]
                    for i in inds.cpu().tolist()
                }
                broad_hits = sum(
                    pid in broad_ids for pid in positives
                )

        records.append(
            {
                "reaction_id": rid,
                "positive_count": len(positives),
                "candidate_budget": k,
                "cage_hits": cage_hits,
                "broad_hits": broad_hits,
            }
        )
        if qi % 250 == 0 or qi == len(positive):
            print("strict-r2e-retrieval", qi, "/", len(positive), flush=True)

    qf = pd.DataFrame(records)
    qf.to_csv(OUT / "query_metrics.csv", index=False)
    summary = {
        "schema": "bridge-relation-unseen-r2e-retrieval-v1",
        "status": "completed",
        "protocol": {
            "mother_benchmark": "bridge_relation_unseen_max_v3",
            "edges": int(len(targets)),
            "queries": int(targets.reaction_id.nunique()),
            "exact_pair_relation_unseen_relative_to_clean2023": True,
            "enzymecage": (
                "published Top-10 similar-reaction gate over the frozen "
                "2023 association graph"
            ),
            "broad": (
                "full 185,918-protein Broad universe; per-query K equals "
                "the native EnzymeCAGE candidate count"
            ),
            "training_known_relations_filtered_from_broad": True,
        },
        "candidate_budget": {
            "mean": float(qf.candidate_budget.mean()),
            "median": float(qf.candidate_budget.median()),
            "sum": int(qf.candidate_budget.sum()),
        },
        "enzymecage": summarize(qf, "cage"),
        "broad_equal_budget": summarize(qf, "broad"),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
