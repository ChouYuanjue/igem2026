from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex

TARGETS = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
TRAIN = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
R2E_GATE = ROOT / "results/bridge_relation_unseen_cage_gate_v1/gate_candidates.csv.gz"
R2E_GATE_Q = ROOT / "results/bridge_relation_unseen_cage_gate_v1/query_gate.csv"
E2R_GATE = ROOT / "results/bridge_relation_unseen_e2r_similarity_gate_v1/candidates.csv.gz"
E2R_GATE_Q = ROOT / "results/bridge_relation_unseen_e2r_similarity_gate_v1/query_gate.csv"
META = ROOT / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv"
OUT = ROOT / "results/bridge_layered_v4_candidates"


def alias_maps(meta: pd.DataFrame) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    pid_to_alias: dict[str, set[str]] = {}
    alias_to_pid: dict[str, set[str]] = {}
    for row in meta[["protein_id", "canonical_accession", "aliases"]].itertuples(index=False):
        pid = str(row.protein_id)
        vals = {pid, str(row.canonical_accession)}
        vals |= {x for x in str(row.aliases).split(";") if x}
        vals = {x for x in vals if x}
        pid_to_alias[pid] = vals
        for value in vals:
            alias_to_pid.setdefault(value, set()).add(pid)
    return pid_to_alias, alias_to_pid


def summarize_budget(frame: pd.DataFrame, column: str) -> dict[str, float | int]:
    x = frame[column].to_numpy(np.int64)
    return {
        "queries": int(len(frame)),
        "sum": int(x.sum()),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "min": int(x.min()),
        "max": int(x.max()),
    }


def build_r2e(targets: pd.DataFrame, train: pd.DataFrame, meta: pd.DataFrame) -> dict[str, object]:
    gate = pd.read_csv(R2E_GATE, dtype=str).fillna("")
    gate_q = pd.read_csv(R2E_GATE_Q, dtype={"reaction_id": str}).fillna("")
    pid_to_alias, alias_to_pid = alias_maps(meta)
    del pid_to_alias

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
    gate_by_query = (
        gate.groupby("reaction_id")["candidate_uid"]
        .agg(lambda x: list(dict.fromkeys(map(str, x))))
        .to_dict()
    )

    index = FibreCandidateIndex(device="cuda")
    native_rows: list[dict[str, object]] = []
    broad_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []

    query_ids = sorted(positive)
    for qi, rid in enumerate(query_ids, 1):
        known_q = known.get(rid, set())
        native_raw = gate_by_query.get(rid, [])
        native: list[str] = []
        for uid in native_raw:
            mapped = alias_to_pid.get(uid, set())
            if mapped & known_q:
                continue
            native.append(uid)
        native = list(dict.fromkeys(native))
        budget = len(native)

        for rank, uid in enumerate(native, 1):
            mapped = sorted(alias_to_pid.get(uid, set()))
            native_rows.append(
                {
                    "reaction_id": rid,
                    "candidate_uid": uid,
                    "candidate_protein_ids": ";".join(mapped),
                    "candidate_rank_in_gate": rank,
                    "label": int(any(pid in positive[rid] for pid in mapped)),
                }
            )

        broad: list[tuple[str, float]] = []
        if budget > 0 and rid in index.reaction_index:
            with torch.no_grad():
                score = (
                    index.reaction_embeddings[index.reaction_index[rid]]
                    @ index.protein_embeddings.T
                ).float()
            for pid in known_q:
                row = index.protein_index.get(pid)
                if row is not None:
                    score[row] = -torch.inf
            k = min(budget, int(torch.isfinite(score).sum().item()))
            if k > 0:
                values, inds = torch.topk(score, k=k, largest=True, sorted=True)
                broad = [
                    (index.protein_ids[int(row)], float(value))
                    for row, value in zip(inds.cpu().tolist(), values.cpu().tolist(), strict=True)
                ]
        for rank, (pid, value) in enumerate(broad, 1):
            broad_rows.append(
                {
                    "reaction_id": rid,
                    "protein_id": pid,
                    "broad_rank": rank,
                    "broad_score": value,
                    "label": int(pid in positive[rid]),
                }
            )

        native_hit = sum(int(row["label"]) for row in native_rows[-len(native):]) if native else 0
        broad_hit = sum(int(row["label"]) for row in broad_rows[-len(broad):]) if broad else 0
        query_rows.append(
            {
                "reaction_id": rid,
                "positive_count": len(positive[rid]),
                "raw_gate_budget": int(
                    gate_q.loc[gate_q.reaction_id.astype(str).eq(rid), "candidate_count"].astype(int).iloc[0]
                ) if (gate_q.reaction_id.astype(str) == rid).any() else len(native_raw),
                "effective_budget_post_clean2023_mask": budget,
                "native_positive_hits": native_hit,
                "broad_positive_hits": broad_hit,
            }
        )
        if qi % 250 == 0 or qi == len(query_ids):
            print("v4-r2e-candidates", qi, "/", len(query_ids), flush=True)

    nf = pd.DataFrame(native_rows)
    bf = pd.DataFrame(broad_rows)
    qf = pd.DataFrame(query_rows)
    nf.to_csv(OUT / "r2e_native_candidates.csv.gz", index=False)
    bf.to_csv(OUT / "r2e_broad_equal_budget_candidates.csv.gz", index=False)
    qf.to_csv(OUT / "r2e_query_budget.csv", index=False)

    return {
        "queries": int(len(qf)),
        "native_candidate_rows": int(len(nf)),
        "broad_candidate_rows": int(len(bf)),
        "native_unique_uids": int(nf.candidate_uid.nunique()) if len(nf) else 0,
        "broad_unique_proteins": int(bf.protein_id.nunique()) if len(bf) else 0,
        "raw_budget": summarize_budget(qf, "raw_gate_budget"),
        "effective_budget": summarize_budget(qf, "effective_budget_post_clean2023_mask"),
        "queries_budget_changed_by_mask": int(
            (
                qf.raw_gate_budget
                != qf.effective_budget_post_clean2023_mask
            ).sum()
        ),
        "removed_candidate_slots": int(
            (
                qf.raw_gate_budget
                - qf.effective_budget_post_clean2023_mask
            ).sum()
        ),
    }


def build_e2r(targets: pd.DataFrame, train: pd.DataFrame) -> dict[str, object]:
    gate = (
        pd.read_csv(E2R_GATE, dtype=str)
        .fillna("")
        .rename(columns={"candidate_reaction_id": "reaction_id"})
    )
    qgate = pd.read_csv(E2R_GATE_Q, dtype={"protein_id": str})
    positive = (
        targets.groupby("protein_id")["reaction_id"]
        .agg(lambda x: set(map(str, x)))
        .to_dict()
    )
    known = (
        train.groupby("protein_id")["reaction_id"]
        .agg(lambda x: set(map(str, x)))
        .to_dict()
    )
    gate_by_query = (
        gate.groupby("protein_id")["reaction_id"]
        .agg(lambda x: list(dict.fromkeys(map(str, x))))
        .to_dict()
    )

    index = FibreCandidateIndex(device="cuda")
    native_rows: list[dict[str, object]] = []
    broad_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []

    for qi, pid in enumerate(sorted(positive), 1):
        known_q = known.get(pid, set())
        native = [rid for rid in gate_by_query.get(pid, []) if rid not in known_q]
        native = list(dict.fromkeys(native))
        budget = len(native)
        for rank, rid in enumerate(native, 1):
            native_rows.append(
                {
                    "protein_id": pid,
                    "reaction_id": rid,
                    "candidate_rank_in_gate": rank,
                    "label": int(rid in positive[pid]),
                }
            )

        broad: list[tuple[str, float]] = []
        if budget > 0 and pid in index.protein_index:
            with torch.no_grad():
                score = (
                    index.protein_embeddings[index.protein_index[pid]]
                    @ index.reaction_embeddings.T
                ).float()
            for rid in known_q:
                row = index.reaction_index.get(rid)
                if row is not None:
                    score[row] = -torch.inf
            k = min(budget, int(torch.isfinite(score).sum().item()))
            if k > 0:
                values, inds = torch.topk(score, k=k, largest=True, sorted=True)
                broad = [
                    (index.reaction_ids[int(row)], float(value))
                    for row, value in zip(inds.cpu().tolist(), values.cpu().tolist(), strict=True)
                ]
        for rank, (rid, value) in enumerate(broad, 1):
            broad_rows.append(
                {
                    "protein_id": pid,
                    "reaction_id": rid,
                    "broad_rank": rank,
                    "broad_score": value,
                    "label": int(rid in positive[pid]),
                }
            )

        query_rows.append(
            {
                "protein_id": pid,
                "positive_count": len(positive[pid]),
                "effective_budget": budget,
                "native_positive_hits": sum(rid in positive[pid] for rid in native),
                "broad_positive_hits": sum(rid in positive[pid] for rid, _ in broad),
            }
        )
        if qi % 1000 == 0 or qi == len(positive):
            print("v4-e2r-candidates", qi, "/", len(positive), flush=True)

    nf = pd.DataFrame(native_rows)
    bf = pd.DataFrame(broad_rows)
    qf = pd.DataFrame(query_rows)
    nf.to_csv(OUT / "e2r_similarity_gate_candidates.csv.gz", index=False)
    bf.to_csv(OUT / "e2r_broad_equal_budget_candidates.csv.gz", index=False)
    qf.to_csv(OUT / "e2r_query_budget.csv", index=False)

    source_q = qgate.set_index("protein_id")["candidate_count"].astype(int).to_dict()
    mismatch = sum(int(source_q.get(str(row.protein_id), -1)) != int(row.effective_budget) for row in qf.itertuples())
    return {
        "queries": int(len(qf)),
        "native_candidate_rows": int(len(nf)),
        "broad_candidate_rows": int(len(bf)),
        "native_unique_reactions": int(nf.reaction_id.nunique()) if len(nf) else 0,
        "broad_unique_reactions": int(bf.reaction_id.nunique()) if len(bf) else 0,
        "effective_budget": summarize_budget(qf, "effective_budget"),
        "source_gate_budget_mismatches": int(mismatch),
    }


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
    meta = pd.read_csv(META, dtype=str).fillna("")

    train_pairs = set(map(tuple, train[["protein_id", "reaction_id"]].itertuples(index=False, name=None)))
    leaked = [
        pair
        for pair in targets[["protein_id", "reaction_id"]].itertuples(index=False, name=None)
        if pair in train_pairs
    ]
    if leaked:
        raise RuntimeError(f"strict mother benchmark leaks clean2023: {leaked[:5]}")

    result = {
        "schema": "bridge-layered-v4-candidates",
        "status": "completed",
        "mother_benchmark": {
            "edges": int(len(targets)),
            "r2e_queries": int(targets.reaction_id.nunique()),
            "e2r_queries": int(targets.protein_id.nunique()),
            "strict_relation_unseen_relative_to_clean2023": True,
        },
        "r2e": build_r2e(targets, train, meta),
        "e2r": build_e2r(targets, train),
        "rules": {
            "known_relation_mask_precedes_budget_definition": True,
            "only_cage_paths_share_budget": True,
            "broad_and_bridge_native_full_space_unchanged": True,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
