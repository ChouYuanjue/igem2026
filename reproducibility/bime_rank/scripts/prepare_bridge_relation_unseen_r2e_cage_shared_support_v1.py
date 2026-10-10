from __future__ import annotations

import json

import pandas as pd
import torch
import yaml

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex
from reproducibility.bime_rank.scripts.evaluate_fibre_cage_broad_shared_pool_v1 import (
    aliases,
    cage_supported_aliases,
)

TARGETS = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
TRAIN = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
META = ROOT / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv"
SEQUENCES = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
GATE = ROOT / "results/bridge_relation_unseen_cage_gate_v1/gate_candidates.csv.gz"
REACTION_ASSET = ROOT / "results/enzymecage_relation_unseen_reaction_assets_v1"
CKPT = ROOT / "external_repos/EnzymeCAGE/checkpoints/pretrain/seed_42"
OUT = ROOT / "results/bridge_relation_unseen_r2e_cage_shared_support_v1"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(TARGETS, dtype=str).fillna("").drop_duplicates(
        ["protein_id", "reaction_id"]
    )
    train = pd.read_csv(TRAIN, dtype=str).fillna("").drop_duplicates(
        ["protein_id", "reaction_id"]
    )
    meta = pd.read_csv(META, dtype=str).fillna("")
    seqf = pd.read_csv(SEQUENCES, sep="\t", dtype=str).fillna("")
    reactions = pd.read_csv(REACTIONS, dtype=str).fillna("")
    gate = pd.read_csv(GATE, dtype=str).fillna("")

    alias_map = aliases(meta)
    sequence = dict(zip(seqf.protein_id.astype(str), seqf.sequence.astype(str)))
    reaction_smiles = dict(
        zip(reactions.reaction_id.astype(str), reactions.reaction_smiles.astype(str))
    )
    cage_alias, support = cage_supported_aliases(alias_map, sequence)
    uid_to_pid = {uid: pid for pid, uid in cage_alias.items()}
    scoreable = sorted(cage_alias)
    scoreable_set = set(scoreable)

    strict_positive = (
        targets[targets.protein_id.isin(scoreable_set)]
        .groupby("reaction_id")["protein_id"]
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

    probe = (
        pd.read_csv(REACTION_ASSET / "probe_pairs.csv", dtype=str)
        .fillna("")
        [["reaction_id"]]
        .drop_duplicates()
    )
    parser_excluded = {"RHEA:63388"}
    reaction_supported = set(probe.reaction_id.astype(str)) - parser_excluded

    index = FibreCandidateIndex(device="cuda")
    scoreable_rows = torch.as_tensor(
        [index.protein_index[p] for p in scoreable],
        dtype=torch.long,
        device=index.device,
    )
    scoreable_emb = index.protein_embeddings.index_select(0, scoreable_rows)

    pair_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []

    query_ids = sorted(
        rid
        for rid, pos in strict_positive.items()
        if pos and rid in reaction_supported and rid in index.reaction_index
    )

    for qi, rid in enumerate(query_ids, 1):
        pos = strict_positive[rid]
        known_q = known.get(rid, set())

        native = []
        for uid in gate_by_query.get(rid, []):
            pid = uid_to_pid.get(uid)
            if pid and pid not in known_q:
                native.append(pid)
        native = list(dict.fromkeys(native))
        budget = len(native)
        if budget == 0:
            continue

        with torch.no_grad():
            score = (
                index.reaction_embeddings[index.reaction_index[rid]]
                @ scoreable_emb.T
            ).float()
        for i, pid in enumerate(scoreable):
            if pid in known_q:
                score[i] = -torch.inf
        k = min(budget, int(torch.isfinite(score).sum().item()))
        values, inds = torch.topk(score, k=k, largest=True, sorted=True)
        broad = [scoreable[int(i)] for i in inds.cpu().tolist()]
        broad_scores = values.cpu().numpy().tolist()

        native_set = set(native)
        broad_set = set(broad)
        union = list(dict.fromkeys(native + broad))
        broad_rank = {pid: rank for rank, pid in enumerate(broad, 1)}
        broad_score = {pid: float(v) for pid, v in zip(broad, broad_scores)}

        for pid in union:
            pair_rows.append(
                {
                    "reaction_id": rid,
                    "protein_id": pid,
                    "UniprotID": cage_alias[pid],
                    "sequence": sequence[pid],
                    "CANO_RXN_SMILES": reaction_smiles[rid],
                    "Label": int(pid in pos),
                }
            )
            membership_rows.append(
                {
                    "reaction_id": rid,
                    "protein_id": pid,
                    "native_pool": int(pid in native_set),
                    "broad_pool": int(pid in broad_set),
                    "broad_rank": broad_rank.get(pid, 0),
                    "broad_score": broad_score.get(pid, float("nan")),
                }
            )

        query_rows.append(
            {
                "reaction_id": rid,
                "scoreable_positive_count": len(pos),
                "native_budget": len(native),
                "broad_budget": len(broad),
                "native_positive_count": len(native_set & pos),
                "broad_positive_count": len(broad_set & pos),
            }
        )
        if qi % 50 == 0 or qi == len(query_ids):
            print("strict-r2e-prepare", qi, "/", len(query_ids), flush=True)

    pairs = pd.DataFrame(pair_rows).drop_duplicates(["reaction_id", "protein_id"])
    membership = pd.DataFrame(membership_rows).drop_duplicates(
        ["reaction_id", "protein_id"]
    )
    query = pd.DataFrame(query_rows)

    pairs.to_csv(OUT / "pairs.csv", index=False)
    membership.to_csv(OUT / "membership.csv.gz", index=False)
    query.to_csv(OUT / "query_audit.csv", index=False)

    config = yaml.safe_load((REACTION_ASSET / "probe.yaml").read_text())
    config["data_path"] = str((OUT / "pairs.csv").resolve())
    config["ckpt_dir"] = str(CKPT.resolve())
    config["model_list"] = ["epoch_19.pth"]
    config["result_dir"] = str((OUT / "cage_inference").resolve())
    config["batch_size"] = 256
    (OUT / "cage_infer.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    summary = {
        "schema": "bridge-relation-unseen-r2e-cage-shared-support-v1",
        "status": "prepared",
        "mother_edges": int(len(targets)),
        "mother_queries": int(targets.reaction_id.nunique()),
        "scoreable_protein_universe": int(len(scoreable)),
        "evaluable_queries": int(len(query)),
        "strict_positive_edges_in_scoreable_support": int(
            targets[
                targets.protein_id.isin(scoreable_set)
                & targets.reaction_id.isin(set(query.reaction_id.astype(str)))
            ].shape[0]
        ),
        "pair_rows": int(len(pairs)),
        "mean_budget": float(query.native_budget.mean()) if len(query) else 0.0,
        "median_budget": float(query.native_budget.median()) if len(query) else 0.0,
        "native_query_hit": float((query.native_positive_count > 0).mean())
        if len(query)
        else 0.0,
        "broad_query_hit": float((query.broad_positive_count > 0).mean())
        if len(query)
        else 0.0,
        "native_macro_positive_recall": float(
            (query.native_positive_count / query.scoreable_positive_count).mean()
        )
        if len(query)
        else 0.0,
        "broad_macro_positive_recall": float(
            (query.broad_positive_count / query.scoreable_positive_count).mean()
        )
        if len(query)
        else 0.0,
        "protein_feature_support": support,
        "selection_uses_model_scores": False,
        "note": (
            "Neural reranking control on the CAGE-scoreable protein support. "
            "Full-space retrieval is evaluated separately on all 1,923 strict queries."
        ),
    }
    (OUT / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
