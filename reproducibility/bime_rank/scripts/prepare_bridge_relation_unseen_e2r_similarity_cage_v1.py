from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex

SCRIPTS = ROOT / "reproducibility/bime_rank/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_enzymecage_internal_reaction_features_v1 as builder
from prepare_enzymecage_full_outer_reaction_probe_v1 import valid_center_queries

TARGETS = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
TRAIN = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
GATE_DIR = ROOT / "results/bridge_relation_unseen_e2r_similarity_gate_v1"
BASE_CAGE = ROOT / "results/bridge_relation_unseen_e2r_cage_v1"
BASE_CONFIG = ROOT / "results/enzymecage_relation_unseen_reaction_assets_v1/probe.yaml"
CKPT = ROOT / "external_repos/EnzymeCAGE/checkpoints/pretrain/seed_42"
OUT = ROOT / "results/bridge_relation_unseen_e2r_similarity_cage_v1"
FEATURE = OUT / "feature/reaction"

# EnzymeCAGE's reaction-center loader raises an out-of-bounds index for this
# reaction even though the earlier center parser accepts it. Keep it in the
# candidate-budget audit but mark it neural-score unsupported.
RUNTIME_UNSUPPORTED_REACTIONS = {"RHEA:58128"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FEATURE.mkdir(parents=True, exist_ok=True)

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
    known = train.groupby("protein_id")["reaction_id"].agg(set).to_dict()
    positives = targets.groupby("protein_id")["reaction_id"].agg(set).to_dict()

    base_pairs = pd.read_csv(
        BASE_CAGE / "pairs.csv",
        usecols=["protein_id", "UniprotID", "sequence"],
        dtype=str,
    ).fillna("").drop_duplicates("protein_id")
    protein_info = {
        str(row.protein_id): (str(row.UniprotID), str(row.sequence))
        for row in base_pairs.itertuples(index=False)
    }
    query_ids = sorted(protein_info)

    gate = pd.read_csv(
        GATE_DIR / "candidates.csv.gz", dtype=str
    ).fillna("")
    gate = gate[gate.protein_id.isin(query_ids)].rename(
        columns={"candidate_reaction_id": "reaction_id"}
    )
    gate["gate_candidate"] = True

    qgate = pd.read_csv(
        GATE_DIR / "query_gate.csv", dtype={"protein_id": str}
    )
    budget = dict(
        zip(qgate.protein_id.astype(str), qgate.candidate_count.astype(int))
    )

    index = FibreCandidateIndex(device="cuda")
    broad_rows: list[dict[str, object]] = []
    batch_size = 64
    for start in range(0, len(query_ids), batch_size):
        batch = query_ids[start : start + batch_size]
        prows = torch.as_tensor(
            [index.protein_index[p] for p in batch],
            dtype=torch.long,
            device=index.device,
        )
        with torch.no_grad():
            scores = (
                index.protein_embeddings.index_select(0, prows)
                @ index.reaction_embeddings.T
            ).float()
        for j, pid in enumerate(batch):
            n = int(budget.get(pid, 0))
            if n <= 0:
                continue
            score = scores[j].clone()
            for rid in known.get(pid, set()):
                row = index.reaction_index.get(rid)
                if row is not None:
                    score[row] = -torch.inf
            k = min(n, int(torch.isfinite(score).sum().item()))
            vals, inds = torch.topk(score, k=k, largest=True, sorted=True)
            for rank, (row, value) in enumerate(
                zip(inds.cpu().tolist(), vals.cpu().tolist()), 1
            ):
                broad_rows.append(
                    {
                        "protein_id": pid,
                        "reaction_id": index.reaction_ids[int(row)],
                        "broad_rank": rank,
                        "broad_score": float(value),
                        "broad_candidate": True,
                    }
                )
    broad = pd.DataFrame(broad_rows)

    membership = (
        gate[["protein_id", "reaction_id", "gate_candidate"]]
        .merge(
            broad[
                [
                    "protein_id",
                    "reaction_id",
                    "broad_candidate",
                    "broad_rank",
                    "broad_score",
                ]
            ],
            on=["protein_id", "reaction_id"],
            how="outer",
        )
        .fillna(
            {
                "gate_candidate": False,
                "broad_candidate": False,
                "broad_rank": 0,
                "broad_score": float("-inf"),
            }
        )
    )
    membership["gate_candidate"] = membership["gate_candidate"].astype(bool)
    membership["broad_candidate"] = membership["broad_candidate"].astype(bool)
    membership["budget"] = membership["protein_id"].map(budget).astype(int)
    membership["label"] = [
        int(str(rid) in positives.get(str(pid), set()))
        for pid, rid in membership[["protein_id", "reaction_id"]].itertuples(
            index=False, name=None
        )
    ]
    membership.to_csv(OUT / "candidate_membership.csv.gz", index=False)

    union_reactions = sorted(set(membership.reaction_id.astype(str)))
    meta = pd.read_csv(REACTIONS, dtype=str).fillna("")
    frame = (
        meta[meta.reaction_id.astype(str).isin(union_reactions)][
            ["reaction_id", "reaction_smiles"]
        ]
        .drop_duplicates("reaction_id")
        .rename(columns={"reaction_smiles": "CANO_RXN_SMILES"})
        .sort_values("reaction_id", kind="stable")
        .reset_index(drop=True)
    )
    missing_meta = sorted(set(union_reactions) - set(frame.reaction_id.astype(str)))
    if missing_meta:
        raise RuntimeError(f"missing reaction metadata: {missing_meta[:10]}")

    builder.OUT = FEATURE
    (FEATURE / "drfp").mkdir(parents=True, exist_ok=True)
    (FEATURE / "reacting_center").mkdir(parents=True, exist_ok=True)

    drfp, drfp_fallbacks = builder.generate_drfp(frame)
    with (FEATURE / "drfp/rxn2fp.pkl").open("wb") as fh:
        pickle.dump(drfp, fh)

    centers, center_failures = builder.generate_centers(frame)
    valid_ids, invalid_centers = valid_center_queries(frame, centers)
    with (FEATURE / "reacting_center/reacting_center.pkl").open("wb") as fh:
        pickle.dump(centers, fh)

    molecules = builder.unique_molecules(frame)
    _, conformer_failures = builder.generate_conformations(molecules)
    failed_mols = {str(x["smiles"]) for x in conformer_failures}
    failed_conformer_queries: set[str] = set()
    if failed_mols:
        for rid, reaction in frame[
            ["reaction_id", "CANO_RXN_SMILES"]
        ].itertuples(index=False):
            left, right = str(reaction).split(">>")
            used = {
                p.replace("*", "C")
                for p in left.split(".") + right.split(".")
                if p
            }
            if used & failed_mols:
                failed_conformer_queries.add(str(rid))
    valid_ids -= failed_conformer_queries
    valid_ids -= RUNTIME_UNSUPPORTED_REACTIONS

    pd.DataFrame(drfp_fallbacks).to_csv(
        FEATURE / "drfp/fallbacks.csv", index=False
    )
    pd.DataFrame(center_failures).to_csv(
        FEATURE / "reacting_center/failures.csv", index=False
    )
    pd.DataFrame(conformer_failures).to_csv(
        FEATURE / "molecule_conformation/failures.csv", index=False
    )

    reaction_smiles = dict(
        zip(frame.reaction_id.astype(str), frame.CANO_RXN_SMILES.astype(str))
    )
    rows: list[dict[str, object]] = []
    for rec in membership.itertuples(index=False):
        pid = str(rec.protein_id)
        rid = str(rec.reaction_id)
        if rid not in valid_ids:
            continue
        uid, seq = protein_info[pid]
        rows.append(
            {
                "reaction_id": rid,
                "protein_id": pid,
                "UniprotID": uid,
                "sequence": seq,
                "CANO_RXN_SMILES": reaction_smiles[rid],
                "Label": int(rec.label),
            }
        )
    pairs = pd.DataFrame(rows)
    pairs.to_csv(OUT / "pairs.csv", index=False)

    config = yaml.safe_load(BASE_CONFIG.read_text())
    config["data_path"] = str((OUT / "pairs.csv").resolve())
    config["protein_gvp_feat"] = str(
        (BASE_CAGE / "protein_features/gvp_protein_feature.pt").resolve()
    )
    config["esm_node_feature"] = str(
        (BASE_CAGE / "protein_features/esm_node_feature.pt").resolve()
    )
    config["esm_mean_feature"] = str(
        (BASE_CAGE / "protein_features/seq2feature.pkl").resolve()
    )
    config["rxn_fp"] = str((FEATURE / "drfp/rxn2fp.pkl").resolve())
    config["mol_conformation"] = str(
        (FEATURE / "molecule_conformation").resolve()
    )
    config["reaction_center"] = str(
        (FEATURE / "reacting_center/reacting_center.pkl").resolve()
    )
    config["ckpt_dir"] = str(CKPT.resolve())
    config["model_list"] = ["epoch_19.pth"]
    config["result_dir"] = str((OUT / "cage_inference").resolve())
    config["batch_size"] = 256
    (OUT / "cage_infer.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )

    membership["cage_supported_reaction"] = membership.reaction_id.isin(valid_ids)
    membership.to_csv(OUT / "candidate_membership.csv.gz", index=False)

    summary = {
        "schema": "bridge-relation-unseen-e2r-similarity-cage-v1",
        "status": "prepared",
        "mother_edges": int(len(targets)),
        "evaluable_queries": int(len(query_ids)),
        "strict_positive_edges_in_evaluable_queries": int(
            targets[targets.protein_id.isin(query_ids)].shape[0]
        ),
        "candidate_membership_rows": int(len(membership)),
        "gate_candidate_rows": int(gate.shape[0]),
        "broad_candidate_rows": int(broad.shape[0]),
        "unique_candidate_reactions": int(len(union_reactions)),
        "cage_supported_candidate_reactions": int(len(valid_ids)),
        "pair_rows_for_cage": int(len(pairs)),
        "invalid_reaction_center_queries": sorted(invalid_centers),
        "failed_conformer_queries": sorted(failed_conformer_queries),
        "runtime_unsupported_reactions": sorted(RUNTIME_UNSUPPORTED_REACTIONS),
        "labels_used_for_candidate_generation": False,
        "exact_relation_unseen_relative_to_clean2023": True,
    }
    (OUT / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
