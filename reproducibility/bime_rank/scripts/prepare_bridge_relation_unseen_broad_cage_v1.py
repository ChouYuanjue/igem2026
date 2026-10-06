from __future__ import annotations

import json
from pathlib import Path

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
REACTION_ASSET = ROOT / "results/enzymecage_relation_unseen_reaction_assets_v1"
CAGE_CKPT_DIR = ROOT / "external_repos/EnzymeCAGE/checkpoints/pretrain/seed_42"
OUT = ROOT / "results/bridge_relation_unseen_broad_cage_v1"
TOPK = 1000


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(TARGETS, dtype=str).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    train = pd.read_csv(TRAIN, dtype=str).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    known = train.groupby("reaction_id")["protein_id"].apply(lambda x: set(map(str, x))).to_dict()
    positives = targets.groupby("reaction_id")["protein_id"].apply(lambda x: set(map(str, x))).to_dict()

    meta = pd.read_csv(META, dtype=str).fillna("")
    seqf = pd.read_csv(SEQUENCES, sep="\t", dtype=str).fillna("")
    reactions = pd.read_csv(REACTIONS, dtype=str).fillna("")
    alias_map = aliases(meta)
    sequence_map = dict(zip(seqf.protein_id.astype(str), seqf.sequence.astype(str)))
    reaction_smiles = dict(zip(reactions.reaction_id.astype(str), reactions.reaction_smiles.astype(str)))
    cage_alias, support = cage_supported_aliases(alias_map, sequence_map)

    probe = pd.read_csv(REACTION_ASSET / "probe_pairs.csv", dtype=str).fillna("")
    reaction_supported = set(probe.reaction_id.astype(str))
    index = FibreCandidateIndex(device="cuda")

    rows = []
    qa = []
    for qi, rid in enumerate(sorted(positives), 1):
        if rid not in index.reaction_index:
            continue
        qrow = torch.as_tensor([index.reaction_index[rid]], dtype=torch.long, device=index.device)
        with torch.no_grad():
            score = (index.reaction_embeddings.index_select(0, qrow) @ index.protein_embeddings.T).flatten()
        masked = [index.protein_index[p] for p in known.get(rid, set()) if p in index.protein_index]
        if masked:
            score[torch.as_tensor(masked, dtype=torch.long, device=index.device)] = -torch.inf
        values, inds = torch.topk(score, k=TOPK, largest=True, sorted=True)
        ids = [index.protein_ids[int(x)] for x in inds.cpu().tolist()]
        vals = values.float().cpu().numpy()

        supported = 0
        supported_positive = 0
        # Exact preflight against the author mol_graph_dict found one
        # reaction whose product molecule graph is absent from the prepared
        # EnzymeCAGE conformer assets. All other reaction-supported queries,
        # including other metal-containing reactions, remain eligible.
        parser_compatible = rid != "RHEA:63388"
        if rid in reaction_supported and parser_compatible:
            for rank, (pid, bscore) in enumerate(zip(ids, vals), 1):
                uid = cage_alias.get(pid)
                if not uid:
                    continue
                label = int(pid in positives[rid])
                supported += 1
                supported_positive += label
                rows.append({
                    "reaction_id": rid,
                    "CANO_RXN_SMILES": reaction_smiles[rid],
                    "protein_id": pid,
                    "UniprotID": uid,
                    "sequence": sequence_map[pid],
                    "Label": label,
                    "broad_score": float(bscore),
                    "broad_rank": rank,
                })
        qa.append({
            "reaction_id": rid,
            "reaction_cage_supported": int(rid in reaction_supported),
            "cage_geometry_parser_compatible": int(parser_compatible),
            "cage_supported_top1000_count": supported,
            "cage_supported_positive_count": supported_positive,
        })
        if qi % 100 == 0 or qi == len(positives):
            print("prepare-broad-cage", qi, "/", len(positives), flush=True)

    pairs = pd.DataFrame(rows)
    audit = pd.DataFrame(qa)
    pairs.to_csv(OUT / "broad_top1000_cage_supported_pairs.csv", index=False)
    audit.to_csv(OUT / "query_audit.csv", index=False)

    config = yaml.safe_load((REACTION_ASSET / "probe.yaml").read_text())
    config["data_path"] = str((OUT / "broad_top1000_cage_supported_pairs.csv").resolve())
    config["batch_size"] = 256
    config["ckpt_dir"] = str(CAGE_CKPT_DIR.resolve())
    config["model_list"] = ["epoch_19.pth"]
    config["result_dir"] = str((OUT / "cage_inference").resolve())
    (OUT / "cage_infer.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    summary = {
        "schema": "bridge-relation-unseen-broad-cage-prep-v1",
        "status": "prepared",
        "queries": int(len(audit)),
        "reaction_supported_queries": int(audit.reaction_cage_supported.sum()),
        "pair_rows": int(len(pairs)),
        "unique_cage_uids": int(pairs.UniprotID.nunique() if len(pairs) else 0),
        "queries_with_supported_positive": int((audit.cage_supported_positive_count > 0).sum()),
        "mean_supported_candidates": float(audit.cage_supported_top1000_count.mean()),
        "protein_feature_support": support,
        "fusion_policy": "CAGE only permutes Broad slots occupied by CAGE-scoreable candidates; unsupported candidates stay at their Broad positions",
    }
    (OUT / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
