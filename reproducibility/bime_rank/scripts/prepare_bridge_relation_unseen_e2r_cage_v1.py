from __future__ import annotations

import json
import pickle as pkl
import re
from pathlib import Path

import pandas as pd
import torch
import yaml

from projects.active.bridge.model.assets import ROOT

TARGETS = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
META = ROOT / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv"
SEQUENCES = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
REACTION_ASSET = ROOT / "results/enzymecage_relation_unseen_reaction_assets_v1"
CKPT = ROOT / "external_repos/EnzymeCAGE/checkpoints/pretrain/seed_42"
OUT = ROOT / "results/bridge_relation_unseen_e2r_cage_v1"

SOURCES = [
    (
        ROOT / "data/external/enzymecage_current/cage_official_features/gvp_feature/gvp_protein_feature.pt",
        ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/pocket_node_feature/esm_node_feature.pt",
        ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/protein_level/seq2feature.pkl",
    ),
    (
        ROOT / "reports/zz_model_workflow_20260817/native_cage/data/feature/protein/gvp_feature/gvp_protein_feature.pt",
        ROOT / "reports/zz_model_workflow_20260817/native_cage/data/feature/protein/ESM-C_600M/pocket_node_feature/esm_node_feature.pt",
        ROOT / "reports/zz_model_workflow_20260817/native_cage/data/feature/protein/ESM-C_600M/protein_level/seq2feature.pkl",
    ),
    *[
        (
            ROOT / f"external_repos/EnzymeCAGE/dataset/external-test-set/{family}/feature/protein/gvp_feature/gvp_protein_feature.pt",
            ROOT / f"external_repos/EnzymeCAGE/dataset/external-test-set/{family}/feature/protein/ESM-C_600M/pocket_node_feature/esm_node_feature.pt",
            ROOT / f"external_repos/EnzymeCAGE/dataset/external-test-set/{family}/feature/protein/ESM-C_600M/protein_level/seq2feature.pkl",
        )
        for family in ("terpene", "p450", "phosphatase")
    ],
]


def alias_map(meta: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in meta[["protein_id", "canonical_accession", "aliases"]].itertuples(index=False):
        vals = [str(row.protein_id), str(row.canonical_accession)]
        vals.extend(x for x in re.split(r";+", str(row.aliases)) if x)
        out[str(row.protein_id)] = list(dict.fromkeys(x for x in vals if x))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feature_dir = OUT / "protein_features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(TARGETS, dtype=str).fillna("").drop_duplicates(
        ["protein_id", "reaction_id"]
    )
    seqf = pd.read_csv(SEQUENCES, sep="\t", dtype=str).fillna("")
    sequence = dict(zip(seqf.protein_id.astype(str), seqf.sequence.astype(str)))
    aliases = alias_map(pd.read_csv(META, dtype=str).fillna(""))

    query_ids = sorted(targets.protein_id.astype(str).unique())
    selected: dict[str, tuple[str, str]] = {}
    gvp_out: dict = {}
    node_out: dict = {}
    mean_out: dict = {}
    source_counts: list[int] = []

    remaining = set(query_ids)
    for gpath, npath, mpath in SOURCES:
        gvp = torch.load(gpath, map_location="cpu", weights_only=False)
        node = torch.load(npath, map_location="cpu", weights_only=False)
        mean = pkl.load(open(mpath, "rb"))
        used = 0
        for pid in sorted(list(remaining)):
            seq = sequence.get(pid, "")
            if not seq or seq not in mean:
                continue
            chosen = None
            for uid in aliases.get(pid, [pid]):
                if uid not in gvp or uid not in node:
                    continue
                if int(gvp[uid][0].shape[0]) != int(node[uid].shape[0]):
                    continue
                chosen = uid
                break
            if chosen is None:
                continue
            selected[pid] = (chosen, seq)
            gvp_out[chosen] = gvp[chosen]
            node_out[chosen] = node[chosen]
            mean_out[seq] = mean[seq]
            remaining.remove(pid)
            used += 1
        source_counts.append(used)

    probe = (
        pd.read_csv(REACTION_ASSET / "probe_pairs.csv", dtype=str)
        .fillna("")
        [["reaction_id", "CANO_RXN_SMILES"]]
        .drop_duplicates("reaction_id")
    )
    parser_excluded = {"RHEA:63388"}
    probe = probe[~probe.reaction_id.isin(parser_excluded)].copy()
    reaction_ids = set(probe.reaction_id.astype(str))
    supported_targets = targets[
        targets.protein_id.isin(selected) & targets.reaction_id.isin(reaction_ids)
    ].copy()
    positive_by_protein = (
        supported_targets.groupby("protein_id")["reaction_id"]
        .agg(lambda x: set(map(str, x)))
        .to_dict()
    )
    keep_proteins = sorted(positive_by_protein)

    gvp_keep = {selected[p][0]: gvp_out[selected[p][0]] for p in keep_proteins}
    node_keep = {selected[p][0]: node_out[selected[p][0]] for p in keep_proteins}
    mean_keep = {selected[p][1]: mean_out[selected[p][1]] for p in keep_proteins}
    torch.save(gvp_keep, feature_dir / "gvp_protein_feature.pt")
    torch.save(node_keep, feature_dir / "esm_node_feature.pt")
    with open(feature_dir / "seq2feature.pkl", "wb") as fh:
        pkl.dump(mean_keep, fh)

    rows: list[dict[str, object]] = []
    probe_rows = list(probe.itertuples(index=False))
    for pid in keep_proteins:
        uid, seq = selected[pid]
        positives = positive_by_protein[pid]
        for rec in probe_rows:
            rid = str(rec.reaction_id)
            rows.append(
                {
                    "reaction_id": rid,
                    "protein_id": pid,
                    "UniprotID": uid,
                    "sequence": seq,
                    "CANO_RXN_SMILES": str(rec.CANO_RXN_SMILES),
                    "Label": int(rid in positives),
                }
            )
    pairs = pd.DataFrame(rows)
    pairs.to_csv(OUT / "pairs.csv", index=False)

    config = yaml.safe_load((REACTION_ASSET / "probe.yaml").read_text())
    config["data_path"] = str((OUT / "pairs.csv").resolve())
    config["protein_gvp_feat"] = str((feature_dir / "gvp_protein_feature.pt").resolve())
    config["esm_node_feature"] = str((feature_dir / "esm_node_feature.pt").resolve())
    config["esm_mean_feature"] = str((feature_dir / "seq2feature.pkl").resolve())
    config["ckpt_dir"] = str(CKPT.resolve())
    config["model_list"] = ["epoch_19.pth"]
    config["result_dir"] = str((OUT / "cage_inference").resolve())
    config["batch_size"] = 256
    (OUT / "cage_infer.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    summary = {
        "schema": "bridge-relation-unseen-e2r-cage-v1",
        "status": "prepared",
        "mother_edges": int(len(targets)),
        "mother_e2r_queries": int(targets.protein_id.nunique()),
        "feature_ready_queries_before_reaction_filter": int(len(selected)),
        "evaluable_queries": int(len(keep_proteins)),
        "strict_positive_edges": int(len(supported_targets)),
        "candidate_reactions": int(len(probe)),
        "parser_excluded_reactions": sorted(parser_excluded),
        "pair_rows": int(len(pairs)),
        "source_cache_new_queries": source_counts,
        "exact_relation_unseen_relative_to_clean2023": True,
        "labels_used_for_feature_selection": False,
        "query_selection": "CAGE protein feature availability plus at least one strict positive in the prebuilt reaction-support domain",
    }
    (OUT / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
