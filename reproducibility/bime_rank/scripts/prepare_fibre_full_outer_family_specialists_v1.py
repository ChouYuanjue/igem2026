from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex

PRED = ROOT / "results/fibre_family_applicability_router_v1/full_outer_predictions.csv"
REACTION_FAMILY = ROOT / "results/enzymecage_reaction_family_response_v1/full_outer/query_features.csv"
OUT = ROOT / "results/enzymecage_family_response_v1/full_outer_specialists"
FEATURE = ROOT / "results/enzymecage_reaction_family_response_v1/full_outer_assets/feature/reaction"
REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
META = ROOT / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv"
SEQUENCES = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
GVP = ROOT / "data/external/enzymecage_current/cage_official_features/gvp_feature/gvp_protein_feature.pt"
ESM_NODE = ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/pocket_node_feature/esm_node_feature.pt"
ESM_MEAN = ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/protein_level/seq2feature.pkl"
MAX_CANDIDATES = 32
THRESHOLD = 0.50


def cage_alias_map():
    meta = pd.read_csv(META, dtype=str).fillna("")
    seq = pd.read_csv(SEQUENCES, sep="	", dtype=str).fillna("")
    seqmap = dict(zip(seq.protein_id.astype(str), seq.sequence.astype(str)))
    gvp = torch.load(GVP, map_location="cpu", weights_only=False)
    node = torch.load(ESM_NODE, map_location="cpu", weights_only=False)
    mean = pickle.load(open(ESM_MEAN, "rb"))
    valid = {
        uid for uid in set(gvp) & set(node)
        if int(gvp[uid][0].shape[0]) == int(node[uid].shape[0])
    }
    mapping = {}
    for rec in meta.to_dict("records"):
        pid = str(rec["protein_id"])
        sequence = seqmap.get(pid, "")
        if not sequence or sequence not in mean:
            continue
        aliases = [pid, str(rec.get("canonical_accession", ""))]
        aliases.extend(str(rec.get("aliases", "")).split(";"))
        for alias in aliases:
            alias = alias.strip()
            if alias and alias in valid:
                mapping[pid] = alias
                break
    return mapping, seqmap


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(PRED, dtype={"reaction_id": str}).fillna("")
    rf = pd.read_csv(REACTION_FAMILY, dtype={"reaction_id": str}).fillna("")
    sel = pred.merge(
        rf[["reaction_id", "reaction_family_winner"]],
        on="reaction_id", how="inner",
    )
    sel["confidence"] = pd.to_numeric(sel["confidence"], errors="raise")
    sel = sel[
        sel["pred"].isin(["p450", "phosphatase", "terpene"])
        & sel["pred"].eq(sel["reaction_family_winner"])
        & (sel["confidence"] >= THRESHOLD)
    ].copy()
    selected = dict(zip(sel.reaction_id.astype(str), sel.pred.astype(str)))

    aliases, seqmap = cage_alias_map()
    rx = pd.read_csv(REACTIONS, dtype=str).fillna("")
    rxmap = dict(zip(rx.reaction_id.astype(str), rx.reaction_smiles.astype(str)))

    index = FibreCandidateIndex(device="cuda")
    rows = []
    support = []
    for q, fam in sorted(selected.items()):
        qr = index.reaction_index.get(q)
        if qr is None:
            continue
        with torch.no_grad():
            score = (
                index.reaction_embeddings[qr] @ index.protein_embeddings.T
            ).float()
            top = torch.topk(score, k=1000, largest=True, sorted=True).indices.cpu().numpy()
        picked = []
        for broad_rank, grow in enumerate(top, 1):
            pid = index.protein_ids[int(grow)]
            alias = aliases.get(pid)
            if alias:
                picked.append((pid, alias, broad_rank))
            if len(picked) >= MAX_CANDIDATES:
                break
        support.append(
            {
                "reaction_id": q,
                "predicted_family": fam,
                "confidence": float(sel.loc[sel.reaction_id.eq(q), "confidence"].iloc[0]),
                "supported_candidates": len(picked),
            }
        )
        if len(picked) < 8 or q not in rxmap:
            continue
        for pid, alias, broad_rank in picked:
            rows.append(
                {
                    "reaction_id": q,
                    "predicted_family": fam,
                    "protein_id": pid,
                    "UniprotID": alias,
                    "sequence": seqmap[pid],
                    "CANO_RXN_SMILES": rxmap[q],
                    "broad_rank_top1000": broad_rank,
                    "Label": 0.0,
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "pairs.csv", index=False)
    pd.DataFrame(support).to_csv(OUT / "support.csv", index=False)

    conf = {
        "model": "EnzymeCAGE",
        "interaction_method": "geo-enhanced-interaction",
        "rxn_inner_interaction": True,
        "pocket_inner_interaction": True,
        "use_prods_info": False,
        "use_structure": True,
        "use_drfp": True,
        "use_esm": True,
        "esm_model": "ESM-C_600M",
        "batch_size": 256,
        "data_path": str((OUT / "pairs.csv").resolve()),
        "rxn_fp": str((FEATURE / "drfp/rxn2fp.pkl").resolve()),
        "mol_conformation": str((FEATURE / "molecule_conformation").resolve()),
        "reaction_center": str((FEATURE / "reacting_center/reacting_center.pkl").resolve()),
        "protein_gvp_feat": str(GVP.resolve()),
        "esm_mean_feature": str(ESM_MEAN.resolve()),
        "esm_node_feature": str(ESM_NODE.resolve()),
    }
    (OUT / "probe.yaml").write_text(yaml.safe_dump(conf, sort_keys=False))
    result = {
        "schema": "fibre-full-outer-family-specialist-probe-v1",
        "status": "prepared",
        "applicability_threshold": THRESHOLD,
        "agreement_required": True,
        "selected_queries": int(len(selected)),
        "selected_by_family": sel["pred"].value_counts().to_dict(),
        "probe_queries": int(frame.reaction_id.nunique()),
        "rows": int(len(frame)),
        "max_candidates_per_query": MAX_CANDIDATES,
        "overall_test_labels_used_for_selection": False,
    }
    (OUT / "prepare_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
