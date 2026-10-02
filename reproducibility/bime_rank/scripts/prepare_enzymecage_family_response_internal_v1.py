from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "data/external/enzymecage_current/authors_drive_current/dataset.zip"
GVP = ROOT / "data/external/enzymecage_current/cage_official_features/gvp_feature/gvp_protein_feature.pt"
ESM_NODE = ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/pocket_node_feature/esm_node_feature.pt"
ESM_MEAN = ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/protein_level/seq2feature.pkl"
META = ROOT / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv"
SEQUENCES = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
PROTEIN_ENTRIES = ROOT / "data/catalyst_candidate_universes/general_merged/proteins/entries.csv"
REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
V4 = ROOT / "results/fibre_dynamic_router_v4/prepared"
OUT = ROOT / "results/enzymecage_family_response_v1/internal_folds"
MIN_CANDIDATES = 8
MAX_CANDIDATES = 32


def ordered_proteins() -> list[str]:
    frame = pd.read_csv(PROTEIN_ENTRIES, dtype=str).fillna("")
    if "row" in frame.columns:
        frame["row"] = pd.to_numeric(frame["row"]).astype(int)
        frame = frame.sort_values("row", kind="stable")
    column = "Entry" if "Entry" in frame.columns else "protein_id"
    return frame[column].astype(str).tolist()


def cage_alias_map() -> tuple[dict[str, str], dict[str, str]]:
    meta = pd.read_csv(META, dtype=str).fillna("")
    seq = pd.read_csv(SEQUENCES, sep="\t", dtype=str).fillna("")
    seqmap = dict(zip(seq["protein_id"].astype(str), seq["sequence"].astype(str)))
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


def build_fold_pairs(
    fold: int,
    pids: list[str],
    aliases: dict[str, str],
    seqmap: dict[str, str],
    reaction_smiles: dict[str, str],
):
    cache = np.load(V4 / f"fold{fold}/cache.npy", allow_pickle=True)
    rows = []
    utility = []
    query_support = []
    for record in cache:
        query_id = str(record["query_id"])
        candidate_rows = np.asarray(record["candidate_rows"], dtype=np.int64)
        core = np.asarray(record["core"], dtype=np.float64)
        positives = set(map(int, record["positive_rows"]))
        selected = []
        for local_rank, idx in enumerate(candidate_rows, 1):
            pid = pids[int(idx)]
            alias = aliases.get(pid)
            if alias:
                selected.append(
                    (pid, alias, int(idx), int(local_rank), float(core[local_rank - 1]))
                )
            if len(selected) >= MAX_CANDIDATES:
                break
        query_support.append((query_id, len(selected)))
        if len(selected) < MIN_CANDIDATES or query_id not in reaction_smiles:
            continue
        smiles = reaction_smiles[query_id]
        for pid, alias, global_row, broad_rank, broad_score in selected:
            rows.append(
                {
                    "reaction_id": query_id,
                    "protein_id": pid,
                    "UniprotID": alias,
                    "sequence": seqmap[pid],
                    "CANO_RXN_SMILES": smiles,
                    "broad_score": broad_score,
                    "broad_rank_top1000": broad_rank,
                    # EnzymeCAGE extraction never sees benchmark labels.
                    "Label": 0.0,
                }
            )
            utility.append(
                {
                    "reaction_id": query_id,
                    "protein_id": pid,
                    "UniprotID": alias,
                    "broad_score": broad_score,
                    "broad_rank_top1000": broad_rank,
                    "is_positive": int(global_row in positives),
                }
            )
    support = pd.DataFrame(
        query_support, columns=["reaction_id", "supported_candidates"]
    )
    return pd.DataFrame(rows), support, pd.DataFrame(utility)


def write_config(tag: str, data_path: Path) -> Path:
    feature = OUT / "feature"
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
        "data_path": str(data_path.resolve()),
        "rxn_fp": str((feature / "reaction/drfp/rxn2fp.pkl").resolve()),
        "mol_conformation": str((feature / "reaction/molecule_conformation").resolve()),
        "reaction_center": str((feature / "reaction/reacting_center/reacting_center.pkl").resolve()),
        "protein_gvp_feat": str(GVP.resolve()),
        "esm_mean_feature": str(ESM_MEAN.resolve()),
        "esm_node_feature": str(ESM_NODE.resolve()),
    }
    path = OUT / f"{tag}/probe.yaml"
    path.write_text(yaml.safe_dump(conf, sort_keys=False))
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pids = ordered_proteins()
    aliases, seqmap = cage_alias_map()
    reactions = pd.read_csv(REACTIONS, dtype=str).fillna("")
    reaction_smiles = dict(zip(reactions["reaction_id"], reactions["reaction_smiles"]))

    fold_summary = {}
    all_queries = set()
    all_frames = []
    all_utility = []
    for fold in range(3):
        frame, support, utility = build_fold_pairs(
            fold, pids, aliases, seqmap, reaction_smiles
        )
        fold_dir = OUT / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        data_path = fold_dir / "pairs.csv"
        frame.to_csv(data_path, index=False)
        support.to_csv(fold_dir / "support.csv", index=False)
        utility.to_csv(fold_dir / "utility_labels.csv", index=False)
        with_fold = frame.copy()
        with_fold["fold"] = fold
        all_frames.append(with_fold)
        utility_with_fold = utility.copy()
        utility_with_fold["fold"] = fold
        all_utility.append(utility_with_fold)
        all_queries.update(frame["CANO_RXN_SMILES"].unique().tolist())
        fold_summary[str(fold)] = {
            "queries_total": int(len(support)),
            "queries_probe_available": int(frame["reaction_id"].nunique()),
            "rows": int(len(frame)),
            "support_mean": float(support["supported_candidates"].mean()),
            "support_median": float(support["supported_candidates"].median()),
            "queries_ge8": int((support["supported_candidates"] >= 8).sum()),
            "queries_ge32": int((support["supported_candidates"] >= 32).sum()),
        }

    all_dir = OUT / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(all_frames, ignore_index=True)
    if combined["reaction_id"].duplicated().any():
        # Each query has multiple candidate rows; this only checks cross-fold ownership.
        owners = combined[["reaction_id", "fold"]].drop_duplicates()
        if owners["reaction_id"].duplicated().any():
            raise RuntimeError("internal fold reaction overlap")
    combined.to_csv(all_dir / "pairs.csv", index=False)
    pd.concat(all_utility, ignore_index=True).to_csv(
        all_dir / "utility_labels.csv", index=False
    )

    configs = {}
    for fold in range(3):
        configs[str(fold)] = str(
            write_config(f"fold{fold}", OUT / f"fold{fold}/pairs.csv").relative_to(ROOT)
        )
    configs["all"] = str(
        write_config("all", all_dir / "pairs.csv").relative_to(ROOT)
    )
    result = {
        "schema": "enzymecage-family-response-internal-folds-v1",
        "status": "prepared",
        "selection": {
            "source_shortlist": "exact frozen V4 Broad Top-1000 cache",
            "min_cage_supported_candidates": MIN_CANDIDATES,
            "max_probe_candidates": MAX_CANDIDATES,
            "labels_used_for_candidate_selection": False,
        },
        "cage_supported_general_proteins": len(aliases),
        "folds": fold_summary,
        "combined": {
            "queries": int(combined["reaction_id"].nunique()),
            "rows": int(len(combined)),
        },
        "reaction_assets": {
            "status": "built_separately",
            "builder": "reproducibility/bime_rank/scripts/build_enzymecage_internal_reaction_features_v1.py",
            "reason": "general-merged reactions are newer than the author 2025 reaction-feature archive",
        },
        "configs": configs,
    }
    (OUT / "prepare_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
