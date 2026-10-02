from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import zipfile
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "data/external/enzymecage_current/authors_drive_current/dataset.zip"
GVP = ROOT / "data/external/enzymecage_current/cage_official_features/gvp_feature/gvp_protein_feature.pt"
ESM_NODE = ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/pocket_node_feature/esm_node_feature.pt"
ESM_MEAN = ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/protein_level/seq2feature.pkl"
OUT = ROOT / "results/enzymecage_family_response_v1/calibration_author_valid"
MIN_CANDIDATES = 8
MAX_CANDIDATES = 64


def hkey(*parts: str) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def valid_structure_uids() -> set[str]:
    gvp = torch.load(GVP, map_location="cpu", weights_only=False)
    node = torch.load(ESM_NODE, map_location="cpu", weights_only=False)
    return {
        uid for uid in set(gvp) & set(node)
        if int(gvp[uid][0].shape[0]) == int(node[uid].shape[0])
    }


def selected_pairs() -> pd.DataFrame:
    valid = valid_structure_uids()
    with zipfile.ZipFile(ARCHIVE) as z, z.open("dataset/training/valid.csv") as f:
        frame = pd.read_csv(
            f,
            usecols=["RHEA_ID", "UniprotID", "CANO_RXN_SMILES", "sequence"],
            dtype=str,
        ).fillna("")
    frame = frame[frame["UniprotID"].isin(valid)].copy()
    frame = frame.drop_duplicates(["CANO_RXN_SMILES", "UniprotID"], keep="first")
    counts = frame.groupby("CANO_RXN_SMILES").size()
    keep_queries = set(counts[counts >= MIN_CANDIDATES].index)
    frame = frame[frame["CANO_RXN_SMILES"].isin(keep_queries)].copy()
    frame["_hash"] = [
        hkey(rxn, uid) for rxn, uid in frame[["CANO_RXN_SMILES", "UniprotID"]].itertuples(index=False)
    ]
    frame = (
        frame.sort_values(["CANO_RXN_SMILES", "_hash"], kind="stable")
        .groupby("CANO_RXN_SMILES", sort=False)
        .head(MAX_CANDIDATES)
        .drop(columns="_hash")
        .reset_index(drop=True)
    )
    frame["Label"] = 0.0
    return frame


def molecule_smiles(reactions: list[str]) -> set[str]:
    values = set()
    for rxn in reactions:
        left, right = rxn.split(">>")
        for smi in left.split(".") + right.split("."):
            values.add(smi.replace("*", "C"))
    return values


def extract_assets(frame: pd.DataFrame) -> dict[str, object]:
    feature = OUT / "feature"
    reaction = feature / "reaction"
    drfp_dir = reaction / "drfp"
    center_dir = reaction / "reacting_center"
    mol_dir = reaction / "molecule_conformation"
    for path in (drfp_dir, center_dir, mol_dir):
        path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ARCHIVE) as z:
        compact = {
            "dataset/RHEA/2025-02-05/feature/reaction/drfp/rxn2fp.pkl": drfp_dir / "rxn2fp.pkl",
            "dataset/RHEA/2025-02-05/feature/reaction/reacting_center/reacting_center.pkl": center_dir / "reacting_center.pkl",
            "dataset/RHEA/2025-02-05/feature/reaction/molecule_conformation/mol2id.csv": mol_dir / "mol2id.csv",
        }
        for member, dest in compact.items():
            with z.open(member) as src, dest.open("wb") as dst:
                dst.write(src.read())

        mol_index = pd.read_csv(mol_dir / "mol2id.csv", dtype={"SMILES": str})
        mapping = dict(zip(mol_index["SMILES"].astype(str), mol_index["ID"].astype(int)))
        wanted_smiles = molecule_smiles(frame["CANO_RXN_SMILES"].drop_duplicates().tolist())
        missing_smiles = sorted(wanted_smiles - set(mapping))
        wanted_ids = sorted({mapping[s] for s in wanted_smiles if s in mapping})
        # Trim the index to the selected calibration molecules.  The CAGE loader
        # eagerly scans every mol2id row, so retaining the full author index would
        # trigger thousands of irrelevant missing-SDF warnings.
        mol_index[mol_index["SMILES"].astype(str).isin(wanted_smiles)].to_csv(
            mol_dir / "mol2id.csv", index=False
        )
        cache = mol_dir / "mol_graph_dict.pt"
        if cache.exists():
            cache.unlink()
        missing_sdf = []
        for idx in wanted_ids:
            member = f"dataset/RHEA/2025-02-05/feature/reaction/molecule_conformation/{idx}.sdf"
            dest = mol_dir / f"{idx}.sdf"
            try:
                with z.open(member) as src, dest.open("wb") as dst:
                    dst.write(src.read())
            except KeyError:
                missing_sdf.append(idx)

    with open(drfp_dir / "rxn2fp.pkl", "rb") as f:
        drfp = pickle.load(f)
    with open(center_dir / "reacting_center.pkl", "rb") as f:
        center = pickle.load(f)
    queries = set(frame["CANO_RXN_SMILES"].astype(str))
    return {
        "wanted_molecules": len(wanted_smiles),
        "mapped_molecules": len(wanted_ids),
        "missing_molecule_smiles": len(missing_smiles),
        "missing_sdf": len(missing_sdf),
        "drfp_query_coverage": sum(q in drfp for q in queries),
        "reaction_center_query_coverage": sum(q in center for q in queries),
    }


def write_config(frame_path: Path) -> Path:
    config = {
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
        "data_path": str(frame_path.resolve()),
        "rxn_fp": str((OUT / "feature/reaction/drfp/rxn2fp.pkl").resolve()),
        "mol_conformation": str((OUT / "feature/reaction/molecule_conformation").resolve()),
        "reaction_center": str((OUT / "feature/reaction/reacting_center/reacting_center.pkl").resolve()),
        "protein_gvp_feat": str(GVP.resolve()),
        "esm_mean_feature": str(ESM_MEAN.resolve()),
        "esm_node_feature": str(ESM_NODE.resolve()),
    }
    path = OUT / "calibration.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = selected_pairs()
    data_path = OUT / "calibration_pairs.csv"
    frame.to_csv(data_path, index=False)
    assets = extract_assets(frame)
    config = write_config(data_path)
    counts = frame.groupby("CANO_RXN_SMILES").size()
    summary = {
        "schema": "enzymecage-family-response-calibration-v1",
        "status": "prepared",
        "source": "author dataset/training/valid.csv",
        "labels_read_for_selection_or_features": False,
        "selection": {
            "min_supported_candidates_per_query": MIN_CANDIDATES,
            "max_candidates_per_query": MAX_CANDIDATES,
            "candidate_selection": "SHA256(reaction|UniprotID), ascending",
        },
        "queries": int(frame["CANO_RXN_SMILES"].nunique()),
        "rows": int(len(frame)),
        "unique_uids": int(frame["UniprotID"].nunique()),
        "candidate_count_median": float(counts.median()),
        "candidate_count_min": int(counts.min()),
        "candidate_count_max": int(counts.max()),
        "assets": assets,
        "config": str(config.relative_to(ROOT)),
    }
    (OUT / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
