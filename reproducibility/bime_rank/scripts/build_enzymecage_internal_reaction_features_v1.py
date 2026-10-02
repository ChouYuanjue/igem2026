from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from drfp import DrfpEncoder
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
CAGE = ROOT / "external_repos/EnzymeCAGE"
FEATURE_DIR = CAGE / "feature"
if str(CAGE) not in sys.path:
    sys.path.insert(0, str(CAGE))
if str(FEATURE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_DIR))

from extract_reacting_center import extract_reacting_center
from utils import cano_rxn

PROBE_ROOT = ROOT / "results/enzymecage_family_response_v1/internal_folds"
MAPPED = ROOT / "data/external/rxnmapper_current/general_merged_v1/mapped_reactions.csv"
OUT = PROBE_ROOT / "feature/reaction"


def legacy_hash(shingling):
    values = [
        int(hashlib.blake2b(token, digest_size=4).hexdigest(), 16)
        for token in shingling
    ]
    return np.asarray(values, dtype=np.uint32).view(np.int32)


def stable_seed(smiles: str) -> int:
    value = int.from_bytes(
        hashlib.sha256(smiles.encode()).digest()[:4], "little", signed=False
    )
    # RDKit expects a signed 32-bit-compatible positive integer.
    return int(value % 2147483647) or 1


def generate_conformer(item: tuple[int, str]) -> tuple[int, str, bool, str]:
    idx, smiles = item
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return idx, smiles, False, "MolFromSmiles"
        ps = AllChem.ETKDGv2()
        ps.randomSeed = stable_seed(smiles)
        rid = -1
        for _ in range(3):
            rid = AllChem.EmbedMolecule(mol, ps)
            if rid == 0:
                break
        if rid == -1:
            ps.useRandomCoords = True
            ps.randomSeed = stable_seed(smiles)
            rid = AllChem.EmbedMolecule(mol, ps)
            if rid == -1:
                AllChem.Compute2DCoords(mol)
            else:
                try:
                    AllChem.MMFFOptimizeMolecule(mol, confId=0)
                except Exception:
                    pass
        else:
            try:
                AllChem.MMFFOptimizeMolecule(mol, confId=0)
            except Exception:
                pass
        return idx, smiles, True, Chem.MolToMolBlock(mol)
    except Exception as exc:
        return idx, smiles, False, f"{type(exc).__name__}:{exc}"


def load_queries() -> pd.DataFrame:
    frames = []
    for fold in range(3):
        path = PROBE_ROOT / f"fold{fold}/pairs.csv"
        frame = pd.read_csv(
            path,
            usecols=["reaction_id", "CANO_RXN_SMILES"],
            dtype=str,
        ).fillna("")
        frames.append(frame.drop_duplicates("reaction_id"))
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates("reaction_id", keep="first")
    if combined["reaction_id"].duplicated().any():
        raise RuntimeError("duplicate reaction_id")
    return combined.sort_values("reaction_id", kind="stable").reset_index(drop=True)


def generate_drfp(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    DrfpEncoder.hash = staticmethod(legacy_hash)
    result = {}
    fallbacks = []
    for reaction_id, reaction in tqdm(
        frame[["reaction_id", "CANO_RXN_SMILES"]].itertuples(index=False),
        desc="DRFP",
        total=len(frame),
    ):
        try:
            encoded = DrfpEncoder.encode([reaction])[0]
        except Exception as exc:
            canonical = cano_rxn(reaction, remove_stereo=True)
            try:
                encoded = DrfpEncoder.encode([canonical])[0]
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"DRFP failed for {reaction_id}: original={type(exc).__name__}; "
                    f"canonical={type(fallback_exc).__name__}"
                ) from fallback_exc
            fallbacks.append(
                {
                    "reaction_id": str(reaction_id),
                    "original_reaction": str(reaction),
                    "fallback_reaction": str(canonical),
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
        result[reaction] = np.asarray(encoded, dtype=np.float32)
    return result, fallbacks


def generate_centers(frame: pd.DataFrame) -> tuple[dict[str, object], list[dict[str, str]]]:
    mapped = pd.read_csv(MAPPED, dtype=str).fillna("")
    mapping = mapped.set_index("reaction_id", drop=False)
    centers = {}
    failures = []
    for reaction_id, reaction in tqdm(
        frame[["reaction_id", "CANO_RXN_SMILES"]].itertuples(index=False),
        total=len(frame),
        desc="reaction-center",
    ):
        if reaction_id not in mapping.index:
            centers[reaction] = [[], []]
            failures.append(
                {"reaction_id": reaction_id, "reason": "missing_rxnmapper"}
            )
            continue
        row = mapping.loc[reaction_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        mapped_rxn = str(row.get("mapped_rxn", "")).strip()
        success = str(row.get("success", "")).lower() == "true"
        if not mapped_rxn or not success:
            centers[reaction] = [[], []]
            failures.append(
                {"reaction_id": reaction_id, "reason": "rxnmapper_unsuccessful"}
            )
            continue
        try:
            centers[reaction] = extract_reacting_center(
                reaction, {reaction: mapped_rxn}
            )
        except Exception as exc:
            centers[reaction] = [[], []]
            failures.append(
                {
                    "reaction_id": reaction_id,
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
    return centers, failures


def unique_molecules(frame: pd.DataFrame) -> list[str]:
    molecules = set()
    for reaction in frame["CANO_RXN_SMILES"].astype(str):
        left, right = reaction.split(">>")
        molecules.update(
            part.replace("*", "C")
            for part in left.split(".") + right.split(".")
            if part
        )
    return sorted(molecules)


def generate_conformations(molecules: list[str], workers: int = 12):
    mol_dir = OUT / "molecule_conformation"
    mol_dir.mkdir(parents=True, exist_ok=True)
    # This is a complete deterministic rebuild.  Remove stale SDFs left by any
    # earlier partial/legacy asset preparation before assigning compact IDs.
    for stale in mol_dir.glob("*.sdf"):
        stale.unlink()
    cache = mol_dir / "mol_graph_dict.pt"
    if cache.exists():
        cache.unlink()
    index = pd.DataFrame({"SMILES": molecules, "ID": range(len(molecules))})
    index.to_csv(mol_dir / "mol2id.csv", index=False)
    failures = []
    items = list(zip(index["ID"].astype(int), index["SMILES"].astype(str)))
    with Pool(processes=workers) as pool:
        iterator = pool.imap_unordered(generate_conformer, items, chunksize=8)
        for idx, smiles, ok, payload in tqdm(
            iterator, total=len(items), desc="conformer"
        ):
            if ok:
                (mol_dir / f"{idx}.sdf").write_text(payload + "\n$$$$\n")
            else:
                failures.append(
                    {"id": int(idx), "smiles": smiles, "reason": payload}
                )
    return index, failures



def _filter_probe_query_ids(tag: str, dropped_query_ids: set[str]) -> None:
    """Apply optional-probe query removal to pairs and train/val label sidecar."""
    for filename in ("pairs.csv", "utility_labels.csv"):
        path = PROBE_ROOT / tag / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype=str).fillna("")
        keep = ~frame["reaction_id"].astype(str).isin(dropped_query_ids)
        frame.loc[keep].to_csv(path, index=False)


def strict_filter_invalid_reaction_centers(
    frame: pd.DataFrame,
    centers: dict[str, object],
) -> dict[str, object]:
    """Drop probe queries whose reaction-center indices cannot index CAGE graphs.

    Broad/FIBRE benchmarks remain untouched.  This filter only marks the optional
    EnzymeCAGE family-response probe unavailable for malformed/new reaction assets.
    """
    invalid_by_smiles: dict[str, str] = {}
    for reaction, payload in centers.items():
        try:
            left_idx, right_idx = payload
            left, right = str(reaction).split(">>")
            def side_atoms(side: str) -> int:
                total = 0
                for part in side.split("."):
                    if not part:
                        continue
                    mol = Chem.MolFromSmiles(part.replace("*", "C"))
                    if mol is None:
                        raise ValueError(f"MolFromSmiles:{part}")
                    total += int(mol.GetNumAtoms())
                return total
            n_left = side_atoms(left)
            n_right = side_atoms(right)
            bad_left = [int(i) for i in left_idx if int(i) < 0 or int(i) >= n_left]
            bad_right = [int(i) for i in right_idx if int(i) < 0 or int(i) >= n_right]
            if bad_left or bad_right:
                invalid_by_smiles[str(reaction)] = (
                    f"center_index_out_of_bounds:left={bad_left},right={bad_right},"
                    f"nodes=({n_left},{n_right})"
                )
        except Exception as exc:
            invalid_by_smiles[str(reaction)] = f"validation_error:{type(exc).__name__}:{exc}"

    if not invalid_by_smiles:
        return {"dropped_query_ids": [], "dropped_by_split": {}, "reasons": {}}

    smiles_to_query = dict(
        zip(frame["CANO_RXN_SMILES"].astype(str), frame["reaction_id"].astype(str))
    )
    invalid_queries = {
        smiles_to_query[s] for s in invalid_by_smiles if s in smiles_to_query
    }
    dropped_by_split = {}
    for tag in ("fold0", "fold1", "fold2", "all"):
        path = PROBE_ROOT / tag / "pairs.csv"
        if not path.exists():
            continue
        probe = pd.read_csv(path, dtype=str).fillna("")
        bad = probe["reaction_id"].astype(str).isin(invalid_queries)
        dropped = sorted(probe.loc[bad, "reaction_id"].astype(str).unique().tolist())
        _filter_probe_query_ids(tag, set(dropped))
        dropped_by_split[tag] = dropped

    return {
        "dropped_query_ids": sorted(invalid_queries),
        "dropped_by_split": dropped_by_split,
        "reasons": {
            smiles_to_query[s]: reason
            for s, reason in invalid_by_smiles.items()
            if s in smiles_to_query
        },
    }


def strict_filter_failed_conformers(failures: list[dict[str, object]]) -> dict[str, object]:
    failed_smiles = {str(row["smiles"]) for row in failures}
    if not failed_smiles:
        return {"dropped_query_ids": [], "dropped_by_split": {}}

    def reaction_uses_failed(reaction: str) -> bool:
        left, right = str(reaction).split(">>")
        return any(
            molecule.replace("*", "C") in failed_smiles
            for molecule in left.split(".") + right.split(".")
            if molecule
        )

    dropped_all = []
    dropped_by_split = {}
    for tag in ("fold0", "fold1", "fold2", "all"):
        path = PROBE_ROOT / tag / "pairs.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype=str).fillna("")
        bad = frame["CANO_RXN_SMILES"].map(reaction_uses_failed)
        dropped = sorted(frame.loc[bad, "reaction_id"].astype(str).unique().tolist())
        _filter_probe_query_ids(tag, set(dropped))
        dropped_by_split[tag] = dropped
        dropped_all.extend(dropped)

    mol_index_path = OUT / "molecule_conformation/mol2id.csv"
    mol_index = pd.read_csv(mol_index_path, dtype={"SMILES": str})
    mol_index = mol_index[~mol_index["SMILES"].astype(str).isin(failed_smiles)].copy()
    mol_index.to_csv(mol_index_path, index=False)
    return {
        "dropped_query_ids": sorted(set(dropped_all)),
        "dropped_by_split": dropped_by_split,
    }

def update_configs() -> None:
    import yaml

    for fold in range(3):
        path = PROBE_ROOT / f"fold{fold}/probe.yaml"
        conf = yaml.safe_load(path.read_text())
        conf["rxn_fp"] = str((OUT / "drfp/rxn2fp.pkl").resolve())
        conf["mol_conformation"] = str((OUT / "molecule_conformation").resolve())
        conf["reaction_center"] = str(
            (OUT / "reacting_center/reacting_center.pkl").resolve()
        )
        path.write_text(yaml.safe_dump(conf, sort_keys=False))


def main() -> None:
    frame = load_queries()
    (OUT / "drfp").mkdir(parents=True, exist_ok=True)
    (OUT / "reacting_center").mkdir(parents=True, exist_ok=True)

    drfp, drfp_fallbacks = generate_drfp(frame)
    with (OUT / "drfp/rxn2fp.pkl").open("wb") as f:
        pickle.dump(drfp, f)
    pd.DataFrame(drfp_fallbacks).to_csv(OUT / "drfp/fallbacks.csv", index=False)

    centers, center_failures = generate_centers(frame)
    center_strict_filter = strict_filter_invalid_reaction_centers(frame, centers)
    with (OUT / "reacting_center/reacting_center.pkl").open("wb") as f:
        pickle.dump(centers, f)
    pd.DataFrame(center_failures).to_csv(
        OUT / "reacting_center/failures.csv", index=False
    )

    molecules = unique_molecules(frame)
    index, conformer_failures = generate_conformations(molecules)
    pd.DataFrame(conformer_failures).to_csv(
        OUT / "molecule_conformation/failures.csv", index=False
    )
    strict_filter = strict_filter_failed_conformers(conformer_failures)
    cache = OUT / "molecule_conformation/mol_graph_dict.pt"
    if cache.exists():
        cache.unlink()

    update_configs()

    nonempty_centers = sum(
        bool(left or right) for left, right in centers.values()
    )
    result = {
        "schema": "enzymecage-internal-reaction-features-v1",
        "status": "completed",
        "probe_query_ids": int(len(frame)),
        "unique_reaction_smiles": int(frame["CANO_RXN_SMILES"].nunique()),
        "drfp_coverage": int(len(drfp)),
        "drfp_fallbacks": int(len(drfp_fallbacks)),
        "drfp_semantics": (
            "DRFP 0.3.6 legacy NumPy-1.x uint32-to-int32 wraparound; "
            "same compatibility used in prior EnzymeCAGE reproduction"
        ),
        "rxnmapper_source": str(MAPPED.relative_to(ROOT)),
        "reaction_center_coverage": int(len(centers)),
        "reaction_center_nonempty": int(nonempty_centers),
        "reaction_center_failures": int(len(center_failures)),
        "molecules": int(len(index)),
        "conformer_failures": int(len(conformer_failures)),
        "strict_probe_filter": strict_filter,
        "strict_reaction_center_filter": center_strict_filter,
        "conformer_method": (
            "EnzymeCAGE ETKDGv2/MMFF flow with deterministic SHA256-derived "
            "RDKit random seed per molecule; maximum 3 deterministic ETKDG attempts "
            "then author-style random-coordinate/2D fallback for hard cofactors"
        ),
        "labels_used": False,
    }
    (OUT / "build_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
