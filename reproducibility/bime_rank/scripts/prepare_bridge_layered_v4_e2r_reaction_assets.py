from __future__ import annotations

import json
import pickle
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from projects.active.bridge.model.assets import ROOT

SCRIPTS = ROOT / "reproducibility/bime_rank/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_enzymecage_internal_reaction_features_v1 as builder

CAND = ROOT / "results/bridge_layered_v4_candidates"
REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
ZIP_PATH = ROOT / "data/external/enzymecage_current/authors_drive_current/dataset.zip"
OUT = ROOT / "results/bridge_layered_v4_e2r_reaction_assets"
FEATURE = OUT / "feature/reaction"
AUTHOR_FEATURE_PREFIX = "dataset/RHEA/2025-02-05/feature/reaction/"
AUTHOR_REGISTRY = "dataset/RHEA/2025-02-05/rhea_rxn2uids.csv"


def candidate_ids() -> set[str]:
    native = pd.read_csv(
        CAND / "e2r_similarity_gate_candidates.csv.gz", dtype=str
    ).fillna("")
    broad = pd.read_csv(
        CAND / "e2r_broad_equal_budget_candidates.csv.gz", dtype=str
    ).fillna("")
    return set(native.reaction_id.astype(str)) | set(broad.reaction_id.astype(str))


def extract_author_features() -> dict[str, int]:
    FEATURE.mkdir(parents=True, exist_ok=True)
    copied = 0
    reused = 0
    with ZipFile(ZIP_PATH) as zf:
        for info in zf.infolist():
            if not info.filename.startswith(AUTHOR_FEATURE_PREFIX) or info.is_dir():
                continue
            rel = Path(info.filename[len(AUTHOR_FEATURE_PREFIX):])
            dst = FEATURE / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and dst.stat().st_size == info.file_size:
                reused += 1
                continue
            with zf.open(info) as fi, dst.open("wb") as fo:
                shutil.copyfileobj(fi, fo, length=1024 * 1024)
            copied += 1
    return {"files_copied": copied, "files_reused": reused}


def author_reaction_map() -> dict[str, str]:
    with ZipFile(ZIP_PATH) as zf:
        with zf.open(AUTHOR_REGISTRY) as fh:
            frame = pd.read_csv(fh, dtype=str).fillna("")
    frame["reaction_id"] = "RHEA:" + frame.RHEA_ID.astype(str).str.split(".").str[0]
    conflict = frame.groupby("reaction_id").CANO_RXN_SMILES.nunique()
    bad = conflict[conflict > 1]
    if len(bad):
        raise RuntimeError(
            f"author RHEA IDs map to multiple strings: {bad.index[:10].tolist()}"
        )
    return (
        frame[["reaction_id", "CANO_RXN_SMILES"]]
        .drop_duplicates("reaction_id")
        .set_index("reaction_id")
        .CANO_RXN_SMILES.astype(str)
        .to_dict()
    )


def molecule_set(reaction: str) -> set[str]:
    left, right = str(reaction).split(">>")
    return {
        x.replace("*", "C")
        for x in left.split(".") + right.split(".")
        if x
    }


def validate_center(reaction: str, payload: object) -> str | None:
    try:
        left_idx, right_idx = payload
        left, right = reaction.split(">>")

        def atoms(side: str) -> int:
            total = 0
            for part in side.split("."):
                if not part:
                    continue
                mol = Chem.MolFromSmiles(part.replace("*", "C"))
                if mol is None:
                    raise ValueError(f"invalid molecule: {part}")
                total += int(mol.GetNumAtoms())
            return total

        nl, nr = atoms(left), atoms(right)
        badl = [int(i) for i in left_idx if int(i) < 0 or int(i) >= nl]
        badr = [int(i) for i in right_idx if int(i) < 0 or int(i) >= nr]
        if badl or badr:
            return f"center_index_out_of_bounds:left={badl},right={badr},nodes=({nl},{nr})"
        return None
    except Exception as exc:
        return f"{type(exc).__name__}:{exc}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    extraction = extract_author_features()

    ids = sorted(candidate_ids())
    current = pd.read_csv(REACTIONS, dtype=str).fillna("")
    current_map = dict(
        zip(current.reaction_id.astype(str), current.reaction_smiles.astype(str))
    )
    author_map = author_reaction_map()

    rows: list[dict[str, str]] = []
    for rid in ids:
        if rid in author_map and author_map[rid]:
            rows.append(
                {
                    "reaction_id": rid,
                    "CANO_RXN_SMILES": author_map[rid],
                    "reaction_source": "author_rhea2025",
                }
            )
        elif rid in current_map and current_map[rid]:
            rows.append(
                {
                    "reaction_id": rid,
                    "CANO_RXN_SMILES": current_map[rid],
                    "reaction_source": "general_merged_fallback",
                }
            )
        else:
            raise RuntimeError(f"missing reaction representation: {rid}")
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "reaction_registry.csv", index=False)

    drfp_path = FEATURE / "drfp/rxn2fp.pkl"
    center_path = FEATURE / "reacting_center/reacting_center.pkl"
    mol_dir = FEATURE / "molecule_conformation"
    mol_index_path = mol_dir / "mol2id.csv"

    with drfp_path.open("rb") as fh:
        drfp = pickle.load(fh)
    with center_path.open("rb") as fh:
        centers = pickle.load(fh)

    missing_drfp = frame[~frame.CANO_RXN_SMILES.isin(set(drfp))].copy()
    drfp_fallbacks: list[dict[str, str]] = []
    if len(missing_drfp):
        extra, drfp_fallbacks = builder.generate_drfp(missing_drfp)
        drfp.update(extra)
        with drfp_path.open("wb") as fh:
            pickle.dump(drfp, fh)

    missing_center = frame[~frame.CANO_RXN_SMILES.isin(set(centers))].copy()
    center_failures: list[dict[str, str]] = []
    if len(missing_center):
        extra, center_failures = builder.generate_centers(missing_center)
        centers.update(extra)
        with center_path.open("wb") as fh:
            pickle.dump(centers, fh)

    pd.DataFrame(drfp_fallbacks).to_csv(OUT / "drfp_incremental_fallbacks.csv", index=False)
    pd.DataFrame(center_failures).to_csv(OUT / "center_incremental_failures.csv", index=False)

    mol_index = pd.read_csv(mol_index_path, dtype=str).fillna("")
    mol_index["ID"] = pd.to_numeric(mol_index["ID"], errors="raise").astype(int)
    existing_molecules = set(mol_index.SMILES.astype(str))
    required_molecules: set[str] = set()
    for reaction in frame.CANO_RXN_SMILES.astype(str):
        required_molecules |= molecule_set(reaction)
    missing_molecules = sorted(required_molecules - existing_molecules)

    conformer_failures: list[dict[str, object]] = []
    if missing_molecules:
        next_id = int(mol_index.ID.max()) + 1 if len(mol_index) else 0
        extra_index = pd.DataFrame(
            {
                "SMILES": missing_molecules,
                "ID": range(next_id, next_id + len(missing_molecules)),
            }
        )
        jobs = list(
            zip(extra_index.ID.astype(int).tolist(), extra_index.SMILES.astype(str).tolist())
        )
        with Pool(processes=12) as pool:
            for idx, smiles, ok, payload in tqdm(
                pool.imap_unordered(builder.generate_conformer, jobs, chunksize=8),
                total=len(jobs),
                desc="incremental-conformer",
            ):
                if ok:
                    (mol_dir / f"{idx}.sdf").write_text(payload + "\n$$$$\n")
                else:
                    conformer_failures.append(
                        {"id": int(idx), "smiles": smiles, "reason": payload}
                    )
        mol_index = pd.concat([mol_index, extra_index], ignore_index=True)
        mol_index.to_csv(mol_index_path, index=False)

    pd.DataFrame(conformer_failures).to_csv(
        OUT / "conformer_incremental_failures.csv", index=False
    )

    failed_molecules = {str(x["smiles"]) for x in conformer_failures}
    audit_rows: list[dict[str, object]] = []
    valid_ids: set[str] = set()
    for rec in frame.itertuples(index=False):
        rid = str(rec.reaction_id)
        reaction = str(rec.CANO_RXN_SMILES)
        errors: list[str] = []
        if reaction not in drfp:
            errors.append("missing_drfp")
        payload = centers.get(reaction)
        if payload is None:
            errors.append("missing_reaction_center")
        else:
            issue = validate_center(reaction, payload)
            if issue:
                errors.append(issue)
        used = molecule_set(reaction)
        if used & failed_molecules:
            errors.append("failed_conformer")
        for molecule in used:
            row = mol_index[mol_index.SMILES.eq(molecule)]
            if row.empty:
                errors.append("missing_molecule_index")
                break
            sdf = mol_dir / f"{int(row.iloc[0].ID)}.sdf"
            if not sdf.exists():
                errors.append("missing_sdf")
                break
        supported = not errors
        if supported:
            valid_ids.add(rid)
        audit_rows.append(
            {
                "reaction_id": rid,
                "CANO_RXN_SMILES": reaction,
                "reaction_source": str(rec.reaction_source),
                "supported": supported,
                "reason": ";".join(errors),
            }
        )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT / "reaction_support_audit.csv", index=False)

    summary = {
        "schema": "bridge-layered-v4-e2r-reaction-assets",
        "status": "completed",
        "candidate_reactions": int(len(frame)),
        "author_reaction_strings": int(frame.reaction_source.eq("author_rhea2025").sum()),
        "fallback_reaction_strings": int(frame.reaction_source.eq("general_merged_fallback").sum()),
        "author_feature_extraction": extraction,
        "incremental_drfp_reactions": int(len(missing_drfp)),
        "incremental_center_reactions": int(len(missing_center)),
        "required_molecules": int(len(required_molecules)),
        "incremental_molecules": int(len(missing_molecules)),
        "incremental_conformer_failures": int(len(conformer_failures)),
        "supported_reactions": int(len(valid_ids)),
        "unsupported_reactions": int(len(frame) - len(valid_ids)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
