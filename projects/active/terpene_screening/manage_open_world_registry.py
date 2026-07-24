from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import fcntl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.prepare_marts_dataset import reaction_signature  # noqa: E402
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    clean_sequence,
    encode_external_enzymes,
    encode_reaction,
    load_feature_schema,
    load_protein_library,
)

DEFAULT_REGISTRY_ROOT = ROOT / "data/terpene_open_world_registry"
DEFAULT_PROTEIN_REGISTRY = DEFAULT_REGISTRY_ROOT / "proteins"
DEFAULT_REACTION_REGISTRY = DEFAULT_REGISTRY_ROOT / "reactions.csv"
DEFAULT_SOURCE_PROTEINS = ROOT / "data/terpene_embeddings/marts_unseen_esmc600m"
DEFAULT_SOURCE_REACTIONS = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu/reaction_registry.csv"
DEFAULT_CURRENT_PROTEINS = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_DEPLOYMENT = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"


@contextmanager
def registry_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".registry.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".csv", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def atomic_write_npy(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".npy", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, matrix)
    os.replace(temporary, path)


def load_registry(protein_dir: Path) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    entries_path = protein_dir / "entries.csv"
    embeddings_path = protein_dir / "embeddings.npy"
    metadata_path = protein_dir / "metadata.csv"
    if not entries_path.exists() or not embeddings_path.exists():
        return (
            np.empty((0, 1152), dtype=np.float32),
            pd.DataFrame(columns=["row", "Entry"]),
            pd.DataFrame(columns=["Entry", "sequence", "source"]),
        )
    entries = pd.read_csv(entries_path, dtype=str).fillna("")
    entries["row"] = pd.to_numeric(entries["row"], errors="raise").astype(int)
    entries = entries.sort_values("row").reset_index(drop=True)
    embeddings = np.load(embeddings_path).astype(np.float32)
    if len(entries) != len(embeddings):
        raise ValueError("Registry entries and embedding matrix differ in length")
    metadata = (
        pd.read_csv(metadata_path, dtype=str).fillna("")
        if metadata_path.exists()
        else pd.DataFrame({"Entry": entries["Entry"], "sequence": "", "source": "legacy"})
    )
    return embeddings, entries, metadata


def save_registry(protein_dir: Path, embeddings: np.ndarray, metadata: pd.DataFrame) -> None:
    metadata = metadata.drop_duplicates("Entry", keep="last").sort_values("Entry").reset_index(drop=True)
    id_to_old_row = {}
    if (protein_dir / "entries.csv").exists():
        old_entries = pd.read_csv(protein_dir / "entries.csv", dtype=str).fillna("")
        id_to_old_row = {
            value: int(row) for value, row in zip(old_entries["Entry"].astype(str), old_entries["row"])
        }
    if len(embeddings) != len(id_to_old_row) and id_to_old_row:
        raise ValueError("Embedding matrix does not match existing entry mapping")
    if id_to_old_row:
        ordered_embeddings = np.stack([embeddings[id_to_old_row[value]] for value in metadata["Entry"]]).astype(np.float32)
    else:
        ordered_embeddings = embeddings.astype(np.float32)
    entries = pd.DataFrame({"row": np.arange(len(metadata)), "Entry": metadata["Entry"].astype(str)})
    atomic_write_npy(ordered_embeddings, protein_dir / "embeddings.npy")
    atomic_write_csv(entries, protein_dir / "entries.csv")
    atomic_write_csv(metadata, protein_dir / "metadata.csv")


def initialize_registry(args: argparse.Namespace) -> dict[str, object]:
    protein_dir = args.protein_registry.resolve()
    reaction_path = args.reaction_registry.resolve()
    if (protein_dir.exists() or reaction_path.exists()) and not args.force:
        raise FileExistsError("Registry already exists; pass --force to reinitialize")
    with registry_lock(args.registry_root.resolve()):
        shutil.rmtree(protein_dir, ignore_errors=True)
        protein_dir.mkdir(parents=True, exist_ok=True)
        source_features, source_ids = load_protein_library(args.source_protein_dir.resolve())
        entries = pd.DataFrame({"row": np.arange(len(source_ids)), "Entry": source_ids})
        metadata = pd.DataFrame({"Entry": source_ids, "sequence": "", "source": "marts_registered"})
        atomic_write_npy(source_features.astype(np.float32), protein_dir / "embeddings.npy")
        atomic_write_csv(entries, protein_dir / "entries.csv")
        atomic_write_csv(metadata, protein_dir / "metadata.csv")

        reactions = pd.read_csv(args.source_reactions.resolve(), dtype=str).fillna("")
        required = {"reaction_id", "reaction_smiles", "source"}
        if required - set(reactions.columns):
            raise ValueError(f"Source reaction registry missing {sorted(required - set(reactions.columns))}")
        reactions = reactions[reactions["source"].astype(str) != "current"].copy()
        reactions["source"] = "marts_registered"
        reactions = reactions.drop_duplicates("reaction_id").sort_values("reaction_id").reset_index(drop=True)
        atomic_write_csv(reactions, reaction_path)
    return registry_status(args)


def enzyme_input(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        frame = pd.read_csv(args.csv, dtype=str).fillna("")
        id_column = "enzyme_id" if "enzyme_id" in frame.columns else "Entry" if "Entry" in frame.columns else None
        sequence_column = "sequence" if "sequence" in frame.columns else "Sequence" if "Sequence" in frame.columns else None
        if id_column is None or sequence_column is None:
            raise ValueError("CSV requires enzyme_id/Entry and sequence/Sequence")
        frame = frame[[id_column, sequence_column]].rename(columns={id_column: "Entry", sequence_column: "sequence"})
    else:
        if not args.enzyme_id or not args.sequence:
            raise ValueError("Provide --csv or both --enzyme-id and --sequence")
        frame = pd.DataFrame({"Entry": [args.enzyme_id], "sequence": [args.sequence]})
    frame["Entry"] = frame["Entry"].astype(str).str.strip()
    frame["sequence"] = frame["sequence"].map(clean_sequence)
    return frame[(frame["Entry"] != "") & (frame["sequence"] != "")].drop_duplicates("Entry")


def add_enzymes(args: argparse.Namespace) -> dict[str, object]:
    additions = enzyme_input(args)
    if additions.empty:
        raise ValueError("No valid enzyme rows")
    current_features, current_ids = load_protein_library(args.current_protein_dir.resolve())
    del current_features
    current_id_set = set(current_ids)
    overlap_current = set(additions["Entry"]) & current_id_set
    if overlap_current and not args.allow_current_id:
        raise ValueError(f"IDs already exist in current protein library: {sorted(overlap_current)}")

    with registry_lock(args.registry_root.resolve()):
        embeddings, entries, metadata = load_registry(args.protein_registry.resolve())
        existing = set(entries["Entry"].astype(str))
        duplicate = set(additions["Entry"]) & existing
        if duplicate and not args.replace:
            raise ValueError(f"IDs already exist in registered protein library: {sorted(duplicate)}")
        encoded = encode_external_enzymes(
            additions.rename(columns={"Entry": "enzyme_id"}),
            args.device,
            args.esmc_model,
        )
        feature_by_id = {value: embeddings[index] for index, value in enumerate(entries["Entry"].astype(str))}
        for index, row in enumerate(additions.itertuples(index=False)):
            feature_by_id[str(row.Entry)] = encoded[index]
        combined_metadata = metadata[~metadata["Entry"].isin(additions["Entry"])].copy()
        new_metadata = additions.copy()
        new_metadata["source"] = args.source_label
        combined_metadata = pd.concat([combined_metadata, new_metadata], ignore_index=True)
        ordered_ids = sorted(feature_by_id)
        ordered_embeddings = np.stack([feature_by_id[value] for value in ordered_ids]).astype(np.float32)
        combined_metadata = combined_metadata.set_index("Entry").loc[ordered_ids].reset_index()
        atomic_write_npy(ordered_embeddings, args.protein_registry.resolve() / "embeddings.npy")
        atomic_write_csv(
            pd.DataFrame({"row": np.arange(len(ordered_ids)), "Entry": ordered_ids}),
            args.protein_registry.resolve() / "entries.csv",
        )
        atomic_write_csv(combined_metadata, args.protein_registry.resolve() / "metadata.csv")
    status = registry_status(args)
    status["added_or_replaced_enzymes"] = additions["Entry"].tolist()
    return status


def reaction_input(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        frame = pd.read_csv(args.csv, dtype=str).fillna("")
        id_column = "reaction_id" if "reaction_id" in frame.columns else None
        smiles_column = "reaction_smiles" if "reaction_smiles" in frame.columns else "smiles" if "smiles" in frame.columns else None
        if id_column is None or smiles_column is None:
            raise ValueError("CSV requires reaction_id and reaction_smiles/smiles")
        frame = frame[[id_column, smiles_column]].rename(columns={smiles_column: "reaction_smiles"})
    else:
        if not args.reaction_id or not args.reaction_smiles:
            raise ValueError("Provide --csv or both --reaction-id and --reaction-smiles")
        frame = pd.DataFrame({"reaction_id": [args.reaction_id], "reaction_smiles": [args.reaction_smiles]})
    frame["reaction_id"] = frame["reaction_id"].astype(str).str.strip()
    frame["reaction_smiles"] = frame["reaction_smiles"].astype(str).str.strip()
    return frame[(frame["reaction_id"] != "") & (frame["reaction_smiles"] != "")].drop_duplicates("reaction_id")


def add_reactions(args: argparse.Namespace) -> dict[str, object]:
    additions = reaction_input(args)
    if additions.empty:
        raise ValueError("No valid reaction rows")
    schema = load_feature_schema(args.deployment_dir.resolve())
    for value in additions["reaction_smiles"]:
        feature = encode_reaction(str(value), schema)
        if not np.isfinite(feature).all():
            raise ValueError(f"Non-finite reaction feature for {value}")
    additions["reaction_signature"] = additions["reaction_smiles"].map(reaction_signature)
    additions["source"] = args.source_label

    with registry_lock(args.registry_root.resolve()):
        path = args.reaction_registry.resolve()
        existing = pd.read_csv(path, dtype=str).fillna("") if path.exists() else pd.DataFrame(columns=additions.columns)
        duplicate = set(additions["reaction_id"]) & set(existing["reaction_id"].astype(str))
        if duplicate and not args.replace:
            raise ValueError(f"Reaction IDs already registered: {sorted(duplicate)}")
        combined = existing[~existing["reaction_id"].isin(additions["reaction_id"])].copy()
        combined = pd.concat([combined, additions], ignore_index=True)
        combined = combined.drop_duplicates("reaction_id", keep="last").sort_values("reaction_id").reset_index(drop=True)
        atomic_write_csv(combined, path)
    status = registry_status(args)
    status["added_or_replaced_reactions"] = additions["reaction_id"].tolist()
    return status


def remove_enzyme(args: argparse.Namespace) -> dict[str, object]:
    with registry_lock(args.registry_root.resolve()):
        embeddings, entries, metadata = load_registry(args.protein_registry.resolve())
        if args.enzyme_id not in set(entries["Entry"].astype(str)):
            raise ValueError(f"Unknown registered enzyme ID: {args.enzyme_id}")
        keep = entries["Entry"].astype(str) != args.enzyme_id
        embeddings = embeddings[keep.to_numpy()]
        metadata = metadata[metadata["Entry"].astype(str) != args.enzyme_id].copy()
        ordered_ids = entries.loc[keep, "Entry"].astype(str).tolist()
        atomic_write_npy(embeddings.astype(np.float32), args.protein_registry.resolve() / "embeddings.npy")
        atomic_write_csv(
            pd.DataFrame({"row": np.arange(len(ordered_ids)), "Entry": ordered_ids}),
            args.protein_registry.resolve() / "entries.csv",
        )
        atomic_write_csv(metadata, args.protein_registry.resolve() / "metadata.csv")
    return registry_status(args)


def remove_reaction(args: argparse.Namespace) -> dict[str, object]:
    with registry_lock(args.registry_root.resolve()):
        path = args.reaction_registry.resolve()
        frame = pd.read_csv(path, dtype=str).fillna("")
        if args.reaction_id not in set(frame["reaction_id"].astype(str)):
            raise ValueError(f"Unknown registered reaction ID: {args.reaction_id}")
        frame = frame[frame["reaction_id"].astype(str) != args.reaction_id].copy()
        atomic_write_csv(frame, path)
    return registry_status(args)


def registry_status(args: argparse.Namespace) -> dict[str, object]:
    embeddings, entries, metadata = load_registry(args.protein_registry.resolve())
    reaction_path = args.reaction_registry.resolve()
    reactions = (
        pd.read_csv(reaction_path, dtype=str).fillna("")
        if reaction_path.exists()
        else pd.DataFrame(columns=["reaction_id", "reaction_smiles", "source"])
    )
    return {
        "registry_root": str(args.registry_root.resolve()),
        "protein_registry": str(args.protein_registry.resolve()),
        "reaction_registry": str(reaction_path),
        "n_registered_proteins": len(entries),
        "protein_embedding_shape": list(embeddings.shape),
        "n_protein_metadata_rows": len(metadata),
        "n_registered_reactions": len(reactions),
        "protein_sources": metadata["source"].value_counts().to_dict() if "source" in metadata else {},
        "reaction_sources": reactions["source"].value_counts().to_dict() if "source" in reactions else {},
    }


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY_ROOT)
    parser.add_argument("--protein-registry", type=Path, default=DEFAULT_PROTEIN_REGISTRY)
    parser.add_argument("--reaction-registry", type=Path, default=DEFAULT_REACTION_REGISTRY)
    parser.add_argument("--current-protein-dir", type=Path, default=DEFAULT_CURRENT_PROTEINS)
    parser.add_argument("--deployment-dir", type=Path, default=DEFAULT_DEPLOYMENT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage persistent open-world TPS enzyme/reaction registries.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    add_common(init_parser)
    init_parser.add_argument("--source-protein-dir", type=Path, default=DEFAULT_SOURCE_PROTEINS)
    init_parser.add_argument("--source-reactions", type=Path, default=DEFAULT_SOURCE_REACTIONS)
    init_parser.add_argument("--force", action="store_true")

    enzyme_parser = subparsers.add_parser("add-enzymes")
    add_common(enzyme_parser)
    enzyme_parser.add_argument("--csv", type=Path, default=None)
    enzyme_parser.add_argument("--enzyme-id", default=None)
    enzyme_parser.add_argument("--sequence", default=None)
    enzyme_parser.add_argument("--replace", action="store_true")
    enzyme_parser.add_argument("--allow-current-id", action="store_true")
    enzyme_parser.add_argument("--source-label", default="user_registered")
    enzyme_parser.add_argument("--device", default="cuda")
    enzyme_parser.add_argument("--esmc-model", default="esmc_600m")

    reaction_parser = subparsers.add_parser("add-reactions")
    add_common(reaction_parser)
    reaction_parser.add_argument("--csv", type=Path, default=None)
    reaction_parser.add_argument("--reaction-id", default=None)
    reaction_parser.add_argument("--reaction-smiles", default=None)
    reaction_parser.add_argument("--replace", action="store_true")
    reaction_parser.add_argument("--source-label", default="user_registered")

    remove_enzyme_parser = subparsers.add_parser("remove-enzyme")
    add_common(remove_enzyme_parser)
    remove_enzyme_parser.add_argument("--enzyme-id", required=True)

    remove_reaction_parser = subparsers.add_parser("remove-reaction")
    add_common(remove_reaction_parser)
    remove_reaction_parser.add_argument("--reaction-id", required=True)

    status_parser = subparsers.add_parser("status")
    add_common(status_parser)

    args = parser.parse_args()
    if args.command == "init":
        result = initialize_registry(args)
    elif args.command == "add-enzymes":
        result = add_enzymes(args)
    elif args.command == "add-reactions":
        result = add_reactions(args)
    elif args.command == "remove-enzyme":
        result = remove_enzyme(args)
    elif args.command == "remove-reaction":
        result = remove_reaction(args)
    else:
        result = registry_status(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
