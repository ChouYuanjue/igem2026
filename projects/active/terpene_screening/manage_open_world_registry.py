from __future__ import annotations

import argparse
import json
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import fcntl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.core.registry_snapshots import (  # noqa: E402
    current_snapshot_name,
    load_snapshot_manifest,
    publish_snapshot,
    resolve_protein_dir,
    resolve_reaction_path,
)
from projects.active.terpene_screening.prepare_marts_dataset import reaction_signature  # noqa: E402
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    clean_sequence,
    encode_external_enzymes_with_audit,
    encode_reaction_with_audit,
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
DEFAULT_DUAL_KERNEL = ROOT / "results/terpene_production_models/marts_dual_kernel_e2r_top20"


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


def load_registry(protein_dir: Path) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    protein_dir = resolve_protein_dir(protein_dir)
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
    metadata = metadata.drop_duplicates("Entry", keep="last")
    metadata = metadata.set_index("Entry").reindex(entries["Entry"].astype(str)).reset_index()
    return embeddings, entries, metadata.fillna("")


def load_reactions(path: Path) -> pd.DataFrame:
    resolved = resolve_reaction_path(path)
    if not resolved.exists():
        return pd.DataFrame(columns=["reaction_id", "reaction_smiles", "reaction_signature", "source"])
    return pd.read_csv(resolved, dtype=str).fillna("")


def publish_registry_state(
    args: argparse.Namespace,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    reactions: pd.DataFrame,
    reason: str,
) -> dict[str, object]:
    return publish_snapshot(
        root=args.registry_root.resolve(),
        embeddings=embeddings,
        metadata=metadata,
        reactions=reactions,
        legacy_protein_dir=args.protein_registry.resolve(),
        legacy_reaction_path=args.reaction_registry.resolve(),
        reason=reason,
    )


def initialize_registry(args: argparse.Namespace) -> dict[str, object]:
    root = args.registry_root.resolve()
    if (
        current_snapshot_name(root)
        or args.protein_registry.exists()
        or args.reaction_registry.exists()
    ) and not args.force:
        raise FileExistsError("Registry already exists; pass --force to reinitialize")
    with registry_lock(root):
        if args.force:
            shutil.rmtree(root / "snapshots", ignore_errors=True)
            (root / "CURRENT").unlink(missing_ok=True)
        source_features, source_ids = load_protein_library(args.source_protein_dir.resolve())
        metadata = pd.DataFrame(
            {"Entry": source_ids, "sequence": "", "source": "marts_registered"}
        )
        reactions = pd.read_csv(args.source_reactions.resolve(), dtype=str).fillna("")
        required = {"reaction_id", "reaction_smiles", "source"}
        if required - set(reactions.columns):
            raise ValueError(f"Source reaction registry missing {sorted(required - set(reactions.columns))}")
        reactions = reactions[reactions["source"].astype(str) != "current"].copy()
        reactions["source"] = "marts_registered"
        reactions = reactions.drop_duplicates("reaction_id").reset_index(drop=True)
        publish_registry_state(
            args,
            source_features.astype(np.float32),
            metadata,
            reactions,
            "initialize_registry",
        )
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
    _, current_ids = load_protein_library(args.current_protein_dir.resolve())
    overlap_current = set(additions["Entry"]) & set(current_ids)
    if overlap_current and not args.allow_current_id:
        raise ValueError(f"IDs already exist in current protein library: {sorted(overlap_current)}")

    with registry_lock(args.registry_root.resolve()):
        embeddings, entries, metadata = load_registry(args.protein_registry.resolve())
        reactions = load_reactions(args.reaction_registry.resolve())
        existing = set(entries["Entry"].astype(str))
        duplicate = set(additions["Entry"]) & existing
        if duplicate and not args.replace:
            raise ValueError(f"IDs already exist in registered protein library: {sorted(duplicate)}")
        encoded, audits = encode_external_enzymes_with_audit(
            additions.rename(columns={"Entry": "enzyme_id"}),
            args.device,
            args.esmc_model,
            input_policy="strict",
        )
        feature_by_id = {
            value: embeddings[index]
            for index, value in enumerate(entries["Entry"].astype(str))
        }
        for index, row in enumerate(additions.itertuples(index=False)):
            feature_by_id[str(row.Entry)] = encoded[index]
        combined_metadata = metadata[~metadata["Entry"].isin(additions["Entry"])].copy()
        new_metadata = additions.copy()
        new_metadata["source"] = args.source_label
        new_metadata["sequence_sha256"] = [audit.sequence_sha256 for audit in audits]
        combined_metadata = pd.concat([combined_metadata, new_metadata], ignore_index=True)
        ordered_ids = sorted(feature_by_id)
        ordered_embeddings = np.stack([feature_by_id[value] for value in ordered_ids]).astype(np.float32)
        combined_metadata = combined_metadata.set_index("Entry").loc[ordered_ids].reset_index()
        publish_registry_state(
            args,
            ordered_embeddings,
            combined_metadata,
            reactions,
            "add_or_replace_enzymes",
        )
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
    audits = []
    for value in additions["reaction_smiles"]:
        feature, audit = encode_reaction_with_audit(
            str(value), schema, failure_policy="strict"
        )
        if not np.isfinite(feature).all():
            raise ValueError(f"Non-finite reaction feature for {value}")
        audits.append(audit)
    additions["reaction_signature"] = additions["reaction_smiles"].map(reaction_signature)
    additions["source"] = args.source_label
    additions["reaction_sha256"] = [audit.reaction_sha256 for audit in audits]

    with registry_lock(args.registry_root.resolve()):
        embeddings, _, metadata = load_registry(args.protein_registry.resolve())
        existing = load_reactions(args.reaction_registry.resolve())
        duplicate = set(additions["reaction_id"]) & set(existing["reaction_id"].astype(str))
        if duplicate and not args.replace:
            raise ValueError(f"Reaction IDs already registered: {sorted(duplicate)}")
        combined = existing[~existing["reaction_id"].isin(additions["reaction_id"])].copy()
        combined = pd.concat([combined, additions], ignore_index=True)
        combined = combined.drop_duplicates("reaction_id", keep="last").reset_index(drop=True)
        publish_registry_state(args, embeddings, metadata, combined, "add_or_replace_reactions")
    status = registry_status(args)
    status["added_or_replaced_reactions"] = additions["reaction_id"].tolist()
    return status


def remove_enzyme(args: argparse.Namespace) -> dict[str, object]:
    with registry_lock(args.registry_root.resolve()):
        embeddings, entries, metadata = load_registry(args.protein_registry.resolve())
        reactions = load_reactions(args.reaction_registry.resolve())
        if args.enzyme_id not in set(entries["Entry"].astype(str)):
            raise ValueError(f"Unknown registered enzyme ID: {args.enzyme_id}")
        keep = entries["Entry"].astype(str) != args.enzyme_id
        publish_registry_state(
            args,
            embeddings[keep.to_numpy()].astype(np.float32),
            metadata[metadata["Entry"].astype(str) != args.enzyme_id].copy(),
            reactions,
            "remove_enzyme",
        )
    return registry_status(args)


def remove_reaction(args: argparse.Namespace) -> dict[str, object]:
    with registry_lock(args.registry_root.resolve()):
        embeddings, _, metadata = load_registry(args.protein_registry.resolve())
        reactions = load_reactions(args.reaction_registry.resolve())
        if args.reaction_id not in set(reactions["reaction_id"].astype(str)):
            raise ValueError(f"Unknown registered reaction ID: {args.reaction_id}")
        reactions = reactions[reactions["reaction_id"].astype(str) != args.reaction_id].copy()
        publish_registry_state(args, embeddings, metadata, reactions, "remove_reaction")
    return registry_status(args)


def snapshot_registry(args: argparse.Namespace) -> dict[str, object]:
    with registry_lock(args.registry_root.resolve()):
        embeddings, _, metadata = load_registry(args.protein_registry.resolve())
        reactions = load_reactions(args.reaction_registry.resolve())
        if not len(metadata) and not len(reactions):
            raise ValueError("Cannot snapshot an empty registry")
        publish_registry_state(args, embeddings, metadata, reactions, "migrate_or_checkpoint_registry")
    return registry_status(args)


def derived_asset_status(
    entries: pd.DataFrame,
    reactions: pd.DataFrame,
    current_protein_dir: Path,
) -> dict[str, str]:
    try:
        _, current_ids = load_protein_library(current_protein_dir)
        asset_proteins = pd.read_csv(DEFAULT_DUAL_KERNEL / "protein_ids.csv", dtype=str).fillna("")["protein_id"].astype(str)
        asset_reactions = pd.read_csv(DEFAULT_DUAL_KERNEL / "reaction_ids.csv", dtype=str).fillna("")["reaction_id"].astype(str)
        protein_ready = set(current_ids) | set(entries["Entry"].astype(str)) == set(asset_proteins)
        schema = load_feature_schema(DEFAULT_DEPLOYMENT)
        reaction_ready = set(map(str, schema["reaction_ids"])) | set(reactions["reaction_id"].astype(str)) == set(asset_reactions)
        dual_kernel = "ready" if protein_ready and reaction_ready else "stale_candidate_universe"
    except Exception:
        dual_kernel = "compatibility_check_failed"
    return {
        "direct_dual_tower": "ready",
        "dual_kernel": dual_kernel,
        "reliability_calibrators": "legacy_unbound" if current_snapshot_name(DEFAULT_REGISTRY_ROOT) else "legacy",
        "registry_batch_outputs": "requires_regeneration_after_change",
        "wetlab_panels": "requires_review_after_change",
    }


def registry_status(args: argparse.Namespace) -> dict[str, object]:
    embeddings, entries, metadata = load_registry(args.protein_registry.resolve())
    reactions = load_reactions(args.reaction_registry.resolve())
    manifest = load_snapshot_manifest(args.registry_root.resolve())
    return {
        "registry_root": str(args.registry_root.resolve()),
        "protein_registry": str(resolve_protein_dir(args.protein_registry.resolve())),
        "reaction_registry": str(resolve_reaction_path(args.reaction_registry.resolve())),
        "registry_version": current_snapshot_name(args.registry_root.resolve()) or "legacy",
        "snapshot_manifest": manifest,
        "n_registered_proteins": len(entries),
        "protein_embedding_shape": list(embeddings.shape),
        "n_protein_metadata_rows": len(metadata),
        "n_registered_reactions": len(reactions),
        "protein_sources": metadata["source"].value_counts().to_dict() if "source" in metadata else {},
        "reaction_sources": reactions["source"].value_counts().to_dict() if "source" in reactions else {},
        "derived_asset_status": derived_asset_status(entries, reactions, args.current_protein_dir.resolve()),
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

    snapshot_parser = subparsers.add_parser("snapshot")
    add_common(snapshot_parser)

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
    elif args.command == "snapshot":
        result = snapshot_registry(args)
    else:
        result = registry_status(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
