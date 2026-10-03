from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256_bytes(values: bytes) -> str:
    return hashlib.sha256(values).hexdigest()


def _id_hash(values: list[str]) -> str:
    return _sha256_bytes("\n".join(sorted(set(map(str, values)))).encode("utf-8"))


def current_snapshot_name(root: Path) -> str | None:
    pointer = root / "CURRENT"
    if not pointer.exists():
        return None
    value = pointer.read_text(encoding="utf-8").strip()
    return value or None


def current_snapshot_root(root: Path) -> Path | None:
    name = current_snapshot_name(root)
    if not name:
        return None
    snapshot = root / "snapshots" / name
    if not (snapshot / "manifest.json").exists():
        raise ValueError(f"Registry CURRENT points to an incomplete snapshot: {snapshot}")
    return snapshot


def resolve_protein_dir(path: Path) -> Path:
    path = path.resolve()
    if path.name != "proteins":
        return path
    snapshot = current_snapshot_root(path.parent)
    return snapshot / "proteins" if snapshot is not None else path


def resolve_reaction_path(path: Path) -> Path:
    path = path.resolve()
    if path.name != "reactions.csv":
        return path
    snapshot = current_snapshot_root(path.parent)
    return snapshot / "reactions.csv" if snapshot is not None else path


def registry_version(root: Path) -> str:
    return current_snapshot_name(root.resolve()) or "legacy"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        with source.open("rb") as reader:
            shutil.copyfileobj(reader, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def publish_snapshot(
    *,
    root: Path,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    reactions: pd.DataFrame,
    legacy_protein_dir: Path,
    legacy_reaction_path: Path,
    reason: str,
) -> dict[str, object]:
    root = root.resolve()
    embeddings = np.asarray(embeddings, dtype=np.float32)
    metadata = metadata.copy().fillna("")
    reactions = reactions.copy().fillna("")
    if "Entry" not in metadata:
        raise ValueError("Protein metadata requires Entry")
    if len(metadata) != len(embeddings):
        raise ValueError("Protein metadata and embeddings differ in length")
    if metadata["Entry"].astype(str).duplicated().any():
        raise ValueError("Protein registry contains duplicate IDs")
    if "reaction_id" not in reactions or "reaction_smiles" not in reactions:
        raise ValueError("Reaction registry requires reaction_id and reaction_smiles")
    if reactions["reaction_id"].astype(str).duplicated().any():
        raise ValueError("Reaction registry contains duplicate IDs")

    order = np.argsort(metadata["Entry"].astype(str).to_numpy(), kind="stable")
    metadata = metadata.iloc[order].reset_index(drop=True)
    embeddings = embeddings[order]
    entries = pd.DataFrame({"row": np.arange(len(metadata)), "Entry": metadata["Entry"].astype(str)})
    reactions = reactions.sort_values("reaction_id").reset_index(drop=True)
    protein_hash = _id_hash(entries["Entry"].astype(str).tolist())
    reaction_hash = _id_hash(reactions["reaction_id"].astype(str).tolist())
    digest = _sha256_bytes((protein_hash + reaction_hash + str(embeddings.shape)).encode("utf-8"))[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    version = f"registry-{timestamp}-{digest}"
    snapshots = root / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    temporary = snapshots / f".{version}.tmp-{os.getpid()}"
    final = snapshots / version
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "proteins").mkdir(parents=True)
    np.save(temporary / "proteins/embeddings.npy", embeddings)
    entries.to_csv(temporary / "proteins/entries.csv", index=False)
    metadata.to_csv(temporary / "proteins/metadata.csv", index=False)
    reactions.to_csv(temporary / "reactions.csv", index=False)
    manifest = {
        "registry_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_version": current_snapshot_name(root),
        "reason": reason,
        "n_proteins": len(entries),
        "n_reactions": len(reactions),
        "protein_embedding_shape": list(embeddings.shape),
        "protein_id_hash": protein_hash,
        "reaction_id_hash": reaction_hash,
        "derived_assets": {
            "direct_dual_tower": "ready",
            "dual_kernel": "requires_compatibility_check",
            "reliability_calibrators": "requires_compatibility_check",
            "registry_batch_outputs": "stale_after_registry_change",
            "wetlab_panels": "stale_after_registry_change",
        },
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, final)
    _atomic_text(root / "CURRENT", version + "\n")

    # Compatibility mirrors are updated after the atomic production pointer.
    for relative in ["embeddings.npy", "entries.csv", "metadata.csv"]:
        _atomic_copy(final / "proteins" / relative, legacy_protein_dir / relative)
    _atomic_copy(final / "reactions.csv", legacy_reaction_path)
    return manifest


def load_snapshot_manifest(root: Path) -> dict[str, object] | None:
    snapshot = current_snapshot_root(root.resolve())
    if snapshot is None:
        return None
    return json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
