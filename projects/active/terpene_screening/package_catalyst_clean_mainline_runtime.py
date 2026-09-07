from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE = ROOT / "results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1"
DEFAULT_FEATURES = ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_runtime(bundle: Path, feature_dir: Path) -> dict[str, object]:
    bundle = bundle.resolve(); feature_dir = feature_dir.resolve()
    schema_path = bundle / "feature_schema.json"
    original_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    entries = pd.read_csv(feature_dir / "entries.csv", dtype={"reaction_id": str}).sort_values("row").reset_index(drop=True)
    if entries["reaction_id"].duplicated().any():
        raise ValueError("registered runtime reaction IDs are not unique")
    matrix_path = feature_dir / "reaction_feature_matrix.npy"
    matrix = np.load(matrix_path, mmap_mode="r")
    expected_dim = int(original_schema["reaction_feature_dimension"])
    if matrix.ndim != 2 or matrix.shape != (len(entries), expected_dim):
        raise ValueError(f"runtime reaction matrix mismatch: {matrix.shape} != {(len(entries), expected_dim)}")
    training_ids = list(map(str, original_schema.get("reaction_ids") or []))
    runtime_ids = entries["reaction_id"].astype(str).tolist()
    missing = sorted(set(training_ids) - set(runtime_ids))
    if missing:
        raise ValueError(f"training schema reactions missing from runtime registry: {missing[:10]}")
    training_schema_path = bundle / "training_feature_schema.json"
    if not training_schema_path.exists():
        shutil.copy2(schema_path, training_schema_path)
    else:
        retained = json.loads(training_schema_path.read_text(encoding="utf-8"))
        if retained != original_schema and original_schema.get("runtime_reaction_registry") is None:
            raise ValueError("existing training_feature_schema differs from current pre-runtime schema")
        # Repackaging an already packaged bundle: derive provenance from retained training schema.
        original_schema = retained
        training_ids = list(map(str, original_schema.get("reaction_ids") or []))
    runtime_schema = dict(original_schema)
    runtime_schema["reaction_ids"] = runtime_ids
    runtime_schema["runtime_reaction_registry"] = {
        "feature_dir": str(feature_dir),
        "entries_sha256": sha256_file(feature_dir / "entries.csv"),
        "matrix_sha256": sha256_file(matrix_path),
        "runtime_reaction_count": len(runtime_ids),
        "training_schema_reaction_count": len(training_ids),
        "training_reactions_are_runtime_subset": True,
        "association_labels_used": False,
    }
    schema_path.write_text(json.dumps(runtime_schema, indent=2) + "\n", encoding="utf-8")
    target = bundle / "reaction_feature_matrix.npy"
    if target.exists() or target.is_symlink():
        target.unlink()
    relative = os.path.relpath(matrix_path, start=bundle)
    target.symlink_to(relative)
    if not target.is_file():
        raise FileNotFoundError(target)
    # Final runtime contract must match load_reaction_library exactly.
    final = json.loads(schema_path.read_text(encoding="utf-8"))
    final_matrix = np.load(target, mmap_mode="r")
    if len(final["reaction_ids"]) != len(final_matrix):
        raise ValueError("packaged runtime schema/matrix row mismatch")
    return {
        "status": "ready",
        "bundle": str(bundle),
        "training_schema": str(training_schema_path),
        "runtime_schema": str(schema_path),
        "runtime_matrix_link": str(target),
        "runtime_matrix_link_target": relative,
        "runtime_reaction_count": len(runtime_ids),
        "training_schema_reaction_count": len(training_ids),
        "reaction_feature_dimension": expected_dim,
        "training_reactions_are_runtime_subset": True,
        "runtime_schema_sha256": sha256_file(schema_path),
        "training_schema_sha256": sha256_file(training_schema_path),
        "runtime_entries_sha256": sha256_file(feature_dir / "entries.csv"),
        "runtime_matrix_sha256": sha256_file(matrix_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Package the confirmed Catalyst clean R2E checkpoint with its full registered reaction runtime library without duplicating the matrix.")
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--reaction-feature-dir", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--manifest", type=Path, default=None)
    args = ap.parse_args()
    result = package_runtime(args.bundle, args.reaction_feature_dir)
    output = args.manifest.resolve() if args.manifest else args.bundle.resolve().parent / "runtime_manifest.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
