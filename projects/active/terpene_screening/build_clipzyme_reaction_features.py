from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAPPING = ROOT / "data/external/rxnmapper_current/general_merged_v1"
DEFAULT_RUNTIME = ROOT / "external_runtime/clipzyme_deps"
DEFAULT_CLIPZYME_ROOT = ROOT / "external/clipzyme"
DEFAULT_OUTPUT = ROOT / "data/external/clipzyme_current/general_merged_reaction_embeddings_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def load_clipzyme(runtime: Path, source: Path):
    for value in [str(source), str(runtime)]:
        if value not in sys.path:
            sys.path.insert(0, value)
    os.environ.setdefault("WANDB_MODE", "disabled")
    from clipzyme.lightning.clipzyme import CLIPZyme  # type: ignore

    return CLIPZyme


def clipzyme_graph_prerequisite(mapped_rxn: str) -> tuple[bool, str]:
    """Check the structural atom-map assumptions used by author difference-graph code.

    CLIPZyme's ``from_mapped_smiles`` requires every atom to have a map number,
    and ``encode_reaction`` adds reactant/product dense edge tensors, so both sides
    must describe the same uniquely mapped atom set. This is an input-domain check,
    not a performance-derived filter.
    """
    if ">>" not in mapped_rxn:
        return False, "missing_reaction_separator"
    left, right = mapped_rxn.split(">>", 1)
    left_mol, right_mol = Chem.MolFromSmiles(left), Chem.MolFromSmiles(right)
    if left_mol is None or right_mol is None:
        return False, "rdkit_parse_failed"
    left_maps = [atom.GetAtomMapNum() for atom in left_mol.GetAtoms()]
    right_maps = [atom.GetAtomMapNum() for atom in right_mol.GetAtoms()]
    if any(value <= 0 for value in left_maps + right_maps):
        return False, "unmapped_atoms"
    if len(left_maps) != len(set(left_maps)) or len(right_maps) != len(set(right_maps)):
        return False, "duplicate_atom_maps"
    if set(left_maps) != set(right_maps):
        return False, "reactant_product_map_sets_differ"
    return True, "compatible"


def inference_args(checkpoint: Path) -> Namespace:
    return Namespace(
        checkpoint_path=str(checkpoint),
        use_as_protein_encoder=False,
        use_as_reaction_encoder=True,
        save_hiddens=False,
        save_predictions=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract official CLIPZyme reaction embeddings from an audited atom-mapping registry. "
            "Unsupported rows remain NaN and are never silently replaced with zeros or another representation."
        )
    )
    parser.add_argument("--mapping-dir", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--clipzyme-root", type=Path, default=DEFAULT_CLIPZYME_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-reactions", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_reactions < 0:
        raise ValueError("batch-size must be positive and max-reactions non-negative")

    mapping_dir = args.mapping_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    runtime = args.runtime.resolve()
    clipzyme_root = args.clipzyme_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for path in [mapping_dir / "mapped_reactions.csv", mapping_dir / "manifest.json", checkpoint]:
        if not path.is_file():
            raise FileNotFoundError(path)

    mapping = pd.read_csv(mapping_dir / "mapped_reactions.csv", dtype=str).fillna("")
    required = {"row", "reaction_id", "mapped_rxn", "confidence", "success", "mapping_status"}
    if not required <= set(mapping):
        raise ValueError(f"mapping registry misses columns: {sorted(required - set(mapping))}")
    mapping["row"] = pd.to_numeric(mapping["row"], errors="raise").astype(int)
    mapping["confidence"] = pd.to_numeric(mapping["confidence"], errors="coerce")
    mapping["success"] = mapping["success"].str.lower().eq("true")
    mapping = mapping.sort_values("row", kind="mergesort").reset_index(drop=True)
    if mapping["reaction_id"].duplicated().any():
        raise ValueError("mapping registry reaction IDs must be unique")
    if args.max_reactions > 0:
        mapping = mapping.iloc[: args.max_reactions].copy()

    CLIPZyme = load_clipzyme(runtime, clipzyme_root)
    from clipzyme.utils.screening import process_mapped_reaction  # type: ignore
    from clipzyme.utils.loading import default_collate  # type: ignore
    import torch

    device = torch.device(args.device)
    wrapper = CLIPZyme(args=inference_args(checkpoint), device=None)
    wrapper.model = wrapper.model.to(device)
    wrapper.model.eval()

    vectors: dict[int, np.ndarray] = {}
    status = np.full(len(mapping), "mapping_unsupported", dtype=object)
    mapping_ok = mapping["success"].to_numpy(dtype=bool)
    graph_compatible = np.zeros(len(mapping), dtype=bool)
    for row in np.flatnonzero(mapping_ok):
        compatible, reason = clipzyme_graph_prerequisite(str(mapping.iloc[row]["mapped_rxn"]))
        graph_compatible[row] = compatible
        status[row] = "pending" if compatible else f"clipzyme_graph_prereq_failed:{reason}"
    success_rows = np.flatnonzero(graph_compatible).tolist()
    feature_dim: int | None = None

    def store(rows: list[int], batch_vectors: object) -> None:
        nonlocal feature_dim
        tensor = batch_vectors.detach().float().cpu()
        if tensor.ndim != 2 or tensor.shape[0] != len(rows):
            raise ValueError(f"unexpected CLIPZyme embedding shape {tuple(tensor.shape)} for {len(rows)} rows")
        values = tensor.numpy().astype(np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("CLIPZyme returned non-finite embeddings")
        local_dim = int(values.shape[1])
        if feature_dim is None:
            feature_dim = local_dim
        elif feature_dim != local_dim:
            raise ValueError(f"CLIPZyme embedding dimension changed: {feature_dim} -> {local_dim}")
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms <= 0):
            raise ValueError("CLIPZyme returned zero-norm embeddings")
        for local_i, row in enumerate(rows):
            vectors[row] = values[local_i]
            status[row] = "encoded"

    with torch.no_grad():
        for start in range(0, len(success_rows), args.batch_size):
            rows = success_rows[start : start + args.batch_size]
            reactions = mapping.iloc[rows]["mapped_rxn"].astype(str).tolist()

            def encode_on_device(batch_reactions: list[str]):
                # CLIPZyme's direct helper builds temporary PyG graphs on CPU. Lightning
                # normally moves dataset batches to the model device, but the direct helper
                # does not. Reproduce the author's graph construction, then perform only the
                # standard batch device transfer before calling the unchanged encoder.
                processed = [process_mapped_reaction(reaction) for reaction in batch_reactions]
                batch = default_collate(
                    [{"reactants": reactants, "products": products} for reactants, products in processed]
                )
                for key, value in list(batch.items()):
                    if hasattr(value, "to"):
                        batch[key] = value.to(device)
                return wrapper.extract_reaction_features(batch=batch)

            try:
                store(rows, encode_on_device(reactions))
            except Exception as batch_error:
                # Mapping succeeded but graph/model processing can still reject a historical
                # chemistry record. Fall back to independent inference so one bad reaction
                # never suppresses the rest of the batch, while keeping failures explicit.
                for row, reaction in zip(rows, reactions, strict=True):
                    try:
                        store([row], encode_on_device([reaction]))
                    except Exception as item_error:
                        status[row] = f"clipzyme_encode_failed:{type(item_error).__name__}"
                print(
                    json.dumps(
                        {
                            "batch_start": start,
                            "batch_size": len(rows),
                            "batch_fallback": True,
                            "batch_error": type(batch_error).__name__,
                        }
                    ),
                    flush=True,
                )
            if start == 0 or (start // args.batch_size + 1) % 25 == 0 or start + len(rows) >= len(success_rows):
                encoded = int(np.count_nonzero(status == "encoded"))
                failed = int(np.count_nonzero(np.char.startswith(status.astype(str), "clipzyme_encode_failed:")))
                print(
                    json.dumps(
                        {
                            "mapping_supported_rows": len(success_rows),
                            "processed_mapping_supported_rows": min(start + len(rows), len(success_rows)),
                            "encoded_rows": encoded,
                            "encoder_failed_rows": failed,
                        }
                    ),
                    flush=True,
                )

    if feature_dim is None:
        raise RuntimeError("CLIPZyme did not produce any embedding")
    matrix = np.full((len(mapping), feature_dim), np.nan, dtype=np.float32)
    for row, vector in vectors.items():
        matrix[row] = vector
    np.save(output / "embeddings.npy", matrix)
    encoded_mask = status == "encoded"
    entries = pd.DataFrame(
        {
            "row": mapping["row"].astype(int),
            "reaction_id": mapping["reaction_id"].astype(str),
            "mapping_success": mapping["success"].astype(bool),
            "mapping_confidence": mapping["confidence"],
            "clipzyme_graph_prerequisite": graph_compatible,
            "clipzyme_supported": encoded_mask,
            "support_status": status,
        }
    )
    entries.to_csv(output / "entries.csv", index=False)
    failed = entries.loc[~entries["clipzyme_supported"], ["reaction_id", "mapping_success", "mapping_confidence", "support_status"]]
    failed.to_csv(output / "unsupported_reactions.csv", index=False)

    mapping_manifest = mapping_dir / "manifest.json"
    manifest = {
        "version": "clipzyme-general-merged-reaction-embeddings-v1",
        "standalone_external_supervised_transfer": True,
        "strict_clean_representation": False,
        "integration_policy": "none; standalone support-aware embedding asset only; no zero-imputation or fallback representation is authorized by this artifact",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "clipzyme_source": str(clipzyme_root),
        "clipzyme_source_git_head": git_head(clipzyme_root),
        "mapping_dir": str(mapping_dir),
        "mapping_manifest_sha256": sha256_file(mapping_manifest),
        "reaction_count": int(len(mapping)),
        "mapping_supported_count": int(mapping_ok.sum()),
        "clipzyme_graph_prerequisite_count": int(graph_compatible.sum()),
        "clipzyme_graph_prerequisite_fraction": float(graph_compatible.mean()),
        "clipzyme_supported_count": int(encoded_mask.sum()),
        "unsupported_count": int((~encoded_mask).sum()),
        "feature_dimension": feature_dim,
        "dtype": "float32",
        "supported_row_norm_mean": float(np.linalg.norm(matrix[encoded_mask], axis=1).mean()),
        "supported_row_norm_max_abs_error_from_one": float(np.max(np.abs(np.linalg.norm(matrix[encoded_mask], axis=1) - 1.0))),
        "unsupported_storage": "NaN rows in embeddings.npy plus explicit support mask/status in entries.csv",
        "mapped_reaction_semantics": "RXNMapper output satisfying the author graph structural prerequisite is passed directly to official CLIPZyme.extract_reaction_features(reaction=...), which builds author graph inputs via process_mapped_reaction",
        "graph_prerequisite_policy": "pre-model input-domain check only: every atom mapped uniquely and reactant/product map sets identical; incompatible reactions remain unsupported and are not repaired or imputed",
        "batch_size": int(args.batch_size),
        "device": str(device),
        "runtime": str(runtime),
        "runtime_boundary": "isolated inference-only dependencies; project .venv is not modified",
        "mapping_confidence_policy": "not filtered; preserved for predeclared stratified diagnostics",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
