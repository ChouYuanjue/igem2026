from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME = ROOT / "external_runtime/clipzyme_deps"
DEFAULT_SOURCE = ROOT / "external/clipzyme"
DEFAULT_CHECKPOINT = ROOT / "external_models/clipzyme_checkpoint/clipzyme_model.ckpt"
DEFAULT_ESM_DIR = ROOT / "external_models/clipzyme_reactzyme_baseline"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Official CLIPZyme native protein embedding builder.")
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    ap.add_argument("--clipzyme-root", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--esm-dir", type=Path, default=DEFAULT_ESM_DIR)
    ap.add_argument("--protein-cache-dir", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    for p in (args.input_csv, args.checkpoint, args.esm_dir / "esm2_t33_650M_UR50D.pt"):
        if not p.is_file():
            raise FileNotFoundError(p)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    # The isolated runtime is a --target installation, not a separate venv.
    # Put it before the project environment so author-compatible PyG/ESM are used.
    for p in (args.clipzyme_root.resolve(), args.runtime.resolve()):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)
    os.environ.setdefault("WANDB_MODE", "disabled")

    from clipzyme import CLIPZyme, ReactionDataset  # type: ignore
    from clipzyme.utils.loading import ignore_None_collate  # type: ignore
    import torch_geometric  # type: ignore

    inputs = pd.read_csv(args.input_csv, dtype=str).fillna("")
    required = {"reaction", "sequence", "protein_id", "cif"}
    if not required <= set(inputs):
        raise ValueError(f"input misses {sorted(required - set(inputs))}")
    if inputs["protein_id"].duplicated().any():
        raise ValueError("protein_id must be unique in one embedding build")
    missing_structure = [p for p in inputs["cif"] if not Path(p).is_file()]
    if missing_structure:
        raise FileNotFoundError(f"missing {len(missing_structure)} structure files; first={missing_structure[0]}")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = args.protein_cache_dir.resolve() if args.protein_cache_dir else out / "protein_graph_cache"
    cache.mkdir(parents=True, exist_ok=True)

    dataset = ReactionDataset(
        dataset_file_path=str(args.input_csv.resolve()),
        esm_dir=str(args.esm_dir.resolve()),
        protein_cache_dir=str(cache),
        use_as_protein_encoder=True,
        use_as_reaction_encoder=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=ignore_None_collate, shuffle=False)

    wrapper = CLIPZyme(checkpoint_path=str(args.checkpoint.resolve())).eval()
    # Official wrapper's optional device setter is incompatible with the installed Lightning
    # because Lightning exposes a read-only `device` property. Moving only the wrapped author
    # network preserves the exact checkpoint computation and avoids modifying CLIPZyme code.
    wrapper.model = wrapper.model.to(args.device)

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                raise RuntimeError("ReactionDataset returned an empty batch")
            batch["graph"] = batch["graph"].to(args.device)
            z = wrapper.extract_protein_features(batch).detach().cpu().numpy().astype(np.float32, copy=False)
            if z.ndim != 2 or z.shape[1] != 1280 or not np.isfinite(z).all():
                raise ValueError(f"invalid CLIPZyme protein embedding batch shape={z.shape}")
            local_ids = [str(x) for x in batch["protein_id"]]
            if len(local_ids) != z.shape[0]:
                raise ValueError("protein id / embedding row mismatch")
            ids.extend(local_ids)
            vectors.append(z)

    if len(ids) != len(inputs) or ids != inputs["protein_id"].astype(str).tolist():
        raise RuntimeError(f"output order/coverage mismatch: encoded={len(ids)} expected={len(inputs)}")
    matrix = np.concatenate(vectors, axis=0)
    norms = np.linalg.norm(matrix, axis=1)
    np.save(out / "embeddings.npy", matrix)
    entries = inputs.copy()
    entries["row"] = np.arange(len(entries), dtype=int)
    entries["status"] = "encoded"
    entries["norm"] = norms
    entries.to_csv(out / "entries.csv", index=False)
    manifest = {
        "version": "clipzyme-native-protein-embeddings-v1",
        "input_csv": str(args.input_csv.resolve()),
        "input_sha256": sha256_file(args.input_csv.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "esm_checkpoint": str((args.esm_dir / "esm2_t33_650M_UR50D.pt").resolve()),
        "protein_count": len(ids),
        "feature_dimension": int(matrix.shape[1]),
        "dtype": str(matrix.dtype),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "batch_size": args.batch_size,
        "device": args.device,
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "runtime": str(args.runtime.resolve()),
        "runtime_policy": "isolated inference-only dependencies; official checkpoint and graph/protein encoder unchanged",
        "device_workaround": "move CLIPZyme.model to requested device because current Lightning exposes wrapper.device as read-only",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
