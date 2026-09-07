from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/bime_rank_unified_v1/clipzyme_r2e_candidate_asset_v1"
PROTEIN_ENTRIES = ROOT / "data/catalyst_candidate_universes/general_merged/proteins/entries.csv"
RELEASED = ROOT / "external_models/clipzyme_audit/clipzyme_data/clipzyme_screening_set.p"
NEW_ROOT = ROOT / "results/clipzyme_native_extension_v1/r2e_strict650_candidate_embeddings_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = pd.read_csv(PROTEIN_ENTRIES, dtype=str).fillna("")
    candidate_ids = entries["Entry"].astype(str).tolist()
    candidate_index = {p: i for i, p in enumerate(candidate_ids)}

    released = pickle.load(RELEASED.open("rb"))
    rel_ids = [str(x) for x in released["uniprots"]]
    h = released["hiddens"]
    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
    h = np.asarray(h, dtype=np.float32)
    rel_index = {p: i for i, p in enumerate(rel_ids)}
    rel_supported = [p for p in candidate_ids if p in rel_index]

    new_entries = pd.read_csv(NEW_ROOT / "new_entries.csv", dtype=str).fillna("")
    new_ids = new_entries["protein_id"].astype(str).tolist()
    new_mat = np.load(NEW_ROOT / "new_embeddings.npy", mmap_mode="r")
    if len(new_ids) != len(new_mat):
        raise RuntimeError("new CLIPZyme entries/embedding length mismatch")
    new_index = {p: i for i, p in enumerate(new_ids)}
    new_supported = [p for p in candidate_ids if p in new_index and p not in rel_index]

    supported_ids = rel_supported + new_supported
    # Preserve global candidate order for deterministic tie handling.
    supported_ids.sort(key=candidate_index.__getitem__)
    mat = np.lib.format.open_memmap(
        OUT / "embeddings.npy", mode="w+", dtype=np.float32,
        shape=(len(supported_ids), h.shape[1]),
    )
    records = []
    batch = 4096
    for st in range(0, len(supported_ids), batch):
        chunk = supported_ids[st:st + batch]
        block = np.empty((len(chunk), h.shape[1]), dtype=np.float32)
        source = []
        for j, p in enumerate(chunk):
            if p in rel_index:
                block[j] = h[rel_index[p]]
                source.append("official_released")
            else:
                block[j] = new_mat[new_index[p]]
                source.append("af_v6_native_extension")
        block = norm_rows(block)
        mat[st:st + len(chunk)] = block
        for j, (p, src) in enumerate(zip(chunk, source, strict=True)):
            records.append({"row": st + j, "protein_id": p, "candidate_row": candidate_index[p], "source": src})
        print(f"asset {st + len(chunk)}/{len(supported_ids)}", flush=True)
    mat.flush()
    pd.DataFrame(records).to_csv(OUT / "entries.csv", index=False)
    manifest = {
        "version": "bime-rank-clipzyme-r2e-candidate-asset-v1",
        "candidate_universe": len(candidate_ids),
        "supported_count": len(supported_ids),
        "official_released_intersection": len(rel_supported),
        "af_v6_native_extension": len(new_supported),
        "feature_dim": int(h.shape[1]),
        "dtype": "float32",
        "normalization": "row_l2",
        "candidate_order": "general_merged protein order; entries.csv records candidate_row",
        "released_asset_sha256": sha256_file(RELEASED),
        "new_embedding_sha256": sha256_file(NEW_ROOT / "new_embeddings.npy"),
        "new_entries_sha256": sha256_file(NEW_ROOT / "new_entries.csv"),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
