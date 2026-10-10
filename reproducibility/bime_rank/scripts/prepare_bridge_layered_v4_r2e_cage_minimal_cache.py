from __future__ import annotations

import gc
import json
import pickle as pkl
from pathlib import Path

import pandas as pd
import torch

from projects.active.bridge.model.assets import ROOT

BASE = ROOT / "results/bridge_layered_v4_r2e_cage"
SHARD_DIR = BASE / "minimal_shards"
OUT = BASE / "minimal_cache"

SOURCE_GVP = BASE / "extra_features/gvp_protein_feature.pt"
SOURCE_NODE = BASE / "extra_features/esm_node_feature.pt"
SOURCE_MEAN = BASE / "extra_features/seq2feature.pkl"

OUT_GVP = OUT / "gvp_protein_feature.pt"
OUT_NODE = OUT / "esm_node_feature.pt"
OUT_MEAN = OUT / "seq2feature.pkl"
OUT_PAIRS = OUT / "pairs_minimal_all.csv"
AUDIT = OUT / "cache_audit.json"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(SHARD_DIR.glob("pairs_minimal_shard_*.csv"))
    if len(paths) != 8:
        raise RuntimeError(f"expected 8 minimal shard inputs, found {len(paths)}")

    frames = [pd.read_csv(path, dtype=str).fillna("") for path in paths]
    pairs = pd.concat(frames, ignore_index=True)
    if pairs.duplicated(["reaction_id", "UniprotID"]).any():
        raise RuntimeError("minimal scorer input contains duplicate pair keys")
    pairs.to_csv(OUT_PAIRS, index=False)

    needed_uids = set(pairs["UniprotID"].astype(str))
    needed_sequences = set(pairs["sequence"].astype(str))
    print(
        f"minimal pairs={len(pairs)} unique_uids={len(needed_uids)} "
        f"unique_sequences={len(needed_sequences)}",
        flush=True,
    )
    del frames
    gc.collect()

    print(f"loading aggregate GVP: {SOURCE_GVP}", flush=True)
    gvp = torch.load(
        SOURCE_GVP,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    gvp_subset = {uid: value for uid, value in gvp.items() if str(uid) in needed_uids}
    print(f"GVP subset {len(gvp_subset)}/{len(gvp)}", flush=True)
    torch.save(gvp_subset, OUT_GVP)
    gvp_total = len(gvp)
    gvp_subset_n = len(gvp_subset)
    del gvp, gvp_subset
    gc.collect()

    print(f"loading aggregate pocket ESM: {SOURCE_NODE}", flush=True)
    node = torch.load(
        SOURCE_NODE,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    node_subset = {uid: value for uid, value in node.items() if str(uid) in needed_uids}
    print(f"node subset {len(node_subset)}/{len(node)}", flush=True)
    torch.save(node_subset, OUT_NODE)
    node_total = len(node)
    node_subset_n = len(node_subset)
    del node, node_subset
    gc.collect()

    print(f"loading aggregate protein mean ESM: {SOURCE_MEAN}", flush=True)
    with open(SOURCE_MEAN, "rb") as handle:
        mean = pkl.load(handle)
    mean_subset = {
        sequence: value
        for sequence, value in mean.items()
        if str(sequence) in needed_sequences
    }
    print(f"mean subset {len(mean_subset)}/{len(mean)}", flush=True)
    with open(OUT_MEAN, "wb") as handle:
        pkl.dump(mean_subset, handle, protocol=pkl.HIGHEST_PROTOCOL)
    mean_total = len(mean)
    mean_subset_n = len(mean_subset)
    del mean, mean_subset
    gc.collect()

    audit = {
        "schema": "bridge-layered-v4-r2e-cage-minimal-cache",
        "pairs": int(len(pairs)),
        "unique_uids": int(len(needed_uids)),
        "unique_sequences": int(len(needed_sequences)),
        "gvp": {"source": gvp_total, "subset": gvp_subset_n},
        "node": {"source": node_total, "subset": node_subset_n},
        "mean": {"source": mean_total, "subset": mean_subset_n},
        "outputs": {
            "pairs": str(OUT_PAIRS.relative_to(ROOT)),
            "gvp": str(OUT_GVP.relative_to(ROOT)),
            "node": str(OUT_NODE.relative_to(ROOT)),
            "mean": str(OUT_MEAN.relative_to(ROOT)),
        },
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
