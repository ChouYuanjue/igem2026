from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz

from projects.active.fibre.geometry.multiscale import (
    fixed_topology_information_refinement_affinity,
    intrinsic_view_information,
)

ROOT=Path(__file__).resolve().parents[4]
GLOBAL=ROOT/"data/terpene_multiresolution_reaction_geometry_v1"
CENTER=ROOT/"data/terpene_multiresolution_reaction_geometry_v2"
OUT=ROOT/"data/terpene_reaction_tangent_information_geometry_v1"


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    global_ids=pd.read_csv(GLOBAL/"reaction_ids.csv",dtype=str).fillna("")
    center_ids=pd.read_csv(CENTER/"reaction_ids.csv",dtype=str).fillna("")
    if not global_ids.equals(center_ids):
        raise RuntimeError("global and center reaction registries are not aligned")

    base=load_npz(GLOBAL/"partial_pullback_affinity.npz").tocsr()
    specs=[
        (
            "center_transition_wasserstein",
            CENTER/"center_transition_wasserstein_distance.npy",
            CENTER/"center_transition_wasserstein_available.npy",
        ),
        (
            "center_transition_token",
            CENTER/"center_token_jaccard_distance.npy",
            CENTER/"center_token_jaccard_available.npy",
        ),
    ]
    distances=[];availabilities=[];information=[]
    for name,dpath,apath in specs:
        d=np.load(dpath).astype(np.float64)
        a=np.load(apath).astype(bool)
        q,_=intrinsic_view_information(d,a)
        distances.append(d);availabilities.append(a);information.append(q)

    refined,diag=fixed_topology_information_refinement_affinity(
        base,distances,availabilities,information,
    )

    OUT.mkdir(parents=True,exist_ok=True)
    save_npz(OUT/"partial_pullback_affinity.npz",refined)
    global_ids.to_csv(OUT/"reaction_ids.csv",index=False)
    for (name,_,_),q in zip(specs,information):
        np.save(OUT/f"{name}_intrinsic_information.npy",q.astype(np.float32))

    manifest={
        "version":"terpene-reaction-tangent-information-geometry-v1",
        "semantic_role":"label-free catalytic-local tangent refinement of the immutable global reaction atlas",
        "reaction_count":int(len(global_ids)),
        "base_topology":"exact edge set of terpene_multiresolution_reaction_geometry_v1/partial_pullback_affinity.npz",
        "refinement_views":[x[0] for x in specs],
        "reliability":"per-node normalized entropy deficit of each self-tuned refinement view; no labels or outcome metrics",
        "edge_rule":"base negative-log affinity energy plus reliability-weighted mean local tangent energy on existing edges only",
        "guarantees":{
            "new_edges_created":False,
            "base_edges_removed":False,
            "missing_refinement_is_neutral":True,
            "duplicate_identical_refinement_is_idempotent":True,
            "refinement_can_shorten_base_edge":False,
        },
        "diagnostics":diag,
        "information":{
            name:{
                "available":int(a.sum()),
                "mean":float(np.mean(q[a])) if np.any(a) else 0.0,
                "median":float(np.median(q[a])) if np.any(a) else 0.0,
                "p90":float(np.quantile(q[a],.9)) if np.any(a) else 0.0,
            }
            for (name,_,_),a,q in zip(specs,availabilities,information)
        },
        "labels_used":False,
        "input_sha256":{
            "global_manifest":sha256(GLOBAL/"manifest.json"),
            "base_affinity":sha256(GLOBAL/"partial_pullback_affinity.npz"),
            "center_manifest":sha256(CENTER/"manifest.json"),
        },
    }
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
