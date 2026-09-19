from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.csgraph import connected_components, shortest_path

from projects.active.fibre.geometry.multiscale import information_product_affinity

ROOT=Path(__file__).resolve().parents[4]
RLOCAL=ROOT/"data/terpene_multiresolution_reaction_geometry_v2"
PLOCAL=ROOT/"data/terpene_multiresolution_protein_geometry_v4"
OUT=ROOT/"data/terpene_catalytic_consensus_geometry_v1"


def intrinsic_geodesic(affinity):
    w=csr_matrix(affinity,dtype=np.float64).maximum(csr_matrix(affinity,dtype=np.float64).T).tocsr()
    w.setdiag(0);w.eliminate_zeros()
    coo=w.tocoo();upper=coo.row<coo.col
    raw=np.sqrt(np.maximum(-np.log(np.clip(coo.data[upper],1e-300,1.0)),1e-12))
    ell=float(np.median(raw))
    if not np.isfinite(ell) or ell<=0:
        raise RuntimeError("invalid local characteristic length")
    edge=np.sqrt(np.maximum(-np.log(np.clip(w.data,1e-300,1.0)),1e-12))/ell
    graph=csr_matrix((edge,w.indices,w.indptr),shape=w.shape)
    ncomp,_=connected_components(graph,directed=False)
    if ncomp!=1:
        raise RuntimeError(f"local geometry disconnected: {ncomp}")
    return np.asarray(shortest_path(graph,directed=False,unweighted=False),dtype=np.float32),ell


def main() -> None:
    reaction_ids=pd.read_csv(RLOCAL/"reaction_ids.csv",dtype=str).fillna("")
    protein_ids=pd.read_csv(PLOCAL/"protein_ids.csv",dtype=str).fillna("")

    rdist=[
        np.load(RLOCAL/"center_transition_wasserstein_distance.npy").astype(np.float64),
        np.load(RLOCAL/"center_token_jaccard_distance.npy").astype(np.float64),
    ]
    ravail=[
        np.load(RLOCAL/"center_transition_wasserstein_available.npy").astype(bool),
        np.load(RLOCAL/"center_token_jaccard_available.npy").astype(bool),
    ]
    rgraph,rdiag=information_product_affinity(rdist,ravail)

    specs=[
        (
            "pocket_local_esmc",
            PLOCAL/"pocket_local_esmc_chordal_distance.npy",
            PLOCAL/"pocket_local_esmc_available.npy",
        ),
        (
            "pocket_3di",
            PLOCAL/"pocket_3di_diffusion_distance.npy",
            PLOCAL/"pocket_3di_relational_available.npy",
        ),
        (
            "pocket_ot",
            PLOCAL/"pocket_ot_diffusion_distance.npy",
            PLOCAL/"pocket_ot_relational_available.npy",
        ),
    ]
    avail=[np.load(a).astype(bool) for _,_,a in specs]
    common=np.logical_and.reduce(avail)
    rows=np.flatnonzero(common)
    if not len(rows):
        raise RuntimeError("no protein has all pocket consensus coordinates")

    OUT.mkdir(parents=True,exist_ok=True)
    save_npz(OUT/"reaction_local_affinity.npz",rgraph)
    rgeo,rell=intrinsic_geodesic(rgraph)
    np.save(OUT/"reaction_local_geodesic.npy",rgeo)
    reaction_ids.to_csv(OUT/"reaction_ids.csv",index=False)
    protein_ids.iloc[rows].reset_index(drop=True).to_csv(
        OUT/"protein_ids.csv",index=False
    )
    np.save(OUT/"protein_global_rows.npy",rows.astype(np.int64))

    views={}
    for name,dpath,_ in specs:
        d=np.load(dpath).astype(np.float64)[np.ix_(rows,rows)]
        mask=np.ones(len(rows),dtype=bool)
        graph,diag=information_product_affinity([d],[mask])
        save_npz(OUT/f"protein_{name}_affinity.npz",graph)
        geo,ell=intrinsic_geodesic(graph)
        np.save(OUT/f"protein_{name}_geodesic.npy",geo)
        ncomp,_=connected_components(graph,directed=False)
        views[name]={**diag,"connected_components":int(ncomp),"characteristic_length":float(ell)}

    rncomp,_=connected_components(rgraph,directed=False)
    manifest={
        "version":"terpene-catalytic-consensus-geometry-v1",
        "scientific_role":"second-resolution FIBRE coordinates for weight-free pocket consensus inside coarse correspondence levels",
        "reaction_views":[
            "reaction_center_transition_wasserstein",
            "reaction_center_transition_token",
        ],
        "protein_coordinates":[x[0] for x in specs],
        "protein_count_total":int(len(protein_ids)),
        "protein_count_common_complete":int(len(rows)),
        "reaction_count":int(len(reaction_ids)),
        "reaction_graph":{**rdiag,"connected_components":int(rncomp),"characteristic_length":float(rell)},
        "protein_coordinate_graphs":views,
        "comparison_contract":"inside one coarse numerical level, local coordinates define only Pareto dominance; no scalar modality fusion weight",
        "missing_policy":"all three pocket coordinates must be observed across the compared coarse level; otherwise the coarse level remains unresolved",
        "motif_role":"family-aware catalytic motifs are a finer mechanistic stratum, not an order-bearing pocket coordinate",
        "labels_used":False,
    }
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+chr(10))
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
