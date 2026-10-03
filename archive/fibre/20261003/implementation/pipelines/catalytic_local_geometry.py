from __future__ import annotations

import argparse
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
MOTIFS=ROOT/"data/terpene_family_aware_motif_coordinates_v1"
OUT=ROOT/"data/terpene_catalytic_local_geometry_v1"



def intrinsic_geodesic(affinity: csr_matrix) -> tuple[np.ndarray,float]:
    w=csr_matrix(affinity,dtype=np.float64).maximum(csr_matrix(affinity,dtype=np.float64).T).tocsr()
    w.setdiag(0);w.eliminate_zeros()
    coo=w.tocoo(); upper=coo.row<coo.col
    edge=np.sqrt(np.maximum(-np.log(np.clip(coo.data[upper],1e-300,1.0)),1e-12))
    if not len(edge):
        raise RuntimeError("local affinity has no undirected edges")
    ell=float(np.median(edge))
    lengths=np.sqrt(np.maximum(-np.log(np.clip(w.data,1e-300,1.0)),1e-12))/ell
    graph=csr_matrix((lengths,w.indices,w.indptr),shape=w.shape)
    dist=np.asarray(shortest_path(graph,directed=False),dtype=np.float64)
    if not np.all(np.isfinite(dist)):
        raise RuntimeError("local affinity graph is disconnected")
    return dist,ell


def chordal_distance(embeddings: np.ndarray, available: np.ndarray) -> np.ndarray:
    x=np.asarray(embeddings,dtype=np.float64)
    a=np.asarray(available,dtype=bool).reshape(-1)
    norm=np.linalg.norm(x,axis=1,keepdims=True)
    good=a&(norm[:,0]>1e-12)
    xn=np.zeros_like(x)
    xn[good]=x[good]/norm[good]
    sim=np.clip(xn@xn.T,-1.0,1.0)
    d=np.sqrt(np.maximum(2.0-2.0*sim,0.0))
    d[~(good[:,None]&good[None,:])]=np.inf
    np.fill_diagonal(d,0.0)
    return d


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--protein-mode",choices=("structure","structure_and_motifs"),default="structure_and_motifs")
    ap.add_argument("--output",type=Path,default=OUT)
    args=ap.parse_args()
    out_dir=Path(args.output)
    reaction_ids=pd.read_csv(RLOCAL/"reaction_ids.csv",dtype=str).fillna("")
    protein_ids=pd.read_csv(PLOCAL/"protein_ids.csv",dtype=str).fillna("")

    reaction_distances=[
        np.load(RLOCAL/"center_transition_wasserstein_distance.npy").astype(np.float64),
        np.load(RLOCAL/"center_token_jaccard_distance.npy").astype(np.float64),
    ]
    reaction_available=[
        np.load(RLOCAL/"center_transition_wasserstein_available.npy").astype(bool),
        np.load(RLOCAL/"center_token_jaccard_available.npy").astype(bool),
    ]
    rgraph,rdiag=information_product_affinity(
        reaction_distances,reaction_available
    )

    protein_specs=[
        ("pocket_local_esmc",
         PLOCAL/"pocket_local_esmc_chordal_distance.npy",
         PLOCAL/"pocket_local_esmc_available.npy"),
        ("pocket_3di",
         PLOCAL/"pocket_3di_diffusion_distance.npy",
         PLOCAL/"pocket_3di_relational_available.npy"),
        ("pocket_ot",
         PLOCAL/"pocket_ot_diffusion_distance.npy",
         PLOCAL/"pocket_ot_relational_available.npy"),
    ]
    protein_names=[x[0] for x in protein_specs]
    protein_full_distances=[
        np.load(dpath).astype(np.float64) for _,dpath,_ in protein_specs
    ]
    protein_available=[
        np.load(apath).astype(bool) for _,_,apath in protein_specs
    ]
    motif_names=["typeI_aspartate","nse_dte","dxdd","qw"]
    if args.protein_mode=="structure_and_motifs":
        for motif_name in motif_names:
            emb=np.load(MOTIFS/f"{motif_name}_embeddings.npy",mmap_mode="r")
            avail=np.load(MOTIFS/f"{motif_name}_available.npy").astype(bool)
            protein_names.append(f"{motif_name}_local_esmc")
            protein_full_distances.append(chordal_distance(emb,avail))
            protein_available.append(avail)

    local_observed=np.logical_or.reduce(protein_available)
    local_rows=np.flatnonzero(local_observed)
    if not len(local_rows):
        raise RuntimeError("no protein has catalytic-local observations")
    protein_distances=[
        d[np.ix_(local_rows,local_rows)] for d in protein_full_distances
    ]
    protein_masks=[a[local_rows] for a in protein_available]
    pgraph,pdiag=information_product_affinity(
        protein_distances,protein_masks
    )

    rncomp,rlabels=connected_components(rgraph,directed=False)
    pncomp,plabels=connected_components(pgraph,directed=False)
    if int(rncomp)!=1 or int(pncomp)!=1:
        raise RuntimeError("catalytic-local factor graphs must be connected")
    reaction_geodesic,reaction_ell=intrinsic_geodesic(rgraph)
    protein_geodesic,protein_ell=intrinsic_geodesic(pgraph)
    pocket3di_global=np.load(PLOCAL/"pocket_3di_relational_available.npy").astype(bool)
    protein_order_authority=pocket3di_global[local_rows]

    out_dir.mkdir(parents=True,exist_ok=True)
    save_npz(out_dir/"reaction_local_affinity.npz",rgraph)
    save_npz(out_dir/"protein_local_affinity.npz",pgraph)
    reaction_ids.to_csv(out_dir/"reaction_ids.csv",index=False)
    protein_ids.iloc[local_rows].reset_index(drop=True).to_csv(
        out_dir/"protein_ids.csv",index=False
    )
    np.save(out_dir/"protein_global_rows.npy",local_rows.astype(np.int64))
    np.save(out_dir/"reaction_local_geodesic.npy",reaction_geodesic.astype(np.float32))
    np.save(out_dir/"protein_local_geodesic.npy",protein_geodesic.astype(np.float32))
    np.save(out_dir/"protein_order_authority.npy",protein_order_authority.astype(bool))
    manifest={
        "version":("terpene-catalytic-pocket-geometry-v1" if args.protein_mode=="structure" else "terpene-catalytic-local-geometry-v1"),
        "protein_mode":args.protein_mode,
        "scientific_role":"second-resolution FIBRE factor geometry; same correspondence operator as coarse field",
        "reaction_views":[
            "reaction_center_transition_wasserstein",
            "reaction_center_transition_token",
        ],
        "protein_views":protein_names,
        "reaction_count":int(len(reaction_ids)),
        "protein_count_total":int(len(protein_ids)),
        "protein_count_with_local_observation":int(len(local_rows)),
        "mechanistic_coordinate_manifest":json.loads((MOTIFS/"manifest.json").read_text())["version"],
        "mechanistic_coordinates_enter_ordering_geometry":bool(args.protein_mode=="structure_and_motifs"),
        "missing_local_observation_is_negative_evidence":False,
        "fusion_rule":"information-product affinity with self-tuned distances and intrinsic entropy-deficit view precision; no fitted modality weights",
        "reaction_graph":{**rdiag,"connected_components":int(rncomp),"characteristic_length":reaction_ell},
        "protein_graph":{**pdiag,"connected_components":int(pncomp),"characteristic_length":protein_ell,"order_authority_count":int(protein_order_authority.sum())},
        "ranking_contract":{
            "coarse_field_can_be_overturned":False,
            "local_field_role":"refine only one coarse numerical level; otherwise remains latent local coordinate",
            "family_motif_role":"chart applicability and mechanistic interpretation, never scalar bonus",
        },
    }
    (out_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
