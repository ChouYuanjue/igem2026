from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.csgraph import connected_components, shortest_path

from projects.active.fibre.geometry.multiscale import information_product_affinity

ROOT=Path(__file__).resolve().parents[4]
MOTIF=ROOT/"data/terpene_family_aware_motif_coordinates_v1"
CATALYTIC=ROOT/"data/terpene_catalytic_consensus_geometry_v1"
OUT=ROOT/"data/terpene_mechanistic_chart_geometry_v1"
COORDS=("typeI_aspartate","nse_dte","dxdd","qw")


def chordal(x: np.ndarray) -> np.ndarray:
    a=np.asarray(x,dtype=np.float64)
    norm=np.linalg.norm(a,axis=1,keepdims=True)
    if np.any(norm<=1e-12):
        raise ValueError("mechanistic embeddings must be non-zero on available rows")
    a=a/norm
    sim=np.clip(a@a.T,-1.0,1.0)
    return np.sqrt(np.maximum(2.0-2.0*sim,0.0))


def intrinsic_geodesic(affinity):
    w=csr_matrix(affinity,dtype=np.float64).maximum(
        csr_matrix(affinity,dtype=np.float64).T
    ).tocsr()
    w.setdiag(0);w.eliminate_zeros()
    coo=w.tocoo();upper=coo.row<coo.col
    raw=np.sqrt(np.maximum(-np.log(np.clip(coo.data[upper],1e-300,1.0)),1e-12))
    ell=float(np.median(raw))
    if not np.isfinite(ell) or ell<=0:
        raise RuntimeError("invalid mechanistic characteristic length")
    edge=np.sqrt(np.maximum(-np.log(np.clip(w.data,1e-300,1.0)),1e-12))/ell
    graph=csr_matrix((edge,w.indices,w.indptr),shape=w.shape)
    ncomp,_=connected_components(graph,directed=False)
    if ncomp!=1:
        raise RuntimeError(f"mechanistic coordinate disconnected: {ncomp}")
    return (
        np.asarray(shortest_path(graph,directed=False,unweighted=False),dtype=np.float32),
        ell,
        int(ncomp),
    )


def as_bool(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.lower().isin({"1","true","yes"}).to_numpy(bool)


def main() -> None:
    ids=pd.read_csv(MOTIF/"protein_ids.csv",dtype=str).fillna("")
    audit=pd.read_csv(MOTIF/"audit.csv",dtype=str).fillna("")
    if ids.protein_id.astype(str).tolist()!=audit.protein_id.astype(str).tolist():
        audit=audit.set_index("protein_id").loc[ids.protein_id.astype(str)].reset_index()
    reaction_ids=pd.read_csv(CATALYTIC/"reaction_ids.csv",dtype=str).fillna("")

    OUT.mkdir(parents=True,exist_ok=True)
    ids.to_csv(OUT/"protein_ids.csv",index=False)
    reaction_ids.to_csv(OUT/"reaction_ids.csv",index=False)

    views={}
    for name in COORDS:
        available=np.load(MOTIF/f"{name}_available.npy").astype(bool)
        applicable=as_bool(audit[f"{name}_applicable"])
        if len(available)!=len(ids) or len(applicable)!=len(ids):
            raise RuntimeError(f"{name}: protein axis mismatch")
        if np.any(available & ~applicable):
            raise RuntimeError(f"{name}: observed motif outside family-applicable chart")
        rows=np.flatnonzero(available)
        if len(rows)<2:
            raise RuntimeError(f"{name}: insufficient observed proteins")
        emb=np.load(MOTIF/f"{name}_embeddings.npy",mmap_mode="r")
        d=chordal(np.asarray(emb[rows],dtype=np.float32))
        graph,diag=information_product_affinity(
            [d],[np.ones(len(rows),dtype=bool)]
        )
        geo,ell,ncomp=intrinsic_geodesic(graph)
        save_npz(OUT/f"protein_{name}_affinity.npz",graph)
        np.save(OUT/f"protein_{name}_geodesic.npy",geo)
        np.save(OUT/f"protein_{name}_global_rows.npy",rows.astype(np.int64))
        np.save(OUT/f"protein_{name}_applicable.npy",applicable)
        np.save(OUT/f"protein_{name}_available.npy",available)
        views[name]={
            **diag,
            "applicable_count":int(applicable.sum()),
            "available_count":int(available.sum()),
            "connected_components":ncomp,
            "characteristic_length":float(ell),
        }

    manifest={
        "version":"terpene-mechanistic-chart-geometry-v1",
        "scientific_role":"third-resolution FIBRE mechanism charts nested inside observed catalytic cells",
        "reaction_geometry_source":str(CATALYTIC/"reaction_local_geodesic.npy"),
        "protein_coordinates":list(COORDS),
        "protein_count_total":int(len(ids)),
        "reaction_count":int(len(reaction_ids)),
        "coordinate_graphs":views,
        "chart_definition":"bit mask of family-applicable motif coordinates; different masks are incomparable local charts",
        "comparison_contract":"within one observed catalytic parent and one chart, motif-specific FIBRE defects define only Pareto dominance; no scalar fusion weight and no canonical total-rank change",
        "missing_policy":"applicable but unobserved motif leaves the chart comparison unresolved; non-applicable motif is outside the chart, never negative evidence",
        "labels_used":False,
    }
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
