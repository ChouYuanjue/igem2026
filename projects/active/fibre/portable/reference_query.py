from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from projects.active.fibre.geometry.correspondence import (
    protein_to_reaction_section_from_support_distances,
    reaction_to_protein_section_from_support_distances,
)
from projects.active.fibre.geometry.extension import (
    QueryAttachment,
    attach_query_to_reference,
    query_geodesic_to_reference,
)


class PortableReferenceBundle:
    def __init__(self, root: str | Path):
        self.root=Path(root)
        self.proteins=pd.read_csv(self.root/"proteins.csv",dtype=str).fillna("")
        self.reactions=pd.read_csv(self.root/"reactions.csv",dtype=str).fillna("")
        self.pids=self.proteins.protein_id.astype(str).tolist()
        self.rids=self.reactions.reaction_id.astype(str).tolist()
        self.pi={x:i for i,x in enumerate(self.pids)}
        self.ri={x:i for i,x in enumerate(self.rids)}
        self.ps=np.load(self.root/"protein_support_indices.npy").astype(np.int64)
        self.rs=np.load(self.root/"reaction_support_indices.npy").astype(np.int64)
        self.pp=np.load(self.root/"positive_pair_support_indices.npy").astype(np.int64)
        self.pgeo=np.load(self.root/"protein_support_geodesic.npy",mmap_mode="r")
        self.rgeo=np.load(self.root/"reaction_support_geodesic.npy",mmap_mode="r")
        self.manifest=json.loads((self.root/"manifest.json").read_text())

    def _external_factor_geodesic_views(
        self,
        factor: str,
        query_views: dict[str,np.ndarray],
        query_distances: dict[str,np.ndarray] | None=None,
    ) -> np.ndarray:
        if factor not in {"protein","reaction"}:
            raise ValueError("factor must be protein or reaction")
        prefix="protein" if factor=="protein" else "reaction"
        graph_path=self.root/f"{prefix}_length_graph.npz"
        view_root=self.root/f"{prefix}_views"
        declared=[
            str(x["name"])
            for x in self.manifest["factor_geometry"][factor].get("views",[])
        ]
        if not declared or not view_root.is_dir():
            raise ValueError(
                f"{factor} factor does not expose feature-compatible OOS attachment; "
                "custom precomputed affinities require a matching attachment implementation"
            )
        query_distances=dict(query_distances or {})
        unknown=sorted((set(query_views)|set(query_distances))-set(declared))
        if unknown:
            raise ValueError(f"{factor} query contains views not present in reference: {unknown}")
        if not query_views and not query_distances:
            raise ValueError(f"{factor} query must provide at least one observed view")

        cross_distances=[]; query_available=[]; reference_available=[]; reference_scales=[]
        n=None
        for name in declared:
            available=np.load(view_root/f"{name}_available.npy").astype(bool)
            scales=np.load(view_root/f"{name}_scales.npy",mmap_mode="r")
            if n is None:
                n=len(available)
            if len(available)!=n or len(scales)!=n:
                raise ValueError(f"{factor} view {name} is not aligned in the bundle")
            feature_path=view_root/f"{name}_features.npy"
            distance_path=view_root/f"{name}_distance.npy"
            if name in query_views and name in query_distances:
                raise ValueError(f"{factor} query view {name} supplied as both feature and distance")
            if name in query_distances:
                if not distance_path.is_file():
                    raise ValueError(
                        f"{factor} view {name} is feature-based; query must provide its feature vector"
                    )
                cross=np.asarray(query_distances[name],dtype=np.float64).reshape(-1)
                if len(cross)!=n:
                    raise ValueError(
                        f"{factor} query distance view {name} must contain {n} distances"
                    )
                observed=cross[available]
                if not np.all(np.isfinite(observed)) or np.any(observed<0):
                    raise ValueError(
                        f"{factor} query distance view {name} must be finite/non-negative "
                        "where the reference coordinate is available"
                    )
                cross_distances.append(cross)
                query_available.append(True)
                reference_available.append(available)
                reference_scales.append(np.asarray(scales))
                continue
            if name not in query_views:
                cross_distances.append(np.zeros(n,dtype=np.float64))
                query_available.append(False)
                reference_available.append(available)
                reference_scales.append(np.asarray(scales))
                continue
            if not feature_path.is_file():
                raise ValueError(
                    f"{factor} view {name} is distance-based; query must provide cross distances"
                )
            reference=np.load(feature_path,mmap_mode="r")
            q=np.asarray(query_views[name],dtype=np.float32).reshape(-1)
            if reference.ndim!=2 or q.shape[0]!=reference.shape[1]:
                raise ValueError(
                    f"{factor} query view {name} dimension {q.shape[0]} does not match "
                    f"reference dimension {reference.shape[1] if reference.ndim==2 else 'invalid'}"
                )
            norm=float(np.linalg.norm(q))
            if not np.isfinite(norm) or norm<=1e-12:
                raise ValueError(f"{factor} query view {name} must be a finite non-zero vector")
            q=q/norm
            similarity=np.clip(np.asarray(reference,dtype=np.float32)@q,-1.0,1.0)
            cross_distances.append(np.sqrt(np.maximum(2.0-2.0*similarity,0.0)))
            query_available.append(True)
            reference_available.append(available)
            reference_scales.append(np.asarray(scales))

        graph_k=int(
            self.manifest["factor_geometry"][factor].get("graph_k")
            or max(1,int(np.ceil(np.sqrt(int(n or 0)))))
        )
        attachment=attach_query_to_reference(
            cross_distances,query_available,reference_available,reference_scales,
            k=graph_k,
        )
        ell=float(self.manifest["factor_geometry"][factor]["characteristic_length"])
        attachment=QueryAttachment(
            reference_indices=attachment.reference_indices,
            edge_lengths=np.asarray(attachment.edge_lengths,dtype=np.float64)/ell,
            query_view_scales=attachment.query_view_scales,
            observed_view_count=attachment.observed_view_count,
            graph_k=attachment.graph_k,
        )
        return query_geodesic_to_reference(load_npz(graph_path).tocsr(),attachment)

    def _external_factor_geodesic(self, factor: str, feature: np.ndarray) -> np.ndarray:
        return self._external_factor_geodesic_views(factor,{"global":feature})

    def enzyme_defect(self,reaction_id: str) -> np.ndarray:
        q=self.ri[str(reaction_id)]
        sec=reaction_to_protein_section_from_support_distances(
            np.square(np.asarray(self.rgeo[:,q],dtype=np.float64)),
            np.square(np.asarray(self.pgeo.T,dtype=np.float64)),
            self.pp,
            query_index=q,
            candidate_indices=np.arange(len(self.pids),dtype=np.int64),
        )
        return np.asarray(sec.defect,dtype=np.float64)

    def reaction_defect(self,protein_id: str) -> np.ndarray:
        q=self.pi[str(protein_id)]
        sec=protein_to_reaction_section_from_support_distances(
            np.square(np.asarray(self.pgeo[:,q],dtype=np.float64)),
            np.square(np.asarray(self.rgeo.T,dtype=np.float64)),
            self.pp,
            query_index=q,
            candidate_indices=np.arange(len(self.rids),dtype=np.int64),
        )
        return np.asarray(sec.defect,dtype=np.float64)

    def enzyme_defect_from_reaction_views(
        self,
        views: dict[str,np.ndarray],
        *,
        distances: dict[str,np.ndarray] | None=None,
    ) -> np.ndarray:
        qgeo=self._external_factor_geodesic_views("reaction",views,distances)
        sec=reaction_to_protein_section_from_support_distances(
            np.square(np.asarray(qgeo[self.rs],dtype=np.float64)),
            np.square(np.asarray(self.pgeo.T,dtype=np.float64)),
            self.pp,
            query_index=-1,
            candidate_indices=np.arange(len(self.pids),dtype=np.int64),
        )
        return np.asarray(sec.defect,dtype=np.float64)

    def reaction_defect_from_protein_views(
        self,
        views: dict[str,np.ndarray],
        *,
        distances: dict[str,np.ndarray] | None=None,
    ) -> np.ndarray:
        qgeo=self._external_factor_geodesic_views("protein",views,distances)
        sec=protein_to_reaction_section_from_support_distances(
            np.square(np.asarray(qgeo[self.ps],dtype=np.float64)),
            np.square(np.asarray(self.rgeo.T,dtype=np.float64)),
            self.pp,
            query_index=-1,
            candidate_indices=np.arange(len(self.rids),dtype=np.int64),
        )
        return np.asarray(sec.defect,dtype=np.float64)

    def rank_enzymes(self,reaction_id: str,top_k: int=10) -> list[dict]:
        q=self.ri[str(reaction_id)]
        sec=reaction_to_protein_section_from_support_distances(
            np.square(np.asarray(self.rgeo[:,q],dtype=np.float64)),
            np.square(np.asarray(self.pgeo.T,dtype=np.float64)),
            self.pp,
            query_index=q,
            candidate_indices=np.arange(len(self.pids),dtype=np.int64),
        )
        order=np.lexsort((np.asarray(self.pids,dtype=str),sec.defect))[:int(top_k)]
        return [
            {
                "rank":rank,
                "protein_id":self.pids[int(i)],
                "correspondence_defect":float(sec.defect[int(i)]),
            }
            for rank,i in enumerate(order,1)
        ]

    def rank_reactions(self,protein_id: str,top_k: int=10) -> list[dict]:
        q=self.pi[str(protein_id)]
        sec=protein_to_reaction_section_from_support_distances(
            np.square(np.asarray(self.pgeo[:,q],dtype=np.float64)),
            np.square(np.asarray(self.rgeo.T,dtype=np.float64)),
            self.pp,
            query_index=q,
            candidate_indices=np.arange(len(self.rids),dtype=np.int64),
        )
        order=np.lexsort((np.asarray(self.rids,dtype=str),sec.defect))[:int(top_k)]
        return [
            {
                "rank":rank,
                "reaction_id":self.rids[int(i)],
                "correspondence_defect":float(sec.defect[int(i)]),
            }
            for rank,i in enumerate(order,1)
        ]

    def rank_enzymes_from_reaction_feature(
        self, feature: np.ndarray, top_k: int=10
    ) -> list[dict]:
        qgeo=self._external_factor_geodesic("reaction",feature)
        sec=reaction_to_protein_section_from_support_distances(
            np.square(np.asarray(qgeo[self.rs],dtype=np.float64)),
            np.square(np.asarray(self.pgeo.T,dtype=np.float64)),
            self.pp,
            query_index=-1,
            candidate_indices=np.arange(len(self.pids),dtype=np.int64),
        )
        order=np.lexsort((np.asarray(self.pids,dtype=str),sec.defect))[:int(top_k)]
        return [
            {
                "rank":rank,
                "protein_id":self.pids[int(i)],
                "correspondence_defect":float(sec.defect[int(i)]),
                "query_mode":"out_of_sample_frozen_reference",
            }
            for rank,i in enumerate(order,1)
        ]

    def rank_reactions_from_protein_feature(
        self, feature: np.ndarray, top_k: int=10
    ) -> list[dict]:
        qgeo=self._external_factor_geodesic("protein",feature)
        sec=protein_to_reaction_section_from_support_distances(
            np.square(np.asarray(qgeo[self.ps],dtype=np.float64)),
            np.square(np.asarray(self.rgeo.T,dtype=np.float64)),
            self.pp,
            query_index=-1,
            candidate_indices=np.arange(len(self.rids),dtype=np.int64),
        )
        order=np.lexsort((np.asarray(self.rids,dtype=str),sec.defect))[:int(top_k)]
        return [
            {
                "rank":rank,
                "reaction_id":self.rids[int(i)],
                "correspondence_defect":float(sec.defect[int(i)]),
                "query_mode":"out_of_sample_frozen_reference",
            }
            for rank,i in enumerate(order,1)
        ]

    def rank_enzymes_from_reaction_views(
        self, views: dict[str,np.ndarray], top_k: int=10,
        distances: dict[str,np.ndarray] | None=None,
    ) -> list[dict]:
        qgeo=self._external_factor_geodesic_views("reaction",views,distances)
        sec=reaction_to_protein_section_from_support_distances(
            np.square(np.asarray(qgeo[self.rs],dtype=np.float64)),
            np.square(np.asarray(self.pgeo.T,dtype=np.float64)),
            self.pp,
            query_index=-1,
            candidate_indices=np.arange(len(self.pids),dtype=np.int64),
        )
        order=np.lexsort((np.asarray(self.pids,dtype=str),sec.defect))[:int(top_k)]
        return [
            {
                "rank":rank,
                "protein_id":self.pids[int(i)],
                "correspondence_defect":float(sec.defect[int(i)]),
                "query_mode":"out_of_sample_frozen_reference",
                "observed_views":sorted(set(views)|set(distances or {})),
            }
            for rank,i in enumerate(order,1)
        ]

    def rank_reactions_from_protein_views(
        self, views: dict[str,np.ndarray], top_k: int=10,
        distances: dict[str,np.ndarray] | None=None,
    ) -> list[dict]:
        qgeo=self._external_factor_geodesic_views("protein",views,distances)
        sec=protein_to_reaction_section_from_support_distances(
            np.square(np.asarray(qgeo[self.ps],dtype=np.float64)),
            np.square(np.asarray(self.rgeo.T,dtype=np.float64)),
            self.pp,
            query_index=-1,
            candidate_indices=np.arange(len(self.rids),dtype=np.int64),
        )
        order=np.lexsort((np.asarray(self.rids,dtype=str),sec.defect))[:int(top_k)]
        return [
            {
                "rank":rank,
                "reaction_id":self.rids[int(i)],
                "correspondence_defect":float(sec.defect[int(i)]),
                "query_mode":"out_of_sample_frozen_reference",
                "observed_views":sorted(set(views)|set(distances or {})),
            }
            for rank,i in enumerate(order,1)
        ]


def load_query_feature(path: str | Path) -> np.ndarray:
    p=Path(path)
    if p.suffix.lower()==".npy":
        x=np.load(p)
    else:
        frame=pd.read_csv(p)
        numeric=frame.select_dtypes(include=[np.number])
        if len(frame)!=1 or numeric.shape[1]==0:
            raise ValueError("query feature CSV must contain exactly one row of numeric features")
        x=numeric.to_numpy()
    x=np.asarray(x,dtype=np.float32)
    if x.ndim==2 and x.shape[0]==1:
        x=x[0]
    if x.ndim!=1:
        raise ValueError("query feature must be one vector")
    return x


def parse_query_views(values: list[str]) -> dict[str,np.ndarray]:
    out={}
    for value in values:
        if "=" not in value:
            raise ValueError(f"query view must be NAME=PATH: {value}")
        name,path=value.split("=",1)
        name=name.strip(); path=path.strip()
        if not name or not path or name in out:
            raise ValueError(f"invalid or duplicated query view: {value}")
        out[name]=load_query_feature(path)
    return out


def parse_query_distances(values: list[str]) -> dict[str,np.ndarray]:
    out={}
    for value in values:
        if "=" not in value:
            raise ValueError(f"query distance must be NAME=PATH: {value}")
        name,path=value.split("=",1)
        name=name.strip(); path=path.strip()
        if not name or not path or name in out:
            raise ValueError(f"invalid or duplicated query distance: {value}")
        x=np.asarray(np.load(path),dtype=np.float64)
        if x.ndim==2 and 1 in x.shape:
            x=x.reshape(-1)
        if x.ndim!=1:
            raise ValueError("query distance file must contain one vector")
        out[name]=x
    return out


def main() -> None:
    ap=argparse.ArgumentParser(description="Query a generic portable FIBRE reference bundle.")
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--reaction-id")
    ap.add_argument("--protein-id")
    ap.add_argument("--reaction-feature",type=Path)
    ap.add_argument("--protein-feature",type=Path)
    ap.add_argument("--reaction-view",action="append",default=[],metavar="NAME=PATH")
    ap.add_argument("--protein-view",action="append",default=[],metavar="NAME=PATH")
    ap.add_argument("--reaction-distance",action="append",default=[],metavar="NAME=NPY")
    ap.add_argument("--protein-distance",action="append",default=[],metavar="NAME=NPY")
    ap.add_argument("--top-k",type=int,default=10)
    a=ap.parse_args()
    b=PortableReferenceBundle(a.bundle)
    reaction_views=parse_query_views(a.reaction_view)
    protein_views=parse_query_views(a.protein_view)
    reaction_distances=parse_query_distances(a.reaction_distance)
    protein_distances=parse_query_distances(a.protein_distance)
    if a.reaction_feature is not None:
        if "global" in reaction_views:
            raise ValueError("--reaction-feature conflicts with --reaction-view global=...")
        reaction_views["global"]=load_query_feature(a.reaction_feature)
    if a.protein_feature is not None:
        if "global" in protein_views:
            raise ValueError("--protein-feature conflicts with --protein-view global=...")
        protein_views["global"]=load_query_feature(a.protein_feature)
    modes=[
        a.reaction_id is not None,
        a.protein_id is not None,
        bool(reaction_views) or bool(reaction_distances),
        bool(protein_views) or bool(protein_distances),
    ]
    if sum(map(bool,modes))!=1:
        raise ValueError(
            "choose exactly one query mode: reaction id, protein id, "
            "reaction feature/views, or protein feature/views"
        )
    if a.reaction_id is not None:
        rows=b.rank_enzymes(a.reaction_id,a.top_k)
    elif a.protein_id is not None:
        rows=b.rank_reactions(a.protein_id,a.top_k)
    elif reaction_views or reaction_distances:
        rows=b.rank_enzymes_from_reaction_views(
            reaction_views,a.top_k,distances=reaction_distances
        )
    else:
        rows=b.rank_reactions_from_protein_views(
            protein_views,a.top_k,distances=protein_distances
        )
    print(json.dumps(rows,indent=2))


if __name__=="__main__":
    main()
