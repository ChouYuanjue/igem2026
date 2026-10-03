from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
from scipy.sparse.csgraph import connected_components, dijkstra

from projects.active.fibre.geometry.multiscale import (
    information_product_affinity,
    partial_observation_pullback_affinity,
)
from .contracts import load_and_validate, load_feature_csv, load_partial_feature_csv


VIEW_NAME_RE=re.compile(r"^[A-Za-z0-9_.-]+$")


def sha256(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def chordal_distance(normalized: np.ndarray) -> np.ndarray:
    x=np.asarray(normalized,dtype=np.float64)
    sim=np.clip(x@x.T,-1.0,1.0)
    return np.sqrt(np.maximum(2.0-2.0*sim,0.0))


def _intrinsic_k(n: int) -> int:
    if n < 2:
        raise ValueError("factor geometry requires at least two entities")
    return min(n-1,max(1,int(math.ceil(math.sqrt(n)))))


def _stabilize_scales(
    raw: np.ndarray,
    available: np.ndarray | None=None,
) -> np.ndarray:
    sigma=np.asarray(raw,dtype=np.float64).reshape(-1)
    mask=(
        np.ones(len(sigma),dtype=bool)
        if available is None
        else np.asarray(available,dtype=bool).reshape(-1)
    )
    if len(mask)!=len(sigma):
        raise ValueError("scale availability length mismatch")
    positive=sigma[mask&np.isfinite(sigma)&(sigma>1e-8)]
    fallback=float(np.median(positive)) if len(positive) else 1.0
    out=np.full(len(sigma),np.nan,dtype=np.float32)
    out[mask]=np.where(
        np.isfinite(sigma[mask])&(sigma[mask]>1e-8),
        sigma[mask],fallback,
    ).astype(np.float32)
    return out


def reference_scales_from_features(
    features: np.ndarray,
    *,
    available: np.ndarray | None=None,
    block_size: int=1024,
) -> np.ndarray:
    """Exact self-tuning scales with bounded memory.

    This computes the same sqrt(N)-neighbour scale used by the canonical
    single-view FIBRE geometry without materialising an N x N distance matrix.
    Runtime remains O(N^2); only peak memory is reduced.
    """
    x=np.asarray(features,dtype=np.float32)
    n=len(x)
    mask=(
        np.ones(n,dtype=bool)
        if available is None
        else np.asarray(available,dtype=bool).reshape(-1)
    )
    if len(mask)!=n:
        raise ValueError("feature availability length mismatch")
    support=int(mask.sum())
    if support<2:
        return np.full(n,np.nan,dtype=np.float32)
    k=min(support-1,max(1,int(math.ceil(math.sqrt(support)))))
    bs=max(1,int(block_size))
    out=np.full(n,np.nan,dtype=np.float32)
    for q0 in range(0,n,bs):
        q1=min(n,q0+bs); q=x[q0:q1]
        best=np.full((q1-q0,k),np.inf,dtype=np.float32)
        for r0 in range(0,n,bs):
            r1=min(n,r0+bs)
            sim=np.clip(q@x[r0:r1].T,-1.0,1.0)
            d2=np.maximum(2.0-2.0*sim,0.0).astype(np.float32,copy=False)
            pair_ok=mask[q0:q1,None]&mask[None,r0:r1]
            d2[~pair_ok]=np.inf
            lo=max(q0,r0); hi=min(q1,r1)
            if lo<hi:
                ids=np.arange(lo,hi)
                d2[ids-q0,ids-r0]=np.inf
            merged=np.concatenate((best,d2),axis=1)
            best=np.partition(merged,k-1,axis=1)[:,:k]
        local=np.sqrt(np.max(best,axis=1)).astype(np.float32)
        local[~mask[q0:q1]]=np.nan
        out[q0:q1]=local
    return _stabilize_scales(out,mask)


def reference_scales_from_distance(
    distance: np.ndarray,
    available: np.ndarray,
    *,
    block_size: int=1024,
) -> np.ndarray:
    d=np.asarray(distance)
    if d.ndim!=2 or d.shape[0]!=d.shape[1]:
        raise ValueError("distance view must be square")
    n=d.shape[0]
    mask=np.asarray(available,dtype=bool).reshape(-1)
    if len(mask)!=n:
        raise ValueError("distance-view availability length mismatch")
    support=int(mask.sum())
    if support<2:
        return np.full(n,np.nan,dtype=np.float32)
    k=min(support-1,max(1,int(math.ceil(math.sqrt(support)))))
    bs=max(1,int(block_size))
    out=np.full(n,np.nan,dtype=np.float32)
    for q0 in range(0,n,bs):
        q1=min(n,q0+bs)
        best=np.full((q1-q0,k),np.inf,dtype=np.float32)
        for r0 in range(0,n,bs):
            r1=min(n,r0+bs)
            tile=np.asarray(d[q0:q1,r0:r1],dtype=np.float32).copy()
            valid=mask[q0:q1,None]&mask[None,r0:r1]&np.isfinite(tile)&(tile>=0)
            tile[~valid]=np.inf
            lo=max(q0,r0); hi=min(q1,r1)
            if lo<hi:
                ids=np.arange(lo,hi)
                tile[ids-q0,ids-r0]=np.inf
            merged=np.concatenate((best,tile),axis=1)
            best=np.partition(merged,k-1,axis=1)[:,:k]
        local=np.max(best,axis=1)
        local[~mask[q0:q1]]=np.nan
        out[q0:q1]=local
    return _stabilize_scales(out,mask)


def _validate_feature_views(
    views: list[tuple[str,np.ndarray,np.ndarray]],
) -> list[tuple[str,np.ndarray,np.ndarray]]:
    if not views:
        raise ValueError("at least one feature view is required")
    n=len(views[0][1])
    seen=set()
    out=[]
    for name,matrix,available in views:
        if not VIEW_NAME_RE.fullmatch(str(name)) or str(name) in seen:
            raise ValueError(f"invalid or duplicated feature view name: {name}")
        seen.add(str(name))
        x=np.asarray(matrix,dtype=np.float32)
        a=np.asarray(available,dtype=bool).reshape(-1)
        if x.ndim!=2 or len(x)!=n or len(a)!=n:
            raise ValueError(f"feature view {name} is not aligned to the reference universe")
        if int(a.sum())<2:
            raise ValueError(f"feature view {name} needs at least two observed entities")
        if not np.all(np.isfinite(x[a])):
            raise ValueError(f"feature view {name} contains non-finite observed vectors")
        out.append((str(name),x,a))
    return out


def _validate_distance_views(
    views: list[tuple[str,np.ndarray,np.ndarray]],
    *,
    expected_n: int | None=None,
) -> list[tuple[str,np.ndarray,np.ndarray]]:
    out=[]; seen=set()
    for name,distance,available in views:
        if not VIEW_NAME_RE.fullmatch(str(name)) or str(name) in seen:
            raise ValueError(f"invalid or duplicated distance view name: {name}")
        seen.add(str(name))
        d=np.asarray(distance)
        a=np.asarray(available,dtype=bool).reshape(-1)
        if d.ndim!=2 or d.shape[0]!=d.shape[1]:
            raise ValueError(f"distance view {name} must be square")
        if expected_n is not None and d.shape!=(expected_n,expected_n):
            raise ValueError(f"distance view {name} is not aligned to the reference universe")
        if len(a)!=d.shape[0] or int(a.sum())<2:
            raise ValueError(
                f"distance view {name} needs an aligned mask with at least two observations"
            )
        ids=np.flatnonzero(a)
        observed=np.asarray(d[np.ix_(ids,ids)],dtype=np.float64)
        if not np.all(np.isfinite(observed)) or np.any(observed<0):
            raise ValueError(f"distance view {name} has invalid observed distances")
        if not np.allclose(observed,observed.T,rtol=1e-5,atol=1e-7):
            raise ValueError(f"distance view {name} must be symmetric")
        if not np.allclose(np.diag(observed),0.0,rtol=0,atol=1e-7):
            raise ValueError(f"distance view {name} must have zero observed diagonal")
        out.append((str(name),d,a))
    return out


def dense_exact_multiview_affinity(
    views: list[tuple[str,np.ndarray,np.ndarray]],
    *,
    k: int | None=None,
    block_size: int=1024,
) -> tuple[csr_matrix,dict,dict[str,np.ndarray]]:
    views=_validate_feature_views(views)
    distances=[]; masks=[]; scales={}
    for name,x,a in views:
        d=chordal_distance(x)
        distances.append(d); masks.append(a)
        scales[name]=reference_scales_from_features(
            x,available=a,block_size=block_size
        )
    affinity,diag=partial_observation_pullback_affinity(distances,masks,k=k)
    return affinity,{
        "backend":"dense_exact_multiview_feature_geometry",
        "exact":True,
        "views":[
            {"name":name,"available_count":int(a.sum()),"dimension":int(x.shape[1])}
            for name,x,a in views
        ],
        **diag,
    },scales


def blockwise_exact_multiview_affinity(
    views: list[tuple[str,np.ndarray,np.ndarray]],
    *,
    k: int | None=None,
    block_size: int=1024,
) -> tuple[csr_matrix,dict,dict[str,np.ndarray]]:
    """Exact missing-neutral multiview affinity with bounded peak memory."""
    views=_validate_feature_views(views)
    n=len(views[0][1])
    graph_k=min(n-1,max(1,int(k) if k is not None else _intrinsic_k(n)))
    bs=max(1,int(block_size))
    scales={
        name:reference_scales_from_features(x,available=a,block_size=bs)
        for name,x,a in views
    }
    rows=[]; cols=[]; vals=[]
    for q0 in range(0,n,bs):
        q1=min(n,q0+bs)
        best_e=np.full((q1-q0,graph_k),np.inf,dtype=np.float32)
        best_i=np.full((q1-q0,graph_k),-1,dtype=np.int64)
        for r0 in range(0,n,bs):
            r1=min(n,r0+bs)
            energy_sum=np.zeros((q1-q0,r1-r0),dtype=np.float32)
            observed=np.zeros((q1-q0,r1-r0),dtype=np.int16)
            for name,x,a in views:
                sim=np.clip(x[q0:q1]@x[r0:r1].T,-1.0,1.0)
                d2=np.maximum(2.0-2.0*sim,0.0).astype(np.float32,copy=False)
                sigma=scales[name]
                denom=sigma[q0:q1,None]*sigma[None,r0:r1]
                valid=(
                    a[q0:q1,None]&a[None,r0:r1]
                    &np.isfinite(denom)&(denom>1e-8)
                )
                e=np.zeros_like(d2)
                e[valid]=d2[valid]/denom[valid]
                energy_sum[valid]+=e[valid]
                observed[valid]+=1
            energy=np.full_like(energy_sum,np.inf)
            valid=observed>0
            energy[valid]=energy_sum[valid]/observed[valid]
            lo=max(q0,r0); hi=min(q1,r1)
            if lo<hi:
                ids=np.arange(lo,hi)
                energy[ids-q0,ids-r0]=np.inf
            tile_i=np.broadcast_to(
                np.arange(r0,r1,dtype=np.int64)[None,:],energy.shape
            )
            all_e=np.concatenate((best_e,energy),axis=1)
            all_i=np.concatenate((best_i,tile_i),axis=1)
            pick=np.argpartition(all_e,graph_k-1,axis=1)[:,:graph_k]
            best_e=np.take_along_axis(all_e,pick,axis=1)
            best_i=np.take_along_axis(all_i,pick,axis=1)
        order=np.argsort(best_e,axis=1,kind="stable")
        best_e=np.take_along_axis(best_e,order,axis=1)
        best_i=np.take_along_axis(best_i,order,axis=1)
        for local in range(q1-q0):
            good=np.isfinite(best_e[local])&(best_i[local]>=0)
            ids=best_i[local,good]
            weights=np.exp(-best_e[local,good]).astype(np.float32)
            rows.extend([q0+local]*len(ids))
            cols.extend(ids.tolist())
            vals.extend(weights.tolist())
    directed=csr_matrix((vals,(rows,cols)),shape=(n,n),dtype=np.float32)
    undirected=directed.maximum(directed.T).tocsr()
    undirected.setdiag(0.0); undirected.eliminate_zeros()
    return undirected,{
        "backend":"blockwise_exact_multiview_feature_geometry",
        "exact":True,
        "graph_k":int(graph_k),
        "block_size":int(bs),
        "views":[
            {"name":name,"available_count":int(a.sum()),"dimension":int(x.shape[1])}
            for name,x,a in views
        ],
        "directed_edges":int(directed.nnz),
        "undirected_edges":int(undirected.nnz//2),
    },scales


def dense_exact_mixed_affinity(
    feature_views: list[tuple[str,np.ndarray,np.ndarray]],
    distance_views: list[tuple[str,np.ndarray,np.ndarray]],
    *,
    k: int | None=None,
    block_size: int=1024,
) -> tuple[csr_matrix,dict,dict[str,np.ndarray]]:
    fviews=_validate_feature_views(feature_views) if feature_views else []
    n=len(fviews[0][1]) if fviews else int(np.asarray(distance_views[0][1]).shape[0])
    dviews=_validate_distance_views(distance_views,expected_n=n)
    distances=[]; masks=[]; scales={}; meta=[]
    for name,x,a in fviews:
        distances.append(chordal_distance(x)); masks.append(a)
        scales[name]=reference_scales_from_features(
            x,available=a,block_size=block_size
        )
        meta.append({
            "name":name,"kind":"feature",
            "available_count":int(a.sum()),"dimension":int(x.shape[1]),
        })
    for name,d,a in dviews:
        distances.append(np.asarray(d,dtype=np.float64)); masks.append(a)
        scales[name]=reference_scales_from_distance(
            d,a,block_size=block_size
        )
        meta.append({
            "name":name,"kind":"distance",
            "available_count":int(a.sum()),
        })
    affinity,diag=partial_observation_pullback_affinity(distances,masks,k=k)
    return affinity,{
        "backend":"dense_exact_mixed_geometry",
        "exact":True,
        "views":meta,
        **diag,
    },scales


def blockwise_exact_mixed_affinity(
    feature_views: list[tuple[str,np.ndarray,np.ndarray]],
    distance_views: list[tuple[str,np.ndarray,np.ndarray]],
    *,
    k: int | None=None,
    block_size: int=1024,
) -> tuple[csr_matrix,dict,dict[str,np.ndarray]]:
    fviews=_validate_feature_views(feature_views) if feature_views else []
    n=len(fviews[0][1]) if fviews else int(np.asarray(distance_views[0][1]).shape[0])
    dviews=_validate_distance_views(distance_views,expected_n=n)
    graph_k=min(n-1,max(1,int(k) if k is not None else _intrinsic_k(n)))
    bs=max(1,int(block_size))
    scales={}
    for name,x,a in fviews:
        scales[name]=reference_scales_from_features(x,available=a,block_size=bs)
    for name,d,a in dviews:
        scales[name]=reference_scales_from_distance(d,a,block_size=bs)
    rows=[]; cols=[]; vals=[]
    for q0 in range(0,n,bs):
        q1=min(n,q0+bs)
        best_e=np.full((q1-q0,graph_k),np.inf,dtype=np.float32)
        best_i=np.full((q1-q0,graph_k),-1,dtype=np.int64)
        for r0 in range(0,n,bs):
            r1=min(n,r0+bs)
            energy_sum=np.zeros((q1-q0,r1-r0),dtype=np.float32)
            observed=np.zeros((q1-q0,r1-r0),dtype=np.int16)
            for name,x,a in fviews:
                sim=np.clip(x[q0:q1]@x[r0:r1].T,-1.0,1.0)
                raw=np.maximum(2.0-2.0*sim,0.0).astype(np.float32,copy=False)
                sigma=scales[name]
                denom=sigma[q0:q1,None]*sigma[None,r0:r1]
                valid=(
                    a[q0:q1,None]&a[None,r0:r1]
                    &np.isfinite(denom)&(denom>1e-8)
                )
                e=np.zeros_like(raw)
                e[valid]=raw[valid]/denom[valid]
                energy_sum[valid]+=e[valid]; observed[valid]+=1
            for name,d,a in dviews:
                raw=np.asarray(d[q0:q1,r0:r1],dtype=np.float32)
                sigma=scales[name]
                denom=sigma[q0:q1,None]*sigma[None,r0:r1]
                valid=(
                    a[q0:q1,None]&a[None,r0:r1]
                    &np.isfinite(raw)&(raw>=0)
                    &np.isfinite(denom)&(denom>1e-8)
                )
                e=np.zeros_like(raw)
                e[valid]=np.square(raw[valid])/denom[valid]
                energy_sum[valid]+=e[valid]; observed[valid]+=1
            energy=np.full_like(energy_sum,np.inf)
            valid=observed>0
            energy[valid]=energy_sum[valid]/observed[valid]
            lo=max(q0,r0); hi=min(q1,r1)
            if lo<hi:
                ids=np.arange(lo,hi)
                energy[ids-q0,ids-r0]=np.inf
            tile_i=np.broadcast_to(
                np.arange(r0,r1,dtype=np.int64)[None,:],energy.shape
            )
            all_e=np.concatenate((best_e,energy),axis=1)
            all_i=np.concatenate((best_i,tile_i),axis=1)
            pick=np.argpartition(all_e,graph_k-1,axis=1)[:,:graph_k]
            best_e=np.take_along_axis(all_e,pick,axis=1)
            best_i=np.take_along_axis(all_i,pick,axis=1)
        order=np.argsort(best_e,axis=1,kind="stable")
        best_e=np.take_along_axis(best_e,order,axis=1)
        best_i=np.take_along_axis(best_i,order,axis=1)
        for local in range(q1-q0):
            good=np.isfinite(best_e[local])&(best_i[local]>=0)
            ids=best_i[local,good]
            weights=np.exp(-best_e[local,good]).astype(np.float32)
            rows.extend([q0+local]*len(ids))
            cols.extend(ids.tolist())
            vals.extend(weights.tolist())
    directed=csr_matrix((vals,(rows,cols)),shape=(n,n),dtype=np.float32)
    undirected=directed.maximum(directed.T).tocsr()
    undirected.setdiag(0.0); undirected.eliminate_zeros()
    meta=[
        {
            "name":name,"kind":"feature",
            "available_count":int(a.sum()),"dimension":int(x.shape[1]),
        }
        for name,x,a in fviews
    ]+[
        {"name":name,"kind":"distance","available_count":int(a.sum())}
        for name,_d,a in dviews
    ]
    return undirected,{
        "backend":"blockwise_exact_mixed_geometry",
        "exact":True,
        "graph_k":int(graph_k),
        "block_size":int(bs),
        "views":meta,
        "directed_edges":int(directed.nnz),
        "undirected_edges":int(undirected.nnz//2),
    },scales


def blockwise_exact_affinity(
    features: np.ndarray,
    *,
    k: int | None=None,
    block_size: int=1024,
) -> tuple[csr_matrix,dict,np.ndarray]:
    """Exact single-view self-tuned kNN affinity with bounded peak memory."""
    x=np.asarray(features,dtype=np.float32)
    n=len(x); graph_k=min(n-1,max(1,int(k) if k is not None else _intrinsic_k(n)))
    bs=max(1,int(block_size))
    sigma=reference_scales_from_features(x,block_size=bs)
    rows=[]; cols=[]; vals=[]
    for q0 in range(0,n,bs):
        q1=min(n,q0+bs); q=x[q0:q1]
        best_e=np.full((q1-q0,graph_k),np.inf,dtype=np.float32)
        best_i=np.full((q1-q0,graph_k),-1,dtype=np.int64)
        for r0 in range(0,n,bs):
            r1=min(n,r0+bs)
            sim=np.clip(q@x[r0:r1].T,-1.0,1.0)
            d2=np.maximum(2.0-2.0*sim,0.0).astype(np.float32,copy=False)
            denom=sigma[q0:q1,None]*sigma[None,r0:r1]
            energy=d2/np.maximum(denom,1e-12)
            lo=max(q0,r0); hi=min(q1,r1)
            if lo<hi:
                ids=np.arange(lo,hi)
                energy[ids-q0,ids-r0]=np.inf
            tile_i=np.broadcast_to(
                np.arange(r0,r1,dtype=np.int64)[None,:],energy.shape
            )
            all_e=np.concatenate((best_e,energy),axis=1)
            all_i=np.concatenate((best_i,tile_i),axis=1)
            pick=np.argpartition(all_e,graph_k-1,axis=1)[:,:graph_k]
            best_e=np.take_along_axis(all_e,pick,axis=1)
            best_i=np.take_along_axis(all_i,pick,axis=1)
        order=np.argsort(best_e,axis=1,kind="stable")
        best_e=np.take_along_axis(best_e,order,axis=1)
        best_i=np.take_along_axis(best_i,order,axis=1)
        for local in range(q1-q0):
            good=np.isfinite(best_e[local])&(best_i[local]>=0)
            ids=best_i[local,good]
            weights=np.exp(-best_e[local,good]).astype(np.float32)
            rows.extend([q0+local]*len(ids))
            cols.extend(ids.tolist())
            vals.extend(weights.tolist())
    directed=csr_matrix((vals,(rows,cols)),shape=(n,n),dtype=np.float32)
    undirected=directed.maximum(directed.T).tocsr()
    undirected.setdiag(0.0); undirected.eliminate_zeros()
    return undirected,{
        "backend":"blockwise_exact_feature_geometry",
        "exact":True,
        "graph_k":int(graph_k),
        "scale_k":int(_intrinsic_k(n)),
        "block_size":int(bs),
        "directed_edges":int(directed.nnz),
        "undirected_edges":int(undirected.nnz//2),
    },sigma


def faiss_hnsw_affinity(
    features: np.ndarray,
    *,
    k: int | None=None,
    query_batch_size: int=1024,
    candidate_multiplier: int=8,
    hnsw_m: int=32,
    ef_construction: int=200,
    ef_search: int=256,
) -> tuple[csr_matrix,dict,np.ndarray]:
    """Approximate large-universe affinity using FAISS HNSW candidate search.

    The returned manifest marks this backend approximate. HNSW retrieves a
    raw-chordal candidate pool; the canonical endpoint self-tuning energy is
    then evaluated exactly inside that pool.
    """
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "geometry backend faiss_hnsw requires the optional faiss-cpu package"
        ) from exc
    x=np.ascontiguousarray(np.asarray(features,dtype=np.float32))
    n,dim=x.shape
    if k is None and _intrinsic_k(n)>256:
        raise ValueError(
            "faiss_hnsw on a large universe requires an explicit graph_k; "
            "the canonical sqrt(N) graph would itself be too large for the "
            "intended ANN deployment regime"
        )
    graph_k=min(n-1,max(1,int(k) if k is not None else _intrinsic_k(n)))
    scale_k=_intrinsic_k(n)
    pool=min(n,max(scale_k+1,graph_k*max(2,int(candidate_multiplier))+1))
    index=faiss.IndexHNSWFlat(dim,max(4,int(hnsw_m)),faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction=max(int(ef_construction),pool)
    index.hnsw.efSearch=max(int(ef_search),pool)
    index.add(x)
    scales=np.empty(n,dtype=np.float32)
    qbs=max(1,int(query_batch_size))
    for q0 in range(0,n,qbs):
        q1=min(n,q0+qbs)
        similarity,indices=index.search(x[q0:q1],pool)
        d2=np.maximum(2.0-2.0*np.clip(similarity,-1.0,1.0),0.0)
        for local,i in enumerate(range(q0,q1)):
            valid=(indices[local]>=0)&(indices[local]!=i)&np.isfinite(d2[local])
            values=np.sort(d2[local,valid])
            if len(values)<scale_k:
                raise RuntimeError("FAISS candidate pool did not recover enough neighbours for scale")
            value=float(values[scale_k-1])
            scales[i]=math.sqrt(value) if value>1e-16 else 0.0
    scales=_stabilize_scales(scales)
    rows=[]; cols=[]; vals=[]
    for q0 in range(0,n,qbs):
        q1=min(n,q0+qbs)
        similarity,indices=index.search(x[q0:q1],pool)
        d2=np.maximum(2.0-2.0*np.clip(similarity,-1.0,1.0),0.0)
        for local,i in enumerate(range(q0,q1)):
            ids=indices[local]
            valid=(ids>=0)&(ids!=i)&np.isfinite(d2[local])
            ids=ids[valid].astype(np.int64)
            energy=d2[local,valid]/np.maximum(scales[i]*scales[ids],1e-12)
            kk=min(graph_k,len(ids))
            pick=np.argpartition(energy,kk-1)[:kk]
            pick=pick[np.argsort(energy[pick],kind="stable")]
            ids=ids[pick]; weights=np.exp(-energy[pick]).astype(np.float32)
            rows.extend([i]*len(ids)); cols.extend(ids.tolist()); vals.extend(weights.tolist())
    directed=csr_matrix((vals,(rows,cols)),shape=(n,n),dtype=np.float32)
    undirected=directed.maximum(directed.T).tocsr()
    undirected.setdiag(0.0); undirected.eliminate_zeros()
    return undirected,{
        "backend":"faiss_hnsw_candidate_geometry",
        "exact":False,
        "approximation":"HNSW candidate retrieval followed by exact canonical energy inside candidate pool",
        "graph_k":int(graph_k),
        "scale_k":int(scale_k),
        "candidate_pool":int(pool),
        "candidate_multiplier":int(candidate_multiplier),
        "query_batch_size":int(qbs),
        "hnsw_m":int(hnsw_m),
        "ef_construction":int(index.hnsw.efConstruction),
        "ef_search":int(index.hnsw.efSearch),
        "directed_edges":int(directed.nnz),
        "undirected_edges":int(undirected.nnz//2),
    },scales


def affinity_to_unit_length_graph(affinity: csr_matrix) -> tuple[csr_matrix,float]:
    w=csr_matrix(affinity,dtype=np.float64).maximum(csr_matrix(affinity,dtype=np.float64).T).tocsr()
    w.setdiag(0.0); w.eliminate_zeros()
    if w.shape[0] < 2:
        raise ValueError("factor graph requires at least two entities")
    ncomp,_=connected_components(w,directed=False)
    if ncomp != 1:
        raise ValueError(f"factor graph must be connected; got {ncomp} components")
    coo=w.tocoo()
    upper=coo.row<coo.col
    raw=np.sqrt(np.maximum(-np.log(np.clip(coo.data[upper],1e-300,1.0)),1e-12))
    ell=float(np.median(raw))
    if not np.isfinite(ell) or ell<=0:
        raise ValueError("invalid graph characteristic length")
    edge=np.sqrt(np.maximum(-np.log(np.clip(w.data,1e-300,1.0)),1e-12))/ell
    graph=csr_matrix((edge,w.indices,w.indptr),shape=w.shape)
    return graph,ell


def graph_from_features(
    features: np.ndarray,
    *,
    k: int | None=None,
    backend: str="auto",
    block_size: int=1024,
    dense_limit: int=5000,
    ann_candidate_multiplier: int=8,
) -> tuple[csr_matrix,dict,np.ndarray]:
    mode=str(backend or "auto")
    if mode=="auto":
        mode="dense_exact" if len(features)<=int(dense_limit) else "blockwise_exact"
    if mode=="dense_exact":
        d=chordal_distance(features)
        affinity,diag=information_product_affinity(
            [d],[np.ones(len(features),dtype=bool)],k=k
        )
        sigma=np.sort(np.where(np.eye(len(features),dtype=bool),np.inf,d),axis=1)[
            :, _intrinsic_k(len(features))-1
        ].astype(np.float32)
        sigma=_stabilize_scales(sigma)
        return affinity,{"backend":"dense_exact_feature_geometry","exact":True,**diag},sigma
    if mode=="blockwise_exact":
        return blockwise_exact_affinity(features,k=k,block_size=block_size)
    if mode=="faiss_hnsw":
        return faiss_hnsw_affinity(
            features,k=k,query_batch_size=block_size,
            candidate_multiplier=ann_candidate_multiplier
        )
    raise ValueError("geometry backend must be auto, dense_exact, blockwise_exact or faiss_hnsw")


def graph_from_feature_views(
    views: list[tuple[str,np.ndarray,np.ndarray]],
    *,
    distance_views: list[tuple[str,np.ndarray,np.ndarray]] | None=None,
    k: int | None=None,
    backend: str="auto",
    block_size: int=1024,
    dense_limit: int=5000,
    ann_candidate_multiplier: int=8,
) -> tuple[csr_matrix,dict,dict[str,np.ndarray]]:
    distance_views=list(distance_views or [])
    if distance_views:
        if not views:
            n=int(np.asarray(distance_views[0][1]).shape[0])
        else:
            n=len(views[0][1])
        _validate_distance_views(distance_views,expected_n=n)
        mode=str(backend or "auto")
        if mode=="auto":
            mode="dense_exact" if n<=int(dense_limit) else "blockwise_exact"
        if mode=="dense_exact":
            return dense_exact_mixed_affinity(
                views,distance_views,k=k,block_size=block_size
            )
        if mode=="blockwise_exact":
            return blockwise_exact_mixed_affinity(
                views,distance_views,k=k,block_size=block_size
            )
        if mode=="faiss_hnsw":
            raise ValueError(
                "faiss_hnsw does not approximate partial distance views; "
                "provide a precomputed sparse affinity whose builder records "
                "the structural/transport approximation policy"
            )
        raise ValueError(
            "geometry backend must be auto, dense_exact, blockwise_exact or faiss_hnsw"
        )
    views=_validate_feature_views(views)
    if len(views)==1 and bool(np.all(views[0][2])):
        name,x,_=views[0]
        affinity,diag,scale=graph_from_features(
            x,k=k,backend=backend,block_size=block_size,
            dense_limit=dense_limit,
            ann_candidate_multiplier=ann_candidate_multiplier,
        )
        diag={**diag,"views":[{
            "name":name,
            "available_count":int(len(x)),
            "dimension":int(x.shape[1]),
        }]}
        return affinity,diag,{name:scale}
    mode=str(backend or "auto")
    if mode=="auto":
        mode="dense_exact" if len(views[0][1])<=int(dense_limit) else "blockwise_exact"
    if mode=="dense_exact":
        return dense_exact_multiview_affinity(
            views,k=k,block_size=block_size
        )
    if mode=="blockwise_exact":
        return blockwise_exact_multiview_affinity(
            views,k=k,block_size=block_size
        )
    if mode=="faiss_hnsw":
        raise ValueError(
            "faiss_hnsw currently supports one fully observed feature view; "
            "for large partial/multiview geometry provide a precomputed sparse "
            "affinity graph whose builder records its approximation policy"
        )
    raise ValueError("geometry backend must be auto, dense_exact, blockwise_exact or faiss_hnsw")


def _load_or_build_graph(
    *,
    feature_views: list[tuple[str,np.ndarray,np.ndarray]] | None,
    distance_views: list[tuple[str,np.ndarray,np.ndarray]] | None,
    affinity_path: Path | None,
    k: int | None,
    backend: str,
    block_size: int,
    dense_limit: int,
    ann_candidate_multiplier: int,
) -> tuple[csr_matrix,dict,dict[str,np.ndarray] | None]:
    if affinity_path is not None:
        affinity=load_npz(affinity_path).tocsr()
        return affinity,{
            "backend":"precomputed_sparse_affinity",
            "exact":None,
            "source":str(affinity_path),
            "source_sha256":sha256(affinity_path),
        },None
    if not feature_views and not distance_views:
        raise ValueError(
            "either feature/distance views or a precomputed affinity graph is required"
        )
    return graph_from_feature_views(
        feature_views or [],distance_views=distance_views,k=k,
        backend=backend,block_size=block_size,
        dense_limit=dense_limit,ann_candidate_multiplier=ann_candidate_multiplier,
    )


def _support_distances(graph: csr_matrix, support: np.ndarray) -> np.ndarray:
    if len(support)==0:
        raise ValueError("positive support may not be empty")
    values=dijkstra(graph,directed=False,indices=np.asarray(support,dtype=np.int64))
    if not np.all(np.isfinite(values)):
        raise ValueError("support distances require a connected factor graph")
    if values.ndim==1:
        values=values[None,:]
    return np.asarray(values,dtype=np.float32)


def _load_extra_views(
    paths: dict[str,str | Path] | None,
    *,
    id_column: str,
    expected_ids: list[str],
) -> list[tuple[str,np.ndarray,np.ndarray]]:
    out=[]
    for raw_name,raw_path in sorted((paths or {}).items()):
        name=str(raw_name).strip()
        if name=="global" or not VIEW_NAME_RE.fullmatch(name):
            raise ValueError(
                f"optional view name must match {VIEW_NAME_RE.pattern} and may not be 'global': {name}"
            )
        matrix,available=load_partial_feature_csv(
            raw_path,id_column=id_column,expected_ids=expected_ids
        )
        out.append((name,matrix,available))
    return out


def _load_distance_views(
    paths: dict[str,tuple[str | Path,str | Path]] | None,
    *,
    expected_n: int,
) -> list[tuple[str,np.ndarray,np.ndarray]]:
    out=[]
    for raw_name,(distance_path,available_path) in sorted((paths or {}).items()):
        name=str(raw_name).strip()
        if name=="global" or not VIEW_NAME_RE.fullmatch(name):
            raise ValueError(
                f"distance view name must match {VIEW_NAME_RE.pattern} and may not be 'global': {name}"
            )
        distance=np.load(distance_path,mmap_mode="r")
        available=np.load(available_path).astype(bool)
        out.extend(_validate_distance_views(
            [(name,distance,available)],expected_n=expected_n
        ))
    return out


def _write_factor_views(
    out: Path,
    factor: str,
    views: list[tuple[str,np.ndarray,np.ndarray]] | None,
    scales: dict[str,np.ndarray] | None,
) -> None:
    if not views or not scales:
        return
    root=out/f"{factor}_views"
    root.mkdir(parents=True,exist_ok=True)
    for name,matrix,available in views:
        if name not in scales:
            raise RuntimeError(f"missing reference scale for {factor} view {name}")
        np.save(root/f"{name}_features.npy",np.asarray(matrix,dtype=np.float32))
        np.save(root/f"{name}_available.npy",np.asarray(available,dtype=bool))
        np.save(root/f"{name}_scales.npy",np.asarray(scales[name],dtype=np.float32))


def _write_factor_distance_views(
    out: Path,
    factor: str,
    views: list[tuple[str,np.ndarray,np.ndarray]] | None,
    scales: dict[str,np.ndarray] | None,
) -> None:
    if not views or not scales:
        return
    root=out/f"{factor}_views"
    root.mkdir(parents=True,exist_ok=True)
    for name,distance,available in views:
        if name not in scales:
            raise RuntimeError(f"missing reference scale for {factor} distance view {name}")
        np.save(root/f"{name}_distance.npy",np.asarray(distance,dtype=np.float32))
        np.save(root/f"{name}_available.npy",np.asarray(available,dtype=bool))
        np.save(root/f"{name}_scales.npy",np.asarray(scales[name],dtype=np.float32))


def build_bundle(
    *,
    proteins_path: str | Path,
    reactions_path: str | Path,
    pairs_path: str | Path,
    output_dir: str | Path,
    protein_features_path: str | Path | None=None,
    reaction_features_path: str | Path | None=None,
    protein_affinity_path: str | Path | None=None,
    reaction_affinity_path: str | Path | None=None,
    protein_view_paths: dict[str,str | Path] | None=None,
    reaction_view_paths: dict[str,str | Path] | None=None,
    protein_distance_view_paths: dict[str,tuple[str | Path,str | Path]] | None=None,
    reaction_distance_view_paths: dict[str,tuple[str | Path,str | Path]] | None=None,
    graph_k: int | None=None,
    graph_backend: str="auto",
    graph_block_size: int=1024,
    graph_dense_limit: int=5000,
    ann_candidate_multiplier: int=8,
) -> dict:
    tables=load_and_validate(proteins_path,reactions_path,pairs_path)
    pids=tables.proteins.protein_id.astype(str).tolist()
    rids=tables.reactions.reaction_id.astype(str).tolist()
    pi={x:i for i,x in enumerate(pids)}
    ri={x:i for i,x in enumerate(rids)}

    pf=(
        load_feature_csv(protein_features_path,id_column="protein_id",expected_ids=pids)
        if protein_features_path is not None else None
    )
    rf=(
        load_feature_csv(reaction_features_path,id_column="reaction_id",expected_ids=rids)
        if reaction_features_path is not None else None
    )
    pa=Path(protein_affinity_path) if protein_affinity_path is not None else None
    ra=Path(reaction_affinity_path) if reaction_affinity_path is not None else None
    p_extra=_load_extra_views(
        protein_view_paths,id_column="protein_id",expected_ids=pids
    )
    r_extra=_load_extra_views(
        reaction_view_paths,id_column="reaction_id",expected_ids=rids
    )
    p_distance=_load_distance_views(
        protein_distance_view_paths,expected_n=len(pids)
    )
    r_distance=_load_distance_views(
        reaction_distance_view_paths,expected_n=len(rids)
    )
    if pa is not None and (p_extra or p_distance):
        raise ValueError(
            "protein extra views cannot be combined with a precomputed protein affinity; "
            "build those views into the supplied affinity instead"
        )
    if ra is not None and (r_extra or r_distance):
        raise ValueError(
            "reaction extra views cannot be combined with a precomputed reaction affinity; "
            "build those views into the supplied affinity instead"
        )
    p_views=(
        ([("global",pf,np.ones(len(pids),dtype=bool))] if pf is not None else [])
        + p_extra
    )
    r_views=(
        ([("global",rf,np.ones(len(rids),dtype=bool))] if rf is not None else [])
        + r_extra
    )
    if pa is None and not p_views and not p_distance:
        raise ValueError("protein geometry needs global features, optional views, or a precomputed affinity")
    if ra is None and not r_views and not r_distance:
        raise ValueError("reaction geometry needs global features, optional views, or a precomputed affinity")
    p_aff,pdiag,pscale=_load_or_build_graph(
        feature_views=p_views or None,distance_views=p_distance or None,
        affinity_path=pa,k=graph_k,backend=graph_backend,
        block_size=graph_block_size,dense_limit=graph_dense_limit,
        ann_candidate_multiplier=ann_candidate_multiplier,
    )
    r_aff,rdiag,rscale=_load_or_build_graph(
        feature_views=r_views or None,distance_views=r_distance or None,
        affinity_path=ra,k=graph_k,backend=graph_backend,
        block_size=graph_block_size,dense_limit=graph_dense_limit,
        ann_candidate_multiplier=ann_candidate_multiplier,
    )
    if p_aff.shape != (len(pids),len(pids)):
        raise ValueError("protein affinity shape does not match protein table")
    if r_aff.shape != (len(rids),len(rids)):
        raise ValueError("reaction affinity shape does not match reaction table")

    p_graph,pell=affinity_to_unit_length_graph(p_aff)
    r_graph,rell=affinity_to_unit_length_graph(r_aff)

    p_support=np.asarray(sorted({pi[x] for x in tables.pairs.protein_id}),dtype=np.int64)
    r_support=np.asarray(sorted({ri[x] for x in tables.pairs.reaction_id}),dtype=np.int64)
    p_support_pos={int(g):i for i,g in enumerate(p_support)}
    r_support_pos={int(g):i for i,g in enumerate(r_support)}
    support_pairs=np.asarray([
        (
            r_support_pos[ri[str(row.reaction_id)]],
            p_support_pos[pi[str(row.protein_id)]],
        )
        for row in tables.pairs.itertuples(index=False)
    ],dtype=np.int64)
    support_pairs=np.unique(support_pairs,axis=0)

    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    tables.proteins.to_csv(out/"proteins.csv",index=False)
    tables.reactions.to_csv(out/"reactions.csv",index=False)
    tables.pairs.to_csv(out/"positive_pairs.csv",index=False)
    save_npz(out/"protein_affinity.npz",p_aff)
    save_npz(out/"reaction_affinity.npz",r_aff)
    save_npz(out/"protein_length_graph.npz",p_graph)
    save_npz(out/"reaction_length_graph.npz",r_graph)
    np.save(out/"protein_support_indices.npy",p_support)
    np.save(out/"reaction_support_indices.npy",r_support)
    np.save(out/"positive_pair_support_indices.npy",support_pairs)
    np.save(out/"protein_support_geodesic.npy",_support_distances(p_graph,p_support))
    np.save(out/"reaction_support_geodesic.npy",_support_distances(r_graph,r_support))
    if pf is not None:
        np.save(out/"protein_features_normalized.npy",pf)
        if pscale is not None and "global" in pscale:
            np.save(out/"protein_reference_scales.npy",pscale["global"])
    if rf is not None:
        np.save(out/"reaction_features_normalized.npy",rf)
        if rscale is not None and "global" in rscale:
            np.save(out/"reaction_reference_scales.npy",rscale["global"])
    _write_factor_views(out,"protein",p_views,pscale)
    _write_factor_views(out,"reaction",r_views,rscale)
    _write_factor_distance_views(out,"protein",p_distance,pscale)
    _write_factor_distance_views(out,"reaction",r_distance,rscale)

    manifest={
        "schema":"fibre-portable-reference-bundle-v1",
        "scientific_role":"generic frozen reference geometry plus accepted positive correspondence support",
        "protein_count":len(pids),
        "reaction_count":len(rids),
        "positive_pair_count":int(len(tables.pairs)),
        "protein_support_count":int(len(p_support)),
        "reaction_support_count":int(len(r_support)),
        "factor_geometry":{
            "protein":{**pdiag,"characteristic_length":pell},
            "reaction":{**rdiag,"characteristic_length":rell},
        },
        "distance_storage":"support-to-all geodesic transforms; dense all-pairs geodesic is not required",
        "external_query_policy":(
            "raw feature out-of-sample attachment is available only for factor geometries "
            "built from the bundled normalized features and reference scales; custom "
            "precomputed affinities require a compatible external attachment implementation"
        ),
        "missing_policy":"missing optional biological observations are unresolved, never negative",
        "pair_policy":"positive_pairs are accepted reference support; runtime few-shot seeds are not merged unless explicitly promoted and rebuilt",
        "labels_used_for_factor_geometry":False,
        "input_sha256":{
            "proteins":sha256(proteins_path),
            "reactions":sha256(reactions_path),
            "positive_pairs":sha256(pairs_path),
            **({"protein_features":sha256(protein_features_path)} if protein_features_path is not None else {}),
            **({"reaction_features":sha256(reaction_features_path)} if reaction_features_path is not None else {}),
            **({"protein_affinity":sha256(pa)} if pa is not None else {}),
            **({"reaction_affinity":sha256(ra)} if ra is not None else {}),
            **{
                f"protein_view:{name}":sha256(path)
                for name,path in sorted((protein_view_paths or {}).items())
            },
            **{
                f"reaction_view:{name}":sha256(path)
                for name,path in sorted((reaction_view_paths or {}).items())
            },
            **{
                f"protein_distance_view:{name}:distance":sha256(paths[0])
                for name,paths in sorted((protein_distance_view_paths or {}).items())
            },
            **{
                f"protein_distance_view:{name}:available":sha256(paths[1])
                for name,paths in sorted((protein_distance_view_paths or {}).items())
            },
            **{
                f"reaction_distance_view:{name}:distance":sha256(paths[0])
                for name,paths in sorted((reaction_distance_view_paths or {}).items())
            },
            **{
                f"reaction_distance_view:{name}:available":sha256(paths[1])
                for name,paths in sorted((reaction_distance_view_paths or {}).items())
            },
        },
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


def main() -> None:
    ap=argparse.ArgumentParser(description="Build a generic portable FIBRE reference bundle.")
    ap.add_argument("--proteins",type=Path,required=True)
    ap.add_argument("--reactions",type=Path,required=True)
    ap.add_argument("--pairs",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--protein-features",type=Path)
    ap.add_argument("--reaction-features",type=Path)
    ap.add_argument("--protein-affinity",type=Path)
    ap.add_argument("--reaction-affinity",type=Path)
    ap.add_argument(
        "--protein-view",action="append",default=[],metavar="NAME=CSV",
        help="optional partially observed protein coordinate; repeatable",
    )
    ap.add_argument(
        "--reaction-view",action="append",default=[],metavar="NAME=CSV",
        help="optional partially observed reaction coordinate; repeatable",
    )
    ap.add_argument(
        "--protein-distance-view",action="append",default=[],metavar="NAME=DISTANCE_NPY,AVAILABLE_NPY",
        help="optional partially observed protein distance coordinate; repeatable",
    )
    ap.add_argument(
        "--reaction-distance-view",action="append",default=[],metavar="NAME=DISTANCE_NPY,AVAILABLE_NPY",
        help="optional partially observed reaction distance coordinate; repeatable",
    )
    ap.add_argument("--graph-k",type=int,default=0)
    ap.add_argument("--graph-backend",choices=["auto","dense_exact","blockwise_exact","faiss_hnsw"],default="auto")
    ap.add_argument("--graph-block-size",type=int,default=1024)
    ap.add_argument("--graph-dense-limit",type=int,default=5000)
    ap.add_argument("--ann-candidate-multiplier",type=int,default=8)
    a=ap.parse_args()
    def parse_views(values: list[str]) -> dict[str,Path]:
        out={}
        for value in values:
            if "=" not in value:
                raise ValueError(f"view must be NAME=CSV: {value}")
            name,path=value.split("=",1)
            name=name.strip(); path=path.strip()
            if not name or not path or name in out:
                raise ValueError(f"invalid or duplicated view argument: {value}")
            out[name]=Path(path)
        return out

    def parse_distance_views(values: list[str]) -> dict[str,tuple[Path,Path]]:
        out={}
        for value in values:
            if "=" not in value:
                raise ValueError(f"distance view must be NAME=DISTANCE_NPY,AVAILABLE_NPY: {value}")
            name,paths=value.split("=",1)
            bits=[x.strip() for x in paths.split(",")]
            name=name.strip()
            if not name or len(bits)!=2 or not all(bits) or name in out:
                raise ValueError(f"invalid or duplicated distance view argument: {value}")
            out[name]=(Path(bits[0]),Path(bits[1]))
        return out

    manifest=build_bundle(
        proteins_path=a.proteins,
        reactions_path=a.reactions,
        pairs_path=a.pairs,
        output_dir=a.output_dir,
        protein_features_path=a.protein_features,
        reaction_features_path=a.reaction_features,
        protein_affinity_path=a.protein_affinity,
        reaction_affinity_path=a.reaction_affinity,
        protein_view_paths=parse_views(a.protein_view),
        reaction_view_paths=parse_views(a.reaction_view),
        protein_distance_view_paths=parse_distance_views(a.protein_distance_view),
        reaction_distance_view_paths=parse_distance_views(a.reaction_distance_view),
        graph_k=None if a.graph_k<=0 else a.graph_k,
        graph_backend=a.graph_backend,
        graph_block_size=a.graph_block_size,
        graph_dense_limit=a.graph_dense_limit,
        ann_candidate_multiplier=a.ann_candidate_multiplier,
    )
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
