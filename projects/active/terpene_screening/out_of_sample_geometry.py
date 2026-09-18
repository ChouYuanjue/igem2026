from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.csgraph import dijkstra


@dataclass(frozen=True)
class QueryAttachment:
    reference_indices: np.ndarray
    edge_lengths: np.ndarray
    query_view_scales: tuple[float | None, ...]
    observed_view_count: np.ndarray
    graph_k: int


def _query_scale(
    cross_distance: np.ndarray,
    reference_available: np.ndarray,
    *,
    epsilon: float,
) -> float | None:
    d = np.asarray(cross_distance, dtype=np.float64).reshape(-1)
    a = np.asarray(reference_available, dtype=bool).reshape(-1)
    valid = a & np.isfinite(d)
    values = d[valid]
    if not len(values):
        return None
    k = min(len(values), max(1, int(math.ceil(math.sqrt(len(values))))))
    sigma = float(np.partition(values, k - 1)[k - 1])
    positive = values[(values > epsilon) & np.isfinite(values)]
    if not np.isfinite(sigma) or sigma <= epsilon:
        sigma = float(np.median(positive)) if len(positive) else 1.0
    return sigma


def attach_query_to_reference(
    cross_distances: list[np.ndarray],
    query_available: list[bool],
    reference_available: list[np.ndarray],
    reference_scales: list[np.ndarray],
    *,
    k: int | None = None,
    epsilon: float = 1e-8,
) -> QueryAttachment:
    """Attach one external point to a frozen partial-observation reference atlas.

    The reference graph and its local scales are immutable. For every query view,
    only a query-side intrinsic scale is estimated from query-to-reference
    distances. A pair energy is the mean over views observed at both endpoints,

        E(q,j) = mean_v d_v(q,j)^2 / (sigma_v(q) sigma_v(j)).

    The sqrt(N) smallest finite energies define query attachment edges. This is
    an out-of-sample Nyström-style chart extension, not a rebuild of the atlas.
    """
    if not cross_distances:
        raise ValueError("at least one cross-distance view is required")
    if not (
        len(cross_distances)
        == len(query_available)
        == len(reference_available)
        == len(reference_scales)
    ):
        raise ValueError("view lists must have equal length")

    distances = [np.asarray(x, dtype=np.float64).reshape(-1) for x in cross_distances]
    n = len(distances[0])
    if any(len(x) != n for x in distances):
        raise ValueError("cross-distance length mismatch")

    energy_sum = np.zeros(n, dtype=np.float64)
    observed_count = np.zeros(n, dtype=np.int16)
    q_scales: list[float | None] = []

    for d, q_ok, r_ok, r_scale in zip(
        distances, query_available, reference_available, reference_scales
    ):
        r_ok = np.asarray(r_ok, dtype=bool).reshape(-1)
        r_scale = np.asarray(r_scale, dtype=np.float64).reshape(-1)
        if len(r_ok) != n or len(r_scale) != n:
            raise ValueError("reference view length mismatch")
        q_scale = _query_scale(d, r_ok, epsilon=epsilon) if q_ok else None
        q_scales.append(q_scale)
        if q_scale is None:
            continue
        valid = (
            r_ok
            & np.isfinite(d)
            & np.isfinite(r_scale)
            & (r_scale > epsilon)
        )
        if not np.any(valid):
            continue
        energy_sum[valid] += np.square(d[valid]) / (q_scale * r_scale[valid])
        observed_count[valid] += 1

    effective = np.full(n, np.inf, dtype=np.float64)
    valid = observed_count > 0
    effective[valid] = energy_sum[valid] / observed_count[valid]
    candidates = np.flatnonzero(np.isfinite(effective))
    if not len(candidates):
        raise ValueError("query has no jointly observed coordinate with the reference atlas")

    graph_k = min(
        len(candidates),
        max(1, int(k) if k is not None else int(math.ceil(math.sqrt(len(candidates))))),
    )
    local = effective[candidates]
    pick = np.argpartition(local, graph_k - 1)[:graph_k]
    selected = candidates[pick]
    selected = selected[np.argsort(effective[selected], kind="stable")]
    lengths = np.sqrt(np.maximum(effective[selected], 0.0))

    return QueryAttachment(
        reference_indices=selected.astype(np.int64),
        edge_lengths=lengths.astype(np.float64),
        query_view_scales=tuple(q_scales),
        observed_view_count=observed_count[selected].astype(np.int16),
        graph_k=int(graph_k),
    )


def query_geodesic_to_reference(
    reference_length_graph: csr_matrix,
    attachment: QueryAttachment,
) -> np.ndarray:
    """Single-source geodesic from an attached query to all reference points.

    This is equivalent to adding one node to the frozen undirected graph, but
    avoids constructing or recomputing any reference-reference distances.
    """
    g = csr_matrix(reference_length_graph, dtype=np.float64)
    if g.shape[0] != g.shape[1]:
        raise ValueError("reference_length_graph must be square")
    n = g.shape[0]
    idx = np.asarray(attachment.reference_indices, dtype=np.int64)
    lengths = np.asarray(attachment.edge_lengths, dtype=np.float64)
    if len(idx) != len(lengths):
        raise ValueError("attachment index/length mismatch")
    if np.any(idx < 0) or np.any(idx >= n):
        raise ValueError("attachment index out of range")

    # Multi-source Dijkstra with source offsets:
    # d(q,j)=min_a [ell(q,a)+d_G(a,j)].
    # scipy's Dijkstra has zero source offsets, so run on the small attachment
    # source set and take the shifted pointwise minimum.
    dist = dijkstra(g, directed=False, indices=idx, return_predecessors=False)
    if dist.ndim == 1:
        dist = dist[None, :]
    out = np.min(dist + lengths[:, None], axis=0)
    if not np.all(np.isfinite(out)):
        raise ValueError("reference atlas is disconnected from the query attachment")
    return np.asarray(out, dtype=np.float64)
