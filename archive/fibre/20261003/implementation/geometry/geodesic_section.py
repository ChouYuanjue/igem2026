from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

from projects.active.fibre.geometry.correspondence import CorrespondenceSection


def _cost_graph(graph: csr_matrix, label: str) -> csr_matrix:
    g = csr_matrix(graph, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError(f"{label} must be square")
    if g.nnz and (not np.all(np.isfinite(g.data)) or np.any(g.data < 0)):
        raise ValueError(f"{label} edge costs must be finite and non-negative")
    if (g != g.T).nnz:
        raise ValueError(f"{label} must be symmetric")
    g = g.copy()
    g.setdiag(0.0)
    g.eliminate_zeros()
    return g


def _pairs(positive_pairs: np.ndarray, nr: int, ne: int) -> np.ndarray:
    pairs = np.asarray(positive_pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
        raise ValueError("positive_pairs must be a non-empty [n,2] array")
    pairs = np.unique(pairs, axis=0)
    if np.any(pairs[:, 0] < 0) or np.any(pairs[:, 0] >= nr):
        raise ValueError("reaction positive index out of range")
    if np.any(pairs[:, 1] < 0) or np.any(pairs[:, 1] >= ne):
        raise ValueError("protein positive index out of range")
    return pairs


def _candidate_indices(size: int, values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.arange(size, dtype=np.int64)
    out = np.asarray(values, dtype=np.int64).reshape(-1)
    if np.any(out < 0) or np.any(out >= size):
        raise ValueError("candidate index out of range")
    return out


def weighted_multisource_geodesic(
    graph: csr_matrix,
    source_indices: np.ndarray,
    source_costs: np.ndarray,
) -> np.ndarray:
    """Exact lower envelope of additive geodesic costs on a sparse graph."""
    g = _cost_graph(graph, "graph")
    source = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    costs = np.asarray(source_costs, dtype=np.float64).reshape(-1)
    if len(source) == 0 or len(source) != len(costs):
        raise ValueError("source indices/costs must be non-empty and aligned")
    if np.any(source < 0) or np.any(source >= g.shape[0]):
        raise ValueError("source index out of range")
    if not np.all(np.isfinite(costs)) or np.any(costs < 0):
        raise ValueError("source costs must be finite and non-negative")

    unique, inverse = np.unique(source, return_inverse=True)
    initial = np.full(len(unique), np.inf, dtype=np.float64)
    np.minimum.at(initial, inverse, costs)

    base = g.tocoo()
    n = g.shape[0]
    rows = np.concatenate([base.row.astype(np.int64), np.full(len(unique), n, dtype=np.int64)])
    cols = np.concatenate([base.col.astype(np.int64), unique])
    data = np.concatenate([base.data.astype(np.float64), initial])
    augmented = coo_matrix((data, (rows, cols)), shape=(n + 1, n + 1), dtype=np.float64).tocsr()
    distance = np.asarray(
        dijkstra(augmented, directed=True, indices=n, return_predecessors=False),
        dtype=np.float64,
    )[:n]
    if not np.all(np.isfinite(distance)):
        raise ValueError("factor cost graph must connect every scored vertex to positive support")
    return distance


def support_geodesic_marginal(graph: csr_matrix, support_indices: np.ndarray) -> np.ndarray:
    support = np.unique(np.asarray(support_indices, dtype=np.int64).reshape(-1))
    return weighted_multisource_geodesic(
        graph, support, np.zeros(len(support), dtype=np.float64)
    )


def reaction_to_protein_geodesic_section(
    reaction_cost_graph: csr_matrix,
    protein_cost_graph: csr_matrix,
    positive_pairs: np.ndarray,
    reaction_index: int,
    *,
    candidate_protein_indices: np.ndarray | None = None,
    protein_marginal_cost: np.ndarray | None = None,
) -> CorrespondenceSection:
    """Exact R2E zero-temperature FIBRE section on additive factor geodesics.

    If C_R and C_E are graph shortest-path energies, define d=sqrt(C_geo).
    The canonical FIBRE d^2 terms are then exactly the graph geodesic costs.
    """
    rg = _cost_graph(reaction_cost_graph, "reaction_cost_graph")
    pg = _cost_graph(protein_cost_graph, "protein_cost_graph")
    pairs = _pairs(positive_pairs, rg.shape[0], pg.shape[0])
    q = int(reaction_index)
    if not 0 <= q < rg.shape[0]:
        raise ValueError("reaction query index out of range")
    candidates = _candidate_indices(pg.shape[0], candidate_protein_indices)

    reaction_cost = np.asarray(
        dijkstra(rg, directed=False, indices=q, return_predecessors=False),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(reaction_cost)):
        raise ValueError("reaction cost graph must be connected")

    reaction_support = np.unique(pairs[:, 0])
    protein_support, inverse = np.unique(pairs[:, 1], return_inverse=True)
    anchor = np.full(len(protein_support), np.inf, dtype=np.float64)
    np.minimum.at(anchor, inverse, reaction_cost[pairs[:, 0]])
    joint = weighted_multisource_geodesic(pg, protein_support, anchor)

    if protein_marginal_cost is None:
        marginal = support_geodesic_marginal(pg, protein_support)
    else:
        marginal = np.asarray(protein_marginal_cost, dtype=np.float64).reshape(-1)
        if len(marginal) != pg.shape[0] or not np.all(np.isfinite(marginal)) or np.any(marginal < 0):
            raise ValueError("protein_marginal_cost must be finite, non-negative and graph-sized")

    return CorrespondenceSection(
        direction="reaction_to_protein",
        query_index=q,
        candidate_indices=candidates,
        joint_sq=joint[candidates],
        query_marginal_sq=float(np.min(reaction_cost[reaction_support])),
        candidate_marginal_sq=marginal[candidates],
    )


def protein_to_reaction_geodesic_section(
    reaction_cost_graph: csr_matrix,
    protein_cost_graph: csr_matrix,
    positive_pairs: np.ndarray,
    protein_index: int,
    *,
    candidate_reaction_indices: np.ndarray | None = None,
    reaction_marginal_cost: np.ndarray | None = None,
) -> CorrespondenceSection:
    """Exact E2R zero-temperature FIBRE section on the same factor graphs."""
    rg = _cost_graph(reaction_cost_graph, "reaction_cost_graph")
    pg = _cost_graph(protein_cost_graph, "protein_cost_graph")
    pairs = _pairs(positive_pairs, rg.shape[0], pg.shape[0])
    q = int(protein_index)
    if not 0 <= q < pg.shape[0]:
        raise ValueError("protein query index out of range")
    candidates = _candidate_indices(rg.shape[0], candidate_reaction_indices)

    protein_cost = np.asarray(
        dijkstra(pg, directed=False, indices=q, return_predecessors=False),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(protein_cost)):
        raise ValueError("protein cost graph must be connected")

    protein_support = np.unique(pairs[:, 1])
    reaction_support, inverse = np.unique(pairs[:, 0], return_inverse=True)
    anchor = np.full(len(reaction_support), np.inf, dtype=np.float64)
    np.minimum.at(anchor, inverse, protein_cost[pairs[:, 1]])
    joint = weighted_multisource_geodesic(rg, reaction_support, anchor)

    if reaction_marginal_cost is None:
        marginal = support_geodesic_marginal(rg, reaction_support)
    else:
        marginal = np.asarray(reaction_marginal_cost, dtype=np.float64).reshape(-1)
        if len(marginal) != rg.shape[0] or not np.all(np.isfinite(marginal)) or np.any(marginal < 0):
            raise ValueError("reaction_marginal_cost must be finite, non-negative and graph-sized")

    return CorrespondenceSection(
        direction="protein_to_reaction",
        query_index=q,
        candidate_indices=candidates,
        joint_sq=joint[candidates],
        query_marginal_sq=float(np.min(protein_cost[protein_support])),
        candidate_marginal_sq=marginal[candidates],
    )
