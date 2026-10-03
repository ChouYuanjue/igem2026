from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags, triu


@dataclass(frozen=True)
class ProductFieldConfig:
    """Regularization parameters for the enzyme-reaction product graph field."""

    reaction_graph_k: int
    protein_graph_k: int
    graph_temperature: float
    reaction_lambda: float
    protein_lambda: float
    steps: int

    @property
    def name(self) -> str:
        return (
            f"rg{self.reaction_graph_k}_pg{self.protein_graph_k}"
            f"_t{self.graph_temperature:g}"
            f"_lr{self.reaction_lambda:g}_lp{self.protein_lambda:g}"
            f"_s{self.steps}"
        )


@dataclass(frozen=True)
class _GraphDifferential:
    """Oriented incidence operator for one weighted graph.

    For a scalar field f, ``B.T @ f`` is the discrete differential on edges and
    ``B @ flux`` is its divergence back on vertices.  This is the standard
    discrete-geometric representation of the same robust energy and avoids
    Python/NumPy scatter accumulation at every flow step.
    """

    incidence: csr_matrix
    weights: np.ndarray


def _graph_differential(affinity: csr_matrix) -> _GraphDifferential:
    rows, cols, weights = _upper_affinity_edges(affinity)
    edge_count = len(weights)
    if edge_count == 0:
        return _GraphDifferential(
            incidence=csr_matrix((affinity.shape[0], 0), dtype=np.float64),
            weights=np.zeros(0, dtype=np.float64),
        )
    edge_ids = np.arange(edge_count, dtype=np.int64)
    incidence = csr_matrix(
        (
            np.concatenate(
                [
                    np.ones(edge_count, dtype=np.float64),
                    -np.ones(edge_count, dtype=np.float64),
                ]
            ),
            (
                np.concatenate([rows, cols]),
                np.concatenate([edge_ids, edge_ids]),
            ),
        ),
        shape=(affinity.shape[0], edge_count),
        dtype=np.float64,
    )
    return _GraphDifferential(incidence=incidence, weights=weights)


def _validate_square(matrix: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{label} must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} contains non-finite values")
    return matrix


def symmetric_knn_operator(
    similarity: np.ndarray,
    *,
    k: int,
    temperature: float,
) -> csr_matrix:
    """Build a symmetric normalized local-geometry operator.

    The directed kNN affinities are converted to an undirected weighted graph,
    then normalized as D^{-1/2} W D^{-1/2}.  The resulting matrix is the
    normalized adjacency S used by the graph-Laplacian regularizer L = I - S.
    """

    similarity = _validate_square(similarity, "similarity")
    if k <= 0:
        raise ValueError("k must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    n = similarity.shape[0]
    if n < 2:
        raise ValueError("at least two nodes are required")
    k = min(k, n - 1)

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row in range(n):
        local = similarity[row].astype(np.float64, copy=True)
        local[row] = -np.inf
        selected = np.argpartition(local, -k)[-k:]
        selected = selected[np.argsort(-local[selected], kind="stable")]
        selected_scores = local[selected]
        finite = np.isfinite(selected_scores)
        selected = selected[finite]
        selected_scores = selected_scores[finite]
        if not len(selected):
            continue
        weights = np.exp((selected_scores - selected_scores.max()) / temperature)
        denominator = weights.sum()
        if denominator <= 0 or not np.isfinite(denominator):
            weights = np.ones(len(selected), dtype=np.float64) / len(selected)
        else:
            weights /= denominator
        rows.extend([row] * len(selected))
        columns.extend(selected.astype(int).tolist())
        values.extend(weights.astype(np.float32).tolist())

    directed = csr_matrix((values, (rows, columns)), shape=(n, n), dtype=np.float32)
    weights = ((directed + directed.T) * 0.5).tocsr()
    weights.eliminate_zeros()
    degree = np.asarray(weights.sum(axis=1)).reshape(-1)
    scale = np.zeros(n, dtype=np.float32)
    positive = degree > 0
    scale[positive] = degree[positive] ** -0.5
    return (diags(scale) @ weights @ diags(scale)).tocsr().astype(np.float32)


def masked_symmetric_knn_operator(
    similarity: np.ndarray,
    available: np.ndarray,
    *,
    k: int,
    temperature: float,
) -> csr_matrix:
    """Build a normalized kNN operator only on observed nodes.

    Unavailable nodes have an exactly zero row/column.  The corresponding
    Laplacian is therefore ``M - S`` with ``M`` the availability diagonal,
    rather than ``I - S``; missing structure is no geometric evidence.
    """

    similarity = _validate_square(similarity, "similarity")
    available = np.asarray(available, dtype=bool).reshape(-1)
    if len(available) != similarity.shape[0]:
        raise ValueError("availability mask length mismatch")
    selected = np.flatnonzero(available)
    if len(selected) < 2:
        return csr_matrix(similarity.shape, dtype=np.float32)
    local = symmetric_knn_operator(
        similarity[np.ix_(selected, selected)],
        k=k,
        temperature=temperature,
    ).tocoo()
    return csr_matrix(
        (local.data, (selected[local.row], selected[local.col])),
        shape=similarity.shape,
        dtype=np.float32,
    )


def symmetric_knn_affinity(
    similarity: np.ndarray,
    *,
    k: int,
    temperature: float,
    available: np.ndarray | None = None,
) -> csr_matrix:
    """Return an undirected weighted kNN graph before Laplacian normalization.

    This is the natural object for nonlinear/anisotropic diffusion: edge weights
    remain explicit, so their conductance can change continuously with the
    compatibility field itself.
    """

    similarity = _validate_square(similarity, "similarity")
    n = similarity.shape[0]
    if available is None:
        selected = np.arange(n, dtype=np.int64)
    else:
        available = np.asarray(available, dtype=bool).reshape(-1)
        if len(available) != n:
            raise ValueError("availability mask length mismatch")
        selected = np.flatnonzero(available)
    if len(selected) < 2:
        return csr_matrix((n, n), dtype=np.float32)
    local_similarity = similarity[np.ix_(selected, selected)]
    local_n = len(selected)
    k = min(max(int(k), 1), local_n - 1)
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row in range(local_n):
        scores = local_similarity[row].astype(np.float64, copy=True)
        scores[row] = -np.inf
        neighbors = np.argpartition(scores, -k)[-k:]
        neighbors = neighbors[np.argsort(-scores[neighbors], kind="stable")]
        local_scores = scores[neighbors]
        finite = np.isfinite(local_scores)
        neighbors = neighbors[finite]
        local_scores = local_scores[finite]
        if not len(neighbors):
            continue
        weights = np.exp((local_scores - local_scores.max()) / temperature)
        total = weights.sum()
        if total <= 0 or not np.isfinite(total):
            weights = np.ones(len(neighbors), dtype=np.float64) / len(neighbors)
        else:
            weights /= total
        rows.extend([row] * len(neighbors))
        columns.extend(neighbors.astype(int).tolist())
        values.extend(weights.astype(np.float32).tolist())

    directed = csr_matrix(
        (values, (rows, columns)), shape=(local_n, local_n), dtype=np.float32
    )
    local = ((directed + directed.T) * 0.5).tocoo()
    return csr_matrix(
        (local.data, (selected[local.row], selected[local.col])),
        shape=(n, n),
        dtype=np.float32,
    )


def self_tuning_distance_knn_affinity(
    distance: np.ndarray,
    *,
    k: int | None = None,
    available: np.ndarray | None = None,
    epsilon: float = 1e-8,
) -> csr_matrix:
    """Density-adaptive local-scaling affinity from an intrinsic distance matrix."""

    distance = _validate_square(distance, "distance").astype(np.float64)
    if np.any(distance < 0):
        raise ValueError("distance must be non-negative")
    n = distance.shape[0]
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if available is None:
        selected = np.arange(n, dtype=np.int64)
    else:
        available = np.asarray(available, dtype=bool).reshape(-1)
        if len(available) != n:
            raise ValueError("availability mask length mismatch")
        selected = np.flatnonzero(available)
    if len(selected) < 2:
        return csr_matrix((n, n), dtype=np.float32)

    local_distance = distance[np.ix_(selected, selected)].copy()
    local_n = len(selected)
    if k is None:
        k = int(np.ceil(np.sqrt(local_n)))
    k = min(max(int(k), 1), local_n - 1)
    np.fill_diagonal(local_distance, np.inf)

    sigma = np.partition(local_distance, k - 1, axis=1)[:, k - 1]
    positive_sigma = sigma[np.isfinite(sigma) & (sigma > epsilon)]
    fallback = float(np.median(positive_sigma)) if len(positive_sigma) else 1.0
    sigma = np.where(np.isfinite(sigma) & (sigma > epsilon), sigma, fallback)

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row in range(local_n):
        neighbors = np.argpartition(local_distance[row], k - 1)[:k]
        neighbors = neighbors[
            np.argsort(local_distance[row, neighbors], kind="stable")
        ]
        d = local_distance[row, neighbors]
        denominator = np.maximum(sigma[row] * sigma[neighbors], epsilon)
        weights = np.exp(-np.square(d) / denominator)
        finite = np.isfinite(weights) & (weights > 0)
        neighbors = neighbors[finite]
        weights = weights[finite]
        rows.extend([row] * len(neighbors))
        columns.extend(neighbors.astype(int).tolist())
        values.extend(weights.astype(np.float32).tolist())

    directed = csr_matrix(
        (values, (rows, columns)), shape=(local_n, local_n), dtype=np.float32
    )
    local = directed.maximum(directed.T).tocoo()
    return csr_matrix(
        (local.data, (selected[local.row], selected[local.col])),
        shape=(n, n),
        dtype=np.float32,
    )


def self_tuning_cosine_knn_affinity(
    similarity: np.ndarray,
    *,
    k: int | None = None,
    available: np.ndarray | None = None,
    epsilon: float = 1e-8,
) -> csr_matrix:
    """Density-adaptive local-scaling affinity for cosine geometry.

    Each vertex receives its own bandwidth from the distance to its k-th
    geometric neighbour and an edge ``i--j`` receives
    ``exp(-d(i,j)^2 / (sigma_i sigma_j))``. Thus one global temperature is not
    imposed on regions with different sampling density. ``k=None`` uses
    ``ceil(sqrt(n_observed))`` as a sampling-scale discretisation rule.

    Missing nodes are omitted exactly when ``available`` is supplied; absence of
    a view therefore creates no artificial dissimilarity.
    """

    similarity = _validate_square(similarity, "similarity")
    cosine_distance = np.sqrt(
        np.maximum(2.0 - 2.0 * np.clip(similarity.astype(np.float64), -1.0, 1.0), 0.0)
    )
    return self_tuning_distance_knn_affinity(
        cosine_distance,
        k=k,
        available=available,
        epsilon=epsilon,
    )


def degree_normalized_affinity(affinity: csr_matrix) -> csr_matrix:
    """Return the symmetric normalized affinity D^{-1/2} W D^{-1/2}.

    Nonlinear product-field energies are sums over graph edges, whereas fidelity
    is a sum over vertices.  Raw kNN affinities therefore change the relative
    scale of those two terms merely when sampling density or chart size changes.
    The normalized affinity is the standard geometry behind the normalized graph
    Laplacian and removes that degree-count artefact without introducing a tuned
    regularization constant.  Isolated/missing vertices remain exact zero rows.
    """

    graph = csr_matrix(affinity, dtype=np.float64)
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("affinity must be square")
    if graph.nnz and (not np.all(np.isfinite(graph.data)) or np.any(graph.data < 0)):
        raise ValueError("affinity weights must be finite and non-negative")
    # Enforce the undirected metric contract before degree normalization.
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    scale = np.zeros_like(degree, dtype=np.float64)
    positive = degree > 0
    scale[positive] = degree[positive] ** -0.5
    normalized = (diags(scale) @ graph @ diags(scale)).tocsr()
    normalized.eliminate_zeros()
    return normalized.astype(np.float32)


def _edge_gradient_rms(
    field: np.ndarray,
    affinity: csr_matrix,
    *,
    axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper = triu(affinity, k=1, format="coo")
    if not len(upper.data):
        return upper.row, upper.col, np.zeros(0, dtype=np.float32)
    if axis == 0:
        difference = field[upper.row] - field[upper.col]
    elif axis == 1:
        difference = field[:, upper.row].T - field[:, upper.col].T
    else:
        raise ValueError("axis must be 0 (reaction) or 1 (protein)")
    gradient = np.sqrt(np.mean(np.square(difference, dtype=np.float64), axis=1))
    return upper.row, upper.col, gradient.astype(np.float32)


def graph_roughness(field: np.ndarray, affinity: csr_matrix, *, axis: int) -> float:
    """Weighted mean squared edge gradient, normalized per field coordinate."""

    rows, columns, gradient = _edge_gradient_rms(field, affinity, axis=axis)
    if not len(gradient):
        return 0.0
    weights = np.asarray(affinity[rows, columns]).reshape(-1).astype(np.float64)
    total = weights.sum()
    if total <= 0:
        return 0.0
    return float(np.sum(weights * np.square(gradient, dtype=np.float64)) / total)


def _upper_affinity_edges(
    affinity: csr_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper = triu(affinity, k=1, format="coo")
    return (
        upper.row.astype(np.int64, copy=False),
        upper.col.astype(np.int64, copy=False),
        upper.data.astype(np.float64, copy=False),
    )


def product_characteristic_scale(
    field: np.ndarray,
    reaction_graph: csr_matrix,
    protein_graphs: list[csr_matrix],
    *,
    epsilon: float = 1e-8,
) -> float:
    """RMS scalar edge gradient under the Cartesian-product metric.

    ``F(r, e)`` is a scalar field on product-graph vertices.  Reaction edges are
    therefore replicated once for every enzyme, while protein edges are
    replicated once for every reaction.  Multiple protein resolutions are
    averaged, matching the direct-sum metric used by the variational energy.
    """

    values = np.asarray(field, dtype=np.float64)
    reaction_rows, reaction_cols, reaction_weights = _upper_affinity_edges(
        reaction_graph
    )
    reaction_square = 0.0
    reaction_mass = 0.0
    if len(reaction_weights):
        difference = values[reaction_rows] - values[reaction_cols]
        reaction_square = float(
            np.sum(reaction_weights[:, None] * np.square(difference))
        )
        reaction_mass = float(reaction_weights.sum() * values.shape[1])

    protein_squares: list[float] = []
    protein_masses: list[float] = []
    for graph in protein_graphs:
        rows, cols, weights = _upper_affinity_edges(graph)
        if not len(weights):
            protein_squares.append(0.0)
            protein_masses.append(0.0)
            continue
        difference = values[:, rows] - values[:, cols]
        protein_squares.append(
            float(np.sum(weights[None, :] * np.square(difference)))
        )
        protein_masses.append(float(weights.sum() * values.shape[0]))

    protein_square = float(np.mean(protein_squares)) if protein_squares else 0.0
    protein_mass = float(np.mean(protein_masses)) if protein_masses else 0.0
    total_square = reaction_square + protein_square
    total_mass = reaction_mass + protein_mass
    if total_mass > 0 and total_square > 0:
        return max(float(np.sqrt(total_square / total_mass)), epsilon)
    return max(float(np.std(values)), epsilon)


def _scalar_graph_energy(
    field: np.ndarray,
    affinity: csr_matrix,
    *,
    axis: int,
    scale: float,
) -> float:
    rows, cols, weights = _upper_affinity_edges(affinity)
    if not len(weights):
        return 0.0
    if axis == 0:
        difference = np.asarray(field[rows] - field[cols], dtype=np.float64)
        weighted_potential = weights[:, None]
    elif axis == 1:
        difference = np.asarray(field[:, rows] - field[:, cols], dtype=np.float64)
        weighted_potential = weights[None, :]
    else:
        raise ValueError("axis must be 0 (reaction) or 1 (protein)")
    ratio = difference / float(scale)
    potential = (scale**2) * (np.sqrt(1.0 + np.square(ratio)) - 1.0)
    return float(np.sum(weighted_potential * potential))


def _scalar_graph_flux(
    field: np.ndarray,
    affinity: csr_matrix,
    *,
    axis: int,
    scale: float,
) -> np.ndarray:
    """Gradient of one scalar Charbonnier Dirichlet term on product edges."""

    rows, cols, weights = _upper_affinity_edges(affinity)
    flux = np.zeros_like(field, dtype=np.float64)
    if not len(weights):
        return flux
    if axis == 0:
        difference = np.asarray(field[rows] - field[cols], dtype=np.float64)
        conductance = 1.0 / np.sqrt(1.0 + np.square(difference / scale))
        contribution = weights[:, None] * conductance * difference
        np.add.at(flux, rows, contribution)
        np.add.at(flux, cols, -contribution)
    elif axis == 1:
        difference = np.asarray(field[:, rows] - field[:, cols], dtype=np.float64)
        conductance = 1.0 / np.sqrt(1.0 + np.square(difference / scale))
        contribution = (weights[None, :] * conductance * difference).T
        transposed = flux.T
        np.add.at(transposed, rows, contribution)
        np.add.at(transposed, cols, -contribution)
    else:
        raise ValueError("axis must be 0 (reaction) or 1 (protein)")
    return flux


def _differential_energy(
    field: np.ndarray,
    differential: _GraphDifferential,
    *,
    axis: int,
    scale: float,
) -> float:
    """Charbonnier energy expressed through the discrete exterior derivative."""

    B = differential.incidence
    if not len(differential.weights):
        return 0.0
    if axis == 0:
        edge_difference = np.asarray(B.T @ field, dtype=np.float64)
        weighted = differential.weights[:, None]
    elif axis == 1:
        edge_difference = np.asarray(field @ B, dtype=np.float64)
        weighted = differential.weights[None, :]
    else:
        raise ValueError("axis must be 0 (reaction) or 1 (protein)")
    ratio = edge_difference / float(scale)
    potential = (scale**2) * (np.sqrt(1.0 + np.square(ratio)) - 1.0)
    return float(np.sum(weighted * potential))


def _differential_flux(
    field: np.ndarray,
    differential: _GraphDifferential,
    *,
    axis: int,
    scale: float,
) -> np.ndarray:
    """Compute d* (w c(dF) dF) with sparse incidence products."""

    B = differential.incidence
    if not len(differential.weights):
        return np.zeros_like(field, dtype=np.float64)
    if axis == 0:
        edge_difference = np.asarray(B.T @ field, dtype=np.float64)
        conductance = 1.0 / np.sqrt(1.0 + np.square(edge_difference / scale))
        edge_flux = differential.weights[:, None] * conductance * edge_difference
        return np.asarray(B @ edge_flux, dtype=np.float64)
    if axis == 1:
        edge_difference = np.asarray(field @ B, dtype=np.float64)
        conductance = 1.0 / np.sqrt(1.0 + np.square(edge_difference / scale))
        edge_flux = differential.weights[None, :] * conductance * edge_difference
        return np.asarray(edge_flux @ B.T, dtype=np.float64)
    raise ValueError("axis must be 0 (reaction) or 1 (protein)")


def _robust_product_energy_terms_differential(
    field: np.ndarray,
    prior: np.ndarray,
    reaction_differential: _GraphDifferential,
    protein_differentials: list[_GraphDifferential],
    *,
    strength: float,
    scale: float,
    anchor_weights: np.ndarray | None = None,
    anchor_targets: np.ndarray | None = None,
    anchor_strength: float = 0.0,
    positive_measure: np.ndarray | None = None,
    positive_strength: float = 0.0,
) -> dict[str, float]:
    field64 = np.asarray(field, dtype=np.float64)
    prior64 = np.asarray(prior, dtype=np.float64)
    fidelity = 0.5 * float(np.square(field64 - prior64).sum())
    reaction_energy = _differential_energy(
        field64, reaction_differential, axis=0, scale=scale
    )
    protein_energy = (
        float(
            np.mean(
                [
                    _differential_energy(field64, geometry, axis=1, scale=scale)
                    for geometry in protein_differentials
                ]
            )
        )
        if protein_differentials
        else 0.0
    )
    geometry = reaction_energy + protein_energy
    anchor_energy = 0.0
    if anchor_strength and anchor_weights is not None and anchor_targets is not None:
        anchor_energy = 0.5 * float(anchor_strength) * float(
            np.sum(
                np.asarray(anchor_weights, dtype=np.float64)
                * np.square(field64 - np.asarray(anchor_targets, dtype=np.float64))
            )
        )
    positive_source = 0.0
    if positive_strength and positive_measure is not None:
        positive_source = -float(positive_strength) * float(scale) * float(
            np.sum(np.asarray(positive_measure, dtype=np.float64) * field64)
        )
    return {
        "fidelity": fidelity,
        "anchors": anchor_energy,
        "positive_source": positive_source,
        "reaction_geometry": reaction_energy,
        "protein_geometry": protein_energy,
        "geometry": geometry,
        "weighted_geometry": float(strength) * geometry,
        "total": fidelity + anchor_energy + positive_source + float(strength) * geometry,
    }


def robust_product_energy(
    field: np.ndarray,
    prior: np.ndarray,
    reaction_graph: csr_matrix,
    protein_graphs: list[csr_matrix],
    *,
    strength: float,
    scale: float,
    anchor_weights: np.ndarray | None = None,
    anchor_targets: np.ndarray | None = None,
    anchor_strength: float = 0.0,
    positive_measure: np.ndarray | None = None,
    positive_strength: float = 0.0,
) -> float:
    terms = robust_product_energy_terms(
        field,
        prior,
        reaction_graph,
        protein_graphs,
        strength=strength,
        scale=scale,
        anchor_weights=anchor_weights,
        anchor_targets=anchor_targets,
        anchor_strength=anchor_strength,
        positive_measure=positive_measure,
        positive_strength=positive_strength,
    )
    return float(terms["total"])


def robust_product_energy_terms(
    field: np.ndarray,
    prior: np.ndarray,
    reaction_graph: csr_matrix,
    protein_graphs: list[csr_matrix],
    *,
    strength: float,
    scale: float,
    anchor_weights: np.ndarray | None = None,
    anchor_targets: np.ndarray | None = None,
    anchor_strength: float = 0.0,
    positive_measure: np.ndarray | None = None,
    positive_strength: float = 0.0,
) -> dict[str, float]:
    """Terms of the scalar robust energy on the Cartesian product graph."""

    field64 = np.asarray(field, dtype=np.float64)
    prior64 = np.asarray(prior, dtype=np.float64)
    fidelity = 0.5 * float(np.square(field64 - prior64).sum())
    reaction_energy = _scalar_graph_energy(
        field64, reaction_graph, axis=0, scale=scale
    )
    protein_energy = (
        float(
            np.mean(
                [
                    _scalar_graph_energy(field64, graph, axis=1, scale=scale)
                    for graph in protein_graphs
                ]
            )
        )
        if protein_graphs
        else 0.0
    )
    geometry = reaction_energy + protein_energy
    anchor_energy = 0.0
    if anchor_strength and anchor_weights is not None and anchor_targets is not None:
        anchor_energy = 0.5 * float(anchor_strength) * float(
            np.sum(
                np.asarray(anchor_weights, dtype=np.float64)
                * np.square(field64 - np.asarray(anchor_targets, dtype=np.float64))
            )
        )
    positive_source = 0.0
    if positive_strength and positive_measure is not None:
        measure = np.asarray(positive_measure, dtype=np.float64)
        positive_source = -float(positive_strength) * float(scale) * float(
            np.sum(measure * field64)
        )
    return {
        "fidelity": fidelity,
        "anchors": anchor_energy,
        "positive_source": positive_source,
        "reaction_geometry": reaction_energy,
        "protein_geometry": protein_energy,
        "geometry": geometry,
        "weighted_geometry": float(strength) * geometry,
        "total": fidelity + anchor_energy + positive_source + float(strength) * geometry,
    }


def anisotropic_product_field(
    prior: np.ndarray,
    reaction_graph: csr_matrix,
    protein_graphs: list[csr_matrix],
    *,
    strength: float = 1.0,
    anchor_weights: np.ndarray | None = None,
    anchor_targets: np.ndarray | None = None,
    anchor_strength: float = 0.0,
    positive_measure: np.ndarray | None = None,
    positive_strength: float = 0.0,
    max_steps: int = 32,
    tolerance: float = 1e-5,
) -> tuple[np.ndarray, dict[str, object]]:
    """Scalar anisotropic diffusion on the enzyme-reaction product graph.

    Every vertex is one enzyme-reaction pair and every product edge receives its
    own Charbonnier conductance from the local scalar compatibility difference.
    Optional observed enzyme-reaction pairs enter the same variational energy.
    ``positive_measure`` is row-normalized on observed fibres and contributes a
    positive-only linear source ``-sigma <A,F>``: observed pairs are pulled up,
    while an unobserved pair receives exactly zero direct observation gradient.
    Zero-shot, one-seed and multi-seed inference therefore differ only by the
    supplied observation measure, not by model type.

    This is the Euler gradient flow of one fixed robust objective, not a query
    router or a view-wise score blend. The explicit step size uses the global
    curvature bound ``1 / (1 + mu a_max + 2 lambda d_max)``.
    """

    prior = np.asarray(prior, dtype=np.float64)
    if prior.ndim != 2:
        raise ValueError("prior must be reaction-by-protein")
    if reaction_graph.shape != (prior.shape[0], prior.shape[0]):
        raise ValueError("reaction graph shape mismatch")
    for graph in protein_graphs:
        if graph.shape != (prior.shape[1], prior.shape[1]):
            raise ValueError("protein graph shape mismatch")
    if strength < 0:
        raise ValueError("strength must be non-negative")
    if anchor_strength < 0:
        raise ValueError("anchor_strength must be non-negative")
    if positive_strength < 0:
        raise ValueError("positive_strength must be non-negative")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    checked_anchor_weights: np.ndarray | None = None
    checked_anchor_targets: np.ndarray | None = None
    if anchor_weights is not None:
        checked_anchor_weights = np.asarray(anchor_weights, dtype=np.float64)
        if checked_anchor_weights.shape != prior.shape:
            raise ValueError("anchor_weights must have the same shape as prior")
        if np.any(checked_anchor_weights < 0) or not np.all(np.isfinite(checked_anchor_weights)):
            raise ValueError("anchor_weights must be finite and non-negative")
        if anchor_targets is None:
            raise ValueError("anchor_targets are required when anchor_weights are supplied")
        checked_anchor_targets = np.asarray(anchor_targets, dtype=np.float64)
        if checked_anchor_targets.shape != prior.shape:
            raise ValueError("anchor_targets must have the same shape as prior")
        if not np.all(np.isfinite(checked_anchor_targets)):
            raise ValueError("anchor_targets must be finite")
    elif anchor_targets is not None:
        raise ValueError("anchor_weights are required when anchor_targets are supplied")

    checked_positive_measure: np.ndarray | None = None
    if positive_measure is not None:
        checked_positive_measure = np.asarray(positive_measure, dtype=np.float64)
        if checked_positive_measure.shape != prior.shape:
            raise ValueError("positive_measure must have the same shape as prior")
        if np.any(checked_positive_measure < 0) or not np.all(np.isfinite(checked_positive_measure)):
            raise ValueError("positive_measure must be finite and non-negative")
        row_mass = checked_positive_measure.sum(axis=1, keepdims=True)
        observed_rows = row_mass[:, 0] > 0
        checked_positive_measure = checked_positive_measure.copy()
        checked_positive_measure[observed_rows] /= row_mass[observed_rows]

    scale = product_characteristic_scale(prior, reaction_graph, protein_graphs)
    reaction_differential = _graph_differential(reaction_graph)
    protein_differentials = [_graph_differential(graph) for graph in protein_graphs]
    reaction_degree = np.asarray(reaction_graph.sum(axis=1)).reshape(-1)
    if protein_graphs:
        protein_degree = np.mean(
            [np.asarray(graph.sum(axis=1)).reshape(-1) for graph in protein_graphs],
            axis=0,
        )
    else:
        protein_degree = np.zeros(prior.shape[1], dtype=np.float64)
    max_product_degree = float(reaction_degree.max(initial=0.0)) + float(
        protein_degree.max(initial=0.0)
    )
    anchor_weight_max = (
        float(np.max(checked_anchor_weights))
        if checked_anchor_weights is not None and checked_anchor_weights.size
        else 0.0
    )
    step_size = 1.0 / (
        1.0
        + float(anchor_strength) * anchor_weight_max
        + 2.0 * float(strength) * max_product_degree
    )

    initial_energy_terms = _robust_product_energy_terms_differential(
        prior,
        prior,
        reaction_differential,
        protein_differentials,
        strength=strength,
        scale=scale,
        anchor_weights=checked_anchor_weights,
        anchor_targets=checked_anchor_targets,
        anchor_strength=anchor_strength,
        positive_measure=checked_positive_measure,
        positive_strength=positive_strength,
    )
    initial_energy = float(initial_energy_terms["total"])
    if (strength == 0 and anchor_strength == 0 and positive_strength == 0) or max_steps == 0:
        return prior.astype(np.float32), {
            "characteristic_scale": scale,
            "max_product_degree": max_product_degree,
            "step_size": step_size,
            "energy_history": [initial_energy],
            "relative_change_history": [],
            "steps": 0,
            "final_energy_terms": initial_energy_terms,
        }

    field = prior.copy()
    energy_history = [initial_energy]
    relative_change_history: list[float] = []
    protein_factor = 1.0 / len(protein_graphs) if protein_graphs else 0.0

    for _ in range(max_steps):
        geometry_gradient = _differential_flux(
            field, reaction_differential, axis=0, scale=scale
        )
        if protein_differentials:
            protein_gradient = np.zeros_like(field)
            for geometry in protein_differentials:
                protein_gradient += _differential_flux(
                    field, geometry, axis=1, scale=scale
                )
            geometry_gradient += protein_factor * protein_gradient
        gradient = (field - prior) + float(strength) * geometry_gradient
        if (
            anchor_strength
            and checked_anchor_weights is not None
            and checked_anchor_targets is not None
        ):
            gradient += (
                float(anchor_strength)
                * checked_anchor_weights
                * (field - checked_anchor_targets)
            )
        if positive_strength and checked_positive_measure is not None:
            gradient -= (
                float(positive_strength)
                * float(scale)
                * checked_positive_measure
            )
        updated = field - step_size * gradient
        relative_change = float(
            np.linalg.norm(updated - field)
            / max(np.linalg.norm(field), np.finfo(np.float64).eps)
        )
        field = updated
        energy_history.append(
            float(
                _robust_product_energy_terms_differential(
                    field,
                    prior,
                    reaction_differential,
                    protein_differentials,
                    strength=strength,
                    scale=scale,
                    anchor_weights=checked_anchor_weights,
                    anchor_targets=checked_anchor_targets,
                    anchor_strength=anchor_strength,
                    positive_measure=checked_positive_measure,
                    positive_strength=positive_strength,
                )["total"]
            )
        )
        relative_change_history.append(relative_change)
        if relative_change <= tolerance:
            break

    return np.asarray(field, dtype=np.float32), {
        "characteristic_scale": scale,
        "max_product_degree": max_product_degree,
        "step_size": step_size,
        "energy_history": energy_history,
        "relative_change_history": relative_change_history,
        "steps": len(relative_change_history),
        "final_energy_terms": _robust_product_energy_terms_differential(
            field,
            prior,
            reaction_differential,
            protein_differentials,
            strength=strength,
            scale=scale,
            anchor_weights=checked_anchor_weights,
            anchor_targets=checked_anchor_targets,
            anchor_strength=anchor_strength,
            positive_measure=checked_positive_measure,
            positive_strength=positive_strength,
        ),
    }


def anisotropic_bregman_product_field(
    prior: np.ndarray,
    reaction_graph: csr_matrix,
    protein_graphs: list[csr_matrix],
    *,
    strength: float = 1.0,
    positive_measure: np.ndarray | None = None,
    positive_strength: float = 1.0,
    normalize_positive_rows: bool = True,
    integral_normalized: bool = False,
    accelerated: bool = False,
    max_steps: int = 64,
    tolerance: float = 1e-5,
) -> tuple[np.ndarray, dict[str, object]]:
    """Observation-driven nonlinear deformation of a compatibility field.

    Let ``Phi`` be the Charbonnier product-graph energy.  This minimizes

        1/2 ||F-F0||^2 + lambda D_Phi(F,F0)
        - mu sigma <A, F-F0>,

    where ``D_Phi`` is the Bregman divergence of ``Phi`` and ``A`` is a
    non-negative positive empirical relation measure.  Raw empirical relation
    rows are normalized by default; callers that already geometrically transport
    a normalized measure can disable this second normalization so transported
    support mass remains informative.  Because a Bregman
    divergence is minimized at its reference, ``F0`` is an exact equilibrium
    when no positive observation is supplied.  With observations, the geometry
    gradient is ``grad Phi(F) - grad Phi(F0)``: conductance is still recomputed
    from the *current* compatibility field, so the deformation remains genuinely
    nonlinear/adaptive rather than becoming a fixed residual smoother.

    Missing pairs receive no direct observation gradient.  Missing geometric
    views remain zero-degree vertices/edges according to their input graphs.
    """

    prior = np.asarray(prior, dtype=np.float64)
    if prior.ndim != 2:
        raise ValueError("prior must be reaction-by-protein")
    if reaction_graph.shape != (prior.shape[0], prior.shape[0]):
        raise ValueError("reaction graph shape mismatch")
    for graph in protein_graphs:
        if graph.shape != (prior.shape[1], prior.shape[1]):
            raise ValueError("protein graph shape mismatch")
    if strength < 0 or positive_strength < 0:
        raise ValueError("strengths must be non-negative")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    measure: np.ndarray | None = None
    if positive_measure is not None:
        measure = np.asarray(positive_measure, dtype=np.float64)
        if measure.shape != prior.shape:
            raise ValueError("positive_measure must have the same shape as prior")
        if np.any(measure < 0) or not np.all(np.isfinite(measure)):
            raise ValueError("positive_measure must be finite and non-negative")
        measure = measure.copy()
        if normalize_positive_rows:
            mass = measure.sum(axis=1, keepdims=True)
            observed = mass[:, 0] > 0
            measure[observed] /= mass[observed]

    scale = product_characteristic_scale(prior, reaction_graph, protein_graphs)
    reaction_differential = _graph_differential(reaction_graph)
    protein_differentials = [_graph_differential(graph) for graph in protein_graphs]
    protein_factor = 1.0 / len(protein_differentials) if protein_differentials else 0.0
    vertex_mass = float(prior.size) if integral_normalized else 1.0
    reaction_edge_mass = float(np.sum(reaction_differential.weights) * prior.shape[1])
    protein_edge_masses = [
        float(np.sum(geometry.weights) * prior.shape[0])
        for geometry in protein_differentials
    ]
    reaction_energy_scale = (
        1.0 / reaction_edge_mass
        if integral_normalized and reaction_edge_mass > 0
        else 1.0
    )
    protein_energy_scales = [
        (1.0 / mass if integral_normalized and mass > 0 else 1.0)
        for mass in protein_edge_masses
    ]

    def geometry_flux(value: np.ndarray) -> np.ndarray:
        flux = reaction_energy_scale * _differential_flux(
            value, reaction_differential, axis=0, scale=scale
        )
        if protein_differentials:
            pflux = np.zeros_like(value, dtype=np.float64)
            for geometry, local_scale in zip(
                protein_differentials, protein_energy_scales
            ):
                pflux += local_scale * _differential_flux(
                    value, geometry, axis=1, scale=scale
                )
            flux += protein_factor * pflux
        return flux

    def geometry_energy(value: np.ndarray) -> float:
        reaction_energy = reaction_energy_scale * _differential_energy(
            value, reaction_differential, axis=0, scale=scale
        )
        protein_energy = (
            float(
                np.mean(
                    [
                        local_scale
                        * _differential_energy(value, geometry, axis=1, scale=scale)
                        for geometry, local_scale in zip(
                            protein_differentials, protein_energy_scales
                        )
                    ]
                )
            )
            if protein_differentials
            else 0.0
        )
        return float(reaction_energy + protein_energy)

    reference_flux = geometry_flux(prior)
    reference_energy = geometry_energy(prior)

    reaction_degree = np.asarray(reaction_graph.sum(axis=1)).reshape(-1)
    reaction_curvature = reaction_energy_scale * float(
        reaction_degree.max(initial=0.0)
    )
    if protein_graphs:
        protein_curvature = float(
            np.mean(
                [
                    local_scale
                    * float(np.asarray(graph.sum(axis=1)).reshape(-1).max(initial=0.0))
                    for graph, local_scale in zip(
                        protein_graphs, protein_energy_scales
                    )
                ]
            )
        )
    else:
        protein_curvature = 0.0
    max_product_degree = reaction_curvature + protein_curvature
    fidelity_curvature = 1.0 / vertex_mass
    step_size = 1.0 / (
        fidelity_curvature + 2.0 * float(strength) * max_product_degree
    )

    def energy_terms(value: np.ndarray) -> dict[str, float]:
        delta = np.asarray(value, dtype=np.float64) - prior
        fidelity = 0.5 * float(np.square(delta).sum()) / vertex_mass
        phi = geometry_energy(value)
        bregman = float(phi - reference_energy - np.sum(reference_flux * delta))
        # Numerical cancellation can create tiny negative values at the reference.
        if bregman < 0 and abs(bregman) <= 1e-9 * max(1.0, abs(phi), abs(reference_energy)):
            bregman = 0.0
        source = 0.0
        if positive_strength and measure is not None:
            source = -float(positive_strength) * float(scale) * float(
                np.sum(measure * delta)
            )
        return {
            "fidelity": fidelity,
            "bregman_geometry": bregman,
            "weighted_bregman_geometry": float(strength) * bregman,
            "positive_source": source,
            "total": fidelity + float(strength) * bregman + source,
        }

    initial_terms = energy_terms(prior)
    initial_energy = float(initial_terms["total"])
    if max_steps == 0 or positive_strength == 0 or measure is None or not np.any(measure):
        return prior.astype(np.float32), {
            "characteristic_scale": scale,
            "max_product_degree": max_product_degree,
            "step_size": step_size,
            "energy_history": [initial_energy],
            "relative_change_history": [],
            "steps": 0,
            "final_energy_terms": initial_terms,
        }

    field = prior.copy()
    previous = field.copy()
    momentum_t = 1.0
    energy_history = [initial_energy]
    relative_change_history: list[float] = []
    for _ in range(max_steps):
        evaluation_point = field
        next_momentum_t = momentum_t
        if accelerated:
            next_momentum_t = 0.5 * (
                1.0 + np.sqrt(1.0 + 4.0 * momentum_t * momentum_t)
            )
            momentum = (momentum_t - 1.0) / next_momentum_t
            evaluation_point = field + momentum * (field - previous)

        gradient = (evaluation_point - prior) / vertex_mass + float(strength) * (
            geometry_flux(evaluation_point) - reference_flux
        )
        gradient -= float(positive_strength) * float(scale) * measure
        updated = evaluation_point - step_size * gradient
        updated_energy = float(energy_terms(updated)["total"])

        # Monotone restart: acceleration may overshoot despite the safe gradient
        # step.  Restarting from the current iterate preserves energy descent and
        # the exact same variational objective.
        if accelerated and updated_energy > energy_history[-1] + 1e-12:
            evaluation_point = field
            momentum_t = 1.0
            next_momentum_t = 0.5 * (1.0 + np.sqrt(5.0))
            gradient = (field - prior) / vertex_mass + float(strength) * (
                geometry_flux(field) - reference_flux
            )
            gradient -= float(positive_strength) * float(scale) * measure
            updated = field - step_size * gradient
            updated_energy = float(energy_terms(updated)["total"])
        relative_change = float(
            np.linalg.norm(updated - field)
            / max(np.linalg.norm(field), np.finfo(np.float64).eps)
        )
        previous, field = field, updated
        momentum_t = next_momentum_t
        energy_history.append(updated_energy)
        relative_change_history.append(relative_change)
        if relative_change <= tolerance:
            break

    return np.asarray(field, dtype=np.float32), {
        "characteristic_scale": scale,
        "max_product_degree": max_product_degree,
        "step_size": step_size,
        "energy_history": energy_history,
        "relative_change_history": relative_change_history,
        "steps": len(relative_change_history),
        "final_energy_terms": energy_terms(field),
    }


def smooth_product_field(
    prior: np.ndarray,
    reaction_operator: csr_matrix,
    protein_operator: csr_matrix,
    *,
    reaction_lambda: float,
    protein_lambda: float,
    steps: int,
) -> np.ndarray:
    """Solve a graph-regularized compatibility field by fixed-point iteration.

    The fixed point satisfies the Euler equation of

        1/2 ||F-F0||_F^2
        + lambda_r/2 tr(F^T L_r F)
        + lambda_p/2 tr(F L_p F^T),

    with normalized graph Laplacians L_r = I-S_r and L_p = I-S_p.
    This is a standard Cartesian-product graph smoothness objective rather than
    an architectural analogy.  ``prior`` is never overwritten; zero steps
    therefore reproduces the exact incoming score field.
    """

    prior = np.asarray(prior, dtype=np.float32)
    if prior.ndim != 2:
        raise ValueError("prior must be a reaction-by-protein matrix")
    if reaction_operator.shape != (prior.shape[0], prior.shape[0]):
        raise ValueError("reaction operator shape mismatch")
    if protein_operator.shape != (prior.shape[1], prior.shape[1]):
        raise ValueError("protein operator shape mismatch")
    if reaction_lambda < 0 or protein_lambda < 0:
        raise ValueError("regularization weights must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    field = prior.copy()
    denominator = 1.0 + reaction_lambda + protein_lambda
    for _ in range(steps):
        reaction_transport = np.asarray(reaction_operator @ field, dtype=np.float32)
        protein_transport = np.asarray(protein_operator @ field.T, dtype=np.float32).T
        field = (
            prior
            + reaction_lambda * reaction_transport
            + protein_lambda * protein_transport
        ) / denominator
    return np.asarray(field, dtype=np.float32)


def smooth_product_field_multiview(
    prior: np.ndarray,
    reaction_operator: csr_matrix,
    protein_terms: list[tuple[csr_matrix, float, np.ndarray]],
    *,
    reaction_lambda: float,
    steps: int,
) -> np.ndarray:
    """Solve a multi-view product-graph Dirichlet field.

    Each protein term is ``(S_m, lambda_m, active_m)`` and contributes

        lambda_m / 2 * tr(F (M_m - S_m) F^T),

    where ``M_m = diag(active_m)``.  A sequence view uses an all-one active
    mask; a structure view uses only proteins with observed structure.  This
    keeps distinct geometric views as distinct Laplacians instead of averaging
    incomparable raw similarities.
    """

    prior = np.asarray(prior, dtype=np.float32)
    if prior.ndim != 2:
        raise ValueError("prior must be a reaction-by-protein matrix")
    if reaction_operator.shape != (prior.shape[0], prior.shape[0]):
        raise ValueError("reaction operator shape mismatch")
    if reaction_lambda < 0:
        raise ValueError("reaction regularization must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    checked_terms: list[tuple[csr_matrix, float, np.ndarray]] = []
    denominator = np.full(prior.shape[1], 1.0 + reaction_lambda, dtype=np.float32)
    for operator, weight, active in protein_terms:
        if operator.shape != (prior.shape[1], prior.shape[1]):
            raise ValueError("protein operator shape mismatch")
        if weight < 0:
            raise ValueError("protein regularization must be non-negative")
        active = np.asarray(active, dtype=np.float32).reshape(-1)
        if len(active) != prior.shape[1] or np.any((active < 0) | (active > 1)):
            raise ValueError("protein active mask must lie in [0,1]")
        checked_terms.append((operator, weight, active))
        denominator += weight * active

    field = prior.copy()
    for _ in range(steps):
        numerator = prior + reaction_lambda * np.asarray(
            reaction_operator @ field, dtype=np.float32
        )
        for operator, weight, _active in checked_terms:
            if weight:
                numerator += weight * np.asarray(operator @ field.T, dtype=np.float32).T
        field = numerator / denominator[None, :]
    return np.asarray(field, dtype=np.float32)


def axis_transport_diagnostics(
    field: np.ndarray,
    reaction_operator: csr_matrix,
    protein_operator: csr_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one-step transport and their local disagreement.

    ``disagreement`` is deliberately not called curvature: without an explicit
    connection between local frames it is only a conservative diagnostic for
    places where reaction-side and protein-side neighborhood evidence disagree.
    """

    field = np.asarray(field, dtype=np.float32)
    reaction_transport = np.asarray(reaction_operator @ field, dtype=np.float32)
    protein_transport = np.asarray(protein_operator @ field.T, dtype=np.float32).T
    scale = np.abs(reaction_transport) + np.abs(protein_transport) + 1e-8
    disagreement = np.abs(reaction_transport - protein_transport) / scale
    return reaction_transport, protein_transport, disagreement.astype(np.float32)


def availability_weighted_similarity(
    base_similarity: np.ndarray,
    views: list[tuple[np.ndarray, np.ndarray, float]],
) -> np.ndarray:
    """Combine optional geometric views without treating missing data as evidence.

    Each optional view is ``(similarity, available_mask, weight)``.  A pair uses
    that view only when both endpoints are available.  The base geometry is
    always present and has unit weight.  This is the hook used by later 3Di,
    dynamic-contact, or explicit-structure resolutions.
    """

    base = _validate_square(base_similarity, "base similarity").astype(np.float64)
    numerator = base.copy()
    denominator = np.ones_like(base, dtype=np.float64)
    for similarity, available, weight in views:
        if weight < 0:
            raise ValueError("view weight must be non-negative")
        local = _validate_square(similarity, "view similarity")
        if local.shape != base.shape:
            raise ValueError("view similarity shape mismatch")
        available = np.asarray(available, dtype=bool).reshape(-1)
        if len(available) != base.shape[0]:
            raise ValueError("availability mask length mismatch")
        pair_available = np.outer(available, available)
        numerator[pair_available] += weight * local[pair_available]
        denominator[pair_available] += weight
    return (numerator / denominator).astype(np.float32)
