from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class ProductKernelDiagnostics:
    node_count: int
    view_count: int
    graph_k: int
    directed_edges: int
    undirected_edges: int


def binary_jaccard_distance(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise Jaccard distance for a binary geometric view.

    A zero row is treated as an unavailable observation, not as a point at
    maximal distance from everything else.  The returned availability mask is
    therefore part of the geometric object.
    """
    x = np.asarray(features)
    if x.ndim != 2:
        raise ValueError("features must be a matrix")
    x = (x > 0).astype(np.float64, copy=False)
    mass = x.sum(axis=1)
    available = mass > 0
    intersection = x @ x.T
    union = mass[:, None] + mass[None, :] - intersection
    similarity = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )
    distance = 1.0 - np.clip(similarity, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    return distance, available


def binary_jaccard_cross_distance(
    query: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, bool, np.ndarray]:
    """Jaccard distance from one binary point to a reference set."""
    q = (np.asarray(query).reshape(-1) > 0).astype(np.float64, copy=False)
    r = (np.asarray(reference) > 0).astype(np.float64, copy=False)
    if r.ndim != 2 or r.shape[1] != q.shape[0]:
        raise ValueError("query/reference dimension mismatch")
    qmass = float(q.sum())
    rmass = r.sum(axis=1)
    q_available = qmass > 0
    r_available = rmass > 0
    intersection = r @ q
    union = rmass + qmass - intersection
    similarity = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )
    return 1.0 - np.clip(similarity, 0.0, 1.0), bool(q_available), r_available


def _self_tuning_scale(
    distance: np.ndarray,
    available: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    n = distance.shape[0]
    sigma = np.full(n, np.nan, dtype=np.float64)
    selected = np.flatnonzero(available)
    if len(selected) < 2:
        return sigma
    local = np.asarray(distance[np.ix_(selected, selected)], dtype=np.float64).copy()
    np.fill_diagonal(local, np.inf)
    k = min(len(selected) - 1, max(1, int(math.ceil(math.sqrt(len(selected))))))
    local_sigma = np.partition(local, k - 1, axis=1)[:, k - 1]
    positive = local_sigma[np.isfinite(local_sigma) & (local_sigma > epsilon)]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    local_sigma = np.where(
        np.isfinite(local_sigma) & (local_sigma > epsilon), local_sigma, fallback
    )
    sigma[selected] = local_sigma
    return sigma


def self_tuning_product_affinity(
    distances: list[np.ndarray],
    availabilities: list[np.ndarray] | None = None,
    *,
    k: int | None = None,
    epsilon: float = 1e-8,
) -> tuple[csr_matrix, ProductKernelDiagnostics]:
    """Geometric-mean local kernel on an availability-neutral product manifold.

    For view v, local conformal scaling gives
        E_v(i,j) = d_v(i,j)^2 / (sigma_v(i) sigma_v(j)).
    On pairs observed in m views, the product kernel is
        K(i,j) = exp(- mean_v E_v(i,j)),
    exactly the geometric mean of the self-tuning Gaussian kernels of the
    observed coordinate views. Missing views do not contribute positive or
    negative evidence. No learned or hand-tuned inter-view weight is used.

    Sparsification uses the same sqrt(N) sampling-scale rule as the existing
    graph discretisation when k is omitted.
    """
    if not distances:
        raise ValueError("at least one distance view is required")
    mats = [np.asarray(d, dtype=np.float64) for d in distances]
    n = mats[0].shape[0]
    if any(d.shape != (n, n) for d in mats):
        raise ValueError("all distance views must have the same square shape")
    if any((not np.all(np.isfinite(d))) or np.any(d < 0) for d in mats):
        raise ValueError("distance views must be finite and non-negative")
    if availabilities is None:
        masks = [np.ones(n, dtype=bool) for _ in mats]
    else:
        if len(availabilities) != len(mats):
            raise ValueError("one availability mask is required per view")
        masks = [np.asarray(a, dtype=bool).reshape(-1) for a in availabilities]
        if any(len(a) != n for a in masks):
            raise ValueError("availability mask length mismatch")
    if n < 2:
        return csr_matrix((n, n), dtype=np.float32), ProductKernelDiagnostics(n, len(mats), 0, 0, 0)
    graph_k = min(n - 1, max(1, int(k) if k is not None else int(math.ceil(math.sqrt(n)))))

    energy_sum = np.zeros((n, n), dtype=np.float64)
    observed_count = np.zeros((n, n), dtype=np.int16)
    for d, available in zip(mats, masks):
        sigma = _self_tuning_scale(d, available, epsilon=epsilon)
        pair_available = available[:, None] & available[None, :]
        denom = sigma[:, None] * sigma[None, :]
        valid = pair_available & np.isfinite(denom) & (denom > epsilon)
        energy = np.zeros((n, n), dtype=np.float64)
        energy[valid] = np.square(d[valid]) / denom[valid]
        energy_sum[valid] += energy[valid]
        observed_count[valid] += 1

    effective = np.full((n, n), np.inf, dtype=np.float64)
    observed = observed_count > 0
    effective[observed] = energy_sum[observed] / observed_count[observed]
    np.fill_diagonal(effective, np.inf)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i in range(n):
        finite = np.flatnonzero(np.isfinite(effective[i]))
        if not len(finite):
            continue
        kk = min(graph_k, len(finite))
        local = effective[i, finite]
        pick = np.argpartition(local, kk - 1)[:kk]
        selected = finite[pick]
        selected = selected[np.argsort(effective[i, selected], kind="stable")]
        weights = np.exp(-effective[i, selected])
        good = np.isfinite(weights) & (weights > 0)
        selected = selected[good]
        weights = weights[good]
        rows.extend([i] * len(selected))
        cols.extend(selected.astype(int).tolist())
        vals.extend(weights.astype(np.float32).tolist())

    directed = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    undirected = directed.maximum(directed.T).tocsr()
    undirected.setdiag(0.0)
    undirected.eliminate_zeros()
    return undirected, ProductKernelDiagnostics(
        node_count=n,
        view_count=len(mats),
        graph_k=graph_k,
        directed_edges=int(directed.nnz),
        undirected_edges=int(undirected.nnz // 2),
    )


def multiview_neighbour_union(
    cross_distances: list[np.ndarray],
    query_available: list[bool],
    reference_available: list[np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    """Union of intrinsic sqrt(N)-scale coordinate neighbourhoods.

    Each observed coordinate view contributes its own local chart. The union is
    an atlas chart rather than a score fusion: membership is geometric, and no
    view receives a learned weight or threshold.
    """
    if not cross_distances:
        return np.zeros(0, dtype=np.int64), []
    if not (len(cross_distances) == len(query_available) == len(reference_available)):
        raise ValueError("view lists must have equal length")
    n = len(np.asarray(cross_distances[0]).reshape(-1))
    selected: set[int] = set()
    counts: list[int] = []
    for distance, q_ok, r_ok in zip(cross_distances, query_available, reference_available):
        d = np.asarray(distance, dtype=np.float64).reshape(-1)
        mask = np.asarray(r_ok, dtype=bool).reshape(-1)
        if len(d) != n or len(mask) != n:
            raise ValueError("reference view length mismatch")
        if not q_ok:
            counts.append(0)
            continue
        valid = np.flatnonzero(mask & np.isfinite(d))
        if not len(valid):
            counts.append(0)
            continue
        kk = min(len(valid), max(1, int(math.ceil(math.sqrt(len(valid))))))
        local = d[valid]
        pick = np.argpartition(local, kk - 1)[:kk]
        chosen = valid[pick]
        chosen = chosen[np.argsort(d[chosen], kind="stable")]
        selected.update(map(int, chosen))
        counts.append(int(len(chosen)))
    return np.asarray(sorted(selected), dtype=np.int64), counts


def self_tuning_product_neighbours(
    cross_distances: list[np.ndarray],
    query_available: list[bool],
    reference_available: list[np.ndarray],
    *,
    k: int | None = None,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, dict[str, object]]:
    """Select one query-centred chart in a locally normalized product metric.

    Each observed coordinate view chooses its intrinsic radial scale as the
    distance from the query to the sqrt(N_v)-th available reference point. For
    reference j the product radial energy is the mean of d_v(q,j)^2/sigma_v(q)^2
    across views observed at both endpoints. The chart contains the sqrt(N)
    smallest finite product energies. This keeps chart cardinality at the same
    discretisation scale as the single-view method while allowing chemically
    complementary coordinates to define the neighbourhood jointly.
    """
    if not cross_distances:
        return np.zeros(0, dtype=np.int64), {
            "graph_k": 0, "view_scales": [], "view_support": [], "finite_reference_count": 0
        }
    if not (len(cross_distances) == len(query_available) == len(reference_available)):
        raise ValueError("view lists must have equal length")
    dists = [np.asarray(d, dtype=np.float64).reshape(-1) for d in cross_distances]
    masks = [np.asarray(a, dtype=bool).reshape(-1) for a in reference_available]
    n = len(dists[0])
    if any(len(d) != n for d in dists) or any(len(a) != n for a in masks):
        raise ValueError("reference view length mismatch")
    if n == 0:
        return np.zeros(0, dtype=np.int64), {
            "graph_k": 0, "view_scales": [], "view_support": [], "finite_reference_count": 0
        }

    energy_sum = np.zeros(n, dtype=np.float64)
    observed_count = np.zeros(n, dtype=np.int16)
    scales: list[float | None] = []
    supports: list[int] = []
    for d, q_ok, mask in zip(dists, query_available, masks):
        valid = mask & np.isfinite(d)
        support = int(valid.sum())
        supports.append(support)
        if (not q_ok) or support == 0:
            scales.append(None)
            continue
        kk = min(support, max(1, int(math.ceil(math.sqrt(support)))))
        values = d[valid]
        sigma = float(np.partition(values, kk - 1)[kk - 1])
        positive = values[np.isfinite(values) & (values > epsilon)]
        if (not np.isfinite(sigma)) or sigma <= epsilon:
            sigma = float(np.median(positive)) if len(positive) else 1.0
        scales.append(sigma)
        e = np.square(d[valid]) / max(sigma * sigma, epsilon)
        energy_sum[valid] += e
        observed_count[valid] += 1

    effective = np.full(n, np.inf, dtype=np.float64)
    finite = observed_count > 0
    effective[finite] = energy_sum[finite] / observed_count[finite]
    valid_indices = np.flatnonzero(np.isfinite(effective))
    if not len(valid_indices):
        return np.zeros(0, dtype=np.int64), {
            "graph_k": 0,
            "view_scales": scales,
            "view_support": supports,
            "finite_reference_count": 0,
        }
    graph_k = min(
        len(valid_indices),
        max(1, int(k) if k is not None else int(math.ceil(math.sqrt(len(valid_indices))))),
    )
    local = effective[valid_indices]
    pick = np.argpartition(local, graph_k - 1)[:graph_k]
    chosen = valid_indices[pick]
    chosen = chosen[np.argsort(effective[chosen], kind="stable")]
    return chosen.astype(np.int64), {
        "graph_k": int(graph_k),
        "view_scales": scales,
        "view_support": supports,
        "finite_reference_count": int(len(valid_indices)),
        "nearest_product_energy": float(effective[chosen[0]]),
        "furthest_chart_product_energy": float(effective[chosen[-1]]),
    }


def _entropy_deficit_from_energy(energy: np.ndarray) -> float:
    """Normalized KL(p || uniform) for p proportional to exp(-energy)."""
    e = np.asarray(energy, dtype=np.float64).reshape(-1)
    e = e[np.isfinite(e)]
    n = len(e)
    if n <= 1:
        return 1.0 if n == 1 else 0.0
    logits = -e
    logits -= float(np.max(logits))
    w = np.exp(logits)
    z = float(w.sum())
    if (not np.isfinite(z)) or z <= 0:
        return 0.0
    p = w / z
    entropy = float(-np.sum(p * np.log(np.maximum(p, 1e-300))))
    return float(np.clip(1.0 - entropy / math.log(n), 0.0, 1.0))


def information_product_neighbours(
    cross_distances: list[np.ndarray],
    query_available: list[bool],
    reference_available: list[np.ndarray],
    *,
    k: int | None = None,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, dict[str, object]]:
    """Query chart under an information-adaptive product metric tensor.

    Each coordinate is first conformally normalized by its intrinsic sqrt(N)
    radial scale. Its tensor coefficient is then the normalized entropy deficit
    KL(p_v || Uniform)/log(N_v), with p_v proportional to exp(-d_v^2/sigma_v^2).
    A concentrated local neighbourhood therefore contributes more metric
    precision, while a flat/uninformative coordinate fades continuously. No
    label, threshold, or learned inter-view weight enters the construction.
    """
    if not cross_distances:
        return np.zeros(0, dtype=np.int64), {"graph_k": 0, "view_information": []}
    if not (len(cross_distances) == len(query_available) == len(reference_available)):
        raise ValueError("view lists must have equal length")
    dists = [np.asarray(d, dtype=np.float64).reshape(-1) for d in cross_distances]
    masks = [np.asarray(a, dtype=bool).reshape(-1) for a in reference_available]
    n = len(dists[0])
    if any(len(d) != n for d in dists) or any(len(a) != n for a in masks):
        raise ValueError("reference view length mismatch")

    per_view_energy: list[np.ndarray] = []
    view_information: list[float] = []
    view_scales: list[float | None] = []
    observed_masks: list[np.ndarray] = []
    for d, q_ok, mask in zip(dists, query_available, masks):
        valid = mask & np.isfinite(d)
        if (not q_ok) or not np.any(valid):
            per_view_energy.append(np.full(n, np.nan, dtype=np.float64))
            view_information.append(0.0)
            view_scales.append(None)
            observed_masks.append(np.zeros(n, dtype=bool))
            continue
        support = int(valid.sum())
        kk = min(support, max(1, int(math.ceil(math.sqrt(support)))))
        values = d[valid]
        sigma = float(np.partition(values, kk - 1)[kk - 1])
        positive = values[np.isfinite(values) & (values > epsilon)]
        if (not np.isfinite(sigma)) or sigma <= epsilon:
            sigma = float(np.median(positive)) if len(positive) else 1.0
        energy = np.full(n, np.nan, dtype=np.float64)
        energy[valid] = np.square(d[valid]) / max(sigma * sigma, epsilon)
        per_view_energy.append(energy)
        view_information.append(_entropy_deficit_from_energy(energy[valid]))
        view_scales.append(sigma)
        observed_masks.append(valid)

    energy_sum = np.zeros(n, dtype=np.float64)
    weight_sum = np.zeros(n, dtype=np.float64)
    fallback_sum = np.zeros(n, dtype=np.float64)
    fallback_count = np.zeros(n, dtype=np.int16)
    for energy, info, valid in zip(per_view_energy, view_information, observed_masks):
        if not np.any(valid):
            continue
        fallback_sum[valid] += energy[valid]
        fallback_count[valid] += 1
        if info > 0:
            energy_sum[valid] += info * energy[valid]
            weight_sum[valid] += info
    effective = np.full(n, np.inf, dtype=np.float64)
    weighted = weight_sum > epsilon
    effective[weighted] = energy_sum[weighted] / weight_sum[weighted]
    fallback = (~weighted) & (fallback_count > 0)
    effective[fallback] = fallback_sum[fallback] / fallback_count[fallback]

    valid_indices = np.flatnonzero(np.isfinite(effective))
    if not len(valid_indices):
        return np.zeros(0, dtype=np.int64), {
            "graph_k": 0, "view_information": view_information, "view_scales": view_scales
        }
    graph_k = min(
        len(valid_indices),
        max(1, int(k) if k is not None else int(math.ceil(math.sqrt(len(valid_indices))))),
    )
    local = effective[valid_indices]
    pick = np.argpartition(local, graph_k - 1)[:graph_k]
    chosen = valid_indices[pick]
    chosen = chosen[np.argsort(effective[chosen], kind="stable")]
    return chosen.astype(np.int64), {
        "graph_k": int(graph_k),
        "view_information": [float(x) for x in view_information],
        "view_scales": view_scales,
        "finite_reference_count": int(len(valid_indices)),
        "nearest_product_energy": float(effective[chosen[0]]),
        "furthest_chart_product_energy": float(effective[chosen[-1]]),
    }


def information_product_affinity(
    distances: list[np.ndarray],
    availabilities: list[np.ndarray] | None = None,
    *,
    k: int | None = None,
    epsilon: float = 1e-8,
) -> tuple[csr_matrix, dict[str, object]]:
    """Sparse product affinity with a spatially varying information metric tensor.

    The per-node coefficient I_v(i) is the normalized entropy deficit of the
    self-tuning Gaussian neighbourhood in view v. For edge i--j, view v receives
    coefficient sqrt(I_v(i) I_v(j)); this is the symmetric tensor interpolation
    between the endpoints. If every observed view is locally flat, the exact
    equal-view product metric is used as the label-free fallback.
    """
    if not distances:
        raise ValueError("at least one distance view is required")
    mats = [np.asarray(d, dtype=np.float64) for d in distances]
    n = mats[0].shape[0]
    if any(d.shape != (n, n) for d in mats):
        raise ValueError("all distance views must have the same square shape")
    if availabilities is None:
        masks = [np.ones(n, dtype=bool) for _ in mats]
    else:
        masks = [np.asarray(a, dtype=bool).reshape(-1) for a in availabilities]
        if len(masks) != len(mats) or any(len(a) != n for a in masks):
            raise ValueError("availability masks do not match views")
    if n < 2:
        return csr_matrix((n, n), dtype=np.float32), {"graph_k": 0, "mean_view_information": []}
    graph_k = min(n - 1, max(1, int(k) if k is not None else int(math.ceil(math.sqrt(n)))))

    energies: list[np.ndarray] = []
    infos: list[np.ndarray] = []
    for d, available in zip(mats, masks):
        sigma = _self_tuning_scale(d, available, epsilon=epsilon)
        pair_available = available[:, None] & available[None, :]
        denom = sigma[:, None] * sigma[None, :]
        valid = pair_available & np.isfinite(denom) & (denom > epsilon)
        energy = np.full((n, n), np.nan, dtype=np.float64)
        energy[valid] = np.square(d[valid]) / denom[valid]
        info = np.zeros(n, dtype=np.float64)
        for i in np.flatnonzero(available):
            row_valid = valid[i].copy(); row_valid[i] = False
            if np.any(row_valid):
                info[i] = _entropy_deficit_from_energy(energy[i, row_valid])
        energies.append(energy)
        infos.append(info)

    effective = np.full((n, n), np.inf, dtype=np.float64)
    weighted_sum = np.zeros((n, n), dtype=np.float64)
    weight_sum = np.zeros((n, n), dtype=np.float64)
    fallback_sum = np.zeros((n, n), dtype=np.float64)
    fallback_count = np.zeros((n, n), dtype=np.int16)
    for energy, info in zip(energies, infos):
        valid = np.isfinite(energy)
        fallback_sum[valid] += energy[valid]
        fallback_count[valid] += 1
        pair_weight = np.sqrt(np.maximum(info[:, None] * info[None, :], 0.0))
        active = valid & (pair_weight > epsilon)
        weighted_sum[active] += pair_weight[active] * energy[active]
        weight_sum[active] += pair_weight[active]
    weighted = weight_sum > epsilon
    effective[weighted] = weighted_sum[weighted] / weight_sum[weighted]
    fallback = (~weighted) & (fallback_count > 0)
    effective[fallback] = fallback_sum[fallback] / fallback_count[fallback]
    np.fill_diagonal(effective, np.inf)

    rows: list[int] = []; cols: list[int] = []; vals: list[float] = []
    for i in range(n):
        finite = np.flatnonzero(np.isfinite(effective[i]))
        if not len(finite):
            continue
        kk = min(graph_k, len(finite))
        local = effective[i, finite]
        pick = np.argpartition(local, kk - 1)[:kk]
        selected = finite[pick]
        selected = selected[np.argsort(effective[i, selected], kind="stable")]
        weights = np.exp(-effective[i, selected])
        good = np.isfinite(weights) & (weights > 0)
        selected = selected[good]; weights = weights[good]
        rows.extend([i] * len(selected)); cols.extend(selected.astype(int).tolist()); vals.extend(weights.astype(np.float32).tolist())
    directed = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    undirected = directed.maximum(directed.T).tocsr(); undirected.setdiag(0.0); undirected.eliminate_zeros()
    return undirected, {
        "graph_k": int(graph_k),
        "directed_edges": int(directed.nnz),
        "undirected_edges": int(undirected.nnz // 2),
        "mean_view_information": [float(np.mean(info[mask])) if np.any(mask) else 0.0 for info, mask in zip(infos, masks)],
    }


def _perplexity_contraction_from_energy(energy: np.ndarray) -> float:
    """Return exp(KL(p||U))-1 = N/perplexity(p)-1 for exp(-energy) weights.

    This is an intrinsic local resolution gain: zero means the coordinate sees
    an effectively uniform neighbourhood, while larger values mean the view has
    contracted the ambient candidate set to a much smaller effective support.
    """
    e = np.asarray(energy, dtype=np.float64).reshape(-1)
    e = e[np.isfinite(e)]
    n = len(e)
    if n <= 1:
        return float(max(n - 1, 0))
    logits = -e
    logits -= float(np.max(logits))
    w = np.exp(logits)
    z = float(w.sum())
    if (not np.isfinite(z)) or z <= 0:
        return 0.0
    p = w / z
    entropy = float(-np.sum(p * np.log(np.maximum(p, 1e-300))))
    kl = float(np.clip(math.log(n) - entropy, 0.0, math.log(n)))
    return float(np.expm1(kl))


def resolution_product_neighbours(
    cross_distances: list[np.ndarray],
    query_available: list[bool],
    reference_available: list[np.ndarray],
    *,
    k: int | None = None,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, dict[str, object]]:
    """Query chart under a perplexity-contraction metric tensor.

    Coordinate precision is exp(KL(p_v||U))-1 = N_v/perplexity(p_v)-1.
    Hence a view contributes in proportion to the number of alternatives it
    actually resolves, rather than a hand-set modality weight. Flat coordinates
    have zero excess precision. Missing views remain absent observations.
    """
    if not cross_distances:
        return np.zeros(0, dtype=np.int64), {"graph_k": 0, "view_resolution": []}
    if not (len(cross_distances) == len(query_available) == len(reference_available)):
        raise ValueError("view lists must have equal length")
    dists = [np.asarray(d, dtype=np.float64).reshape(-1) for d in cross_distances]
    masks = [np.asarray(a, dtype=bool).reshape(-1) for a in reference_available]
    n = len(dists[0])
    if any(len(d) != n for d in dists) or any(len(a) != n for a in masks):
        raise ValueError("reference view length mismatch")

    energy_sum = np.zeros(n, dtype=np.float64)
    precision_sum = np.zeros(n, dtype=np.float64)
    fallback_sum = np.zeros(n, dtype=np.float64)
    fallback_count = np.zeros(n, dtype=np.int16)
    resolutions: list[float] = []
    scales: list[float | None] = []
    for d, q_ok, mask in zip(dists, query_available, masks):
        valid = mask & np.isfinite(d)
        if (not q_ok) or not np.any(valid):
            resolutions.append(0.0); scales.append(None); continue
        support = int(valid.sum())
        kk = min(support, max(1, int(math.ceil(math.sqrt(support)))))
        values = d[valid]
        sigma = float(np.partition(values, kk - 1)[kk - 1])
        positive = values[np.isfinite(values) & (values > epsilon)]
        if (not np.isfinite(sigma)) or sigma <= epsilon:
            sigma = float(np.median(positive)) if len(positive) else 1.0
        e = np.square(d[valid]) / max(sigma * sigma, epsilon)
        resolution = _perplexity_contraction_from_energy(e)
        resolutions.append(resolution); scales.append(sigma)
        fallback_sum[valid] += e; fallback_count[valid] += 1
        if resolution > epsilon:
            energy_sum[valid] += resolution * e
            precision_sum[valid] += resolution

    effective = np.full(n, np.inf, dtype=np.float64)
    weighted = precision_sum > epsilon
    effective[weighted] = energy_sum[weighted] / precision_sum[weighted]
    fallback = (~weighted) & (fallback_count > 0)
    effective[fallback] = fallback_sum[fallback] / fallback_count[fallback]
    valid_indices = np.flatnonzero(np.isfinite(effective))
    if not len(valid_indices):
        return np.zeros(0, dtype=np.int64), {"graph_k": 0, "view_resolution": resolutions, "view_scales": scales}
    graph_k = min(len(valid_indices), max(1, int(k) if k is not None else int(math.ceil(math.sqrt(len(valid_indices))))))
    local = effective[valid_indices]
    pick = np.argpartition(local, graph_k - 1)[:graph_k]
    chosen = valid_indices[pick]
    chosen = chosen[np.argsort(effective[chosen], kind="stable")]
    return chosen.astype(np.int64), {
        "graph_k": int(graph_k),
        "view_resolution": [float(x) for x in resolutions],
        "view_scales": scales,
        "finite_reference_count": int(len(valid_indices)),
        "nearest_product_energy": float(effective[chosen[0]]),
        "furthest_chart_product_energy": float(effective[chosen[-1]]),
    }



def resolution_product_partition(
    cross_distances: list[np.ndarray],
    reference_distances: list[np.ndarray],
    query_availabilities: list[np.ndarray],
    reference_availabilities: list[np.ndarray],
    *,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, dict[str, object]]:
    """Mass-conserving partition of unity under the resolution-product metric.

    This is the cross-set counterpart of :func:`resolution_product_affinity`.
    Reference points define the sampled manifold chart, while query points are
    atoms to be restricted onto that chart.  For each coordinate view, both
    query and reference endpoints receive self-tuning local scales and an
    intrinsic precision

        exp(KL(p_v || U)) - 1 = support / perplexity - 1.

    Pair energies are averaged with the geometric mean of the endpoint
    precisions. Missing coordinate observations are omitted exactly.  The final
    ``exp(-energy)`` rows are normalized to one, so every input atom keeps its
    mass and no missing view becomes negative evidence.
    """
    if not cross_distances:
        raise ValueError("at least one distance view is required")
    if not (
        len(cross_distances)
        == len(reference_distances)
        == len(query_availabilities)
        == len(reference_availabilities)
    ):
        raise ValueError("all view lists must have equal length")

    cross = [np.asarray(d, dtype=np.float64) for d in cross_distances]
    refs = [np.asarray(d, dtype=np.float64) for d in reference_distances]
    nq, nr = cross[0].shape
    if any(d.shape != (nq, nr) for d in cross):
        raise ValueError("all cross-distance views must have the same shape")
    if any(d.shape != (nr, nr) for d in refs):
        raise ValueError("all reference-distance views must be square and aligned")
    qmasks = [np.asarray(a, dtype=bool).reshape(-1) for a in query_availabilities]
    rmasks = [np.asarray(a, dtype=bool).reshape(-1) for a in reference_availabilities]
    if any(len(a) != nq for a in qmasks) or any(len(a) != nr for a in rmasks):
        raise ValueError("availability masks do not align to query/reference sets")
    if nq == 0:
        return np.zeros((0, nr), dtype=np.float64), {
            "mean_query_view_resolution": [0.0] * len(cross),
            "mean_reference_view_resolution": [0.0] * len(cross),
        }
    if nr == 0:
        raise ValueError("reference chart must contain at least one point")

    weighted_sum = np.zeros((nq, nr), dtype=np.float64)
    precision_sum = np.zeros((nq, nr), dtype=np.float64)
    fallback_sum = np.zeros((nq, nr), dtype=np.float64)
    fallback_count = np.zeros((nq, nr), dtype=np.int16)
    query_resolutions: list[np.ndarray] = []
    reference_resolutions: list[np.ndarray] = []

    for d_cross, d_ref, q_available, r_available in zip(cross, refs, qmasks, rmasks):
        reference_sigma = _self_tuning_scale(d_ref, r_available, epsilon=epsilon)
        query_sigma = np.full(nq, np.nan, dtype=np.float64)
        for qi in np.flatnonzero(q_available):
            valid = r_available & np.isfinite(d_cross[qi])
            values = d_cross[qi, valid]
            if not len(values):
                continue
            kk = min(len(values), max(1, int(math.ceil(math.sqrt(len(values))))))
            sigma = float(np.partition(values, kk - 1)[kk - 1])
            positive = values[np.isfinite(values) & (values > epsilon)]
            if (not np.isfinite(sigma)) or sigma <= epsilon:
                sigma = float(np.median(positive)) if len(positive) else 1.0
            query_sigma[qi] = sigma

        denom = query_sigma[:, None] * reference_sigma[None, :]
        valid_cross = (
            q_available[:, None]
            & r_available[None, :]
            & np.isfinite(d_cross)
            & np.isfinite(denom)
            & (denom > epsilon)
        )
        cross_energy = np.full((nq, nr), np.nan, dtype=np.float64)
        cross_energy[valid_cross] = np.square(d_cross[valid_cross]) / denom[valid_cross]

        query_resolution = np.zeros(nq, dtype=np.float64)
        for qi in np.flatnonzero(q_available):
            rv = valid_cross[qi]
            if np.any(rv):
                query_resolution[qi] = _perplexity_contraction_from_energy(
                    cross_energy[qi, rv]
                )

        ref_denom = reference_sigma[:, None] * reference_sigma[None, :]
        valid_ref = (
            r_available[:, None]
            & r_available[None, :]
            & np.isfinite(d_ref)
            & np.isfinite(ref_denom)
            & (ref_denom > epsilon)
        )
        ref_energy = np.full((nr, nr), np.nan, dtype=np.float64)
        ref_energy[valid_ref] = np.square(d_ref[valid_ref]) / ref_denom[valid_ref]
        reference_resolution = np.zeros(nr, dtype=np.float64)
        for ri in np.flatnonzero(r_available):
            rv = valid_ref[ri].copy()
            rv[ri] = False
            if np.any(rv):
                reference_resolution[ri] = _perplexity_contraction_from_energy(
                    ref_energy[ri, rv]
                )

        valid = np.isfinite(cross_energy)
        fallback_sum[valid] += cross_energy[valid]
        fallback_count[valid] += 1
        pair_precision = np.sqrt(
            np.maximum(query_resolution[:, None] * reference_resolution[None, :], 0.0)
        )
        active = valid & (pair_precision > epsilon)
        weighted_sum[active] += pair_precision[active] * cross_energy[active]
        precision_sum[active] += pair_precision[active]
        query_resolutions.append(query_resolution)
        reference_resolutions.append(reference_resolution)

    effective = np.full((nq, nr), np.inf, dtype=np.float64)
    weighted = precision_sum > epsilon
    effective[weighted] = weighted_sum[weighted] / precision_sum[weighted]
    fallback = (~weighted) & (fallback_count > 0)
    effective[fallback] = fallback_sum[fallback] / fallback_count[fallback]

    weights = np.zeros((nq, nr), dtype=np.float64)
    for qi in range(nq):
        finite = np.isfinite(effective[qi])
        if not np.any(finite):
            raise ValueError(f"query row {qi} has no observed geometric coordinate on reference chart")
        local = effective[qi, finite]
        local -= float(np.min(local))
        w = np.exp(-local)
        z = float(w.sum())
        if (not np.isfinite(z)) or z <= 0:
            raise RuntimeError("resolution-product partition failed to normalize")
        weights[qi, finite] = w / z

    return weights, {
        "mean_query_view_resolution": [
            float(np.mean(r[m])) if np.any(m) else 0.0
            for r, m in zip(query_resolutions, qmasks)
        ],
        "mean_reference_view_resolution": [
            float(np.mean(r[m])) if np.any(m) else 0.0
            for r, m in zip(reference_resolutions, rmasks)
        ],
    }


def partial_observation_pullback_affinity(
    distances: list[np.ndarray],
    availabilities: list[np.ndarray] | None = None,
    *,
    k: int | None = None,
    epsilon: float = 1e-8,
) -> tuple[csr_matrix, dict[str, object]]:
    """Build one local metric from every jointly observed molecular coordinate.

    The statistical setting is sparse pair supervision but heterogeneous side
    information.  Each distance matrix is one measurement map of the *same*
    underlying molecular state (global sequence, whole structure, motif context,
    pocket sequence, pocket structure, pocket OT, ...).  A view contributes to
    an edge only when both endpoints are observed in that view.

    Every view is first made dimensionless by its own self-tuning local scale,

        E_m(i,j) = d_m(i,j)^2 / (sigma_m(i) sigma_m(j)).

    The product-manifold protein energy is the arithmetic mean of the available
    normalized pullback energies for that pair.  Consequently:

    * a missing view changes nothing about a pair;
    * no endpoint is rewarded merely for having more measured modalities;
    * duplicating an identical view leaves the effective energy unchanged;
    * rescaling one coordinate system by a positive constant leaves its energy
      unchanged through the corresponding rescaling of ``sigma_m``;
    * no learned or manually tuned inter-view weight is introduced.

    The returned affinity uses the same intrinsic ``ceil(sqrt(n))`` local graph
    discretisation as the rest of the geometry stack unless ``k`` is supplied
    explicitly for a diagnostic.  Pairs with no jointly observed coordinate are
    absent, not negative.
    """
    if not distances:
        raise ValueError("at least one distance view is required")
    mats = [np.asarray(d, dtype=np.float64) for d in distances]
    n = mats[0].shape[0]
    if any(d.shape != (n, n) for d in mats):
        raise ValueError("all distance views must have the same square shape")
    if any(np.any((d[np.isfinite(d)] < -epsilon)) for d in mats):
        raise ValueError("distance views must be non-negative where finite")
    if availabilities is None:
        masks = [np.ones(n, dtype=bool) for _ in mats]
    else:
        masks = [np.asarray(a, dtype=bool).reshape(-1) for a in availabilities]
        if len(masks) != len(mats) or any(len(a) != n for a in masks):
            raise ValueError("availability masks do not match views")
    if n < 2:
        return csr_matrix((n, n), dtype=np.float32), {
            "graph_k": 0,
            "view_count": len(mats),
            "view_available_counts": [int(m.sum()) for m in masks],
            "pair_observed_view_count_mean": 0.0,
        }

    energy_sum = np.zeros((n, n), dtype=np.float64)
    observed_count = np.zeros((n, n), dtype=np.int16)
    view_scales: list[np.ndarray] = []
    for d, available in zip(mats, masks):
        sigma = _self_tuning_scale(d, available, epsilon=epsilon)
        view_scales.append(sigma)
        denom = sigma[:, None] * sigma[None, :]
        pair_available = (
            available[:, None]
            & available[None, :]
            & np.isfinite(d)
            & np.isfinite(denom)
            & (denom > epsilon)
        )
        e = np.zeros((n, n), dtype=np.float64)
        e[pair_available] = np.square(np.maximum(d[pair_available], 0.0)) / denom[pair_available]
        energy_sum[pair_available] += e[pair_available]
        observed_count[pair_available] += 1

    effective = np.full((n, n), np.inf, dtype=np.float64)
    observed = observed_count > 0
    effective[observed] = energy_sum[observed] / observed_count[observed]
    np.fill_diagonal(effective, np.inf)

    graph_k = min(n - 1, max(1, int(k) if k is not None else int(math.ceil(math.sqrt(n)))))
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i in range(n):
        finite = np.flatnonzero(np.isfinite(effective[i]))
        if not len(finite):
            continue
        kk = min(graph_k, len(finite))
        local = effective[i, finite]
        pick = np.argpartition(local, kk - 1)[:kk]
        selected = finite[pick]
        selected = selected[np.argsort(effective[i, selected], kind="stable")]
        weights = np.exp(-effective[i, selected])
        good = np.isfinite(weights) & (weights > 0)
        selected = selected[good]
        weights = weights[good]
        rows.extend([i] * len(selected))
        cols.extend(selected.astype(int).tolist())
        vals.extend(weights.astype(np.float32).tolist())
    directed = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    undirected = directed.maximum(directed.T).tocsr()
    undirected.setdiag(0.0)
    undirected.eliminate_zeros()

    offdiag = ~np.eye(n, dtype=bool)
    jointly_observed = observed & offdiag
    counts = observed_count[jointly_observed]
    return undirected, {
        "graph_k": int(graph_k),
        "view_count": int(len(mats)),
        "view_available_counts": [int(m.sum()) for m in masks],
        "directed_edges": int(directed.nnz),
        "undirected_edges": int(undirected.nnz // 2),
        "pair_observed_view_count_mean": float(np.mean(counts)) if len(counts) else 0.0,
        "pair_observed_view_count_median": float(np.median(counts)) if len(counts) else 0.0,
        "pairs_with_any_observation": int(np.count_nonzero(jointly_observed) // 2),
    }


def hierarchical_pullback_refinement_affinity(
    base_distance: np.ndarray,
    base_available: np.ndarray,
    refinement_distances: list[np.ndarray],
    refinement_availabilities: list[np.ndarray] | None = None,
    refinement_reliabilities: list[np.ndarray] | None = None,
    *,
    k: int | None = None,
    epsilon: float = 1e-8,
    fallback_distance: np.ndarray | None = None,
    fallback_available: np.ndarray | None = None,
) -> tuple[csr_matrix, dict[str, object]]:
    """Refine a complete/coarse atlas without allowing partial views to rewrite topology.

    ``base_distance`` defines the molecular atlas: its intrinsic sqrt(N)-scale
    neighbours are the only ordinary edges.  Additional measurements are local
    pullback coordinates on that atlas.  On a base edge i--j,

        E(i,j) = E_base(i,j) + mean_m E_m(i,j),

    where the mean contains only refinement views observed at both endpoints.
    If no refinement is jointly observed, ``E == E_base`` exactly.  This makes
    missing measurements neutral and duplicate refinement views idempotent, while
    ensuring that local catalytic/structural evidence can only *lengthen/refine*
    a coarse edge rather than create a cross-manifold shortcut.

    A ``fallback_distance`` is used only for nodes with no base observation at
    all.  It supplies their local atlas neighbourhood; it never changes the
    topology of base-observed nodes.  This is intended for genuine missing-base
    cases, not as a score router.
    """
    base=np.asarray(base_distance,dtype=np.float64)
    avail=np.asarray(base_available,dtype=bool).reshape(-1)
    if base.ndim!=2 or base.shape[0]!=base.shape[1] or len(avail)!=base.shape[0]:
        raise ValueError('base distance/availability shape mismatch')
    n=base.shape[0]
    refs=[np.asarray(d,dtype=np.float64) for d in refinement_distances]
    if any(d.shape!=(n,n) for d in refs):
        raise ValueError('refinement distance shape mismatch')
    if refinement_availabilities is None:
        rmasks=[np.ones(n,dtype=bool) for _ in refs]
    else:
        rmasks=[np.asarray(a,dtype=bool).reshape(-1) for a in refinement_availabilities]
        if len(rmasks)!=len(refs) or any(len(a)!=n for a in rmasks):
            raise ValueError('refinement availability shape mismatch')
    if refinement_reliabilities is None:
        reliabilities=[np.ones(n,dtype=np.float64) for _ in refs]
    else:
        reliabilities=[np.asarray(q,dtype=np.float64).reshape(-1) for q in refinement_reliabilities]
        if len(reliabilities)!=len(refs) or any(len(q)!=n for q in reliabilities):
            raise ValueError('refinement reliability shape mismatch')
        if any(np.any((q < 0) | (q > 1) | (~np.isfinite(q))) for q in reliabilities):
            raise ValueError('refinement reliabilities must be finite in [0,1]')
    if np.any(base[np.isfinite(base)] < -epsilon) or any(np.any(d[np.isfinite(d)] < -epsilon) for d in refs):
        raise ValueError('distances must be non-negative where finite')
    if n<2:
        return csr_matrix((n,n),dtype=np.float32), {'graph_k':0,'base_available':int(avail.sum())}
    graph_k=min(n-1,max(1,int(k) if k is not None else int(math.ceil(math.sqrt(n)))))
    bsigma=_self_tuning_scale(base,avail,epsilon=epsilon)
    ref_sigmas=[_self_tuning_scale(d,a,epsilon=epsilon) for d,a in zip(refs,rmasks)]

    directed_edges:set[tuple[int,int]]=set()
    base_energy={}
    for i in np.flatnonzero(avail):
        valid=avail & np.isfinite(base[i]) & np.isfinite(bsigma) & (bsigma>epsilon)
        valid[i]=False
        js=np.flatnonzero(valid)
        if not len(js):
            continue
        e=np.square(np.maximum(base[i,js],0.0))/(bsigma[i]*bsigma[js])
        kk=min(graph_k,len(js)); pick=np.argpartition(e,kk-1)[:kk]
        for pos in pick:
            j=int(js[pos]); directed_edges.add((int(i),j)); base_energy[(int(i),j)]=float(e[pos])

    fallback_nodes=np.flatnonzero(~avail)
    fallback_edge_count=0
    if len(fallback_nodes):
        if fallback_distance is None or fallback_available is None:
            raise ValueError('base-missing nodes require an explicit fallback geometry')
        fd=np.asarray(fallback_distance,dtype=np.float64); fa=np.asarray(fallback_available,dtype=bool).reshape(-1)
        if fd.shape!=(n,n) or len(fa)!=n:
            raise ValueError('fallback geometry shape mismatch')
        fsigma=_self_tuning_scale(fd,fa,epsilon=epsilon)
        for i in fallback_nodes:
            if not fa[i] or not np.isfinite(fsigma[i]) or fsigma[i]<=epsilon:
                continue
            valid=fa & np.isfinite(fd[i]) & np.isfinite(fsigma) & (fsigma>epsilon); valid[i]=False
            js=np.flatnonzero(valid)
            if not len(js): continue
            e=np.square(np.maximum(fd[i,js],0.0))/(fsigma[i]*fsigma[js])
            kk=min(graph_k,len(js)); pick=np.argpartition(e,kk-1)[:kk]
            for pos in pick:
                j=int(js[pos]); directed_edges.add((int(i),j)); base_energy[(int(i),j)]=float(e[pos]); fallback_edge_count+=1

    # Symmetric topology is the union of directed base/fallback local charts.
    undirected_pairs={tuple(sorted((i,j))) for i,j in directed_edges if i!=j}
    rows=[]; cols=[]; vals=[]; local_counts=[]
    for i,j in sorted(undirected_pairs):
        e0=[]
        for a,b in [(i,j),(j,i)]:
            if (a,b) in base_energy: e0.append(base_energy[(a,b)])
        if not e0:
            continue
        # Symmetric interpolation of directed self-tuned base energy.
        eb=float(np.mean(e0))
        local=[]
        for d,a,sigma,q in zip(refs,rmasks,ref_sigmas,reliabilities):
            if not (a[i] and a[j] and np.isfinite(d[i,j]) and np.isfinite(sigma[i]) and np.isfinite(sigma[j]) and sigma[i]>epsilon and sigma[j]>epsilon):
                continue
            pair_reliability=float(np.sqrt(q[i]*q[j])); local.append(pair_reliability*float(max(d[i,j],0.0)**2/(sigma[i]*sigma[j])))
        energy=eb+(float(np.mean(local)) if local else 0.0)
        w=float(np.exp(-energy))
        if np.isfinite(w) and w>0:
            rows.extend([i,j]); cols.extend([j,i]); vals.extend([w,w]); local_counts.append(len(local))
    graph=csr_matrix((np.asarray(vals,dtype=np.float32),(rows,cols)),shape=(n,n),dtype=np.float32)
    graph.eliminate_zeros()
    degree=np.asarray((graph>0).sum(axis=1)).reshape(-1)
    return graph, {
        'graph_k':int(graph_k),
        'base_available':int(avail.sum()),
        'base_missing':int((~avail).sum()),
        'refinement_view_count':int(len(refs)),
        'refinement_available_counts':[int(a.sum()) for a in rmasks],
        'refinement_mean_reliability':[float(np.mean(q[a])) if np.any(a) else 0.0 for q,a in zip(reliabilities,rmasks)],
        'undirected_edges':int(graph.nnz//2),
        'fallback_directed_edges':int(fallback_edge_count),
        'edge_refinement_count_mean':float(np.mean(local_counts)) if local_counts else 0.0,
        'edge_refinement_count_median':float(np.median(local_counts)) if local_counts else 0.0,
        'isolated_nodes':int(np.sum(degree==0)),
    }

def resolution_product_affinity(
    distances: list[np.ndarray],
    availabilities: list[np.ndarray] | None = None,
    *,
    k: int | None = None,
    epsilon: float = 1e-8,
) -> tuple[csr_matrix, dict[str, object]]:
    """Local affinity with symmetric perplexity-contraction tensor precision."""
    if not distances:
        raise ValueError("at least one distance view is required")
    mats = [np.asarray(d, dtype=np.float64) for d in distances]
    n = mats[0].shape[0]
    if any(d.shape != (n, n) for d in mats):
        raise ValueError("all distance views must have the same square shape")
    if availabilities is None:
        masks = [np.ones(n, dtype=bool) for _ in mats]
    else:
        masks = [np.asarray(a, dtype=bool).reshape(-1) for a in availabilities]
        if len(masks) != len(mats) or any(len(a) != n for a in masks):
            raise ValueError("availability masks do not match views")
    if n < 2:
        return csr_matrix((n, n), dtype=np.float32), {"graph_k": 0, "mean_view_resolution": []}
    graph_k = min(n - 1, max(1, int(k) if k is not None else int(math.ceil(math.sqrt(n)))))

    energies: list[np.ndarray] = []
    resolutions: list[np.ndarray] = []
    for d, available in zip(mats, masks):
        sigma = _self_tuning_scale(d, available, epsilon=epsilon)
        pair_available = available[:, None] & available[None, :]
        denom = sigma[:, None] * sigma[None, :]
        valid = pair_available & np.isfinite(denom) & (denom > epsilon)
        energy = np.full((n, n), np.nan, dtype=np.float64)
        energy[valid] = np.square(d[valid]) / denom[valid]
        resolution = np.zeros(n, dtype=np.float64)
        for i in np.flatnonzero(available):
            rv = valid[i].copy(); rv[i] = False
            if np.any(rv):
                resolution[i] = _perplexity_contraction_from_energy(energy[i, rv])
        energies.append(energy); resolutions.append(resolution)

    weighted_sum = np.zeros((n, n), dtype=np.float64)
    precision_sum = np.zeros((n, n), dtype=np.float64)
    fallback_sum = np.zeros((n, n), dtype=np.float64)
    fallback_count = np.zeros((n, n), dtype=np.int16)
    for energy, resolution in zip(energies, resolutions):
        valid = np.isfinite(energy)
        fallback_sum[valid] += energy[valid]; fallback_count[valid] += 1
        pair_precision = np.sqrt(np.maximum(resolution[:, None] * resolution[None, :], 0.0))
        active = valid & (pair_precision > epsilon)
        weighted_sum[active] += pair_precision[active] * energy[active]
        precision_sum[active] += pair_precision[active]
    effective = np.full((n, n), np.inf, dtype=np.float64)
    weighted = precision_sum > epsilon
    effective[weighted] = weighted_sum[weighted] / precision_sum[weighted]
    fallback = (~weighted) & (fallback_count > 0)
    effective[fallback] = fallback_sum[fallback] / fallback_count[fallback]
    np.fill_diagonal(effective, np.inf)

    rows: list[int] = []; cols: list[int] = []; vals: list[float] = []
    for i in range(n):
        finite = np.flatnonzero(np.isfinite(effective[i]))
        if not len(finite): continue
        kk = min(graph_k, len(finite))
        local = effective[i, finite]
        pick = np.argpartition(local, kk - 1)[:kk]
        selected = finite[pick]
        selected = selected[np.argsort(effective[i, selected], kind="stable")]
        weights = np.exp(-effective[i, selected])
        good = np.isfinite(weights) & (weights > 0)
        selected = selected[good]; weights = weights[good]
        rows.extend([i]*len(selected)); cols.extend(selected.astype(int).tolist()); vals.extend(weights.astype(np.float32).tolist())
    directed = csr_matrix((vals,(rows,cols)),shape=(n,n),dtype=np.float32)
    undirected = directed.maximum(directed.T).tocsr(); undirected.setdiag(0.0); undirected.eliminate_zeros()
    return undirected, {
        "graph_k": int(graph_k),
        "directed_edges": int(directed.nnz),
        "undirected_edges": int(undirected.nnz//2),
        "mean_view_resolution": [float(np.mean(r[m])) if np.any(m) else 0.0 for r,m in zip(resolutions,masks)],
    }


def one_step_diffusion_distance(affinity: csr_matrix, *, epsilon: float = 1e-12) -> np.ndarray:
    """Exact t=1 diffusion distance of an undirected nonnegative graph.

    If P=D^{-1}A and pi is its stationary measure, this returns
        D_1(i,j)^2 = sum_k (P_ik-P_jk)^2 / pi_k.
    It compares local transition neighbourhoods rather than ambient coordinates,
    making the geometry intrinsic to the sampled manifold and reducing hubness.
    """
    a = csr_matrix(affinity, dtype=np.float64)
    if a.shape[0] != a.shape[1]:
        raise ValueError("affinity must be square")
    if a.nnz and np.min(a.data) < 0:
        raise ValueError("affinity must be non-negative")
    n = a.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    degree = np.asarray(a.sum(axis=1)).reshape(-1)
    safe_degree = np.maximum(degree, epsilon)
    total = float(degree.sum())
    if total <= epsilon:
        return np.zeros((n, n), dtype=np.float64)
    pi = np.maximum(degree / total, epsilon)
    p = a.multiply(1.0 / safe_degree[:, None]).toarray()
    # Weighted Gram matrix avoids materialising an n x n x n tensor.
    weighted = p / np.sqrt(pi[None, :])
    gram = weighted @ weighted.T
    norm = np.diag(gram)
    d2 = norm[:, None] + norm[None, :] - 2.0 * gram
    d2 = np.maximum(d2, 0.0)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2, dtype=np.float64)


def one_step_diffusion_affinity(
    affinity: csr_matrix,
    *,
    k: int | None = None,
) -> tuple[csr_matrix, dict[str, object]]:
    """Re-chart a local manifold using its exact one-step diffusion distance."""
    d = one_step_diffusion_distance(affinity)
    available = np.asarray(np.asarray(affinity.sum(axis=1)).reshape(-1) > 0, dtype=bool)
    # Reuse the same self-tuning local-kernel discretisation principle, now in
    # intrinsic diffusion coordinates. The implementation here is intentionally
    # local to avoid coupling this module back to the field solver.
    n = d.shape[0]
    if n < 2:
        return csr_matrix((n, n), dtype=np.float32), {"graph_k": 0, "edges": 0}
    graph_k = min(n - 1, max(1, int(k) if k is not None else int(math.ceil(math.sqrt(n)))))
    sigma = _self_tuning_scale(d, available, epsilon=1e-12)
    rows: list[int] = []; cols: list[int] = []; vals: list[float] = []
    for i in np.flatnonzero(available):
        valid = np.flatnonzero(available)
        valid = valid[valid != i]
        if not len(valid): continue
        denom = sigma[i] * sigma[valid]
        good = np.isfinite(denom) & (denom > 1e-12)
        valid = valid[good]; denom = denom[good]
        if not len(valid): continue
        energy = np.square(d[i, valid]) / denom
        kk = min(graph_k, len(valid))
        pick = np.argpartition(energy, kk - 1)[:kk]
        selected = valid[pick]; e = energy[pick]
        order = np.argsort(e, kind="stable"); selected=selected[order]; e=e[order]
        w=np.exp(-e); keep=np.isfinite(w)&(w>0); selected=selected[keep]; w=w[keep]
        rows.extend([int(i)]*len(selected)); cols.extend(selected.astype(int).tolist()); vals.extend(w.astype(np.float32).tolist())
    directed=csr_matrix((vals,(rows,cols)),shape=(n,n),dtype=np.float32)
    undirected=directed.maximum(directed.T).tocsr(); undirected.setdiag(0); undirected.eliminate_zeros()
    deg=np.asarray((undirected>0).sum(axis=0)).reshape(-1)
    return undirected, {
        "graph_k": int(graph_k),
        "edges": int(undirected.nnz//2),
        "degree_median": float(np.median(deg)) if len(deg) else 0.0,
        "degree_max": int(deg.max()) if len(deg) else 0,
    }


def diffusion_conformal_affinity(
    affinity: csr_matrix,
    *,
    epsilon: float = 1e-12,
) -> tuple[csr_matrix, dict[str, object]]:
    """Conformally correct existing graph conductances by intrinsic diffusion.

    Let K0 be the original self-tuning local kernel and D1 its exact t=1
    diffusion distance. On the *same edge set*, define another self-tuning heat
    kernel K1 from D1 and return K0*K1. Equivalently, the edge action is the sum
    of ambient-local and intrinsic-diffusion energies. No new edge is created:
    direct local proximity remains the atlas topology, while hub-like edges whose
    Markov neighbourhoods disagree are continuously suppressed.
    """
    a = csr_matrix(affinity, dtype=np.float64)
    if a.shape[0] != a.shape[1]:
        raise ValueError("affinity must be square")
    if a.nnz and np.min(a.data) < 0:
        raise ValueError("affinity must be non-negative")
    n = a.shape[0]
    if n < 2 or a.nnz == 0:
        return a.astype(np.float32), {"edges": int(a.nnz // 2), "degree_median": 0.0, "degree_max": 0}
    d = one_step_diffusion_distance(a, epsilon=epsilon)
    available = np.asarray(a.sum(axis=1)).reshape(-1) > epsilon
    sigma = _self_tuning_scale(d, available, epsilon=epsilon)
    coo = a.tocoo()
    denom = sigma[coo.row] * sigma[coo.col]
    valid = np.isfinite(denom) & (denom > epsilon) & (coo.row != coo.col)
    correction = np.zeros_like(coo.data, dtype=np.float64)
    correction[valid] = np.exp(-np.square(d[coo.row[valid], coo.col[valid]]) / denom[valid])
    data = coo.data * correction
    keep = np.isfinite(data) & (data > 0)
    out = csr_matrix((data[keep].astype(np.float32), (coo.row[keep], coo.col[keep])), shape=a.shape)
    out = out.maximum(out.T).tocsr(); out.setdiag(0.0); out.eliminate_zeros()
    deg = np.asarray((out > 0).sum(axis=0)).reshape(-1)
    weighted_degree = np.asarray(out.sum(axis=0)).reshape(-1)
    return out, {
        "edges": int(out.nnz // 2),
        "degree_median": float(np.median(deg)) if len(deg) else 0.0,
        "degree_max": int(deg.max()) if len(deg) else 0,
        "weighted_degree_cv": float(np.std(weighted_degree) / max(np.mean(weighted_degree), epsilon)) if len(weighted_degree) else 0.0,
        "mean_diffusion_correction": float(np.mean(correction[valid])) if np.any(valid) else 0.0,
    }
