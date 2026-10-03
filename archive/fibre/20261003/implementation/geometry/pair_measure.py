from __future__ import annotations

"""Positive relation measure on the enzyme × reaction product space.

This module contains no benchmark split, no BiME route, and no learned gate.  It is a
small mathematical primitive used by the unified geometry mainline: observed catalytic
pairs define a positive empirical measure on the Cartesian product.  Geometry may
transport that measure along reaction and protein neighbourhoods, and R2E / E2R are
simply the two conditional views of the same transported joint measure.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, triu


@dataclass(frozen=True)
class ConditionalMeasure:
    joint: np.ndarray
    r2e: np.ndarray
    e2r: np.ndarray
    reaction_mass: np.ndarray
    protein_mass: np.ndarray


def positive_joint_intensity(
    reaction_transport: csr_matrix,
    relation: csr_matrix,
    protein_transport: csr_matrix,
) -> np.ndarray:
    """Transport a positive empirical relation measure on both geometric axes.

    If ``A`` is the observed positive relation matrix, this computes

        J = K_R A K_E^T.

    The expression is not a heuristic fusion rule: it is the push-forward / kernel
    smoothing of one positive measure on the product space under the two marginal
    geometric transport operators.  Missing observations remain absence of positive
    mass; they are never converted into negative labels.
    """

    if relation.ndim != 2:
        raise ValueError("relation must be a reaction-by-protein matrix")
    nr, ne = relation.shape
    if reaction_transport.shape != (nr, nr):
        raise ValueError("reaction transport shape mismatch")
    if protein_transport.shape != (ne, ne):
        raise ValueError("protein transport shape mismatch")
    if relation.nnz and np.any(relation.data < 0):
        raise ValueError("positive relation measure cannot contain negative mass")
    joint = (reaction_transport @ relation @ protein_transport.T).toarray().astype(np.float64)
    joint[joint < 0] = 0.0
    return joint


def anisotropic_reaction_pushforward(
    field: np.ndarray,
    relation: np.ndarray,
    reaction_affinity: csr_matrix,
    *,
    scale: float,
    include_identity: bool = True,
    mass_conserving: bool = False,
) -> np.ndarray:
    """Push positive mass across reaction product-edges using field conductance.

    A reaction edge is not allowed to move its whole protein fibre with one
    scalar weight.  At product vertex ``(r,e)`` the transported mass from a
    neighbouring reaction ``r'`` is weighted by

        w(r,r') / sqrt(1 + ((F(r,e)-F(r',e))/scale)^2).

    Thus chemical proximity and local compatibility continuity must agree for a
    positive observation to cross the reaction edge.  This is the same
    Charbonnier conductance used by the nonlinear field energy.  All operations
    are positive; an unobserved pair never contributes negative evidence.
    """

    values = np.asarray(field, dtype=np.float64)
    mass = np.asarray(relation, dtype=np.float64)
    if values.ndim != 2 or mass.shape != values.shape:
        raise ValueError("field and relation must be aligned reaction-by-protein matrices")
    if reaction_affinity.shape != (values.shape[0], values.shape[0]):
        raise ValueError("reaction affinity shape mismatch")
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("scale must be finite and positive")
    if np.any(mass < 0) or not np.all(np.isfinite(mass)):
        raise ValueError("relation must be finite and non-negative")

    out = mass.copy() if include_identity else np.zeros_like(mass)
    upper = triu(reaction_affinity, k=1, format="coo")
    if not len(upper.data):
        return out
    rows = upper.row.astype(np.int64, copy=False)
    cols = upper.col.astype(np.int64, copy=False)
    weights = upper.data.astype(np.float64, copy=False)
    difference = values[rows] - values[cols]
    conductance = 1.0 / np.sqrt(1.0 + np.square(difference / float(scale)))
    edge_weight = weights[:, None] * conductance
    if mass_conserving:
        denominator = np.ones_like(mass) if include_identity else np.zeros_like(mass)
        np.add.at(denominator, rows, edge_weight)
        np.add.at(denominator, cols, edge_weight)
        safe = np.where(denominator > 0, denominator, 1.0)
        out = (mass / safe) if include_identity else np.zeros_like(mass)
        np.add.at(out, rows, edge_weight * (mass[cols] / safe[cols]))
        np.add.at(out, cols, edge_weight * (mass[rows] / safe[rows]))
        return out
    np.add.at(out, rows, edge_weight * mass[cols])
    np.add.at(out, cols, edge_weight * mass[rows])
    return out


def anisotropic_protein_pushforward(
    field: np.ndarray,
    relation: np.ndarray,
    protein_affinity: csr_matrix,
    *,
    scale: float,
    include_identity: bool = True,
    mass_conserving: bool = False,
) -> np.ndarray:
    """Push positive mass across protein product-edges using field conductance.

    This is the protein-axis counterpart of ``anisotropic_reaction_pushforward``.
    For every reaction fibre, positive mass may move from protein ``e'`` to a
    geometric neighbour ``e`` only in proportion to both protein affinity and
    Charbonnier continuity of the current compatibility field on that edge.
    """

    values = np.asarray(field, dtype=np.float64)
    mass = np.asarray(relation, dtype=np.float64)
    if values.ndim != 2 or mass.shape != values.shape:
        raise ValueError("field and relation must be aligned reaction-by-protein matrices")
    if protein_affinity.shape != (values.shape[1], values.shape[1]):
        raise ValueError("protein affinity shape mismatch")
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("scale must be finite and positive")
    if np.any(mass < 0) or not np.all(np.isfinite(mass)):
        raise ValueError("relation must be finite and non-negative")

    out = mass.copy() if include_identity else np.zeros_like(mass)
    upper = triu(protein_affinity, k=1, format="coo")
    if not len(upper.data):
        return out
    rows = upper.row.astype(np.int64, copy=False)
    cols = upper.col.astype(np.int64, copy=False)
    weights = upper.data.astype(np.float64, copy=False)
    difference = values[:, rows] - values[:, cols]
    conductance = 1.0 / np.sqrt(1.0 + np.square(difference / float(scale)))
    edge_weight = weights[None, :] * conductance
    if mass_conserving:
        denominator = np.ones_like(mass) if include_identity else np.zeros_like(mass)
        transposed_denominator = denominator.T
        np.add.at(transposed_denominator, rows, edge_weight.T)
        np.add.at(transposed_denominator, cols, edge_weight.T)
        safe = np.where(denominator > 0, denominator, 1.0)
        out = (mass / safe) if include_identity else np.zeros_like(mass)
        transposed = out.T
        np.add.at(
            transposed,
            rows,
            (edge_weight * (mass[:, cols] / safe[:, cols])).T,
        )
        np.add.at(
            transposed,
            cols,
            (edge_weight * (mass[:, rows] / safe[:, rows])).T,
        )
        return out
    transposed = out.T
    np.add.at(transposed, rows, (edge_weight * mass[:, cols]).T)
    np.add.at(transposed, cols, (edge_weight * mass[:, rows]).T)
    return out


def conditional_measure(joint: np.ndarray) -> ConditionalMeasure:
    """Return both conditionals of one non-negative product-space measure.

    ``r2e[r, e]`` is p(e | r), while ``e2r[r, e]`` is p(r | e).  Unsupported rows or
    columns remain exactly zero.  No smoothing constant is inserted because zero mass
    means no geometric evidence, not weak negative evidence.
    """

    joint = np.asarray(joint, dtype=np.float64)
    if joint.ndim != 2:
        raise ValueError("joint must be two-dimensional")
    if not np.all(np.isfinite(joint)) or np.any(joint < 0):
        raise ValueError("joint measure must be finite and non-negative")
    reaction_mass = joint.sum(axis=1)
    protein_mass = joint.sum(axis=0)
    r2e = np.zeros_like(joint)
    e2r = np.zeros_like(joint)
    valid_r = reaction_mass > 0
    valid_e = protein_mass > 0
    if np.any(valid_r):
        r2e[valid_r] = joint[valid_r] / reaction_mass[valid_r, None]
    if np.any(valid_e):
        e2r[:, valid_e] = joint[:, valid_e] / protein_mass[None, valid_e]
    return ConditionalMeasure(
        joint=joint,
        r2e=r2e,
        e2r=e2r,
        reaction_mass=reaction_mass,
        protein_mass=protein_mass,
    )


def normalized_information_support(conditionals: np.ndarray, *, axis: int) -> np.ndarray:
    """Intrinsic concentration of a conditional measure relative to uniform support.

    The value is ``1 - H(q)/log(N)`` in [0,1].  It is zero for a uniform conditional
    and approaches one as the mass concentrates.  This is a diagnostic/support field,
    not a routing threshold: downstream geometry can use it continuously as a measure
    of how informative local positive evidence is.
    """

    q = np.asarray(conditionals, dtype=np.float64)
    if q.ndim != 2 or axis not in (0, 1):
        raise ValueError("conditionals must be 2D and axis must be 0 or 1")
    if not np.all(np.isfinite(q)) or np.any(q < 0):
        raise ValueError("conditionals must be finite and non-negative")
    if axis == 1:
        q = q.T
    n = q.shape[1]
    if n <= 1:
        return np.zeros(q.shape[0], dtype=np.float32)
    mass = q.sum(axis=1)
    out = np.zeros(q.shape[0], dtype=np.float64)
    valid = mass > 0
    if not np.any(valid):
        return out.astype(np.float32)
    qq = np.zeros_like(q)
    qq[valid] = q[valid] / mass[valid, None]
    positive = qq > 0
    terms = np.zeros_like(qq)
    terms[positive] = -qq[positive] * np.log(qq[positive])
    entropy = terms.sum(axis=1)
    out[valid] = 1.0 - entropy[valid] / np.log(float(n))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def binary_relation_from_indices(
    reaction_rows: np.ndarray,
    protein_rows: np.ndarray,
    shape: tuple[int, int],
) -> csr_matrix:
    """Construct the empirical positive relation measure with duplicate pairs collapsed."""

    rr = np.asarray(reaction_rows, dtype=np.int64).reshape(-1)
    ee = np.asarray(protein_rows, dtype=np.int64).reshape(-1)
    if rr.shape != ee.shape:
        raise ValueError("reaction/protein row arrays must align")
    if len(rr) == 0:
        return csr_matrix(shape, dtype=np.float32)
    if np.any(rr < 0) or np.any(rr >= shape[0]) or np.any(ee < 0) or np.any(ee >= shape[1]):
        raise ValueError("relation index out of range")
    relation = csr_matrix((np.ones(len(rr), dtype=np.float32), (rr, ee)), shape=shape)
    if relation.nnz:
        relation.data[:] = 1.0
    relation.eliminate_zeros()
    return relation


def anisotropic_product_pushforward(
    field: np.ndarray,
    relation: np.ndarray,
    reaction_affinity: csr_matrix,
    protein_affinity: csr_matrix,
    *,
    scale: float,
    include_identity: bool = True,
) -> np.ndarray:
    """One mass-conserving Markov push-forward on the Cartesian product graph.

    At a product vertex x=(r,e), all incident reaction-axis and protein-axis
    conductances are normalized *together*.  Thus the empirical measure is
    mollified by one lazy random walk on G_R square G_E, rather than by an
    order-dependent composition K_R K_E.  Edge conductance uses the same
    Charbonnier continuity of the reference field as the variational energy.

    For every source vertex x, outgoing transition weights sum to one exactly;
    total positive mass is therefore preserved without assigning any negative
    evidence to unobserved pairs.
    """
    values = np.asarray(field, dtype=np.float64)
    mass = np.asarray(relation, dtype=np.float64)
    if values.ndim != 2 or mass.shape != values.shape:
        raise ValueError("field and relation must align on reaction-by-protein product space")
    nr, ne = values.shape
    if reaction_affinity.shape != (nr, nr):
        raise ValueError("reaction affinity shape mismatch")
    if protein_affinity.shape != (ne, ne):
        raise ValueError("protein affinity shape mismatch")
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("scale must be finite and positive")
    if np.any(mass < 0) or not np.all(np.isfinite(mass)):
        raise ValueError("relation must be finite and non-negative")

    denominator = np.ones_like(mass) if include_identity else np.zeros_like(mass)

    r_upper = triu(reaction_affinity, k=1, format="coo")
    if len(r_upper.data):
        rr = r_upper.row.astype(np.int64, copy=False)
        rc = r_upper.col.astype(np.int64, copy=False)
        rw = r_upper.data.astype(np.float64, copy=False)
        rdiff = values[rr] - values[rc]
        rconductance = 1.0 / np.sqrt(1.0 + np.square(rdiff / float(scale)))
        redge = rw[:, None] * rconductance
        np.add.at(denominator, rr, redge)
        np.add.at(denominator, rc, redge)
    else:
        rr = rc = np.zeros(0, dtype=np.int64)
        redge = np.zeros((0, ne), dtype=np.float64)

    p_upper = triu(protein_affinity, k=1, format="coo")
    if len(p_upper.data):
        pr = p_upper.row.astype(np.int64, copy=False)
        pc = p_upper.col.astype(np.int64, copy=False)
        pw = p_upper.data.astype(np.float64, copy=False)
        pdiff = values[:, pr] - values[:, pc]
        pconductance = 1.0 / np.sqrt(1.0 + np.square(pdiff / float(scale)))
        pedge = pw[None, :] * pconductance
        denom_t = denominator.T
        np.add.at(denom_t, pr, pedge.T)
        np.add.at(denom_t, pc, pedge.T)
    else:
        pr = pc = np.zeros(0, dtype=np.int64)
        pedge = np.zeros((nr, 0), dtype=np.float64)

    safe = np.where(denominator > 0, denominator, 1.0)
    out = (mass / safe) if include_identity else np.zeros_like(mass)

    if len(rr):
        np.add.at(out, rr, redge * (mass[rc] / safe[rc]))
        np.add.at(out, rc, redge * (mass[rr] / safe[rr]))
    if len(pr):
        out_t = out.T
        np.add.at(out_t, pr, (pedge * (mass[:, pc] / safe[:, pc])).T)
        np.add.at(out_t, pc, (pedge * (mass[:, pr] / safe[:, pr])).T)

    return out


def symmetrized_anisotropic_product_pushforward(
    field: np.ndarray,
    relation: np.ndarray,
    reaction_affinity: csr_matrix,
    protein_affinity: csr_matrix,
    *,
    scale: float,
) -> np.ndarray:
    """Order-symmetric two-factor mollification of a product-space measure.

    Let K_R and K_E be the mass-conserving Charbonnier Markov kernels on the two
    factor directions of the Cartesian product graph.  This returns

        1/2 (K_R K_E + K_E K_R) mu.

    Each branch can traverse one edge in both factor directions, so diagonal
    product neighbourhoods are represented, while averaging the two Lie
    splittings removes the arbitrary choice of which factor acts first.  Since
    both branches are positive and mass-conserving, their average is as well.
    """
    values = np.asarray(field, dtype=np.float64)
    mass = np.asarray(relation, dtype=np.float64)
    if values.ndim != 2 or mass.shape != values.shape:
        raise ValueError("field and relation must align on product space")

    e_then_r = anisotropic_reaction_pushforward(
        values,
        anisotropic_protein_pushforward(
            values, mass, protein_affinity, scale=scale,
            include_identity=True, mass_conserving=True,
        ),
        reaction_affinity,
        scale=scale,
        include_identity=True,
        mass_conserving=True,
    )
    r_then_e = anisotropic_protein_pushforward(
        values,
        anisotropic_reaction_pushforward(
            values, mass, reaction_affinity, scale=scale,
            include_identity=True, mass_conserving=True,
        ),
        protein_affinity,
        scale=scale,
        include_identity=True,
        mass_conserving=True,
    )
    return 0.5 * (e_then_r + r_then_e)


def second_order_anisotropic_product_pushforward(
    field: np.ndarray,
    relation: np.ndarray,
    reaction_affinity: csr_matrix,
    protein_affinity: csr_matrix,
    *,
    scale: float,
) -> np.ndarray:
    """Two-step lazy Markov mollification on the Cartesian product graph.

    The one-step product kernel P is defined by jointly normalizing all
    Charbonnier-weighted reaction- and protein-axis edges incident to each
    product vertex.  This returns P^2 mu.  Two steps are the minimal Cartesian
    graph order that can reach a neighbour differing in both factors, so the
    construction captures diagonal product neighbourhoods without choosing an
    axis order or adding a mixing weight.
    """
    once = anisotropic_product_pushforward(
        field, relation, reaction_affinity, protein_affinity,
        scale=scale, include_identity=True,
    )
    return anisotropic_product_pushforward(
        field, once, reaction_affinity, protein_affinity,
        scale=scale, include_identity=True,
    )


def anisotropic_product_heat_pushforward(
    field: np.ndarray,
    relation: np.ndarray,
    reaction_affinity: csr_matrix,
    protein_affinity: csr_matrix,
    *,
    scale: float,
    numerical_tolerance: float = 1e-12,
) -> np.ndarray:
    """Unit-time heat flow of a positive measure on the Cartesian product graph.

    The product conductance graph contains both reaction-axis and protein-axis
    edges, with Charbonnier conductance determined by the same reference field.
    Let ``P`` be the source-normalized Markov operator on this graph *without an
    arbitrary self-loop*.  Its normalized continuous-time generator is ``P-I``.
    We return the canonical unit-time semigroup

        exp(P - I) mu = exp(-1) sum_{k>=0} P^k mu / k!.

    The Poisson path expansion has three useful consequences for the mainline:
    it is invariant to exchanging the two factor manifolds, it reaches genuine
    two-factor diagonal neighbourhoods through multi-edge paths, and it introduces
    no reaction-vs-protein ordering or learned mixing weight.  The diffusion time
    is the intrinsic unit of the normalized generator (one expected jump), not a
    tuned model hyperparameter.

    ``numerical_tolerance`` only truncates the convergent Poisson series.  The
    output is rescaled by the accumulated scalar mass at machine precision so
    total positive mass is exactly preserved numerically.
    """
    values = np.asarray(field, dtype=np.float64)
    mass = np.asarray(relation, dtype=np.float64)
    if values.ndim != 2 or mass.shape != values.shape:
        raise ValueError("field and relation must align on product space")
    if numerical_tolerance <= 0:
        raise ValueError("numerical_tolerance must be positive")

    # P has no explicit self-loop.  Continuous time supplies the canonical
    # exp(-1) no-jump probability through the k=0 Poisson term.
    def markov_step(current: np.ndarray) -> np.ndarray:
        moved = anisotropic_product_pushforward(
            values,
            current,
            reaction_affinity,
            protein_affinity,
            scale=scale,
            include_identity=False,
        )
        # In the degenerate event that a product vertex has no incident edge,
        # it is an absorbing state of the continuous-time chain.
        # Detect lost mass per connected-isolation pattern by rebuilding only
        # the zero-degree mask from a unit probe when necessary.
        current_total = float(current.sum())
        moved_total = float(moved.sum())
        if current_total > 0 and moved_total < current_total - 1e-10 * max(1.0, current_total):
            probe = np.ones_like(current, dtype=np.float64)
            probe_moved = anisotropic_product_pushforward(
                values,
                probe,
                reaction_affinity,
                protein_affinity,
                scale=scale,
                include_identity=False,
            )
            isolated = probe_moved <= 0
            moved = moved.copy()
            moved[isolated] += current[isolated]
        return moved

    weight = float(np.exp(-1.0))
    accumulated_weight = weight
    term = mass.copy()
    out = weight * term
    k = 0
    while 1.0 - accumulated_weight > numerical_tolerance:
        k += 1
        term = markov_step(term)
        weight /= float(k)
        out += weight * term
        accumulated_weight += weight
        if k > 64:
            raise RuntimeError("unit-time Poisson heat series failed to converge")

    source_mass = float(mass.sum())
    out_mass = float(out.sum())
    if source_mass > 0 and out_mass > 0:
        out *= source_mass / out_mass
    out[out < 0] = 0.0
    return out


def anisotropic_factor_sum_heat_pushforward(
    field: np.ndarray,
    relation: np.ndarray,
    reaction_affinity: csr_matrix,
    protein_affinity: csr_matrix,
    *,
    scale: float,
    numerical_tolerance: float = 1e-12,
) -> np.ndarray:
    """Heat flow generated by the sum of the two normalized factor generators.

    For the Cartesian product manifold, the canonical diffusion generator is

        L_x = L_R + L_E,

    with each factor measured in its own normalized unit of diffusion time.  Let
    P_R and P_E be the source-normalized, no-self-loop Charbonnier Markov
    operators along the reaction and protein fibres.  Then

        L_R = P_R - I,  L_E = P_E - I,
        exp(L_R + L_E) = exp(m (P_bar - I)),

    where P_bar is the equal average of the *active* factor Markov operators and
    m is the number of active factors.  The equal coefficient is not a tunable
    fusion weight: it is the uniformization identity for the sum generator.

    The Poisson expansion therefore gives one expected jump per active factor,
    includes arbitrary alternating product paths, is invariant to swapping the
    factors, and removes the arbitrary Lie-splitting order of sequential
    mollification.
    """
    values = np.asarray(field, dtype=np.float64)
    mass = np.asarray(relation, dtype=np.float64)
    if values.ndim != 2 or mass.shape != values.shape:
        raise ValueError("field and relation must align on product space")
    if numerical_tolerance <= 0:
        raise ValueError("numerical_tolerance must be positive")

    active_reaction = reaction_affinity.nnz > 0
    active_protein = protein_affinity.nnz > 0
    factor_count = int(active_reaction) + int(active_protein)
    if factor_count == 0:
        return mass.copy()

    # The reference field and factor graphs are fixed throughout the Poisson
    # series.  Precompute all Charbonnier edge conductances and source
    # normalizers once; recomputing them at every P^k term is numerically
    # redundant and dominates runtime on the canonical 90 x 432 chart.
    if active_reaction:
        r_upper = triu(reaction_affinity, k=1, format="coo")
        r_rows = r_upper.row.astype(np.int64, copy=False)
        r_cols = r_upper.col.astype(np.int64, copy=False)
        r_weights = r_upper.data.astype(np.float64, copy=False)
        r_difference = values[r_rows] - values[r_cols]
        r_edge = r_weights[:, None] / np.sqrt(
            1.0 + np.square(r_difference / float(scale))
        )
        r_denominator = np.zeros_like(values)
        np.add.at(r_denominator, r_rows, r_edge)
        np.add.at(r_denominator, r_cols, r_edge)
        r_safe = np.where(r_denominator > 0, r_denominator, 1.0)

    if active_protein:
        p_upper = triu(protein_affinity, k=1, format="coo")
        p_rows = p_upper.row.astype(np.int64, copy=False)
        p_cols = p_upper.col.astype(np.int64, copy=False)
        p_weights = p_upper.data.astype(np.float64, copy=False)
        p_difference = values[:, p_rows] - values[:, p_cols]
        p_edge = p_weights[None, :] / np.sqrt(
            1.0 + np.square(p_difference / float(scale))
        )
        p_denominator = np.zeros_like(values)
        p_denominator_t = p_denominator.T
        np.add.at(p_denominator_t, p_rows, p_edge.T)
        np.add.at(p_denominator_t, p_cols, p_edge.T)
        p_safe = np.where(p_denominator > 0, p_denominator, 1.0)

    def factor_average_step(current: np.ndarray) -> np.ndarray:
        parts: list[np.ndarray] = []
        if active_reaction:
            reaction_part = np.zeros_like(current)
            np.add.at(
                reaction_part, r_rows,
                r_edge * (current[r_cols] / r_safe[r_cols]),
            )
            np.add.at(
                reaction_part, r_cols,
                r_edge * (current[r_rows] / r_safe[r_rows]),
            )
            parts.append(reaction_part)
        if active_protein:
            protein_part = np.zeros_like(current)
            protein_part_t = protein_part.T
            np.add.at(
                protein_part_t, p_rows,
                (p_edge * (current[:, p_cols] / p_safe[:, p_cols])).T,
            )
            np.add.at(
                protein_part_t, p_cols,
                (p_edge * (current[:, p_rows] / p_safe[:, p_rows])).T,
            )
            parts.append(protein_part)
        # kNN factor graphs in the canonical evaluator have no isolated vertices.
        # Preserve the previous robustness semantics for degenerate external uses.
        moved = sum(parts) / float(factor_count)
        current_total = float(current.sum())
        moved_total = float(moved.sum())
        if current_total > 0 and moved_total < current_total - 1e-10 * max(1.0, current_total):
            moved = moved.copy()
            moved *= current_total / max(moved_total, np.finfo(np.float64).tiny)
        return moved

    rate = float(factor_count)
    weight = float(np.exp(-rate))
    accumulated_weight = weight
    term = mass.copy()
    out = weight * term
    k = 0
    while 1.0 - accumulated_weight > numerical_tolerance:
        k += 1
        term = factor_average_step(term)
        weight *= rate / float(k)
        out += weight * term
        accumulated_weight += weight
        if k > 96:
            raise RuntimeError("factor-sum heat Poisson series failed to converge")

    source_mass = float(mass.sum())
    out_mass = float(out.sum())
    if source_mass > 0 and out_mass > 0:
        out *= source_mass / out_mass
    out[out < 0] = 0.0
    return out
