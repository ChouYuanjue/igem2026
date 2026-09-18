from __future__ import annotations

"""Clean-dev v8 evaluator with an order-symmetric two-factor mollifier on one reaction-protein product manifold.

BiME-Rank is only a matched-protocol reference.  The score used here is one scalar
compatibility field.  For each held-out reaction we take a numerical local chart of
that product manifold: the target reaction plus nearby *training* reactions and the
target fibre's leading proteins under the smooth base field.  Training positives are
a positive empirical measure on neighbour fibres, never labels on the held-out fibre
and never targets hand-set to a particular score.  A Charbonnier product-graph flow
then updates the same scalar field.

Both chart radii use the sampling-scale rule ceil(sqrt(N)); they are not tuned score
thresholds.  The chart restriction is computational: outside the chart the exact
full-support base score is retained and the final readout re-sorts all 185,918
candidates.  It is not a router, expert gate, or fixed-prefix readout.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening import run_bime_r2e_clipzyme_expert_v1 as clip
from projects.active.terpene_screening import run_r2e_lambdarank_fusion_v1 as base
from projects.active.terpene_screening.runtime.ranking_metrics import (
    DEFAULT_BUDGETS,
    DEFAULT_TOP_PERCENTS,
    evaluate_full_candidate_ranks,
    summarize_query_metrics,
)
from projects.active.terpene_screening.geometry.product_field import (
    anisotropic_bregman_product_field,
    availability_weighted_similarity,
    degree_normalized_affinity,
    product_characteristic_scale,
    self_tuning_cosine_knn_affinity,
    self_tuning_distance_knn_affinity,
)
from projects.active.terpene_screening.geometry.pair_measure import (
    symmetrized_anisotropic_product_pushforward,
)
from projects.active.terpene_screening.geometry.multiscale import (
    binary_jaccard_distance,
    diffusion_conformal_affinity,
    resolution_product_affinity,
    resolution_product_neighbours,
)

OUT = ROOT / "results/geometric_product_flow_clean_dev_v8"


def _logmeanexp_two(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.logaddexp(a, b) - math.log(2.0)


def _binary_jaccard_cross_matrix(
    query_features: np.ndarray,
    train_features: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU Jaccard distances from query reactions to train reactions."""
    q = torch.as_tensor(np.asarray(query_features, dtype=np.float32), device=device)
    t = torch.as_tensor(np.asarray(train_features, dtype=np.float32), device=device)
    with torch.no_grad():
        q = (q > 0).float(); t = (t > 0).float()
        inter = q @ t.T
        qmass = q.sum(dim=1, keepdim=True)
        tmass = t.sum(dim=1)[None, :]
        union = qmass + tmass - inter
        sim = torch.where(union > 0, inter / union.clamp_min(1e-12), torch.zeros_like(union))
        dist = (1.0 - sim.clamp(0.0, 1.0)).cpu().numpy().astype(np.float64)
        q_available = (qmass[:, 0] > 0).cpu().numpy().astype(bool)
        t_available = (tmass[0] > 0).cpu().numpy().astype(bool)
    return dist, q_available, t_available


def _metric_map(frame: pd.DataFrame) -> dict[str, float]:
    m = summarize_query_metrics(frame, budgets=DEFAULT_BUDGETS, top_percents=DEFAULT_TOP_PERCENTS)
    return {
        "mrr": float(m["mrr"]),
        "map": float(m["map"]),
        "macro_roc_auc": float(m["macro_roc_auc"]),
        "ndcg_at_10": float(m["ndcg_at_10"]),
        "hit_at_10": float(m["hit_at_10"]),
        "hit_at_20": float(m["hit_at_20"]),
        "hit_at_50": float(m["hit_at_50"]),
        "median_best_positive_rank": float(m["median_best_positive_rank"]),
    }


def _moments(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = points.float()
    mean = x.mean(dim=0)
    second = (x.T @ x) / x.shape[0]
    covariance = second - torch.outer(mean, mean)
    return mean, covariance


def _selected_z(
    queries: torch.Tensor,
    points: torch.Tensor,
    mean: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    q = queries.float()
    raw = q @ points.float().T
    centre = q @ mean
    variance = torch.einsum("bi,ij,bj->b", q, covariance, q).clamp_min(1e-12)
    return (raw - centre[:, None]) / torch.sqrt(variance)[:, None]


def _full_base_fibre(
    q: str,
    qrow: int,
    *,
    pe0: torch.Tensor,
    pe1: torch.Tensor,
    re0: torch.Tensor,
    re1: torch.Tensor,
    clip_pt: torch.Tensor,
    clip_rows: np.ndarray,
    clip_rmat: np.ndarray,
    clip_ridx: dict[str, int],
) -> np.ndarray:
    with torch.no_grad():
        s0 = (re0[qrow] @ pe0.T).float().cpu().numpy().astype(np.float64, copy=False)
        s1 = (re1[qrow] @ pe1.T).float().cpu().numpy().astype(np.float64, copy=False)
    pz = (s0 - float(s0.mean())) / max(float(s0.std()), 1e-6)
    sz = (s1 - float(s1.mean())) / max(float(s1.std()), 1e-6)
    potential = _logmeanexp_two(pz, sz)
    if q in clip_ridx:
        cq = torch.as_tensor(np.asarray(clip_rmat[clip_ridx[q]], dtype=np.float32), device=clip_pt.device)
        with torch.no_grad():
            cs = (cq @ clip_pt.T).float().cpu().numpy().astype(np.float64, copy=False)
        cz = (cs - float(cs.mean())) / max(float(cs.std()), 1e-6)
        potential[clip_rows] = (
            np.logaddexp(np.logaddexp(pz[clip_rows], sz[clip_rows]), cz) - math.log(3.0)
        )
    return potential


def _local_prior(
    local_reaction_ids: list[str],
    local_reaction_rows: np.ndarray,
    candidate_rows: np.ndarray,
    *,
    pe0: torch.Tensor,
    pe1: torch.Tensor,
    re0: torch.Tensor,
    re1: torch.Tensor,
    p0_mean: torch.Tensor,
    p0_cov: torch.Tensor,
    p1_mean: torch.Tensor,
    p1_cov: torch.Tensor,
    clip_pt: torch.Tensor,
    clip_lookup: np.ndarray,
    clip_rmat: np.ndarray,
    clip_ridx: dict[str, int],
    clip_mean: torch.Tensor,
    clip_cov: torch.Tensor,
) -> np.ndarray:
    device = pe0.device
    rr = torch.as_tensor(local_reaction_rows, dtype=torch.long, device=device)
    cr = torch.as_tensor(candidate_rows, dtype=torch.long, device=device)
    with torch.no_grad():
        z0 = _selected_z(re0[rr], pe0[cr], p0_mean, p0_cov).cpu().numpy().astype(np.float64)
        z1 = _selected_z(re1[rr], pe1[cr], p1_mean, p1_cov).cpu().numpy().astype(np.float64)
    prior = _logmeanexp_two(z0, z1)

    cpos = clip_lookup[candidate_rows]
    observed_columns = np.flatnonzero(cpos >= 0)
    supported_rows = [i for i, rid in enumerate(local_reaction_ids) if rid in clip_ridx]
    if len(observed_columns) and supported_rows:
        cp = torch.as_tensor(cpos[observed_columns], dtype=torch.long, device=device)
        rq = torch.as_tensor(
            np.stack([
                np.asarray(clip_rmat[clip_ridx[local_reaction_ids[i]]], dtype=np.float32)
                for i in supported_rows
            ]),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            cz = _selected_z(rq, clip_pt[cp], clip_mean, clip_cov).cpu().numpy().astype(np.float64)
        for j, row in enumerate(supported_rows):
            cols = observed_columns
            prior[row, cols] = (
                np.logaddexp(np.logaddexp(z0[row, cols], z1[row, cols]), cz[j])
                - math.log(3.0)
            )
    return prior


def _project_positive_vertices_to_chart(
    observation_rows: np.ndarray,
    candidate_rows: np.ndarray,
    chart_similarity: np.ndarray,
    *,
    esmc: np.ndarray,
    clip_pt: torch.Tensor,
    clip_lookup: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Mass-conserving geometric restriction of positive atoms to a protein chart.

    Each observed protein is a unit atom of the empirical positive measure.  Its
    restriction to the finite chart is a partition of unity defined by a
    self-tuning Gaussian kernel.  Local bandwidths use the same sqrt(N) sampling
    scale as the graph discretisation.  ESM-C geometry is always present; when
    both endpoints have CLIPZyme structure-dependent embeddings, that observed
    geometry refines the pair similarity by equal availability-neutral averaging.

    No observation is discarded and no score/rank threshold is used: every input
    row returns non-negative chart weights summing to one.
    """

    obs = np.asarray(observation_rows, dtype=np.int64).reshape(-1)
    chart = np.asarray(candidate_rows, dtype=np.int64).reshape(-1)
    if len(obs) == 0:
        return np.zeros((0, len(chart)), dtype=np.float64)
    if len(chart) < 2:
        return np.ones((len(obs), len(chart)), dtype=np.float64)

    with torch.no_grad():
        ot = torch.as_tensor(np.asarray(esmc[obs], dtype=np.float32).copy(), device=device)
        ct = torch.as_tensor(np.asarray(esmc[chart], dtype=np.float32).copy(), device=device)
        ot = F.normalize(ot, dim=1)
        ct = F.normalize(ct, dim=1)
        similarity = (ot @ ct.T).cpu().numpy().astype(np.float64)

    obs_clip = clip_lookup[obs]
    chart_clip = clip_lookup[chart]
    oi = np.flatnonzero(obs_clip >= 0)
    cj = np.flatnonzero(chart_clip >= 0)
    if len(oi) and len(cj):
        with torch.no_grad():
            op = F.normalize(
                clip_pt[torch.as_tensor(obs_clip[oi], dtype=torch.long, device=device)].float(),
                dim=1,
            )
            cp = F.normalize(
                clip_pt[torch.as_tensor(chart_clip[cj], dtype=torch.long, device=device)].float(),
                dim=1,
            )
            structural = (op @ cp.T).cpu().numpy().astype(np.float64)
        similarity[np.ix_(oi, cj)] = 0.5 * (
            similarity[np.ix_(oi, cj)] + structural
        )

    similarity = np.clip(similarity, -1.0, 1.0)
    distance = np.sqrt(np.maximum(2.0 - 2.0 * similarity, 0.0))
    chart_distance = np.sqrt(
        np.maximum(2.0 - 2.0 * np.clip(chart_similarity.astype(np.float64), -1.0, 1.0), 0.0)
    )
    np.fill_diagonal(chart_distance, np.inf)
    scale_k = min(len(chart) - 1, max(1, int(math.ceil(math.sqrt(len(chart))))))
    chart_sigma = np.partition(chart_distance, scale_k - 1, axis=1)[:, scale_k - 1]
    obs_sigma = np.partition(distance, scale_k - 1, axis=1)[:, scale_k - 1]
    finite_scale = np.concatenate([
        chart_sigma[np.isfinite(chart_sigma) & (chart_sigma > 1e-8)],
        obs_sigma[np.isfinite(obs_sigma) & (obs_sigma > 1e-8)],
    ])
    fallback = float(np.median(finite_scale)) if len(finite_scale) else 1.0
    chart_sigma = np.where(np.isfinite(chart_sigma) & (chart_sigma > 1e-8), chart_sigma, fallback)
    obs_sigma = np.where(np.isfinite(obs_sigma) & (obs_sigma > 1e-8), obs_sigma, fallback)
    denominator = np.maximum(obs_sigma[:, None] * chart_sigma[None, :], 1e-8)
    weights = np.exp(-np.square(distance) / denominator)
    mass = weights.sum(axis=1, keepdims=True)
    bad = ~(np.isfinite(mass[:, 0]) & (mass[:, 0] > 0))
    if np.any(bad):
        nearest = np.argmin(distance[bad], axis=1)
        weights[bad] = 0.0
        weights[np.flatnonzero(bad), nearest] = 1.0
        mass = weights.sum(axis=1, keepdims=True)
    return weights / mass


def evaluate_fold(
    fold: int,
    device_name: str,
    chart_k: int,
    reaction_neighbours: int,
    max_steps: int,
    max_queries: int | None,
    ablation: str,
) -> dict[str, object]:
    device = torch.device(device_name)
    ids, reaction_features, reaction_ids, pe0, pe1, re0, re1 = base._load_fold_embeddings(fold, device)
    if len(ids) != 185918:
        raise RuntimeError("clean-dev candidate universe drifted")
    pindex = {p: i for i, p in enumerate(ids)}
    rindex = {r: i for i, r in enumerate(reaction_ids)}
    lex = base._lexical_rank(ids)

    clip_pt, clip_rows, clip_lookup, clip_rmat, clip_ridx = clip._load_clip_assets(ids, device)
    p0_mean, p0_cov = _moments(pe0)
    p1_mean, p1_cov = _moments(pe1)
    clip_mean, clip_cov = _moments(clip_pt)
    esmc = np.load(base.PRIMARY_PROTEINS / "embeddings.npy", mmap_mode="r")
    if esmc.shape != (len(ids), 1152):
        raise RuntimeError(f"unexpected ESM-C geometry shape: {esmc.shape}")

    split = base.DEV_ROOT / "baseline_base" / f"fold{fold}"
    train = pd.read_csv(split / "training_pairs.csv", dtype=str).fillna("")
    dev = pd.read_csv(split / "dev_pairs.csv", dtype=str).fillna("")
    train_reactions = sorted(train.reaction_id.astype(str).unique())
    query_ids = sorted(dev.reaction_id.astype(str).unique())
    if max_queries is not None:
        query_ids = query_ids[: int(max_queries)]
    if set(train_reactions) & set(query_ids):
        raise RuntimeError("reaction-level train/dev leakage")
    positives = dev.groupby("reaction_id").protein_id.apply(lambda x: sorted(set(map(str, x)))).to_dict()
    train_positive_rows = {
        str(r): {pindex[p] for p in g.protein_id.astype(str) if p in pindex}
        for r, g in train.groupby("reaction_id")
    }

    train_rows_np = np.asarray([rindex[r] for r in train_reactions], dtype=np.int64)

    # The deterministic reaction representation has three chemically distinct
    # coordinate scales: global DRFP reaction difference, whole-reaction
    # RDKitPlus structure, and atom-mapped local reaction center.  They define
    # one product manifold; the small categorical block is intentionally not
    # used because its one-hot partitions would create hard chemical gates.
    reaction_manifest = json.loads((base.REACTIONS / "manifest.json").read_text())
    center_dim = int(reaction_manifest["reaction_center_dimension"])
    base_dim = int(reaction_manifest["base_dimension"])
    rdkit_manifest = json.loads((Path(reaction_manifest["base_feature_dir"]) / "manifest.json").read_text())
    pre_rdkit_dim = int(rdkit_manifest["base_dimension"])
    rdkit_dim = int(rdkit_manifest["rdkitplus_dimension"])
    categorical_manifest = json.loads((Path(rdkit_manifest["base_feature_dir"]) / "manifest.json").read_text())
    drfp_dim = int(categorical_manifest["contract"]["drfp_dimension"])
    if not (0 < drfp_dim < pre_rdkit_dim < base_dim < reaction_features.shape[1]):
        raise RuntimeError("reaction feature block geometry drifted")
    if base_dim - pre_rdkit_dim != rdkit_dim or reaction_features.shape[1] - base_dim != center_dim:
        raise RuntimeError("reaction feature block dimensions drifted")

    reaction_views = [
        np.asarray(reaction_features[:, :drfp_dim], dtype=np.float32),
        np.asarray(reaction_features[:, pre_rdkit_dim:base_dim], dtype=np.float32),
        np.asarray(reaction_features[:, base_dim:base_dim + center_dim], dtype=np.float32),
    ]
    reaction_view_names = ["drfp_global", "rdkitplus_whole", "mapped_center"]
    train_rows_np = np.asarray([rindex[r] for r in train_reactions], dtype=np.int64)
    query_rows_np = np.asarray([rindex[q] for q in query_ids], dtype=np.int64)
    effective_chart_k = min(
        len(ids), int(chart_k) if int(chart_k) > 0 else int(math.ceil(math.sqrt(len(ids))))
    )
    effective_reaction_neighbours = min(
        len(train_reactions),
        int(reaction_neighbours)
        if int(reaction_neighbours) > 0
        else int(math.ceil(math.sqrt(max(1, len(train_reactions))))),
    )

    reaction_cross_distances: list[np.ndarray] = []
    query_view_available: list[np.ndarray] = []
    train_view_available: list[np.ndarray] = []
    for view in reaction_views:
        dist, q_available, t_available = _binary_jaccard_cross_matrix(
            view[query_rows_np], view[train_rows_np], device=device
        )
        reaction_cross_distances.append(dist)
        query_view_available.append(q_available)
        train_view_available.append(t_available)

    base_records: list[dict[str, object]] = []
    flow_records: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    for qi, q in enumerate(query_ids):
        qrow = rindex[q]
        full_base = _full_base_fibre(
            q,
            qrow,
            pe0=pe0,
            pe1=pe1,
            re0=re0,
            re1=re1,
            clip_pt=clip_pt,
            clip_rows=clip_rows,
            clip_rmat=clip_rmat,
            clip_ridx=clip_ridx,
        )
        base_order = np.lexsort((lex, -full_base)).astype(np.int32)
        base_inv = np.empty(len(ids), dtype=np.int32)
        base_inv[base_order] = np.arange(1, len(ids) + 1, dtype=np.int32)
        prows = np.asarray([pindex[p] for p in positives[q]], dtype=np.int32)
        base_records.append({"fold": fold, "query_id": q, **evaluate_full_candidate_ranks(base_inv[prows], len(ids))})

        kk = effective_chart_k
        base_chart = base_order[:kk]
        selected_train, reaction_chart_info = resolution_product_neighbours(
            [d[qi] for d in reaction_cross_distances],
            [bool(a[qi]) for a in query_view_available],
            train_view_available,
            k=effective_reaction_neighbours,
        )
        neighbour_ids = [train_reactions[int(i)] for i in selected_train]
        nk = len(neighbour_ids)
        local_ids = [q, *neighbour_ids]
        local_rows = np.asarray([qrow, *[rindex[r] for r in neighbour_ids]], dtype=np.int64)
        nearest_by_view = []
        for dmat, qavail, tavail in zip(reaction_cross_distances, query_view_available, train_view_available):
            valid = np.asarray(tavail, dtype=bool) & np.isfinite(dmat[qi])
            nearest_by_view.append(
                float(1.0 - np.min(dmat[qi][valid])) if bool(qavail[qi]) and np.any(valid) else float("nan")
            )
        nearest_similarity = nearest_by_view[-1]

        # Numerical restriction of the protein manifold.  Positive observations
        # outside the sqrt(N) target chart are not dropped or inserted as extra
        # vertices; below, their unit masses are restricted to this chart by a
        # mass-conserving self-tuning geometric partition of unity.
        candidates = np.asarray(base_chart, dtype=np.int32)
        domain_size = len(candidates)
        observation_vertices: set[int] = set()
        for rid in neighbour_ids:
            observation_vertices.update(train_positive_rows[rid])

        prior = _local_prior(
            local_ids,
            local_rows,
            candidates,
            pe0=pe0,
            pe1=pe1,
            re0=re0,
            re1=re1,
            p0_mean=p0_mean,
            p0_cov=p0_cov,
            p1_mean=p1_mean,
            p1_cov=p1_cov,
            clip_pt=clip_pt,
            clip_lookup=clip_lookup,
            clip_rmat=clip_rmat,
            clip_ridx=clip_ridx,
            clip_mean=clip_mean,
            clip_cov=clip_cov,
        )

        local_reaction_distances: list[np.ndarray] = []
        local_reaction_availability: list[np.ndarray] = []
        for view in reaction_views:
            distance, available = binary_jaccard_distance(view[local_rows])
            local_reaction_distances.append(distance)
            local_reaction_availability.append(available)
        # One protein-state geometry.  ESM-C sequence geometry is always present.
        # CLIPZyme's author-native representation depends on protein structure and
        # refines only pairs for which both structures are observed.  Missing
        # structure therefore leaves the ESM-C geometry exactly unchanged.
        with torch.no_grad():
            esmc_local = torch.as_tensor(
                np.asarray(esmc[candidates], dtype=np.float32).copy(), device=device
            )
            esmc_local = F.normalize(esmc_local, dim=1)
            protein_similarity = (esmc_local @ esmc_local.T).cpu().numpy()
        clip_positions = clip_lookup[candidates]
        structure_available = clip_positions >= 0
        if int(structure_available.sum()) >= 2:
            observed = np.flatnonzero(structure_available)
            cpt = clip_pt[
                torch.as_tensor(clip_positions[observed], dtype=torch.long, device=device)
            ]
            with torch.no_grad():
                cpt = F.normalize(cpt.float(), dim=1)
                observed_similarity = (cpt @ cpt.T).cpu().numpy()
            structure_similarity = np.eye(domain_size, dtype=np.float32)
            structure_similarity[np.ix_(observed, observed)] = observed_similarity
            protein_similarity = availability_weighted_similarity(
                protein_similarity,
                [(structure_similarity, structure_available, 1.0)],
            )

        reaction_affinity, reaction_kernel_info = resolution_product_affinity(
            local_reaction_distances, local_reaction_availability
        )
        reaction_graph = degree_normalized_affinity(reaction_affinity)
        ambient_protein_affinity = self_tuning_cosine_knn_affinity(protein_similarity)
        conformal_protein_affinity, protein_conformal_info = diffusion_conformal_affinity(
            ambient_protein_affinity
        )
        protein_graphs = [degree_normalized_affinity(conformal_protein_affinity)]
        if ablation == "reaction_only":
            protein_graphs = []
        elif ablation == "protein_only":
            reaction_graph = csr_matrix(reaction_graph.shape, dtype=np.float32)

        positive_measure = np.zeros_like(prior, dtype=np.float64)
        observation_count = sum(len(train_positive_rows[rid]) for rid in neighbour_ids)
        observed_reactions = sum(bool(train_positive_rows[rid]) for rid in neighbour_ids)
        observation_rows = np.asarray(sorted(observation_vertices), dtype=np.int64)
        projected_atoms = _project_positive_vertices_to_chart(
            observation_rows,
            candidates,
            protein_similarity,
            esmc=esmc,
            clip_pt=clip_pt,
            clip_lookup=clip_lookup,
            device=device,
        )
        projected_lookup = {int(row): i for i, row in enumerate(observation_rows)}
        for ri, rid in enumerate(neighbour_ids, start=1):
            rows = [projected_lookup[int(p)] for p in train_positive_rows[rid]]
            if rows:
                # Sum unit atomic measures: total mass on this fibre equals the
                # number of observed positives for that reaction exactly.
                positive_measure[ri] = projected_atoms[rows].sum(axis=0)

        # Treat the observed relation as a weighted bipartite geometric object,
        # not raw annotation counts.  Symmetric degree normalization
        # A_re/sqrt(d_r d_e) is direction-neutral: densely annotated reactions
        # and promiscuous proteins cannot dominate merely through degree, while
        # R2E and E2R remain slices of the same normalized joint relation.
        empirical_measure = positive_measure.copy()
        reaction_degree = empirical_measure.sum(axis=1)
        protein_degree = empirical_measure.sum(axis=0)
        denominator = np.sqrt(
            np.maximum(reaction_degree[:, None] * protein_degree[None, :], 0.0)
        )
        empirical_measure = np.divide(
            empirical_measure,
            denominator,
            out=np.zeros_like(empirical_measure),
            where=denominator > 0,
        )
        empirical_total = float(empirical_measure.sum())
        if empirical_total > 0:
            empirical_measure /= empirical_total
        transport_scale = product_characteristic_scale(
            prior, reaction_graph, protein_graphs
        )
        if protein_graphs:
            transported_measure = symmetrized_anisotropic_product_pushforward(
                prior, empirical_measure, reaction_graph, protein_graphs[0],
                scale=transport_scale,
            )
        else:
            zero_protein = csr_matrix((prior.shape[1], prior.shape[1]), dtype=np.float32)
            transported_measure = symmetrized_anisotropic_product_pushforward(
                prior, empirical_measure, reaction_graph, zero_protein,
                scale=transport_scale,
            )

        effective_positive_strength = 0.0 if ablation == "no_observations" else 1.0
        flowed, info = anisotropic_bregman_product_field(
            prior,
            reaction_graph,
            protein_graphs,
            strength=1.0,
            positive_measure=transported_measure,
            positive_strength=effective_positive_strength,
            normalize_positive_rows=False,
            integral_normalized=True,
            accelerated=True,
            max_steps=max_steps,
            tolerance=1e-5,
        )
        # Exact full-support readout: the local chart modifies scores, not a fixed
        # prefix.  Candidates may naturally cross the chart boundary either way.
        flow_scores = full_base.copy()
        flow_scores[candidates] = flowed[0]
        flow_order = np.lexsort((lex, -flow_scores)).astype(np.int32)
        flow_inv = np.empty(len(ids), dtype=np.int32)
        flow_inv[flow_order] = np.arange(1, len(ids) + 1, dtype=np.int32)
        flow_records.append({"fold": fold, "query_id": q, **evaluate_full_candidate_ranks(flow_inv[prows], len(ids))})

        base_local = prior[0]
        target_exact = full_base[candidates]
        diagnostics.append(
            {
                "fold": fold,
                "query_id": q,
                "reaction_neighbours": nk,
                "base_chart_size": kk,
                "protein_domain_size": domain_size,
                "observation_vertices_projected": len(observation_vertices),
                "observation_count": observation_count,
                "observed_reactions": observed_reactions,
                "target_transported_positive_mass": float(transported_measure[0].sum()),
                "target_transported_positive_support": int(np.count_nonzero(transported_measure[0])),
                "nearest_reaction_similarity": nearest_similarity,
                "nearest_drfp_similarity": nearest_by_view[0],
                "nearest_rdkitplus_similarity": nearest_by_view[1],
                "nearest_center_similarity": nearest_by_view[2],
                "nearest_product_energy": float(reaction_chart_info.get("nearest_product_energy", float("nan"))),
                "furthest_chart_product_energy": float(reaction_chart_info.get("furthest_chart_product_energy", float("nan"))),
                "reaction_product_view_scales": json.dumps(reaction_chart_info.get("view_scales", [])),
                "reaction_product_view_resolution": json.dumps(reaction_chart_info.get("view_resolution", [])),
                "reaction_graph_mean_view_resolution": json.dumps(reaction_kernel_info.get("mean_view_resolution", [])),
                "reaction_graph_undirected_edges": int(reaction_kernel_info.get("undirected_edges", 0)),
                "protein_conformal_edges": int(protein_conformal_info["edges"]),
                "protein_conformal_degree_median": float(protein_conformal_info["degree_median"]),
                "protein_conformal_degree_max": int(protein_conformal_info["degree_max"]),
                "protein_conformal_weighted_degree_cv": float(protein_conformal_info["weighted_degree_cv"]),
                "protein_mean_diffusion_correction": float(protein_conformal_info["mean_diffusion_correction"]),
                "analytic_base_max_abs_error": float(np.max(np.abs(base_local - target_exact))),
                "flow_steps": int(info["steps"]),
                "final_relative_change": float(info["relative_change_history"][-1]) if info["relative_change_history"] else 0.0,
                "solver_step_size": float(info["step_size"]),
                "characteristic_scale": float(info["characteristic_scale"]),
                "initial_energy": float(info["energy_history"][0]),
                "final_energy": float(info["energy_history"][-1]),
                "target_fibre_l2_change": float(np.linalg.norm(flowed[0] - prior[0])),
                "positive_measure_strength": effective_positive_strength,
            }
        )
        if (qi + 1) % 16 == 0 or qi + 1 == len(query_ids):
            print(f"geometry-flow fold={fold} {qi + 1}/{len(query_ids)}", flush=True)

    base_frame = pd.DataFrame(base_records)
    flow_frame = pd.DataFrame(flow_records)
    diag_frame = pd.DataFrame(diagnostics)
    suffix = "full" if max_queries is None else f"q{len(query_ids)}"
    out = OUT / f"fold{fold}" / f"{suffix}_{ablation}"
    out.mkdir(parents=True, exist_ok=True)
    base_frame.to_csv(out / "base_query_metrics.csv", index=False)
    flow_frame.to_csv(out / "flow_query_metrics.csv", index=False)
    diag_frame.to_csv(out / "diagnostics.csv", index=False)
    summary = {
        "method": "order_symmetric_product_measure_charbonnier_bregman_field_v8",
        "fold": fold,
        "queries": len(query_ids),
        "candidate_count": len(ids),
        "chart_k": int(effective_chart_k),
        "chart_rule": "ceil(sqrt(full_candidate_support)) unless explicitly overridden for diagnostics",
        "protein_domain_rule": "target sqrt(N) high-potential chart; all train-positive atoms are mass-conservingly restricted into the chart by self-tuning protein geometry",
        "reaction_neighbours": int(effective_reaction_neighbours),
        "reaction_chart_rule": "query-centred locally conformal product metric; ceil(sqrt(fold-train support)) nearest reactions unless explicitly overridden",
        "reaction_geometry": "locally conformal product metric on DRFP global, RDKitPlus whole-reaction, and atom-mapped reaction-center Jaccard coordinates with perplexity-contraction tensor precision",
        "reaction_view_dimensions": {"drfp_global": drfp_dim, "rdkitplus_whole": rdkit_dim, "mapped_center": center_dim},
        "reaction_view_policy": "each coordinate precision is exp(KL(local self-tuning kernel || uniform))-1 = ambient-support/effective-perplexity - 1; flat coordinates vanish continuously and sharply resolving coordinates dominate by intrinsic resolution gain; no labels, thresholds, or learned inter-view weights; categorical one-hot block excluded to avoid hard partitions",
        "protein_geometry": "self-tuning ESM-C/observed-CLIP atlas topology with exact t=1 diffusion-conformal conductance on the same edges; no new protein edges and no learned scale",
        "flow_strength": 1.0,
        "positive_measure_strength": 0.0 if ablation == "no_observations" else 1.0,
        "positive_measure_transport": "order-symmetric product mollifier 1/2(K_R K_E + K_E K_R) using mass-conserving F0-dependent Charbonnier Markov kernels on the two factor directions; preserves two-axis product paths without privileging an axis order",
        "variational_discretization": "integral-normalized: fidelity averaged over product vertices and each geometric axis averaged over its product-edge mass",
        "ablation": ablation,
        "max_steps": int(max_steps),
        "base_metrics": _metric_map(base_frame),
        "flow_metrics": _metric_map(flow_frame),
        "median_observation_count": float(diag_frame.observation_count.median()),
        "queries_with_observation": int((diag_frame.observation_count > 0).sum()),
        "median_protein_domain_size": float(diag_frame.protein_domain_size.median()),
        "median_observation_vertices_projected": float(diag_frame.observation_vertices_projected.median()),
        "max_analytic_base_error": float(diag_frame.analytic_base_max_abs_error.max()),
        "labels_used_for_scoring": False,
        "positive_measure_from_training_fold_only": True,
        "external_metrics_used": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chart-k", type=int, default=0, help="0 = ceil(sqrt(candidate support))")
    ap.add_argument("--reaction-neighbours", type=int, default=0, help="0 = ceil(sqrt(train reaction support))")
    ap.add_argument("--max-steps", type=int, default=64)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument(
        "--ablation",
        choices=("full", "reaction_only", "protein_only", "no_observations"),
        default="full",
    )
    args = ap.parse_args()
    result = evaluate_fold(
        args.fold,
        args.device,
        args.chart_k,
        args.reaction_neighbours,
        args.max_steps,
        args.max_queries,
        args.ablation,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
