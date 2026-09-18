from __future__ import annotations

"""One-time frozen external retention for the unified geometric product field.

The method definition is imported unchanged from the internally frozen clean-dev
implementation.  External Rhea128->141 test labels are used only for the final
metrics.  The benchmark train-side relation plays exactly the role of fold-train
positive observations used during internal cross-validation.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening import run_bime_r2e_clipzyme_expert_v1 as clip
from projects.active.terpene_screening import run_r2e_lambdarank_fusion_v1 as base
from projects.active.terpene_screening.broad_rhea_metrics import (
    DEFAULT_BUDGETS,
    DEFAULT_TOP_PERCENTS,
    evaluate_full_candidate_ranks,
    summarize_query_metrics,
)
from projects.active.terpene_screening.evaluate_broad_rhea_benchmark import encode_chunks
from projects.active.terpene_screening.evaluate_geometric_product_flow_clean_dev_v1 import (
    _full_base_fibre,
    _local_prior,
    _moments,
    _project_positive_vertices_to_chart,
)
from projects.active.terpene_screening.geometric_pair_measure import (
    anisotropic_protein_pushforward,
    anisotropic_reaction_pushforward,
)
from projects.active.terpene_screening.geometric_product_field import (
    anisotropic_bregman_product_field,
    availability_weighted_similarity,
    degree_normalized_affinity,
    product_characteristic_scale,
    self_tuning_cosine_knn_affinity,
    self_tuning_distance_knn_affinity,
)
from projects.active.terpene_screening.rank_open_world import (
    load_feature_schema,
    load_models,
    load_protein_library,
    load_registered_reaction_feature_library,
)

OUT = ROOT / "results/geometric_product_flow_external_retention_v1"
PRIMARY = ROOT / "results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1"
SECONDARY = ROOT / "results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1"
PRIMARY_PROTEINS = ROOT / "data/catalyst_candidate_universes/general_merged/proteins"
SECONDARY_PROTEINS = ROOT / "data/external/enzgfm_current/general_merged_650m_mean_v1"
REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1"
BENCH = ROOT / "results/rhea128_to141_external_v2/rhea128_to141_sprot_strict_double_cold_v2"
QFILE = ROOT / "results/clipzyme_native_extension_v1/r2e_strict650_same_support_v1/mutual_cold_query_ids.txt"
PAIR = ROOT / "results/clipzyme_native_extension_v1/r2e_strict650_same_support_v1/mutual_cold_test_pairs.csv"
SUP = ROOT / "results/clipzyme_native_extension_v1/r2e_strict650_candidate_ids.txt"
BIME = ROOT / "results/bime_rank_unified_v1/r2e_structure_external_confirmation_v1/query_metrics.csv"


def metric_map(frame: pd.DataFrame) -> dict[str, float]:
    m = summarize_query_metrics(frame, budgets=DEFAULT_BUDGETS, top_percents=DEFAULT_TOP_PERCENTS)
    keys = (
        "mrr", "map", "macro_roc_auc", "ndcg_at_10", "hit_at_10",
        "hit_at_20", "hit_at_50", "median_best_positive_rank",
    )
    return {key: float(m[key]) for key in keys}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pf, pids = load_protein_library(PRIMARY_PROTEINS)
    sf, sids = load_protein_library(SECONDARY_PROTEINS)
    if pids != sids or len(pids) != 185918:
        raise RuntimeError("production protein candidate universe drifted")
    pindex = {p: i for i, p in enumerate(pids)}
    lex = base._lexical_rank(pids)

    ps = load_feature_schema(PRIMARY)
    ss = load_feature_schema(SECONDARY)
    rf, rids = load_registered_reaction_feature_library(REACTIONS, ps)
    rf2, rids2 = load_registered_reaction_feature_library(REACTIONS, ss)
    if rids != rids2 or not np.array_equal(rf, rf2):
        raise RuntimeError("production reaction feature libraries drifted")
    rindex = {r: i for i, r in enumerate(rids)}

    pm = load_models(PRIMARY / "models", "production", device)
    sm = load_models(SECONDARY / "models", "production", device)
    if len(pm) != 1 or len(sm) != 1:
        raise RuntimeError("expected one frozen production model per representation")
    pe0 = encode_chunks(pm[0], pf, kind="protein", device=device, chunk_size=8192)
    pe1 = encode_chunks(sm[0], sf, kind="protein", device=device, chunk_size=8192)
    re0 = encode_chunks(pm[0], rf, kind="reaction", device=device, chunk_size=8192)
    re1 = encode_chunks(sm[0], rf, kind="reaction", device=device, chunk_size=8192)

    clip_pt, clip_rows, clip_lookup, clip_rmat, clip_ridx = clip._load_clip_assets(pids, device)
    p0_mean, p0_cov = _moments(pe0)
    p1_mean, p1_cov = _moments(pe1)
    clip_mean, clip_cov = _moments(clip_pt)
    esmc = np.load(PRIMARY_PROTEINS / "embeddings.npy", mmap_mode="r")
    if esmc.shape != (len(pids), 1152):
        raise RuntimeError(f"unexpected ESM-C geometry shape {esmc.shape}")

    qids = [x.strip() for x in QFILE.read_text().splitlines() if x.strip()]
    test = pd.read_csv(PAIR, dtype=str).fillna("")
    train = pd.read_csv(BENCH / "train_pairs.csv", dtype=str).fillna("")
    if len(qids) != 144 or test.reaction_id.nunique() != 144 or len(test) != 309:
        raise RuntimeError("frozen external query set drifted")
    if set(train.reaction_id) & set(qids):
        raise RuntimeError("external reaction-level train/test leakage")
    positives = test.groupby("reaction_id").protein_id.apply(
        lambda x: sorted(set(map(str, x)))
    ).to_dict()

    support = [x.strip() for x in SUP.read_text().splitlines() if x.strip()]
    if len(support) != 166202 or support != sorted(support):
        raise RuntimeError("frozen common candidate support drifted")
    support_set = set(support)
    support_mask = np.asarray([p in support_set for p in pids], dtype=bool)
    support_index = {p: i for i, p in enumerate(support)}

    train_reactions = sorted(set(train.reaction_id.astype(str)) & set(rindex))
    train_positive_rows = {
        str(r): {pindex[p] for p in g.protein_id.astype(str) if p in pindex}
        for r, g in train.groupby("reaction_id")
        if str(r) in rindex
    }
    train_rows = np.asarray([rindex[r] for r in train_reactions], dtype=np.int64)

    manifest = json.loads((REACTIONS / "manifest.json").read_text())
    center_dim = int(manifest["reaction_center_dimension"])
    reaction_center = np.asarray(rf[:, -center_dim:], dtype=np.float32)
    center_available = np.any(reaction_center != 0, axis=1)
    valid_mask = center_available[train_rows]
    valid_train_rows = train_rows[valid_mask]
    valid_train_ids = [r for r, ok in zip(train_reactions, valid_mask) if ok]
    chart_k = int(math.ceil(math.sqrt(len(pids))))
    reaction_neighbours = min(
        len(valid_train_ids), int(math.ceil(math.sqrt(max(1, len(valid_train_ids)))))
    )

    center_t = torch.as_tensor(reaction_center, dtype=torch.float32, device=device)
    qt = torch.as_tensor([rindex[q] for q in qids], dtype=torch.long, device=device)
    tt = torch.as_tensor(valid_train_rows, dtype=torch.long, device=device)
    with torch.no_grad():
        qcenter = center_t[qt]
        tcenter = center_t[tt]
        inter = qcenter @ tcenter.T
        qmass = qcenter.sum(dim=1, keepdim=True)
        tmass = tcenter.sum(dim=1)[None, :]
        union = qmass + tmass - inter
        similarity_to_train = torch.where(
            union > 0, inter / union.clamp_min(1e-12), torch.zeros_like(union)
        ).cpu().numpy()

    base_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    for qi, q in enumerate(qids):
        qrow = rindex[q]
        full_base = _full_base_fibre(
            q, qrow, pe0=pe0, pe1=pe1, re0=re0, re1=re1,
            clip_pt=clip_pt, clip_rows=clip_rows, clip_rmat=clip_rmat,
            clip_ridx=clip_ridx,
        )
        base_order = np.lexsort((lex, -full_base)).astype(np.int32)
        base_projected = base_order[support_mask[base_order]]
        base_inv = np.empty(len(support), dtype=np.int32)
        for rank, row in enumerate(base_projected, 1):
            base_inv[support_index[pids[int(row)]]] = rank
        pr = np.asarray([base_inv[support_index[p]] for p in positives[q]], dtype=np.int32)
        base_rows.append({"query_id": q, **evaluate_full_candidate_ranks(pr, len(support))})

        candidates = np.asarray(base_order[:chart_k], dtype=np.int32)
        sim = similarity_to_train[qi]
        nearest = np.lexsort((np.asarray(valid_train_ids, dtype=object), -sim))[:reaction_neighbours]
        neighbour_ids = [valid_train_ids[i] for i in nearest]
        local_ids = [q, *neighbour_ids]
        local_rows = np.asarray([qrow, *[rindex[r] for r in neighbour_ids]], dtype=np.int64)

        prior = _local_prior(
            local_ids, local_rows, candidates,
            pe0=pe0, pe1=pe1, re0=re0, re1=re1,
            p0_mean=p0_mean, p0_cov=p0_cov, p1_mean=p1_mean, p1_cov=p1_cov,
            clip_pt=clip_pt, clip_lookup=clip_lookup, clip_rmat=clip_rmat,
            clip_ridx=clip_ridx, clip_mean=clip_mean, clip_cov=clip_cov,
        )

        lc = reaction_center[local_rows].astype(np.float64, copy=False)
        li = lc @ lc.T
        lm = lc.sum(axis=1)
        lu = lm[:, None] + lm[None, :] - li
        rsim = np.divide(li, lu, out=np.zeros_like(li), where=lu > 0)
        rdist = 1.0 - np.clip(rsim, 0.0, 1.0)
        np.fill_diagonal(rdist, 0.0)
        reaction_graph = degree_normalized_affinity(
            self_tuning_distance_knn_affinity(
                rdist, available=center_available[local_rows]
            )
        )

        with torch.no_grad():
            et = torch.as_tensor(
                np.asarray(esmc[candidates], dtype=np.float32).copy(), device=device
            )
            et = F.normalize(et, dim=1)
            protein_similarity = (et @ et.T).cpu().numpy()
        cpos = clip_lookup[candidates]
        savail = cpos >= 0
        if int(savail.sum()) >= 2:
            observed = np.flatnonzero(savail)
            cp = clip_pt[torch.as_tensor(cpos[observed], dtype=torch.long, device=device)]
            with torch.no_grad():
                cp = F.normalize(cp.float(), dim=1)
                ss = (cp @ cp.T).cpu().numpy()
            structural = np.eye(len(candidates), dtype=np.float32)
            structural[np.ix_(observed, observed)] = ss
            protein_similarity = availability_weighted_similarity(
                protein_similarity, [(structural, savail, 1.0)]
            )
        protein_graph = degree_normalized_affinity(
            self_tuning_cosine_knn_affinity(protein_similarity)
        )

        observation_vertices: set[int] = set()
        for rid in neighbour_ids:
            observation_vertices.update(train_positive_rows.get(rid, set()))
        observation_rows = np.asarray(sorted(observation_vertices), dtype=np.int64)
        projected = _project_positive_vertices_to_chart(
            observation_rows, candidates, protein_similarity,
            esmc=esmc, clip_pt=clip_pt, clip_lookup=clip_lookup, device=device,
        )
        plook = {int(row): i for i, row in enumerate(observation_rows)}
        relation = np.zeros_like(prior, dtype=np.float64)
        observation_count = 0
        for ri, rid in enumerate(neighbour_ids, start=1):
            source = train_positive_rows.get(rid, set())
            observation_count += len(source)
            idx = [plook[int(p)] for p in source]
            if idx:
                relation[ri] = projected[idx].sum(axis=0)

        dr = relation.sum(axis=1)
        de = relation.sum(axis=0)
        denom = np.sqrt(np.maximum(dr[:, None] * de[None, :], 0.0))
        measure = np.divide(
            relation, denom, out=np.zeros_like(relation), where=denom > 0
        )
        total = float(measure.sum())
        if total > 0:
            measure /= total

        scale = product_characteristic_scale(prior, reaction_graph, [protein_graph])
        measure = anisotropic_protein_pushforward(
            prior, measure, protein_graph, scale=scale,
            include_identity=True, mass_conserving=True,
        )
        measure = anisotropic_reaction_pushforward(
            prior, measure, reaction_graph, scale=scale,
            include_identity=True, mass_conserving=True,
        )
        flowed, info = anisotropic_bregman_product_field(
            prior, reaction_graph, [protein_graph], strength=1.0,
            positive_measure=measure, positive_strength=1.0,
            normalize_positive_rows=False, integral_normalized=True,
            accelerated=True, max_steps=128, tolerance=1e-5,
        )

        scores = full_base.copy()
        scores[candidates] = flowed[0]
        order = np.lexsort((lex, -scores)).astype(np.int32)
        projected_order = order[support_mask[order]]
        inv = np.empty(len(support), dtype=np.int32)
        for rank, row in enumerate(projected_order, 1):
            inv[support_index[pids[int(row)]]] = rank
        pr = np.asarray([inv[support_index[p]] for p in positives[q]], dtype=np.int32)
        flow_rows.append({"query_id": q, **evaluate_full_candidate_ranks(pr, len(support))})
        diagnostics.append({
            "query_id": q,
            "reaction_neighbours": reaction_neighbours,
            "nearest_reaction_similarity": float(sim[nearest[0]]),
            "observation_count": int(observation_count),
            "observation_vertices_projected": int(len(observation_vertices)),
            "target_transported_positive_mass": float(measure[0].sum()),
            "flow_steps": int(info["steps"]),
            "final_relative_change": float(info["relative_change_history"][-1]) if info["relative_change_history"] else 0.0,
            "initial_energy": float(info["energy_history"][0]),
            "final_energy": float(info["energy_history"][-1]),
        })
        if (qi + 1) % 16 == 0 or qi + 1 == len(qids):
            print(f"external geometry {qi+1}/{len(qids)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    base_frame = pd.DataFrame(base_rows)
    flow_frame = pd.DataFrame(flow_rows)
    diag = pd.DataFrame(diagnostics)
    base_frame.to_csv(OUT / "base_query_metrics.csv", index=False)
    flow_frame.to_csv(OUT / "query_metrics.csv", index=False)
    diag.to_csv(OUT / "diagnostics.csv", index=False)
    bime = pd.read_csv(BIME, dtype={"query_id": str})
    if set(bime.query_id.astype(str)) != set(flow_frame.query_id.astype(str)):
        raise RuntimeError("BiME/geometry external query keys differ")
    summary = {
        "status": "one_time_external_retention",
        "protocol": "Frozen geometry method from internal clean-dev; Rhea128->141 strict double-cold mutual-train-cold 144 queries x 166202 common protein support; test labels used only after scoring",
        "queries": len(qids),
        "full_candidates_ranked": len(pids),
        "common_candidates": len(support),
        "positive_pairs": len(test),
        "external_metrics_used_for_selection": False,
        "external_metrics_used_for_retuning": False,
        "train_reaction_test_reaction_overlap": int(len(set(train.reaction_id) & set(qids))),
        "chart_k": chart_k,
        "reaction_neighbours": reaction_neighbours,
        "base_metrics": metric_map(base_frame),
        "geometry_metrics": metric_map(flow_frame),
        "bime_metrics": metric_map(bime),
        "delta_geometry_vs_base": {
            k: metric_map(flow_frame)[k] - metric_map(base_frame)[k]
            for k in metric_map(flow_frame)
        },
        "delta_geometry_vs_bime": {
            k: metric_map(flow_frame)[k] - metric_map(bime)[k]
            for k in metric_map(flow_frame)
        },
        "convergence": {
            "converged": int((diag.final_relative_change <= 1e-5 + 1e-12).sum()),
            "hit_max_steps": int((diag.flow_steps >= 128).sum()),
            "median_steps": float(diag.flow_steps.median()),
            "p90_steps": float(diag.flow_steps.quantile(0.9)),
            "max_final_relative_change": float(diag.final_relative_change.max()),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
