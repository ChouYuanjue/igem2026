from __future__ import annotations

"""Same-field E2R fibre audit for the canonical product-manifold method.

This does not train an E2R model and does not compare against a differently split
historical E2R benchmark.  It takes the exact scalar prior F0(r,e) used by the
canonical R2E product field, holds e fixed, and reads F*(.,e) on the *same*
fold-train empirical relation and the same double-cold dev pairs.

Numerically it is the factor-swapped chart construction of canonical v8:
  R2E: (target reaction + nearby train reactions) x top-F0 proteins
  E2R: top-F0 reactions x (target protein + nearby train proteins)
Off-chart positive atoms are restricted by a mass-conserving partition of unity in
the corresponding intrinsic factor geometry.  No dev label enters scoring.
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
from projects.active.terpene_screening.broad_rhea_metrics import (
    DEFAULT_BUDGETS,
    DEFAULT_TOP_PERCENTS,
    evaluate_full_candidate_ranks,
    summarize_query_metrics,
)
from projects.active.terpene_screening.evaluate_geometric_product_flow_clean_dev_v8 import (
    _binary_jaccard_cross_matrix,
    _local_prior,
    _metric_map,
    _moments,
)
from projects.active.terpene_screening.geometric_pair_measure import (
    symmetrized_anisotropic_product_pushforward,
)
from projects.active.terpene_screening.geometric_product_field import (
    anisotropic_bregman_product_field,
    availability_weighted_similarity,
    degree_normalized_affinity,
    product_characteristic_scale,
    self_tuning_cosine_knn_affinity,
)
from projects.active.terpene_screening.multiscale_geometry import (
    binary_jaccard_distance,
    diffusion_conformal_affinity,
    resolution_product_affinity,
    resolution_product_partition,
)

OUT = ROOT / "results/product_field_e2r_section_probe_v1"


def _reaction_views(reaction_features: np.ndarray) -> tuple[list[np.ndarray], dict[str, int]]:
    manifest = json.loads((base.REACTIONS / "manifest.json").read_text())
    center_dim = int(manifest["reaction_center_dimension"])
    base_dim = int(manifest["base_dimension"])
    rdkit_manifest = json.loads((Path(manifest["base_feature_dir"]) / "manifest.json").read_text())
    pre_rdkit_dim = int(rdkit_manifest["base_dimension"])
    rdkit_dim = int(rdkit_manifest["rdkitplus_dimension"])
    categorical_manifest = json.loads((Path(rdkit_manifest["base_feature_dir"]) / "manifest.json").read_text())
    drfp_dim = int(categorical_manifest["contract"]["drfp_dimension"])
    if not (0 < drfp_dim < pre_rdkit_dim < base_dim < reaction_features.shape[1]):
        raise RuntimeError("reaction feature block geometry drifted")
    if base_dim - pre_rdkit_dim != rdkit_dim or reaction_features.shape[1] - base_dim != center_dim:
        raise RuntimeError("reaction feature block dimensions drifted")
    return [
        np.asarray(reaction_features[:, :drfp_dim], dtype=np.float32),
        np.asarray(reaction_features[:, pre_rdkit_dim:base_dim], dtype=np.float32),
        np.asarray(reaction_features[:, base_dim:base_dim + center_dim], dtype=np.float32),
    ], {"drfp_global": drfp_dim, "rdkitplus_whole": rdkit_dim, "mapped_center": center_dim}


def _protein_similarity_block(
    rows: np.ndarray,
    *,
    esmc: np.ndarray,
    clip_pt: torch.Tensor,
    clip_lookup: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    with torch.no_grad():
        x = F.normalize(
            torch.as_tensor(np.asarray(esmc[rows], dtype=np.float32).copy(), device=device),
            dim=1,
        )
        similarity = (x @ x.T).cpu().numpy().astype(np.float32)
    cpos = clip_lookup[rows]
    available = cpos >= 0
    if int(available.sum()) >= 2:
        obs = np.flatnonzero(available)
        with torch.no_grad():
            sx = F.normalize(
                clip_pt[torch.as_tensor(cpos[obs], dtype=torch.long, device=device)].float(), dim=1
            )
            ss = (sx @ sx.T).cpu().numpy().astype(np.float32)
        structure = np.eye(len(rows), dtype=np.float32)
        structure[np.ix_(obs, obs)] = ss
        similarity = availability_weighted_similarity(similarity, [(structure, available, 1.0)])
    return similarity, available


def _protein_query_neighbours(
    query_row: int,
    train_rows: np.ndarray,
    *,
    k: int,
    esmc: np.ndarray,
    clip_pt: torch.Tensor,
    clip_lookup: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    train_rows = np.asarray(train_rows, dtype=np.int64)
    with torch.no_grad():
        q = F.normalize(
            torch.as_tensor(np.asarray(esmc[[query_row]], dtype=np.float32).copy(), device=device), dim=1
        )
        t = F.normalize(
            torch.as_tensor(np.asarray(esmc[train_rows], dtype=np.float32).copy(), device=device), dim=1
        )
        esmc_sim = (q @ t.T).cpu().numpy()[0].astype(np.float64)
    combined = esmc_sim.copy()
    qclip = int(clip_lookup[query_row])
    tclip = clip_lookup[train_rows]
    supported = np.flatnonzero(tclip >= 0)
    structure_used = False
    if qclip >= 0 and len(supported):
        with torch.no_grad():
            qp = F.normalize(clip_pt[qclip:qclip+1].float(), dim=1)
            tp = F.normalize(
                clip_pt[torch.as_tensor(tclip[supported], dtype=torch.long, device=device)].float(), dim=1
            )
            structural = (qp @ tp.T).cpu().numpy()[0].astype(np.float64)
        combined[supported] = 0.5 * (combined[supported] + structural)
        structure_used = True
    distance = np.sqrt(np.maximum(2.0 - 2.0 * np.clip(combined, -1.0, 1.0), 0.0))
    kk = min(len(train_rows), max(1, int(k)))
    pick = np.argpartition(distance, kk - 1)[:kk]
    pick = pick[np.argsort(distance[pick], kind="stable")]
    return train_rows[pick].astype(np.int64), {
        "query_structure_observed": bool(qclip >= 0),
        "structure_coordinate_used": bool(structure_used),
        "nearest_combined_similarity": float(combined[pick[0]]),
        "furthest_chart_combined_similarity": float(combined[pick[-1]]),
    }


def _project_reaction_atoms_to_chart(
    observation_rows: np.ndarray,
    chart_rows: np.ndarray,
    reaction_views: list[np.ndarray],
    *,
    device: torch.device,
) -> np.ndarray:
    obs = np.asarray(observation_rows, dtype=np.int64)
    chart = np.asarray(chart_rows, dtype=np.int64)
    if len(obs) == 0:
        return np.zeros((0, len(chart)), dtype=np.float64)
    cross_distances: list[np.ndarray] = []
    ref_distances: list[np.ndarray] = []
    q_available: list[np.ndarray] = []
    r_available: list[np.ndarray] = []
    for view in reaction_views:
        cross, qa, ra = _binary_jaccard_cross_matrix(view[obs], view[chart], device=device)
        ref, ref_av = binary_jaccard_distance(view[chart])
        cross_distances.append(cross)
        ref_distances.append(ref)
        q_available.append(qa)
        r_available.append(ref_av & ra)
    weights, _ = resolution_product_partition(
        cross_distances, ref_distances, q_available, r_available
    )
    return weights


def evaluate_fold(
    fold: int,
    device_name: str,
    reaction_chart_k: int,
    protein_neighbours: int,
    max_steps: int,
    max_queries: int | None,
    query_file: Path | None = None,
) -> dict[str, object]:
    device = torch.device(device_name)
    ids, reaction_features, reaction_ids, pe0, pe1, re0, re1 = base._load_fold_embeddings(fold, device)
    if len(ids) != 185918 or len(reaction_ids) != 11081:
        raise RuntimeError("canonical support drifted")
    pindex = {p: i for i, p in enumerate(ids)}
    rindex = {r: i for i, r in enumerate(reaction_ids)}
    rlex = base._lexical_rank(reaction_ids)

    clip_pt, _, clip_lookup, clip_rmat, clip_ridx = clip._load_clip_assets(ids, device)
    p0_mean, p0_cov = _moments(pe0)
    p1_mean, p1_cov = _moments(pe1)
    clip_mean, clip_cov = _moments(clip_pt)
    esmc = np.load(base.PRIMARY_PROTEINS / "embeddings.npy", mmap_mode="r")
    reaction_views, reaction_dims = _reaction_views(reaction_features)

    split = base.DEV_ROOT / "baseline_base" / f"fold{fold}"
    train = pd.read_csv(split / "training_pairs.csv", dtype=str).fillna("")
    dev = pd.read_csv(split / "dev_pairs.csv", dtype=str).fillna("")
    if set(train.reaction_id) & set(dev.reaction_id):
        raise RuntimeError("reaction-level train/dev leakage")
    if set(train.protein_id) & set(dev.protein_id):
        raise RuntimeError("protein-level train/dev leakage")
    if set(map(tuple, train[["protein_id","reaction_id"]].to_numpy())) & set(map(tuple, dev[["protein_id","reaction_id"]].to_numpy())):
        raise RuntimeError("pair-level train/dev leakage")

    positives = dev.groupby("protein_id").reaction_id.apply(lambda x: sorted(set(map(str, x)))).to_dict()
    query_proteins = sorted(positives)
    if query_file is not None:
        requested = [x.strip() for x in query_file.read_text().splitlines() if x.strip()]
        missing = sorted(set(requested) - set(query_proteins))
        if missing:
            raise ValueError(f"query-file proteins absent from fold dev query set: {missing[:5]}")
        requested_set = set(requested)
        query_proteins = [x for x in query_proteins if x in requested_set]
    if max_queries is not None:
        query_proteins = query_proteins[: int(max_queries)]
    train_by_protein = {
        str(p): {rindex[r] for r in g.reaction_id.astype(str) if r in rindex}
        for p, g in train.groupby("protein_id")
    }
    train_proteins = sorted(p for p in train_by_protein if p in pindex)
    train_protein_rows = np.asarray([pindex[p] for p in train_proteins], dtype=np.int64)
    train_protein_by_row = {pindex[p]: p for p in train_proteins}

    effective_rk = min(
        len(reaction_ids), int(reaction_chart_k) if reaction_chart_k > 0 else int(math.ceil(math.sqrt(len(reaction_ids))))
    )
    effective_pk = min(
        len(train_proteins), int(protein_neighbours) if protein_neighbours > 0 else int(math.ceil(math.sqrt(len(train_proteins))))
    )

    all_reaction_rows = np.arange(len(reaction_ids), dtype=np.int64)
    base_records: list[dict[str, object]] = []
    flow_records: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    for qi, protein in enumerate(query_proteins):
        prow = pindex[protein]
        # SAME scalar prior as R2E: every reaction row is standardized against the
        # full protein population. Holding e fixed reads a column; it does not
        # re-standardize over reaction candidates.
        full_base = _local_prior(
            reaction_ids, all_reaction_rows, np.asarray([prow], dtype=np.int64),
            pe0=pe0, pe1=pe1, re0=re0, re1=re1,
            p0_mean=p0_mean, p0_cov=p0_cov, p1_mean=p1_mean, p1_cov=p1_cov,
            clip_pt=clip_pt, clip_lookup=clip_lookup, clip_rmat=clip_rmat, clip_ridx=clip_ridx,
            clip_mean=clip_mean, clip_cov=clip_cov,
        )[:, 0]
        base_order = np.lexsort((rlex, -full_base)).astype(np.int32)
        base_inv = np.empty(len(reaction_ids), dtype=np.int32)
        base_inv[base_order] = np.arange(1, len(reaction_ids)+1, dtype=np.int32)
        positive_rows = np.asarray([rindex[r] for r in positives[protein]], dtype=np.int32)
        base_records.append({"fold":fold,"query_id":protein,**evaluate_full_candidate_ranks(base_inv[positive_rows], len(reaction_ids))})

        reaction_chart = np.asarray(base_order[:effective_rk], dtype=np.int32)
        neighbour_rows, protein_chart_info = _protein_query_neighbours(
            prow, train_protein_rows, k=effective_pk, esmc=esmc,
            clip_pt=clip_pt, clip_lookup=clip_lookup, device=device,
        )
        # Query protein first, mirroring target reaction first in R2E.
        local_protein_rows = np.concatenate([np.asarray([prow],dtype=np.int64), neighbour_rows])
        local_protein_ids = [protein, *[train_protein_by_row[int(x)] for x in neighbour_rows]]
        prior = _local_prior(
            [reaction_ids[int(r)] for r in reaction_chart], reaction_chart, local_protein_rows,
            pe0=pe0, pe1=pe1, re0=re0, re1=re1,
            p0_mean=p0_mean, p0_cov=p0_cov, p1_mean=p1_mean, p1_cov=p1_cov,
            clip_pt=clip_pt, clip_lookup=clip_lookup, clip_rmat=clip_rmat, clip_ridx=clip_ridx,
            clip_mean=clip_mean, clip_cov=clip_cov,
        )
        # Exact target-fibre consistency check against full F0.
        analytic_error = float(np.max(np.abs(prior[:,0] - full_base[reaction_chart])))

        local_reaction_distances=[]; local_reaction_availability=[]
        for view in reaction_views:
            d,a=binary_jaccard_distance(view[reaction_chart]); local_reaction_distances.append(d); local_reaction_availability.append(a)
        reaction_affinity, reaction_kernel_info = resolution_product_affinity(local_reaction_distances, local_reaction_availability)
        reaction_graph = degree_normalized_affinity(reaction_affinity)

        protein_similarity, structure_available = _protein_similarity_block(
            local_protein_rows, esmc=esmc, clip_pt=clip_pt, clip_lookup=clip_lookup, device=device
        )
        ambient_protein_affinity = self_tuning_cosine_knn_affinity(protein_similarity)
        conformal_protein_affinity, protein_conformal_info = diffusion_conformal_affinity(ambient_protein_affinity)
        protein_graphs=[degree_normalized_affinity(conformal_protein_affinity)]

        observation_reaction_rows = np.asarray(sorted({r for pid in local_protein_ids[1:] for r in train_by_protein[pid]}), dtype=np.int64)
        projected = _project_reaction_atoms_to_chart(observation_reaction_rows, reaction_chart, reaction_views, device=device)
        projected_lookup={int(r):i for i,r in enumerate(observation_reaction_rows)}
        positive_measure=np.zeros_like(prior,dtype=np.float64)
        observation_count=0
        for j,pid in enumerate(local_protein_ids[1:],start=1):
            rows=[projected_lookup[int(r)] for r in train_by_protein[pid]]
            observation_count += len(rows)
            if rows:
                positive_measure[:,j]=projected[rows].sum(axis=0)

        empirical=positive_measure.copy()
        reaction_degree=empirical.sum(axis=1); protein_degree=empirical.sum(axis=0)
        denom=np.sqrt(np.maximum(reaction_degree[:,None]*protein_degree[None,:],0.0))
        empirical=np.divide(empirical,denom,out=np.zeros_like(empirical),where=denom>0)
        total=float(empirical.sum())
        if total>0: empirical/=total
        scale=product_characteristic_scale(prior,reaction_graph,protein_graphs)
        transported=symmetrized_anisotropic_product_pushforward(
            prior,empirical,reaction_graph,protein_graphs[0],scale=scale
        )
        flowed,info=anisotropic_bregman_product_field(
            prior,reaction_graph,protein_graphs,strength=1.0,
            positive_measure=transported,positive_strength=1.0,
            normalize_positive_rows=False,integral_normalized=True,accelerated=True,
            max_steps=max_steps,tolerance=1e-5,
        )
        flow_scores=full_base.copy(); flow_scores[reaction_chart]=flowed[:,0]
        flow_order=np.lexsort((rlex,-flow_scores)).astype(np.int32)
        flow_inv=np.empty(len(reaction_ids),dtype=np.int32); flow_inv[flow_order]=np.arange(1,len(reaction_ids)+1,dtype=np.int32)
        flow_records.append({"fold":fold,"query_id":protein,**evaluate_full_candidate_ranks(flow_inv[positive_rows],len(reaction_ids))})
        reaction_pos = {int(row): int(i) for i, row in enumerate(reaction_chart)}
        positive_nodal = []
        for rid in positives[protein]:
            rr = int(rindex[rid])
            if rr in reaction_pos:
                ci = reaction_pos[rr]
                positive_nodal.append({
                    "reaction_id": str(rid),
                    "prior_score": float(prior[ci, 0]),
                    "flow_score": float(flowed[ci, 0]),
                    "delta": float(flowed[ci, 0] - prior[ci, 0]),
                    "chart_index": int(ci),
                })
        diagnostics.append({
            "fold":fold,"query_id":protein,"reaction_chart_size":len(reaction_chart),
            "protein_neighbours":len(neighbour_rows),"protein_chart_size":len(local_protein_rows),
            "observation_reaction_atoms":len(observation_reaction_rows),"observation_count":observation_count,
            "target_transported_positive_mass":float(transported[:,0].sum()),
            "query_structure_observed":protein_chart_info["query_structure_observed"],
            "nearest_train_protein_similarity":protein_chart_info["nearest_combined_similarity"],
            "furthest_protein_chart_similarity":protein_chart_info["furthest_chart_combined_similarity"],
            "reaction_graph_mean_view_resolution":json.dumps(reaction_kernel_info.get("mean_view_resolution",[])),
            "protein_conformal_edges":int(protein_conformal_info["edges"]),
            "analytic_base_max_abs_error":analytic_error,
            "flow_steps":int(info["steps"]),
            "final_relative_change":float(info["relative_change_history"][-1]) if info["relative_change_history"] else 0.0,
            "characteristic_scale":float(info["characteristic_scale"]),
            "positive_nodal_scores_json":json.dumps(positive_nodal,sort_keys=True),
            "top10_predicted_reactions":json.dumps([reaction_ids[int(x)] for x in flow_order[:10]]),
        })
        if (qi+1)%8==0 or qi+1==len(query_proteins):
            print(f"same-field E2R fold={fold} {qi+1}/{len(query_proteins)}",flush=True)

    b=pd.DataFrame(base_records); f=pd.DataFrame(flow_records); d=pd.DataFrame(diagnostics)
    suffix="full" if max_queries is None else f"q{len(query_proteins)}"
    out=OUT/f"fold{fold}"/suffix; out.mkdir(parents=True,exist_ok=True)
    b.to_csv(out/'base_query_metrics.csv',index=False); f.to_csv(out/'flow_query_metrics.csv',index=False); d.to_csv(out/'diagnostics.csv',index=False)
    result={
        "method":"e2r_section_consistency_probe_v1",
        "direction":"E2R",
        "role":"bidirectional completeness audit on exactly the current R2E clean double-cold pairs; not a comparison to historical E2R splits",
        "fold":fold,"queries":len(query_proteins),"reaction_candidates":len(reaction_ids),
        "reaction_chart_k":effective_rk,"protein_neighbours":effective_pk,
        "same_prior_definition_as_r2e":True,"e2r_specific_model":False,"e2r_specific_ranker":False,"e2r_specific_router":False,
        "base_metrics":_metric_map(b),"flow_metrics":_metric_map(f),
        "max_analytic_base_error":float(d.analytic_base_max_abs_error.max()),
        "converged_queries":int((d.final_relative_change<1e-5).sum()),
        "hit_max_steps":int((d.flow_steps>=max_steps).sum()),
        "train_dev_reaction_overlap":0,"train_dev_protein_overlap":0,"train_dev_pair_overlap":0,
        "labels_used_for_scoring":False,"external_metrics_used":False,
        "reaction_view_dimensions":reaction_dims,
    }
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True, choices=(0, 1, 2))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--reaction-chart-k', type=int, default=0)
    ap.add_argument('--protein-neighbours', type=int, default=0)
    ap.add_argument('--max-steps', type=int, default=128)
    ap.add_argument('--max-queries', type=int, default=None)
    ap.add_argument('--query-file', type=Path, default=None)
    a = ap.parse_args()
    print(json.dumps(evaluate_fold(
        a.fold, a.device, a.reaction_chart_k, a.protein_neighbours,
        a.max_steps, a.max_queries, query_file=a.query_file,
    ), indent=2), flush=True)

if __name__=='__main__': main()
